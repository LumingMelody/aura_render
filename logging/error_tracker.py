"""
Error Tracking and Logging System
错误追踪和日志系统 - 提供统一的错误处理、追踪和分析功能
"""
import asyncio
import logging
import traceback
import sys
import os
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque
import threading
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
import inspect

from cache.redis_cache_manager import get_cache_manager


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ErrorCategory(Enum):
    """错误分类"""
    SYSTEM = "system"           # 系统错误
    APPLICATION = "application" # 应用错误
    DATABASE = "database"       # 数据库错误
    NETWORK = "network"         # 网络错误
    VALIDATION = "validation"   # 验证错误
    PERMISSION = "permission"   # 权限错误
    RESOURCE = "resource"       # 资源错误
    TIMEOUT = "timeout"         # 超时错误
    UNKNOWN = "unknown"         # 未知错误


@dataclass
class ErrorContext:
    """错误上下文信息"""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """错误记录"""
    error_id: str
    timestamp: datetime
    level: LogLevel
    category: ErrorCategory
    message: str
    exception_type: str
    exception_message: str
    traceback_text: str
    module: str
    function: str
    line_number: int
    context: ErrorContext
    tags: List[str] = field(default_factory=list)
    count: int = 1
    first_seen: datetime = None
    last_seen: datetime = None

    def __post_init__(self):
        if self.first_seen is None:
            self.first_seen = self.timestamp
        if self.last_seen is None:
            self.last_seen = self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["first_seen"] = self.first_seen.isoformat()
        result["last_seen"] = self.last_seen.isoformat()
        result["level"] = self.level.value
        result["category"] = self.category.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorRecord':
        """从字典创建实例"""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["first_seen"] = datetime.fromisoformat(data["first_seen"])
        data["last_seen"] = datetime.fromisoformat(data["last_seen"])
        data["level"] = LogLevel(data["level"])
        data["category"] = ErrorCategory(data["category"])
        data["context"] = ErrorContext(**data["context"])
        return cls(**data)


@dataclass
class LogEntry:
    """日志条目"""
    log_id: str
    timestamp: datetime
    level: LogLevel
    message: str
    module: str
    function: str
    line_number: int
    context: Optional[ErrorContext] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["level"] = self.level.value
        if self.context:
            result["context"] = asdict(self.context)
        return result


class CustomLogHandler(logging.Handler):
    """自定义日志处理器"""

    def __init__(self, error_tracker):
        super().__init__()
        self.error_tracker = error_tracker

    def emit(self, record):
        """处理日志记录"""
        try:
            # 获取调用栈信息
            frame = sys._getframe()
            while frame:
                if frame.f_code.co_filename != __file__:
                    module = frame.f_globals.get('__name__', 'unknown')
                    function = frame.f_code.co_name
                    line_number = frame.f_lineno
                    break
                frame = frame.f_back
            else:
                module = 'unknown'
                function = 'unknown'
                line_number = 0

            # 创建日志条目
            log_entry = LogEntry(
                log_id=str(uuid.uuid4()),
                timestamp=datetime.fromtimestamp(record.created),
                level=LogLevel(record.levelname),
                message=record.getMessage(),
                module=module,
                function=function,
                line_number=line_number,
                extra_data=getattr(record, 'extra_data', {})
            )

            # 如果是错误级别，创建错误记录
            if record.levelno >= logging.ERROR:
                if hasattr(record, 'exc_info') and record.exc_info:
                    self.error_tracker._create_error_from_log(record, log_entry)

            # 记录日志
            asyncio.create_task(self.error_tracker._store_log_entry(log_entry))

        except Exception:
            self.handleError(record)


class ErrorTracker:
    """错误追踪器"""

    def __init__(self,
                 log_directory: str = "/tmp/aura_render_logs",
                 max_log_files: int = 10,
                 max_file_size_mb: int = 100,
                 retention_days: int = 30):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        self.max_log_files = max_log_files
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.retention_period = timedelta(days=retention_days)

        # 错误存储
        self.error_records: Dict[str, ErrorRecord] = {}
        self.error_patterns: Dict[str, List[str]] = defaultdict(list)
        self.recent_logs: deque = deque(maxlen=1000)

        # 错误统计
        self.error_counts: Dict[ErrorCategory, int] = defaultdict(int)
        self.hourly_error_counts: Dict[str, int] = defaultdict(int)

        # 回调函数
        self.error_handlers: List[Callable] = []

        # 缓存
        self.cache = get_cache_manager()

        # 配置日志记录器
        self.logger = self._setup_logger()

        # 启动清理任务
        self.cleanup_task = None

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("aura_render")
        logger.setLevel(logging.DEBUG)

        # 移除已存在的处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # 文件处理器 - 详细日志
        log_file = self.log_directory / "application.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size_bytes,
            backupCount=self.max_log_files
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        # 错误文件处理器 - 仅错误
        error_file = self.log_directory / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=self.max_file_size_bytes,
            backupCount=self.max_log_files
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # 自定义处理器
        custom_handler = CustomLogHandler(self)
        custom_handler.setLevel(logging.DEBUG)

        logger.addHandler(file_handler)
        logger.addHandler(error_handler)
        logger.addHandler(console_handler)
        logger.addHandler(custom_handler)

        return logger

    def start_cleanup_task(self):
        """启动清理任务"""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    def stop_cleanup_task(self):
        """停止清理任务"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            self.cleanup_task = None

    async def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # 每小时清理一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(3600)

    async def _cleanup_old_data(self):
        """清理过期数据"""
        cutoff_time = datetime.now() - self.retention_period

        # 清理内存中的错误记录
        expired_errors = [
            error_id for error_id, error in self.error_records.items()
            if error.last_seen < cutoff_time
        ]

        for error_id in expired_errors:
            del self.error_records[error_id]

        # 清理最近日志
        while self.recent_logs and self.recent_logs[0].timestamp < cutoff_time:
            self.recent_logs.popleft()

        # 清理小时统计
        current_hour = datetime.now().strftime("%Y%m%d%H")
        expired_hours = [
            hour for hour in self.hourly_error_counts.keys()
            if hour < current_hour and
               datetime.strptime(hour, "%Y%m%d%H") < cutoff_time
        ]

        for hour in expired_hours:
            del self.hourly_error_counts[hour]

        if expired_errors or expired_hours:
            self.logger.info(f"Cleaned up {len(expired_errors)} error records and {len(expired_hours)} hourly stats")

    def track_error(self,
                   exception: Exception,
                   category: ErrorCategory = ErrorCategory.APPLICATION,
                   context: Optional[ErrorContext] = None,
                   tags: Optional[List[str]] = None) -> str:
        """追踪错误"""
        # 获取异常信息
        exc_type = type(exception).__name__
        exc_message = str(exception)
        tb_text = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))

        # 获取调用栈信息
        frame = inspect.currentframe().f_back
        module = frame.f_globals.get('__name__', 'unknown')
        function = frame.f_code.co_name
        line_number = frame.f_lineno

        # 生成错误特征（用于聚合相同错误）
        error_signature = self._generate_error_signature(exc_type, module, function, line_number)

        # 检查是否已存在相同错误
        if error_signature in self.error_records:
            error_record = self.error_records[error_signature]
            error_record.count += 1
            error_record.last_seen = datetime.now()
            error_id = error_record.error_id
        else:
            # 创建新错误记录
            error_id = str(uuid.uuid4())
            error_record = ErrorRecord(
                error_id=error_id,
                timestamp=datetime.now(),
                level=LogLevel.ERROR,
                category=category,
                message=exc_message,
                exception_type=exc_type,
                exception_message=exc_message,
                traceback_text=tb_text,
                module=module,
                function=function,
                line_number=line_number,
                context=context or ErrorContext(),
                tags=tags or []
            )
            self.error_records[error_signature] = error_record

        # 更新统计
        self.error_counts[category] += 1
        hour_key = datetime.now().strftime("%Y%m%d%H")
        self.hourly_error_counts[hour_key] += 1

        # 存储到缓存和日志
        asyncio.create_task(self._store_error_record(error_record))

        # 触发错误处理回调
        asyncio.create_task(self._trigger_error_handlers(error_record))

        # 记录到系统日志
        self.logger.error(f"Error tracked: {exc_message}", exc_info=exception)

        return error_id

    def _generate_error_signature(self, exc_type: str, module: str, function: str, line_number: int) -> str:
        """生成错误特征签名"""
        return f"{exc_type}:{module}:{function}:{line_number}"

    async def _store_error_record(self, error_record: ErrorRecord):
        """存储错误记录"""
        try:
            # 存储到缓存
            cache_key = f"error:{error_record.error_id}"
            await self.cache.set(cache_key, error_record.to_dict(), ttl=86400)  # 24小时

            # 存储到文件（JSON格式，便于分析）
            error_file = self.log_directory / "errors.json"

            # 读取现有错误
            existing_errors = []
            if error_file.exists():
                try:
                    with open(error_file, 'r', encoding='utf-8') as f:
                        existing_errors = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_errors = []

            # 添加新错误或更新现有错误
            error_dict = error_record.to_dict()
            updated = False
            for i, existing_error in enumerate(existing_errors):
                if existing_error.get('error_id') == error_record.error_id:
                    existing_errors[i] = error_dict
                    updated = True
                    break

            if not updated:
                existing_errors.append(error_dict)

            # 限制文件大小（保留最近的1000个错误）
            if len(existing_errors) > 1000:
                existing_errors = existing_errors[-1000:]

            # 写回文件
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(existing_errors, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Failed to store error record: {e}")

    async def _store_log_entry(self, log_entry: LogEntry):
        """存储日志条目"""
        try:
            self.recent_logs.append(log_entry)

            # 存储到文件（结构化日志）
            log_file = self.log_directory / "structured.log"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry.to_dict(), ensure_ascii=False) + '\n')

        except Exception as e:
            print(f"Failed to store log entry: {e}")

    async def _trigger_error_handlers(self, error_record: ErrorRecord):
        """触发错误处理回调"""
        for handler in self.error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error_record)
                else:
                    handler(error_record)
            except Exception as e:
                self.logger.error(f"Error in error handler: {e}")

    def _create_error_from_log(self, record, log_entry: LogEntry):
        """从日志记录创建错误记录"""
        if hasattr(record, 'exc_info') and record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            if exc_value:
                self.track_error(
                    exc_value,
                    category=ErrorCategory.APPLICATION,
                    context=log_entry.context
                )

    def add_error_handler(self, handler: Callable):
        """添加错误处理回调"""
        self.error_handlers.append(handler)

    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """获取错误统计"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # 过滤指定时间范围内的错误
        recent_errors = [
            error for error in self.error_records.values()
            if error.last_seen >= cutoff_time
        ]

        # 按类别统计
        category_stats = defaultdict(int)
        for error in recent_errors:
            category_stats[error.category.value] += error.count

        # 按小时统计
        hourly_stats = {}
        for hour_key, count in self.hourly_error_counts.items():
            hour_time = datetime.strptime(hour_key, "%Y%m%d%H")
            if hour_time >= cutoff_time:
                hourly_stats[hour_key] = count

        # Top错误
        top_errors = sorted(
            recent_errors,
            key=lambda x: x.count,
            reverse=True
        )[:10]

        return {
            "time_range_hours": hours,
            "total_errors": sum(error.count for error in recent_errors),
            "unique_errors": len(recent_errors),
            "category_breakdown": dict(category_stats),
            "hourly_breakdown": hourly_stats,
            "top_errors": [
                {
                    "error_id": error.error_id,
                    "message": error.message,
                    "count": error.count,
                    "category": error.category.value,
                    "first_seen": error.first_seen.isoformat(),
                    "last_seen": error.last_seen.isoformat()
                }
                for error in top_errors
            ]
        }

    def get_recent_logs(self, count: int = 100, level: Optional[LogLevel] = None) -> List[Dict[str, Any]]:
        """获取最近日志"""
        logs = list(self.recent_logs)

        if level:
            logs = [log for log in logs if log.level == level]

        # 按时间倒序排列
        logs.sort(key=lambda x: x.timestamp, reverse=True)

        return [log.to_dict() for log in logs[:count]]

    def search_errors(self,
                     query: Optional[str] = None,
                     category: Optional[ErrorCategory] = None,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """搜索错误"""
        results = []

        for error in self.error_records.values():
            # 时间过滤
            if start_time and error.last_seen < start_time:
                continue
            if end_time and error.first_seen > end_time:
                continue

            # 类别过滤
            if category and error.category != category:
                continue

            # 文本搜索
            if query:
                search_text = f"{error.message} {error.exception_message} {error.module} {error.function}".lower()
                if query.lower() not in search_text:
                    continue

            results.append(error.to_dict())

        # 按最后发生时间排序
        results.sort(key=lambda x: x['last_seen'], reverse=True)
        return results

    async def export_error_report(self, output_path: str, hours: int = 24):
        """导出错误报告"""
        try:
            statistics = self.get_error_statistics(hours)
            recent_logs = self.get_recent_logs(200, LogLevel.ERROR)

            report = {
                "generated_at": datetime.now().isoformat(),
                "statistics": statistics,
                "recent_error_logs": recent_logs,
                "system_info": {
                    "log_directory": str(self.log_directory),
                    "retention_days": self.retention_period.days,
                    "active_error_records": len(self.error_records),
                    "recent_log_entries": len(self.recent_logs)
                }
            }

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Error report exported to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export error report: {e}")
            raise

    # 装饰器和上下文管理器

    def track_exceptions(self, category: ErrorCategory = ErrorCategory.APPLICATION):
        """异常追踪装饰器"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    self.track_error(e, category)
                    raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.track_error(e, category)
                    raise

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        return decorator

    @asynccontextmanager
    async def error_context(self, context: ErrorContext):
        """错误上下文管理器"""
        # 设置上下文到线程局部存储
        import threading
        local = threading.local()
        original_context = getattr(local, 'error_context', None)
        local.error_context = context

        try:
            yield
        except Exception as e:
            self.track_error(e, context=context)
            raise
        finally:
            local.error_context = original_context


# 全局错误追踪器实例
_global_error_tracker: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """获取全局错误追踪器实例"""
    global _global_error_tracker
    if _global_error_tracker is None:
        _global_error_tracker = ErrorTracker()
        _global_error_tracker.start_cleanup_task()
    return _global_error_tracker


# 便捷装饰器
def track_errors(category: ErrorCategory = ErrorCategory.APPLICATION):
    """错误追踪装饰器（使用全局追踪器）"""
    tracker = get_error_tracker()
    return tracker.track_exceptions(category)


# 便捷日志函数
def log_info(message: str, **kwargs):
    """记录信息日志"""
    tracker = get_error_tracker()
    tracker.logger.info(message, extra={'extra_data': kwargs})


def log_warning(message: str, **kwargs):
    """记录警告日志"""
    tracker = get_error_tracker()
    tracker.logger.warning(message, extra={'extra_data': kwargs})


def log_error(message: str, exception: Optional[Exception] = None, **kwargs):
    """记录错误日志"""
    tracker = get_error_tracker()
    if exception:
        tracker.logger.error(message, exc_info=exception, extra={'extra_data': kwargs})
    else:
        tracker.logger.error(message, extra={'extra_data': kwargs})


# 测试代码
async def test_error_tracker():
    """测试错误追踪系统"""
    print("🐛 测试错误追踪和日志系统")
    print("=" * 50)

    tracker = ErrorTracker(log_directory="/tmp/aura_render_logs")

    # 添加错误处理回调
    def error_alert_handler(error_record):
        print(f"🚨 Error Alert: {error_record.message} ({error_record.count} times)")

    tracker.add_error_handler(error_alert_handler)

    try:
        # 启动清理任务
        tracker.start_cleanup_task()

        # 测试基本日志记录
        log_info("System started", component="main")
        log_warning("High memory usage detected", memory_usage=85.5)

        # 测试错误追踪
        try:
            raise ValueError("Test validation error")
        except Exception as e:
            context = ErrorContext(
                request_id="req_123",
                user_id="user_456",
                endpoint="/api/test"
            )
            tracker.track_error(e, ErrorCategory.VALIDATION, context, ["test", "validation"])

        # 测试装饰器
        @tracker.track_exceptions(ErrorCategory.APPLICATION)
        def test_function():
            raise RuntimeError("Test runtime error")

        try:
            test_function()
        except Exception:
            pass

        # 模拟相同错误多次发生
        for i in range(3):
            try:
                raise ConnectionError("Database connection failed")
            except Exception as e:
                tracker.track_error(e, ErrorCategory.DATABASE)

        # 等待一下让异步操作完成
        await asyncio.sleep(2)

        # 获取错误统计
        stats = tracker.get_error_statistics(24)
        print(f"\n📊 错误统计:")
        print(f"  总错误数: {stats['total_errors']}")
        print(f"  唯一错误数: {stats['unique_errors']}")
        print(f"  类别分布: {stats['category_breakdown']}")

        print(f"\n🔥 Top错误:")
        for error in stats['top_errors'][:3]:
            print(f"  - {error['message'][:50]}... ({error['count']}次)")

        # 获取最近日志
        recent_logs = tracker.get_recent_logs(10)
        print(f"\n📋 最近日志数: {len(recent_logs)}")

        # 导出错误报告
        report_path = "/tmp/aura_render_outputs/error_report.json"
        await tracker.export_error_report(report_path, 24)
        print(f"📄 错误报告已导出: {report_path}")

        print("\n✅ 错误追踪和日志系统测试完成")

    finally:
        tracker.stop_cleanup_task()


if __name__ == "__main__":
    # 需要导入logging.handlers
    import logging.handlers
    asyncio.run(test_error_tracker())