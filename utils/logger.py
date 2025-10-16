#!/usr/bin/env python3
"""
Enhanced Logging System for Aura Render

Provides structured logging with performance tracking, error correlation,
and advanced debugging capabilities.
"""

import logging
import json
import time
import traceback
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
import contextvars
import functools
import sys
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import os


class LogLevel(Enum):
    """Enhanced log levels"""
    TRACE = 5
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogCategory(Enum):
    """Log categories for better organization"""
    SYSTEM = "system"
    API = "api" 
    RENDERING = "rendering"
    AI = "ai"
    MATERIALS = "materials"
    USER = "user"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DATABASE = "database"


@dataclass
class LogContext:
    """Log context information"""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    category: LogCategory = LogCategory.SYSTEM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self).items() if v is not None}


@dataclass
class PerformanceMetrics:
    """Performance tracking data"""
    start_time: float
    end_time: Optional[float] = None
    cpu_usage_start: Optional[float] = None
    cpu_usage_end: Optional[float] = None
    memory_usage_start: Optional[int] = None
    memory_usage_end: Optional[int] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def cpu_delta(self) -> Optional[float]:
        """Calculate CPU usage delta"""
        if self.cpu_usage_start and self.cpu_usage_end:
            return self.cpu_usage_end - self.cpu_usage_start
        return None
    
    @property
    def memory_delta(self) -> Optional[int]:
        """Calculate memory usage delta"""
        if self.memory_usage_start and self.memory_usage_end:
            return self.memory_usage_end - self.memory_usage_start
        return None


# Context variables for request tracking
log_context: contextvars.ContextVar[LogContext] = contextvars.ContextVar('log_context', default=LogContext())
performance_tracker: contextvars.ContextVar[Dict[str, PerformanceMetrics]] = contextvars.ContextVar('performance_tracker', default={})


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter"""
    
    def __init__(self, include_performance: bool = True):
        super().__init__()
        self.include_performance = include_performance
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        # Get current context
        context = log_context.get()
        
        # Base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": threading.current_thread().name,
            "process": os.getpid()
        }
        
        # Add context information
        log_data.update(context.to_dict())
        
        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        # Add performance metrics if available
        if self.include_performance:
            tracker = performance_tracker.get()
            if tracker:
                log_data["performance_context"] = {
                    name: {
                        "duration": metrics.duration,
                        "cpu_delta": metrics.cpu_delta,
                        "memory_delta": metrics.memory_delta
                    } for name, metrics in tracker.items() if metrics.duration is not None
                }
        
        return json.dumps(log_data, ensure_ascii=False, separators=(',', ':'))


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for human-readable output"""
    
    # Color codes
    COLORS = {
        'TRACE': '\033[90m',      # Dark gray
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[91m',   # Bright red
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, show_context: bool = True, show_performance: bool = False):
        super().__init__()
        self.show_context = show_context
        self.show_performance = show_performance
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors and context"""
        context = log_context.get()
        
        # Color the level
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        colored_level = f"{level_color}{record.levelname:<8}{self.COLORS['RESET']}"
        
        # Base format
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        base_msg = f"[{timestamp}] {colored_level} {record.name:<20} | {record.getMessage()}"
        
        # Add context information
        if self.show_context and any(v for v in context.to_dict().values()):
            context_parts = []
            ctx_dict = context.to_dict()
            
            if ctx_dict.get('request_id'):
                context_parts.append(f"req:{ctx_dict['request_id'][:8]}")
            if ctx_dict.get('user_id'):
                context_parts.append(f"user:{ctx_dict['user_id']}")
            if ctx_dict.get('operation'):
                context_parts.append(f"op:{ctx_dict['operation']}")
            if ctx_dict.get('component'):
                context_parts.append(f"comp:{ctx_dict['component']}")
            
            if context_parts:
                context_str = f" [{' | '.join(context_parts)}]"
                base_msg += f"\033[90m{context_str}\033[0m"
        
        # Add location info for errors
        if record.levelno >= logging.ERROR:
            location = f" ({record.filename}:{record.lineno})"
            base_msg += f"\033[90m{location}\033[0m"
        
        # Add exception information
        if record.exc_info:
            base_msg += f"\n{traceback.format_exception(*record.exc_info)[-1].strip()}"
        
        # Add performance info
        if self.show_performance:
            tracker = performance_tracker.get()
            if tracker:
                perf_info = []
                for name, metrics in tracker.items():
                    if metrics.duration:
                        perf_info.append(f"{name}:{metrics.duration:.3f}s")
                
                if perf_info:
                    perf_str = f" [⏱️  {' | '.join(perf_info)}]"
                    base_msg += f"\033[90m{perf_str}\033[0m"
        
        return base_msg


class AuraLogger:
    """Enhanced logger with context and performance tracking"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_done = False
    
    def _ensure_setup(self):
        """Ensure logger is set up (lazy initialization)"""
        if not self._setup_done:
            LoggerManager.setup_logger(self.logger.name)
            self._setup_done = True
    
    def with_context(self, **kwargs) -> 'AuraLogger':
        """Create logger with additional context"""
        current_context = log_context.get()
        new_context = LogContext(
            request_id=kwargs.get('request_id', current_context.request_id),
            user_id=kwargs.get('user_id', current_context.user_id),
            session_id=kwargs.get('session_id', current_context.session_id),
            operation=kwargs.get('operation', current_context.operation),
            component=kwargs.get('component', current_context.component),
            category=kwargs.get('category', current_context.category)
        )
        
        # Set context for this logger
        log_context.set(new_context)
        return self
    
    def trace(self, msg: str, **kwargs):
        """Log trace message"""
        self._ensure_setup()
        self.logger.log(LogLevel.TRACE.value, msg, extra=kwargs)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message"""
        self._ensure_setup()
        self.logger.debug(msg, extra=kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message"""
        self._ensure_setup()
        self.logger.info(msg, extra=kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message"""
        self._ensure_setup()
        self.logger.warning(msg, extra=kwargs)
    
    def error(self, msg: str, exc_info: bool = True, **kwargs):
        """Log error message"""
        self._ensure_setup()
        self.logger.error(msg, exc_info=exc_info, extra=kwargs)
    
    def critical(self, msg: str, exc_info: bool = True, **kwargs):
        """Log critical message"""
        self._ensure_setup()
        self.logger.critical(msg, exc_info=exc_info, extra=kwargs)
    
    def performance(self, msg: str, duration: float, **kwargs):
        """Log performance metric"""
        self._ensure_setup()
        perf_data = {"duration": duration, "metric_type": "performance"}
        perf_data.update(kwargs)
        self.logger.info(f"⏱️  {msg}", extra=perf_data)


class PerformanceTracker:
    """Context manager for performance tracking"""
    
    def __init__(self, operation_name: str, logger: Optional[AuraLogger] = None):
        self.operation_name = operation_name
        self.logger = logger
        self.metrics = PerformanceMetrics(start_time=time.time())
    
    def __enter__(self):
        """Start tracking"""
        # Record initial system metrics
        try:
            process = psutil.Process()
            self.metrics.cpu_usage_start = process.cpu_percent()
            self.metrics.memory_usage_start = process.memory_info().rss
        except:
            pass
        
        # Add to context
        tracker = performance_tracker.get()
        tracker[self.operation_name] = self.metrics
        performance_tracker.set(tracker)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End tracking"""
        self.metrics.end_time = time.time()
        
        # Record final system metrics
        try:
            process = psutil.Process()
            self.metrics.cpu_usage_end = process.cpu_percent()
            self.metrics.memory_usage_end = process.memory_info().rss
        except:
            pass
        
        # Log performance if logger provided
        if self.logger and self.metrics.duration:
            perf_msg = f"{self.operation_name} completed"
            perf_kwargs = {
                "cpu_delta": self.metrics.cpu_delta,
                "memory_delta": self.metrics.memory_delta
            }
            self.logger.performance(perf_msg, self.metrics.duration, **perf_kwargs)


class LoggerManager:
    """Centralized logger management"""
    
    _instance = None
    _setup_done = False
    
    def __init__(self):
        self.loggers: Dict[str, AuraLogger] = {}
        self.handlers: List[logging.Handler] = []
    
    @classmethod
    def get_instance(cls) -> 'LoggerManager':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def setup(cls, log_dir: Path, log_level: Union[str, int] = logging.INFO,
              enable_console: bool = True, enable_json: bool = True,
              enable_performance: bool = True, max_file_size: int = 100 * 1024 * 1024):
        """Setup logging system"""
        if cls._setup_done:
            return
        
        instance = cls.get_instance()
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add TRACE level
        logging.addLevelName(LogLevel.TRACE.value, "TRACE")
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(LogLevel.TRACE.value)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler (colored, human-readable)
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(ColoredConsoleFormatter(
                show_context=True,
                show_performance=enable_performance
            ))
            instance.handlers.append(console_handler)
            root_logger.addHandler(console_handler)
        
        # Main log file (human-readable text format)
        from logging.handlers import RotatingFileHandler

        main_handler = RotatingFileHandler(
            log_dir / "aura_render.log",
            maxBytes=max_file_size,
            backupCount=5,
            encoding='utf-8'
        )
        main_handler.setLevel(log_level)
        main_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        instance.handlers.append(main_handler)
        root_logger.addHandler(main_handler)

        # Error log file (errors and above)
        error_handler = RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=max_file_size // 2,
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        instance.handlers.append(error_handler)
        root_logger.addHandler(error_handler)
        
        cls._setup_done = True
    
    @classmethod
    def setup_logger(cls, name: str) -> None:
        """Setup individual logger (called lazily)"""
        # Individual logger configuration is handled by global setup
        pass
    
    @classmethod
    def get_logger(cls, name: str) -> AuraLogger:
        """Get or create logger"""
        instance = cls.get_instance()
        
        if name not in instance.loggers:
            instance.loggers[name] = AuraLogger(name)
        
        return instance.loggers[name]
    
    @classmethod
    def set_log_level(cls, level: Union[str, int]):
        """Change log level for all handlers"""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        for handler in root_logger.handlers:
            if handler.level > logging.ERROR:  # Don't change error handlers
                continue
            handler.setLevel(level)


def get_logger(name: str) -> AuraLogger:
    """Get logger instance"""
    return LoggerManager.get_logger(name)


def setup_logging(log_dir: Path, log_level: Union[str, int] = logging.INFO, **kwargs):
    """Setup logging system"""
    LoggerManager.setup(log_dir, log_level, **kwargs)


def performance_track(operation_name: str):
    """Decorator for automatic performance tracking"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(f"{func.__module__}.{func.__name__}")
            with PerformanceTracker(operation_name, logger):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(f"{func.__module__}.{func.__name__}")
            with PerformanceTracker(operation_name, logger):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def log_context_manager(**context_kwargs):
    """Context manager for setting log context"""
    class LogContextManager:
        def __init__(self, **kwargs):
            self.new_context = LogContext(**kwargs)
            self.token = None
        
        def __enter__(self):
            self.token = log_context.set(self.new_context)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.token:
                log_context.reset(self.token)
    
    return LogContextManager(**context_kwargs)


# Convenience function for common use case
def with_request_context(request_id: str, user_id: Optional[str] = None, operation: Optional[str] = None):
    """Set request context for logging"""
    return log_context_manager(
        request_id=request_id,
        user_id=user_id,
        operation=operation
    )