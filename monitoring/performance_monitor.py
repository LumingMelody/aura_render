"""
性能监控系统 - 实时监控系统性能、资源使用和任务状态
"""
from typing import Dict, List, Any, Optional, Callable
import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json
import statistics

from database.database_manager import DatabaseManager
from cache.cache_manager import CacheManager


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"         # 计数器
    GAUGE = "gauge"            # 瞬时值
    HISTOGRAM = "histogram"    # 直方图
    TIMER = "timer"           # 计时器


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """性能指标"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class Alert:
    """告警信息"""
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold_value: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class SystemHealth:
    """系统健康状态"""
    overall_status: str         # healthy, warning, critical
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    response_time_p95: float
    error_rate: float
    timestamp: datetime


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.collectors: Dict[str, Callable] = {}
        self.collection_interval = 5  # 秒
        self.running = False
        self.collection_task: Optional[asyncio.Task] = None

    def register_collector(self, name: str, collector_func: Callable):
        """注册指标收集器"""
        self.collectors[name] = collector_func
        print(f"✅ Metric collector registered: {name}")

    async def start_collection(self):
        """开始指标收集"""
        if self.running:
            return

        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        print("🚀 Metrics collection started")

    async def stop_collection(self):
        """停止指标收集"""
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        print("🛑 Metrics collection stopped")

    async def _collection_loop(self):
        """指标收集循环"""
        while self.running:
            try:
                # 收集系统指标
                await self._collect_system_metrics()

                # 收集注册的自定义指标
                for name, collector in self.collectors.items():
                    try:
                        if asyncio.iscoroutinefunction(collector):
                            metrics = await collector()
                        else:
                            metrics = collector()

                        if isinstance(metrics, list):
                            for metric in metrics:
                                self.add_metric(metric)
                        elif isinstance(metrics, Metric):
                            self.add_metric(metrics)

                    except Exception as e:
                        print(f"❌ Error collecting metrics from {name}: {e}")

                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)

    async def _collect_system_metrics(self):
        """收集系统基础指标"""
        timestamp = datetime.now()

        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=None)
        self.add_metric(Metric(
            name="system.cpu.usage_percent",
            type=MetricType.GAUGE,
            value=cpu_percent,
            timestamp=timestamp,
            description="CPU usage percentage"
        ))

        # 内存使用
        memory = psutil.virtual_memory()
        self.add_metric(Metric(
            name="system.memory.usage_percent",
            type=MetricType.GAUGE,
            value=memory.percent,
            timestamp=timestamp,
            description="Memory usage percentage"
        ))

        self.add_metric(Metric(
            name="system.memory.available_gb",
            type=MetricType.GAUGE,
            value=memory.available / (1024**3),
            timestamp=timestamp,
            description="Available memory in GB"
        ))

        # 磁盘使用
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.add_metric(Metric(
            name="system.disk.usage_percent",
            type=MetricType.GAUGE,
            value=disk_percent,
            timestamp=timestamp,
            description="Disk usage percentage"
        ))

        # 网络IO
        net_io = psutil.net_io_counters()
        self.add_metric(Metric(
            name="system.network.bytes_sent",
            type=MetricType.COUNTER,
            value=net_io.bytes_sent,
            timestamp=timestamp,
            description="Network bytes sent"
        ))

        self.add_metric(Metric(
            name="system.network.bytes_recv",
            type=MetricType.COUNTER,
            value=net_io.bytes_recv,
            timestamp=timestamp,
            description="Network bytes received"
        ))

    def add_metric(self, metric: Metric):
        """添加指标"""
        self.metrics[metric.name].append(metric)

    def get_metric_history(self, metric_name: str, duration_minutes: int = 60) -> List[Metric]:
        """获取指标历史"""
        if metric_name not in self.metrics:
            return []

        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [
            metric for metric in self.metrics[metric_name]
            if metric.timestamp >= cutoff_time
        ]

    def get_latest_metric(self, metric_name: str) -> Optional[Metric]:
        """获取最新指标值"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        return self.metrics[metric_name][-1]

    def calculate_metric_stats(self, metric_name: str, duration_minutes: int = 60) -> Dict[str, float]:
        """计算指标统计信息"""
        history = self.get_metric_history(metric_name, duration_minutes)
        if not history:
            return {}

        values = [metric.value for metric in history]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'p95': self._percentile(values, 95),
            'p99': self._percentile(values, 99)
        }

    def _percentile(self, values: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        values_sorted = sorted(values)
        index = (percentile / 100) * (len(values_sorted) - 1)
        if index.is_integer():
            return values_sorted[int(index)]
        else:
            lower = values_sorted[int(index)]
            upper = values_sorted[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


class AlertManager:
    """告警管理器"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, Dict] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_handlers: List[Callable] = []
        self.check_interval = 10  # 秒
        self.running = False
        self.alert_task: Optional[asyncio.Task] = None

    def add_alert_rule(self, rule_id: str, metric_name: str, condition: str,
                      threshold: float, level: AlertLevel, message: str):
        """添加告警规则"""
        self.alert_rules[rule_id] = {
            'metric_name': metric_name,
            'condition': condition,  # >, <, >=, <=, ==
            'threshold': threshold,
            'level': level,
            'message': message,
            'enabled': True
        }
        print(f"✅ Alert rule added: {rule_id}")

    def remove_alert_rule(self, rule_id: str):
        """移除告警规则"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            print(f"✅ Alert rule removed: {rule_id}")

    def add_alert_handler(self, handler: Callable):
        """添加告警处理器"""
        self.alert_handlers.append(handler)

    async def start_monitoring(self):
        """开始告警监控"""
        if self.running:
            return

        self.running = True
        self.alert_task = asyncio.create_task(self._monitoring_loop())
        print("🚨 Alert monitoring started")

    async def stop_monitoring(self):
        """停止告警监控"""
        self.running = False
        if self.alert_task:
            self.alert_task.cancel()
            try:
                await self.alert_task
            except asyncio.CancelledError:
                pass
        print("🛑 Alert monitoring stopped")

    async def _monitoring_loop(self):
        """告警监控循环"""
        while self.running:
            try:
                await self._check_alert_rules()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Alert monitoring error: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_alert_rules(self):
        """检查告警规则"""
        for rule_id, rule in self.alert_rules.items():
            if not rule['enabled']:
                continue

            try:
                metric = self.metrics_collector.get_latest_metric(rule['metric_name'])
                if not metric:
                    continue

                # 检查条件
                triggered = self._evaluate_condition(
                    metric.value, rule['condition'], rule['threshold']
                )

                if triggered:
                    # 触发告警
                    if rule_id not in self.active_alerts:
                        alert = Alert(
                            id=rule_id,
                            level=rule['level'],
                            message=rule['message'],
                            metric_name=rule['metric_name'],
                            threshold_value=rule['threshold'],
                            current_value=metric.value,
                            timestamp=datetime.now()
                        )
                        self.active_alerts[rule_id] = alert
                        self.alert_history.append(alert)
                        await self._trigger_alert(alert)
                else:
                    # 解除告警
                    if rule_id in self.active_alerts:
                        alert = self.active_alerts[rule_id]
                        alert.resolved = True
                        alert.resolved_at = datetime.now()
                        del self.active_alerts[rule_id]
                        await self._resolve_alert(alert)

            except Exception as e:
                print(f"❌ Error checking alert rule {rule_id}: {e}")

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """评估告警条件"""
        if condition == '>':
            return value > threshold
        elif condition == '<':
            return value < threshold
        elif condition == '>=':
            return value >= threshold
        elif condition == '<=':
            return value <= threshold
        elif condition == '==':
            return abs(value - threshold) < 1e-6
        else:
            return False

    async def _trigger_alert(self, alert: Alert):
        """触发告警"""
        print(f"🚨 Alert triggered: {alert.message} (Value: {alert.current_value})")

        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert, 'triggered')
                else:
                    handler(alert, 'triggered')
            except Exception as e:
                print(f"❌ Alert handler error: {e}")

    async def _resolve_alert(self, alert: Alert):
        """解除告警"""
        print(f"✅ Alert resolved: {alert.message}")

        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert, 'resolved')
                else:
                    handler(alert, 'resolved')
            except Exception as e:
                print(f"❌ Alert handler error: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())

    def get_alert_history(self, duration_hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        cutoff_time = datetime.now() - timedelta(hours=duration_hours)
        return [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]


class PerformanceMonitor:
    """性能监控主类"""

    def __init__(self, database_manager: Optional[DatabaseManager] = None,
                 cache_manager: Optional[CacheManager] = None):
        self.database_manager = database_manager
        self.cache_manager = cache_manager

        # 核心组件
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)

        # 应用级别指标
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.error_counts: Dict[str, int] = defaultdict(int)

        # 系统状态
        self.start_time = datetime.now()
        self.is_running = False

        # 注册默认的指标收集器
        self._register_default_collectors()
        self._setup_default_alerts()

    def _register_default_collectors(self):
        """注册默认指标收集器"""

        async def collect_application_metrics():
            """收集应用程序指标"""
            timestamp = datetime.now()
            metrics = []

            # 请求计数
            total_requests = sum(self.request_counts.values())
            metrics.append(Metric(
                name="app.requests.total",
                type=MetricType.COUNTER,
                value=total_requests,
                timestamp=timestamp,
                description="Total number of requests"
            ))

            # 错误率
            total_errors = sum(self.error_counts.values())
            error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
            metrics.append(Metric(
                name="app.errors.rate_percent",
                type=MetricType.GAUGE,
                value=error_rate,
                timestamp=timestamp,
                description="Error rate percentage"
            ))

            # 平均响应时间
            all_response_times = []
            for times in self.response_times.values():
                all_response_times.extend(times)

            if all_response_times:
                avg_response_time = statistics.mean(all_response_times)
                p95_response_time = self.metrics_collector._percentile(all_response_times, 95)

                metrics.append(Metric(
                    name="app.response_time.average_ms",
                    type=MetricType.GAUGE,
                    value=avg_response_time,
                    timestamp=timestamp,
                    description="Average response time in milliseconds"
                ))

                metrics.append(Metric(
                    name="app.response_time.p95_ms",
                    type=MetricType.GAUGE,
                    value=p95_response_time,
                    timestamp=timestamp,
                    description="95th percentile response time in milliseconds"
                ))

            return metrics

        async def collect_service_metrics():
            """收集服务状态指标"""
            timestamp = datetime.now()
            metrics = []

            # 数据库连接状态
            if self.database_manager:
                try:
                    db_status = await self.database_manager.health_check()
                    db_healthy = 1 if db_status.get('status') == 'healthy' else 0
                    metrics.append(Metric(
                        name="service.database.healthy",
                        type=MetricType.GAUGE,
                        value=db_healthy,
                        timestamp=timestamp,
                        description="Database health status (1=healthy, 0=unhealthy)"
                    ))
                except Exception:
                    metrics.append(Metric(
                        name="service.database.healthy",
                        type=MetricType.GAUGE,
                        value=0,
                        timestamp=timestamp,
                        description="Database health status (1=healthy, 0=unhealthy)"
                    ))

            # 缓存状态
            if self.cache_manager:
                try:
                    cache_status = await self.cache_manager.health_check()
                    cache_healthy = 1 if cache_status.get('status') == 'healthy' else 0
                    metrics.append(Metric(
                        name="service.cache.healthy",
                        type=MetricType.GAUGE,
                        value=cache_healthy,
                        timestamp=timestamp,
                        description="Cache health status (1=healthy, 0=unhealthy)"
                    ))
                except Exception:
                    metrics.append(Metric(
                        name="service.cache.healthy",
                        type=MetricType.GAUGE,
                        value=0,
                        timestamp=timestamp,
                        description="Cache health status (1=healthy, 0=unhealthy)"
                    ))

            return metrics

        # 注册收集器
        self.metrics_collector.register_collector('application', collect_application_metrics)
        self.metrics_collector.register_collector('services', collect_service_metrics)

    def _setup_default_alerts(self):
        """设置默认告警规则"""
        # CPU使用率告警
        self.alert_manager.add_alert_rule(
            'high_cpu_usage',
            'system.cpu.usage_percent',
            '>',
            80.0,
            AlertLevel.WARNING,
            'CPU usage is high (>80%)'
        )

        # 内存使用率告警
        self.alert_manager.add_alert_rule(
            'high_memory_usage',
            'system.memory.usage_percent',
            '>',
            85.0,
            AlertLevel.WARNING,
            'Memory usage is high (>85%)'
        )

        # 磁盘使用率告警
        self.alert_manager.add_alert_rule(
            'high_disk_usage',
            'system.disk.usage_percent',
            '>',
            90.0,
            AlertLevel.ERROR,
            'Disk usage is critically high (>90%)'
        )

        # 错误率告警
        self.alert_manager.add_alert_rule(
            'high_error_rate',
            'app.errors.rate_percent',
            '>',
            5.0,
            AlertLevel.ERROR,
            'Error rate is high (>5%)'
        )

        # 响应时间告警
        self.alert_manager.add_alert_rule(
            'slow_response_time',
            'app.response_time.p95_ms',
            '>',
            2000.0,
            AlertLevel.WARNING,
            'Response time is slow (P95 >2s)'
        )

    async def start(self):
        """启动性能监控"""
        if self.is_running:
            return

        self.is_running = True
        await self.metrics_collector.start_collection()
        await self.alert_manager.start_monitoring()
        print("🚀 Performance monitoring started")

    async def stop(self):
        """停止性能监控"""
        if not self.is_running:
            return

        self.is_running = False
        await self.metrics_collector.stop_collection()
        await self.alert_manager.stop_monitoring()
        print("🛑 Performance monitoring stopped")

    def record_request(self, endpoint: str, response_time_ms: float, success: bool = True):
        """记录请求"""
        self.request_counts[endpoint] += 1
        self.response_times[endpoint].append(response_time_ms)

        if not success:
            self.error_counts[endpoint] += 1

        # 记录到指标收集器
        timestamp = datetime.now()
        self.metrics_collector.add_metric(Metric(
            name=f"endpoint.{endpoint}.requests",
            type=MetricType.COUNTER,
            value=1,
            timestamp=timestamp,
            tags={'endpoint': endpoint, 'success': str(success)}
        ))

        self.metrics_collector.add_metric(Metric(
            name=f"endpoint.{endpoint}.response_time_ms",
            type=MetricType.TIMER,
            value=response_time_ms,
            timestamp=timestamp,
            tags={'endpoint': endpoint}
        ))

    def get_system_health(self) -> SystemHealth:
        """获取系统健康状态"""
        # 获取最新的系统指标
        cpu_metric = self.metrics_collector.get_latest_metric('system.cpu.usage_percent')
        memory_metric = self.metrics_collector.get_latest_metric('system.memory.usage_percent')
        disk_metric = self.metrics_collector.get_latest_metric('system.disk.usage_percent')
        error_rate_metric = self.metrics_collector.get_latest_metric('app.errors.rate_percent')
        response_time_metric = self.metrics_collector.get_latest_metric('app.response_time.p95_ms')

        # 计算整体状态
        status_factors = []

        cpu_usage = cpu_metric.value if cpu_metric else 0
        memory_usage = memory_metric.value if memory_metric else 0
        disk_usage = disk_metric.value if disk_metric else 0
        error_rate = error_rate_metric.value if error_rate_metric else 0
        response_time_p95 = response_time_metric.value if response_time_metric else 0

        # 状态评估
        if cpu_usage > 90 or memory_usage > 95 or disk_usage > 95 or error_rate > 10:
            overall_status = "critical"
        elif cpu_usage > 80 or memory_usage > 85 or disk_usage > 90 or error_rate > 5:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        return SystemHealth(
            overall_status=overall_status,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io={
                'bytes_sent': 0,  # 可以从指标获取
                'bytes_recv': 0
            },
            active_connections=0,  # 需要从连接池获取
            response_time_p95=response_time_p95,
            error_rate=error_rate,
            timestamp=datetime.now()
        )

    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取监控面板数据"""
        health = self.get_system_health()
        active_alerts = self.alert_manager.get_active_alerts()

        return {
            'system_health': {
                'status': health.overall_status,
                'cpu_usage': health.cpu_usage,
                'memory_usage': health.memory_usage,
                'disk_usage': health.disk_usage,
                'error_rate': health.error_rate,
                'response_time_p95': health.response_time_p95
            },
            'alerts': {
                'active_count': len(active_alerts),
                'critical_count': len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
                'warning_count': len([a for a in active_alerts if a.level == AlertLevel.WARNING]),
                'recent_alerts': [
                    {
                        'message': alert.message,
                        'level': alert.level.value,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in active_alerts[:5]  # 最近5个告警
                ]
            },
            'performance': {
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'total_requests': sum(self.request_counts.values()),
                'total_errors': sum(self.error_counts.values()),
                'top_endpoints': [
                    {'endpoint': endpoint, 'count': count}
                    for endpoint, count in sorted(
                        self.request_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                ]
            },
            'timestamp': datetime.now().isoformat()
        }

    def get_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """获取指标摘要"""
        summary = {}

        # 系统指标
        system_metrics = [
            'system.cpu.usage_percent',
            'system.memory.usage_percent',
            'system.disk.usage_percent'
        ]

        for metric_name in system_metrics:
            stats = self.metrics_collector.calculate_metric_stats(metric_name, duration_minutes)
            if stats:
                summary[metric_name] = stats

        # 应用指标
        app_metrics = [
            'app.requests.total',
            'app.errors.rate_percent',
            'app.response_time.average_ms',
            'app.response_time.p95_ms'
        ]

        for metric_name in app_metrics:
            stats = self.metrics_collector.calculate_metric_stats(metric_name, duration_minutes)
            if stats:
                summary[metric_name] = stats

        return summary

    def export_metrics(self, format: str = 'json') -> str:
        """导出指标数据"""
        if format == 'json':
            data = {
                'timestamp': datetime.now().isoformat(),
                'system_health': self.get_system_health().__dict__,
                'metrics_summary': self.get_metrics_summary(),
                'active_alerts': [
                    {
                        'id': alert.id,
                        'level': alert.level.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in self.alert_manager.get_active_alerts()
                ]
            }
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def add_custom_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                         tags: Optional[Dict[str, str]] = None, description: str = ""):
        """添加自定义指标"""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            description=description
        )
        self.metrics_collector.add_metric(metric)