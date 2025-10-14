"""
监控系统模块 - 全面的系统监控、健康检查和性能跟踪
"""

# 原有的监控组件
from .health_checks import (
    HealthChecker,
    HealthStatus,
    ServiceType,
    HealthCheckResult,
    SystemMetrics,
    get_health_checker
)

from .error_handler import (
    ErrorHandler,
    ErrorSeverity,
    ErrorCategory,
    ErrorRecord,
    get_error_handler
)

from .metrics_collector import (
    MetricsCollector,
    get_metrics_collector
)

# 新增的性能监控组件
from .performance_monitor import (
    PerformanceMonitor,
    MetricsCollector as NewMetricsCollector,
    AlertManager,
    Metric,
    Alert,
    SystemHealth,
    MetricType,
    AlertLevel
)

from .analytics_dashboard import AnalyticsDashboard
from .user_behavior_tracker import UserBehaviorTracker

__all__ = [
    # 原有组件
    'HealthChecker',
    'HealthStatus',
    'ServiceType',
    'HealthCheckResult',
    'SystemMetrics',
    'get_health_checker',
    'ErrorHandler',
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorRecord',
    'get_error_handler',
    'MetricsCollector',
    'get_metrics_collector',
    # 新增组件
    'PerformanceMonitor',
    'NewMetricsCollector',
    'AlertManager',
    'Metric',
    'Alert',
    'SystemHealth',
    'MetricType',
    'AlertLevel',
    'AnalyticsDashboard',
    'UserBehaviorTracker'
]