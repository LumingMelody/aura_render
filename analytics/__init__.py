"""
Analytics Package

Comprehensive analytics and metrics collection system for video generation,
user behavior tracking, and system performance monitoring.
"""

from .metrics_system import (
    MetricsCollector,
    VideoAnalytics,
    SystemMonitor,
    get_metrics_collector,
    get_video_analytics,
    get_system_monitor,
    MetricType,
    EventType,
    PerformanceMetrics,
    VideoMetrics,
    UserMetrics,
    BusinessMetrics
)

__all__ = [
    'MetricsCollector',
    'VideoAnalytics', 
    'SystemMonitor',
    'get_metrics_collector',
    'get_video_analytics',
    'get_system_monitor',
    'MetricType',
    'EventType',
    'PerformanceMetrics',
    'VideoMetrics',
    'UserMetrics',
    'BusinessMetrics'
]