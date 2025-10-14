"""
Production Monitoring and Metrics Collection

Comprehensive metrics collection for Aura Render platform including:
- Application performance metrics
- Business logic metrics
- System resource metrics
- Custom metrics for AI optimization
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import functools
import asyncio

logger = logging.getLogger(__name__)

# Application Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

# Video Generation Metrics
video_generation_total = Counter(
    'video_generation_total',
    'Total video generation requests',
    ['theme', 'user_id']
)

video_generation_failures_total = Counter(
    'video_generation_failures_total',
    'Total video generation failures',
    ['error_type', 'stage']
)

video_generation_duration_seconds = Histogram(
    'video_generation_duration_seconds',
    'Video generation duration in seconds',
    ['theme', 'duration_category']
)

video_quality_score = Gauge(
    'video_quality_score',
    'Generated video quality score',
    ['theme', 'optimization_level']
)

# AI Optimization Metrics
ai_optimization_total = Counter(
    'ai_optimization_total',
    'Total AI optimization requests',
    ['optimization_type', 'level']
)

ai_optimization_failures_total = Counter(
    'ai_optimization_failures_total',
    'Total AI optimization failures',
    ['optimization_type', 'error_type']
)

optimization_processing_time_seconds = Histogram(
    'optimization_processing_time_seconds',
    'AI optimization processing time in seconds',
    ['optimization_type', 'level']
)

optimization_size_reduction_percent = Histogram(
    'optimization_size_reduction_percent',
    'Video size reduction percentage after optimization',
    ['optimization_type']
)

# Queue Metrics
celery_queue_length = Gauge(
    'celery_queue_length',
    'Number of tasks in Celery queue',
    ['queue']
)

celery_worker_load_average = Gauge(
    'celery_worker_load_average',
    'Celery worker load average',
    ['worker_id']
)

# System Metrics
system_cpu_usage_percent = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

system_memory_usage_bytes = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

system_disk_usage_percent = Gauge(
    'system_disk_usage_percent',
    'System disk usage percentage',
    ['mount_point']
)

# Business Metrics
active_users_total = Gauge(
    'active_users_total',
    'Total number of active users',
    ['time_period']
)

revenue_total = Counter(
    'revenue_total',
    'Total revenue',
    ['subscription_type', 'currency']
)

# Application Info
app_info = Info(
    'aura_render_app',
    'Aura Render application information'
)


class MetricsCollector:
    """Central metrics collection and management"""
    
    def __init__(self):
        self.start_time = time.time()
        self._system_metrics_interval = 30  # seconds
        self._running = False
        
    async def start(self, port: int = 8080):
        """Start the metrics server and collection"""
        try:
            start_http_server(port)
            logger.info(f"Metrics server started on port {port}")
            
            # Set application info
            app_info.info({
                'version': '1.0.0',
                'environment': 'production',
                'start_time': str(self.start_time)
            })
            
            self._running = True
            await self._collect_system_metrics_loop()
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    async def stop(self):
        """Stop metrics collection"""
        self._running = False
        logger.info("Metrics collection stopped")
    
    async def _collect_system_metrics_loop(self):
        """Continuously collect system metrics"""
        while self._running:
            try:
                self._collect_system_metrics()
                await asyncio.sleep(self._system_metrics_interval)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(self._system_metrics_interval)
    
    def _collect_system_metrics(self):
        """Collect current system metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        system_cpu_usage_percent.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        system_memory_usage_bytes.set(memory.used)
        
        # Disk usage
        for partition in psutil.disk_partitions():
            try:
                disk_usage = psutil.disk_usage(partition.mountpoint)
                usage_percent = (disk_usage.used / disk_usage.total) * 100
                system_disk_usage_percent.labels(mount_point=partition.mountpoint).set(usage_percent)
            except (PermissionError, FileNotFoundError):
                # Skip partitions that can't be accessed
                continue


# Decorators for automatic metrics collection

def track_requests(endpoint: str = None):
    """Decorator to track HTTP requests"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            method = getattr(kwargs.get('request', args[0] if args else None), 'method', 'UNKNOWN')
            endpoint_name = endpoint or func.__name__
            
            start_time = time.time()
            status = '200'
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = getattr(e, 'status_code', '500')
                raise
            finally:
                duration = time.time() - start_time
                http_requests_total.labels(method=method, endpoint=endpoint_name, status=status).inc()
                http_request_duration_seconds.labels(method=method, endpoint=endpoint_name).observe(duration)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            method = getattr(kwargs.get('request', args[0] if args else None), 'method', 'UNKNOWN')
            endpoint_name = endpoint or func.__name__
            
            start_time = time.time()
            status = '200'
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = getattr(e, 'status_code', '500')
                raise
            finally:
                duration = time.time() - start_time
                http_requests_total.labels(method=method, endpoint=endpoint_name, status=status).inc()
                http_request_duration_seconds.labels(method=method, endpoint=endpoint_name).observe(duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def track_video_generation(theme: str = None, user_id: str = None):
    """Decorator to track video generation metrics"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            theme_label = theme or kwargs.get('theme', 'unknown')
            user_label = user_id or kwargs.get('user_id', 'anonymous')
            
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Track successful generation
                video_generation_total.labels(theme=theme_label, user_id=user_label).inc()
                
                # Track duration
                duration = time.time() - start_time
                duration_category = 'short' if duration < 30 else 'medium' if duration < 120 else 'long'
                video_generation_duration_seconds.labels(theme=theme_label, duration_category=duration_category).observe(duration)
                
                # Track quality if available in result
                if hasattr(result, 'quality_score') and result.quality_score is not None:
                    video_quality_score.labels(theme=theme_label, optimization_level='standard').set(result.quality_score)
                
                return result
                
            except Exception as e:
                # Track failure
                error_type = type(e).__name__
                stage = getattr(e, 'stage', 'unknown')
                video_generation_failures_total.labels(error_type=error_type, stage=stage).inc()
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            theme_label = theme or kwargs.get('theme', 'unknown')
            user_label = user_id or kwargs.get('user_id', 'anonymous')
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Track successful generation
                video_generation_total.labels(theme=theme_label, user_id=user_label).inc()
                
                # Track duration
                duration = time.time() - start_time
                duration_category = 'short' if duration < 30 else 'medium' if duration < 120 else 'long'
                video_generation_duration_seconds.labels(theme=theme_label, duration_category=duration_category).observe(duration)
                
                return result
                
            except Exception as e:
                # Track failure
                error_type = type(e).__name__
                stage = getattr(e, 'stage', 'unknown')
                video_generation_failures_total.labels(error_type=error_type, stage=stage).inc()
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def track_ai_optimization(optimization_type: str = None, level: str = None):
    """Decorator to track AI optimization metrics"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            opt_type = optimization_type or kwargs.get('optimization_type', 'unknown')
            opt_level = level or kwargs.get('level', 'standard')
            
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Track successful optimization
                ai_optimization_total.labels(optimization_type=opt_type, level=opt_level).inc()
                
                # Track processing time
                duration = time.time() - start_time
                optimization_processing_time_seconds.labels(optimization_type=opt_type, level=opt_level).observe(duration)
                
                # Track size reduction if available
                if hasattr(result, 'size_reduction_percent') and result.size_reduction_percent is not None:
                    optimization_size_reduction_percent.labels(optimization_type=opt_type).observe(result.size_reduction_percent)
                
                return result
                
            except Exception as e:
                # Track failure
                error_type = type(e).__name__
                ai_optimization_failures_total.labels(optimization_type=opt_type, error_type=error_type).inc()
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else async_wrapper
    return decorator


@contextmanager
def track_processing_time(metric: Histogram, labels: Dict[str, str] = None):
    """Context manager to track processing time"""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        if labels:
            metric.labels(**labels).observe(duration)
        else:
            metric.observe(duration)


def update_queue_metrics(queue_name: str, queue_length: int):
    """Update queue length metrics"""
    celery_queue_length.labels(queue=queue_name).set(queue_length)


def update_worker_load(worker_id: str, load_average: float):
    """Update worker load metrics"""
    celery_worker_load_average.labels(worker_id=worker_id).set(load_average)


def track_business_metric(metric_name: str, value: float, labels: Dict[str, str] = None):
    """Track custom business metrics"""
    if metric_name == 'active_users':
        time_period = labels.get('time_period', 'daily') if labels else 'daily'
        active_users_total.labels(time_period=time_period).set(value)
    elif metric_name == 'revenue':
        subscription_type = labels.get('subscription_type', 'basic') if labels else 'basic'
        currency = labels.get('currency', 'USD') if labels else 'USD'
        revenue_total.labels(subscription_type=subscription_type, currency=currency).inc(value)


# Global metrics collector instance
metrics_collector = MetricsCollector()


async def start_metrics_collection(port: int = 8080):
    """Start the global metrics collection"""
    await metrics_collector.start(port)


async def stop_metrics_collection():
    """Stop the global metrics collection"""
    await metrics_collector.stop()