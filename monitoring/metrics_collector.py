"""
Metrics Collector

Collects and aggregates system performance metrics.
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import psutil
import logging
from config import Settings


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metrics"""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_point(self, value: float, tags: Dict[str, str] = None):
        """Add a metric point"""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            tags=tags or {}
        )
        self.points.append(point)
        
    def get_latest(self) -> Optional[MetricPoint]:
        """Get latest metric point"""
        return self.points[-1] if self.points else None
        
    def get_average(self, minutes: int = 5) -> float:
        """Get average value for last N minutes"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent_points = [p for p in self.points if p.timestamp > cutoff]
        
        if not recent_points:
            return 0.0
            
        return sum(p.value for p in recent_points) / len(recent_points)
        
    def get_max(self, minutes: int = 5) -> float:
        """Get max value for last N minutes"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent_points = [p for p in self.points if p.timestamp > cutoff]
        
        if not recent_points:
            return 0.0
            
        return max(p.value for p in recent_points)


class MetricsCollector:
    """System metrics collector"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.logger = logging.getLogger(__name__)
        self.metrics: Dict[str, MetricSeries] = {}
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.collection_enabled = self.settings.monitoring.metrics_enabled
        self._collection_task = None
        
    async def start_collection(self):
        """Start automatic metrics collection"""
        if not self.collection_enabled:
            self.logger.info("Metrics collection is disabled")
            return
            
        if self._collection_task is not None:
            return  # Already started
            
        self._collection_task = asyncio.create_task(self._collection_loop())
        self.logger.info("Started metrics collection")
        
    async def stop_collection(self):
        """Stop automatic metrics collection"""
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None
            self.logger.info("Stopped metrics collection")
            
    async def _collection_loop(self):
        """Main collection loop"""
        while True:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(30)
                
    async def _collect_system_metrics(self):
        """Collect system resource metrics"""
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.gauge("system.cpu.usage_percent", cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.gauge("system.memory.usage_percent", memory.percent)
        self.gauge("system.memory.used_mb", memory.used / 1024 / 1024)
        self.gauge("system.memory.available_mb", memory.available / 1024 / 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.gauge("system.disk.usage_percent", (disk.used / disk.total) * 100)
        self.gauge("system.disk.used_gb", disk.used / 1024 / 1024 / 1024)
        self.gauge("system.disk.free_gb", disk.free / 1024 / 1024 / 1024)
        
        # Network I/O
        net_io = psutil.net_io_counters()
        self.gauge("system.network.bytes_sent", net_io.bytes_sent)
        self.gauge("system.network.bytes_recv", net_io.bytes_recv)
        
        # Process count
        self.gauge("system.processes.count", len(psutil.pids()))
        
    # Metric recording methods
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric (current value)"""
        if not self.collection_enabled:
            return
            
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name)
            
        self.metrics[name].add_point(value, tags)
        
    def counter(self, name: str, increment: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        if not self.collection_enabled:
            return
            
        key = f"{name}:{tags}" if tags else name
        self.counters[key] += increment
        
    def timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer metric (duration)"""
        if not self.collection_enabled:
            return
            
        key = f"{name}:{tags}" if tags else name
        self.timers[key].append(duration)
        
        # Keep only recent timings (last 100)
        if len(self.timers[key]) > 100:
            self.timers[key] = self.timers[key][-100:]
            
        # Also record as gauge for latest value
        self.gauge(f"{name}.duration_ms", duration * 1000, tags)
        
    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric"""
        self.gauge(name, value, tags)
        
    # Context managers for timing
    def time_context(self, name: str, tags: Dict[str, str] = None):
        """Context manager for timing operations"""
        return TimingContext(self, name, tags)
        
    # Application-specific metrics
    def record_task_started(self, task_id: str, task_type: str = "unknown"):
        """Record task start"""
        self.counter("tasks.started", tags={"type": task_type})
        self.gauge("tasks.active", self._get_active_tasks_count())
        
    def record_task_completed(self, task_id: str, task_type: str = "unknown", duration: float = 0):
        """Record task completion"""
        self.counter("tasks.completed", tags={"type": task_type})
        self.timer("tasks.duration", duration, tags={"type": task_type})
        self.gauge("tasks.active", self._get_active_tasks_count())
        
    def record_task_failed(self, task_id: str, task_type: str = "unknown", error_type: str = "unknown"):
        """Record task failure"""
        self.counter("tasks.failed", tags={"type": task_type, "error": error_type})
        self.gauge("tasks.active", self._get_active_tasks_count())
        
    def record_api_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Record API request metrics"""
        self.counter("api.requests", tags={
            "endpoint": endpoint,
            "method": method,
            "status": str(status_code)
        })
        self.timer("api.response_time", duration, tags={
            "endpoint": endpoint,
            "method": method
        })
        
    def record_ai_service_call(self, service: str, model: str, duration: float, success: bool):
        """Record AI service call metrics"""
        status = "success" if success else "error"
        self.counter("ai.requests", tags={"service": service, "model": model, "status": status})
        self.timer("ai.response_time", duration, tags={"service": service, "model": model})
        
    def record_render_progress(self, task_id: str, progress: float, stage: str = "unknown"):
        """Record rendering progress"""
        self.gauge("render.progress", progress, tags={"stage": stage})
        
    def record_material_search(self, provider: str, query_type: str, result_count: int, duration: float):
        """Record material search metrics"""
        self.counter("materials.searches", tags={"provider": provider, "type": query_type})
        self.gauge("materials.results", result_count, tags={"provider": provider})
        self.timer("materials.search_time", duration, tags={"provider": provider})
        
    def record_cache_operation(self, operation: str, hit: bool = None, duration: float = 0):
        """Record cache operation metrics"""
        self.counter(f"cache.{operation}")
        if hit is not None:
            status = "hit" if hit else "miss"
            self.counter("cache.requests", tags={"status": status})
        if duration > 0:
            self.timer(f"cache.{operation}_time", duration)
            
    def record_database_query(self, table: str, operation: str, duration: float):
        """Record database query metrics"""
        self.counter("database.queries", tags={"table": table, "operation": operation})
        self.timer("database.query_time", duration, tags={"table": table, "operation": operation})
        
    def _get_active_tasks_count(self) -> int:
        """Get current active tasks count (placeholder)"""
        # This would integrate with the actual task management system
        return 0
        
    # Data retrieval methods
    def get_metric_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get summary of all metrics for last N minutes"""
        
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "period_minutes": minutes,
            "gauges": {},
            "counters": dict(self.counters),
            "timers": {}
        }
        
        # Process gauges
        for name, series in self.metrics.items():
            latest = series.get_latest()
            if latest:
                summary["gauges"][name] = {
                    "latest": latest.value,
                    "average": series.get_average(minutes),
                    "max": series.get_max(minutes),
                    "timestamp": latest.timestamp.isoformat()
                }
                
        # Process timers
        for name, durations in self.timers.items():
            if durations:
                summary["timers"][name] = {
                    "count": len(durations),
                    "average": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "p95": self._percentile(durations, 95),
                    "p99": self._percentile(durations, 99)
                }
                
        return summary
        
    def get_system_health_metrics(self) -> Dict[str, Any]:
        """Get system health metrics"""
        
        health = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check system resources
        for metric_name in ["system.cpu.usage_percent", "system.memory.usage_percent", "system.disk.usage_percent"]:
            if metric_name in self.metrics:
                series = self.metrics[metric_name]
                latest = series.get_latest()
                if latest:
                    value = latest.value
                    status = "healthy"
                    
                    # Define thresholds
                    if "cpu" in metric_name and value > 80:
                        status = "warning" if value < 90 else "critical"
                    elif "memory" in metric_name and value > 85:
                        status = "warning" if value < 95 else "critical"
                    elif "disk" in metric_name and value > 90:
                        status = "warning" if value < 98 else "critical"
                        
                    health["checks"][metric_name] = {
                        "status": status,
                        "value": value,
                        "threshold_warning": 80 if "cpu" in metric_name else 85 if "memory" in metric_name else 90,
                        "threshold_critical": 90 if "cpu" in metric_name else 95 if "memory" in metric_name else 98
                    }
                    
                    if status == "critical":
                        health["status"] = "critical"
                    elif status == "warning" and health["status"] == "healthy":
                        health["status"] = "warning"
                        
        return health
        
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
            
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
            
        return sorted_data[index]
        
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.counters.clear()
        self.timers.clear()
        self.logger.info("All metrics reset")


class TimingContext:
    """Context manager for timing operations"""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Dict[str, str] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.timer(self.name, duration, self.tags)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(settings: Optional[Settings] = None) -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(settings)
    return _metrics_collector