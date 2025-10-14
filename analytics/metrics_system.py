"""
Video Analytics and Metrics Collection System

Provides comprehensive analytics for video generation, user behavior,
system performance, and business intelligence.
"""

import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path

from pydantic import BaseModel, Field
import logging


class MetricType(str, Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class EventType(str, Enum):
    """Types of events to track"""
    VIDEO_GENERATION_START = "video_generation_start"
    VIDEO_GENERATION_COMPLETE = "video_generation_complete"
    VIDEO_GENERATION_FAILED = "video_generation_failed"
    USER_LOGIN = "user_login"
    TEMPLATE_USED = "template_used"
    PRESET_APPLIED = "preset_applied"
    API_REQUEST = "api_request"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"
    USER_INTERACTION = "user_interaction"


@dataclass
class MetricPoint:
    """Individual metric data point"""
    timestamp: datetime
    metric_name: str
    metric_type: MetricType
    value: Union[int, float]
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class Event:
    """Analytics event"""
    id: str
    timestamp: datetime
    event_type: EventType
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    data: Dict[str, Any] = None
    metadata: Dict[str, str] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.metadata is None:
            self.metadata = {}


class PerformanceMetrics(BaseModel):
    """System performance metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    active_tasks: int = 0
    queue_length: int = 0
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0


class VideoMetrics(BaseModel):
    """Video generation specific metrics"""
    total_videos_generated: int = 0
    total_generation_time: float = 0.0
    average_generation_time: float = 0.0
    success_rate: float = 0.0
    most_popular_template: Optional[str] = None
    most_popular_style: Optional[str] = None
    total_render_time: float = 0.0
    average_video_duration: float = 0.0


class UserMetrics(BaseModel):
    """User behavior metrics"""
    total_users: int = 0
    active_users_daily: int = 0
    active_users_weekly: int = 0
    active_users_monthly: int = 0
    average_session_duration: float = 0.0
    videos_per_user: float = 0.0
    retention_rate: float = 0.0


class BusinessMetrics(BaseModel):
    """Business intelligence metrics"""
    conversion_rate: float = 0.0
    revenue: float = 0.0
    premium_users: int = 0
    trial_to_paid_conversion: float = 0.0
    churn_rate: float = 0.0
    customer_lifetime_value: float = 0.0


class MetricsCollector:
    """Core metrics collection and storage system"""
    
    def __init__(self, storage_dir: str = "analytics_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.metrics: List[MetricPoint] = []
        self.events: List[Event] = []
        
        # In-memory metric storage for fast access
        self.metric_values: Dict[str, Any] = {}
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = {}
        
        # Performance tracking
        self.start_time = datetime.now()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a metric data point"""
        metric = MetricPoint(
            timestamp=datetime.now(),
            metric_name=name,
            metric_type=metric_type,
            value=value,
            tags=tags or {}
        )
        
        self.metrics.append(metric)
        
        # Update in-memory storage
        if metric_type == MetricType.COUNTER:
            self.counters[name] = self.counters.get(name, 0) + int(value)
        elif metric_type == MetricType.GAUGE:
            self.gauges[name] = float(value)
        elif metric_type == MetricType.TIMER:
            if name not in self.timers:
                self.timers[name] = []
            self.timers[name].append(float(value))
        
        # Periodically flush to disk
        if len(self.metrics) >= 100:
            await self._flush_metrics()
    
    async def record_event(
        self,
        event_type: EventType,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None
    ):
        """Record an analytics event"""
        event = Event(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            data=data or {},
            metadata=metadata or {}
        )
        
        self.events.append(event)
        
        # Also create corresponding metrics
        await self.record_metric(
            f"event.{event_type.value}",
            1,
            MetricType.COUNTER,
            {"user_id": user_id, "session_id": session_id}
        )
        
        # Periodically flush events
        if len(self.events) >= 50:
            await self._flush_events()
    
    async def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        await self.record_metric(name, value, MetricType.COUNTER, tags)
    
    async def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        await self.record_metric(name, value, MetricType.GAUGE, tags)
    
    async def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric"""
        await self.record_metric(name, duration, MetricType.TIMER, tags)
    
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        return TimerContext(self, name, tags)
    
    async def get_counter(self, name: str) -> int:
        """Get current counter value"""
        return self.counters.get(name, 0)
    
    async def get_gauge(self, name: str) -> Optional[float]:
        """Get current gauge value"""
        return self.gauges.get(name)
    
    async def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a timer metric"""
        if name not in self.timers or not self.timers[name]:
            return {}
        
        values = self.timers[name]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "total": sum(values)
        }
    
    async def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        return {
            "time_range": f"Last {hours} hours",
            "metrics_count": len(recent_metrics),
            "events_count": len(recent_events),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "timer_stats": {name: await self.get_timer_stats(name) for name in self.timers},
            "top_events": self._get_top_events(recent_events),
            "system_uptime": (datetime.now() - self.start_time).total_seconds()
        }
    
    def _get_top_events(self, events: List[Event], limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequent events"""
        event_counts = {}
        for event in events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return [
            {"event_type": event_type, "count": count}
            for event_type, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]
    
    async def _flush_metrics(self):
        """Flush metrics to disk"""
        if not self.metrics:
            return
        
        metrics_file = self.storage_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H')}.json"
        
        try:
            metrics_data = [asdict(metric) for metric in self.metrics]
            
            with open(metrics_file, 'a', encoding='utf-8') as f:
                for metric_data in metrics_data:
                    # Convert datetime to string for JSON serialization
                    metric_data['timestamp'] = metric_data['timestamp'].isoformat()
                    f.write(json.dumps(metric_data) + '\n')
            
            self.metrics.clear()
            self.logger.debug(f"Flushed {len(metrics_data)} metrics to {metrics_file}")
        
        except Exception as e:
            self.logger.error(f"Error flushing metrics: {e}")
    
    async def _flush_events(self):
        """Flush events to disk"""
        if not self.events:
            return
        
        events_file = self.storage_dir / f"events_{datetime.now().strftime('%Y%m%d_%H')}.json"
        
        try:
            events_data = [asdict(event) for event in self.events]
            
            with open(events_file, 'a', encoding='utf-8') as f:
                for event_data in events_data:
                    # Convert datetime to string for JSON serialization
                    event_data['timestamp'] = event_data['timestamp'].isoformat()
                    f.write(json.dumps(event_data) + '\n')
            
            self.events.clear()
            self.logger.debug(f"Flushed {len(events_data)} events to {events_file}")
        
        except Exception as e:
            self.logger.error(f"Error flushing events: {e}")


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            await self.collector.record_timer(self.name, duration, self.tags)


class VideoAnalytics:
    """Specialized analytics for video generation"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    async def track_video_generation_start(
        self,
        user_id: str,
        template_id: Optional[str] = None,
        duration: Optional[float] = None
    ):
        """Track start of video generation"""
        await self.metrics.record_event(
            EventType.VIDEO_GENERATION_START,
            user_id=user_id,
            data={
                "template_id": template_id,
                "expected_duration": duration
            }
        )
        
        await self.metrics.increment_counter("video.generations_started")
        
        if template_id:
            await self.metrics.increment_counter(
                "video.template_usage",
                tags={"template_id": template_id}
            )
    
    async def track_video_generation_complete(
        self,
        user_id: str,
        generation_time: float,
        video_duration: float,
        template_id: Optional[str] = None,
        file_size: Optional[int] = None
    ):
        """Track successful video generation completion"""
        await self.metrics.record_event(
            EventType.VIDEO_GENERATION_COMPLETE,
            user_id=user_id,
            data={
                "generation_time": generation_time,
                "video_duration": video_duration,
                "template_id": template_id,
                "file_size": file_size
            }
        )
        
        await self.metrics.increment_counter("video.generations_completed")
        await self.metrics.record_timer("video.generation_time", generation_time)
        await self.metrics.record_timer("video.duration", video_duration)
        
        if file_size:
            await self.metrics.record_timer("video.file_size", file_size)
    
    async def track_video_generation_failed(
        self,
        user_id: str,
        error_type: str,
        error_message: str,
        generation_time: Optional[float] = None
    ):
        """Track failed video generation"""
        await self.metrics.record_event(
            EventType.VIDEO_GENERATION_FAILED,
            user_id=user_id,
            data={
                "error_type": error_type,
                "error_message": error_message,
                "partial_generation_time": generation_time
            }
        )
        
        await self.metrics.increment_counter("video.generations_failed")
        await self.metrics.increment_counter(
            "video.errors",
            tags={"error_type": error_type}
        )
    
    async def get_video_metrics(self) -> VideoMetrics:
        """Get comprehensive video generation metrics"""
        total_started = await self.metrics.get_counter("video.generations_started")
        total_completed = await self.metrics.get_counter("video.generations_completed")
        total_failed = await self.metrics.get_counter("video.generations_failed")
        
        generation_stats = await self.metrics.get_timer_stats("video.generation_time")
        duration_stats = await self.metrics.get_timer_stats("video.duration")
        
        success_rate = 0.0
        if total_started > 0:
            success_rate = total_completed / total_started
        
        return VideoMetrics(
            total_videos_generated=total_completed,
            total_generation_time=generation_stats.get("total", 0),
            average_generation_time=generation_stats.get("avg", 0),
            success_rate=success_rate,
            average_video_duration=duration_stats.get("avg", 0)
        )


class SystemMonitor:
    """System performance monitoring"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring_active = False
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous system monitoring"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logging.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU and memory usage (placeholder - would use psutil in real implementation)
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            await self.metrics.set_gauge("system.cpu_usage", cpu_percent)
            await self.metrics.set_gauge("system.memory_usage", memory.percent)
            await self.metrics.set_gauge("system.memory_available", memory.available)
            await self.metrics.set_gauge("system.disk_usage", disk.percent)
            await self.metrics.set_gauge("system.disk_free", disk.free)
            
        except ImportError:
            # Fallback if psutil not available
            await self.metrics.set_gauge("system.cpu_usage", 0.0)
            await self.metrics.set_gauge("system.memory_usage", 0.0)
    
    async def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return PerformanceMetrics(
            cpu_usage=await self.metrics.get_gauge("system.cpu_usage") or 0.0,
            memory_usage=await self.metrics.get_gauge("system.memory_usage") or 0.0,
            disk_usage=await self.metrics.get_gauge("system.disk_usage") or 0.0,
            active_tasks=await self.metrics.get_gauge("system.active_tasks") or 0,
            queue_length=await self.metrics.get_gauge("system.queue_length") or 0,
            response_time=await self.metrics.get_gauge("api.response_time") or 0.0,
            error_rate=await self.metrics.get_gauge("api.error_rate") or 0.0
        )


# Global instances
_metrics_collector = None
_video_analytics = None
_system_monitor = None

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

def get_video_analytics() -> VideoAnalytics:
    """Get the global video analytics instance"""
    global _video_analytics
    if _video_analytics is None:
        _video_analytics = VideoAnalytics(get_metrics_collector())
    return _video_analytics

def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor instance"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor(get_metrics_collector())
    return _system_monitor