"""
Analytics API Endpoints

REST API endpoints for metrics collection and analytics data access.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from analytics import (
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

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


class MetricRequest(BaseModel):
    """Request model for recording metrics"""
    name: str = Field(..., min_length=1, max_length=100)
    value: float
    metric_type: MetricType
    tags: Dict[str, str] = Field(default_factory=dict)


class EventRequest(BaseModel):
    """Request model for recording events"""
    event_type: EventType
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)


class MetricsSummaryResponse(BaseModel):
    """Response model for metrics summary"""
    time_range: str
    metrics_count: int
    events_count: int
    counters: Dict[str, int]
    gauges: Dict[str, float]
    timer_stats: Dict[str, Dict[str, float]]
    top_events: List[Dict[str, Any]]
    system_uptime: float


@router.post("/metrics")
async def record_metric(request: MetricRequest):
    """Record a metric data point"""
    try:
        metrics = get_metrics_collector()
        
        await metrics.record_metric(
            name=request.name,
            value=request.value,
            metric_type=request.metric_type,
            tags=request.tags
        )
        
        return {
            "success": True,
            "message": "Metric recorded successfully",
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record metric: {str(e)}"
        )


@router.post("/events")
async def record_event(request: EventRequest):
    """Record an analytics event"""
    try:
        metrics = get_metrics_collector()
        
        await metrics.record_event(
            event_type=request.event_type,
            user_id=request.user_id,
            session_id=request.session_id,
            data=request.data,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "message": "Event recorded successfully",
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record event: {str(e)}"
        )


@router.get("/summary", response_model=MetricsSummaryResponse)
async def get_metrics_summary(hours: int = 24):
    """Get comprehensive metrics summary"""
    try:
        if hours <= 0 or hours > 168:  # Max 1 week
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Hours must be between 1 and 168"
            )
        
        metrics = get_metrics_collector()
        summary = await metrics.get_metrics_summary(hours)
        
        return MetricsSummaryResponse(**summary)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics summary: {str(e)}"
        )


@router.get("/counters/{counter_name}")
async def get_counter_value(counter_name: str):
    """Get current value of a counter metric"""
    try:
        metrics = get_metrics_collector()
        value = await metrics.get_counter(counter_name)
        
        return {
            "counter_name": counter_name,
            "value": value,
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get counter value: {str(e)}"
        )


@router.get("/gauges/{gauge_name}")
async def get_gauge_value(gauge_name: str):
    """Get current value of a gauge metric"""
    try:
        metrics = get_metrics_collector()
        value = await metrics.get_gauge(gauge_name)
        
        if value is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Gauge not found"
            )
        
        return {
            "gauge_name": gauge_name,
            "value": value,
            "timestamp": datetime.now()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get gauge value: {str(e)}"
        )


@router.get("/timers/{timer_name}")
async def get_timer_stats(timer_name: str):
    """Get statistics for a timer metric"""
    try:
        metrics = get_metrics_collector()
        stats = await metrics.get_timer_stats(timer_name)
        
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Timer not found"
            )
        
        return {
            "timer_name": timer_name,
            "stats": stats,
            "timestamp": datetime.now()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get timer stats: {str(e)}"
        )


@router.get("/video", response_model=VideoMetrics)
async def get_video_metrics():
    """Get video generation specific metrics"""
    try:
        video_analytics = get_video_analytics()
        metrics = await video_analytics.get_video_metrics()
        
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get video metrics: {str(e)}"
        )


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """Get current system performance metrics"""
    try:
        system_monitor = get_system_monitor()
        metrics = await system_monitor.get_performance_metrics()
        
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@router.post("/video/generation/start")
async def track_video_generation_start(
    user_id: str,
    template_id: Optional[str] = None,
    duration: Optional[float] = None
):
    """Track start of video generation"""
    try:
        video_analytics = get_video_analytics()
        
        await video_analytics.track_video_generation_start(
            user_id=user_id,
            template_id=template_id,
            duration=duration
        )
        
        return {
            "success": True,
            "message": "Video generation start tracked",
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to track video generation start: {str(e)}"
        )


@router.post("/video/generation/complete")
async def track_video_generation_complete(
    user_id: str,
    generation_time: float,
    video_duration: float,
    template_id: Optional[str] = None,
    file_size: Optional[int] = None
):
    """Track successful video generation completion"""
    try:
        video_analytics = get_video_analytics()
        
        await video_analytics.track_video_generation_complete(
            user_id=user_id,
            generation_time=generation_time,
            video_duration=video_duration,
            template_id=template_id,
            file_size=file_size
        )
        
        return {
            "success": True,
            "message": "Video generation completion tracked",
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to track video generation completion: {str(e)}"
        )


@router.post("/video/generation/failed")
async def track_video_generation_failed(
    user_id: str,
    error_type: str,
    error_message: str,
    generation_time: Optional[float] = None
):
    """Track failed video generation"""
    try:
        video_analytics = get_video_analytics()
        
        await video_analytics.track_video_generation_failed(
            user_id=user_id,
            error_type=error_type,
            error_message=error_message,
            generation_time=generation_time
        )
        
        return {
            "success": True,
            "message": "Video generation failure tracked",
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to track video generation failure: {str(e)}"
        )


@router.get("/dashboard")
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        metrics = get_metrics_collector()
        video_analytics = get_video_analytics()
        system_monitor = get_system_monitor()
        
        # Get data from all systems
        metrics_summary = await metrics.get_metrics_summary(24)
        video_metrics = await video_analytics.get_video_metrics()
        performance_metrics = await system_monitor.get_performance_metrics()
        
        return {
            "overview": {
                "timestamp": datetime.now(),
                "time_range": "Last 24 hours"
            },
            "metrics": metrics_summary,
            "video": video_metrics.dict(),
            "performance": performance_metrics.dict(),
            "alerts": [],  # Could be populated with system alerts
            "recommendations": []  # Could include optimization recommendations
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dashboard data: {str(e)}"
        )


@router.get("/event-types")
async def get_event_types():
    """Get all available event types"""
    return {
        "event_types": [
            {"value": event_type.value, "label": event_type.value.replace("_", " ").title()}
            for event_type in EventType
        ]
    }


@router.get("/metric-types")
async def get_metric_types():
    """Get all available metric types"""
    return {
        "metric_types": [
            {"value": metric_type.value, "label": metric_type.value.replace("_", " ").title()}
            for metric_type in MetricType
        ]
    }