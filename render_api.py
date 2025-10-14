"""
Render Engine API

RESTful API endpoints for the render engine system,
providing access to FFmpeg rendering, quality validation, and render queue management.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from render_engine import (
    FFmpegRenderer, 
    get_render_manager,
    QualityValidator
)
from render_engine.ffmpeg_renderer import RenderConfig
from render_engine.render_manager import RenderStatus

# Configure router
render_router = APIRouter(prefix="/render", tags=["render"])
logger = logging.getLogger(__name__)

# Pydantic models
class RenderRequest(BaseModel):
    """Render request model"""
    composition_data: Dict[str, Any] = Field(..., description="Video composition data")
    output_filename: str = Field(..., description="Output filename", min_length=1)
    render_config: Optional[Dict[str, Any]] = Field(None, description="Render configuration")
    priority: int = Field(0, description="Task priority (higher = more priority)")
    
    class Config:
        schema_extra = {
            "example": {
                "composition_data": {
                    "layers": [
                        {
                            "layer_type": "video",
                            "source_path": "/path/to/video.mp4",
                            "start_time": 0.0,
                            "duration": 10.0
                        }
                    ],
                    "duration": 10.0,
                    "resolution": [1920, 1080],
                    "fps": 30
                },
                "output_filename": "final_video.mp4",
                "priority": 1
            }
        }


class RenderResponse(BaseModel):
    """Render response model"""
    task_id: str
    status: str
    message: str
    timestamp: datetime


class QualityRequest(BaseModel):
    """Quality validation request"""
    video_path: str = Field(..., description="Path to video file for validation")


class RenderConfigModel(BaseModel):
    """Render configuration model"""
    video_codec: str = Field("libx264", description="Video codec")
    audio_codec: str = Field("aac", description="Audio codec")
    pixel_format: str = Field("yuv420p", description="Pixel format")
    frame_rate: int = Field(30, description="Frame rate")
    crf: int = Field(23, description="Constant Rate Factor (0-51)")
    preset: str = Field("medium", description="Encoding preset")
    resolution: Optional[List[int]] = Field(None, description="Video resolution [width, height]")
    video_bitrate: Optional[str] = Field(None, description="Video bitrate (e.g., '2M', '5000k')")
    audio_bitrate: str = Field("128k", description="Audio bitrate")
    hardware_accel: Optional[str] = Field(None, description="Hardware acceleration")
    two_pass: bool = Field(False, description="Enable two-pass encoding")


# API Endpoints

@render_router.post("/submit", response_model=RenderResponse)
async def submit_render_task(request: RenderRequest):
    """Submit a new render task to the queue"""
    try:
        render_manager = get_render_manager()
        
        # Build output path
        output_path = f"/tmp/aura_render_outputs/{request.output_filename}"
        
        # Create render config
        render_config = RenderConfig()
        if request.render_config:
            # Update config with provided values
            for key, value in request.render_config.items():
                if hasattr(render_config, key):
                    setattr(render_config, key, value)
        
        # Submit task
        task_id = await render_manager.submit_render_task(
            composition_data=request.composition_data,
            output_path=output_path,
            render_config=render_config,
            priority=request.priority
        )
        
        return RenderResponse(
            task_id=task_id,
            status="queued",
            message="Render task submitted successfully",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to submit render task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit render task: {str(e)}"
        )


@render_router.get("/task/{task_id}")
async def get_render_task_status(task_id: str):
    """Get render task status and progress"""
    try:
        render_manager = get_render_manager()
        task_status = await render_manager.get_task_status(task_id)
        
        if task_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Render task not found"
            )
        
        return task_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )


@render_router.delete("/task/{task_id}")
async def cancel_render_task(task_id: str):
    """Cancel a render task"""
    try:
        render_manager = get_render_manager()
        success = await render_manager.cancel_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Render task not found or cannot be cancelled"
            )
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Render task cancelled successfully",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}"
        )


@render_router.get("/queue")
async def get_render_queue_status():
    """Get render queue status and statistics"""
    try:
        render_manager = get_render_manager()
        queue_status = await render_manager.get_queue_status()
        return queue_status
        
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue status: {str(e)}"
        )


@render_router.post("/validate-quality")
async def validate_video_quality(request: QualityRequest):
    """Validate video quality and get metrics"""
    try:
        validator = QualityValidator()
        metrics = await validator.validate_video(request.video_path)
        
        return {
            "quality_metrics": {
                "overall_quality": metrics.overall_quality,
                "video_quality": metrics.video_quality,
                "audio_quality": metrics.audio_quality,
                "file_size_mb": metrics.file_size_mb,
                "duration_seconds": metrics.duration_seconds,
                "resolution": metrics.resolution,
                "frame_rate": metrics.frame_rate,
                "bitrate_kbps": metrics.bitrate_kbps,
                "video_codec": metrics.video_codec,
                "audio_codec": metrics.audio_codec
            },
            "analysis": {
                "has_issues": metrics.has_issues,
                "issues": metrics.issues,
                "recommendations": metrics.recommendations
            },
            "validation_timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Quality validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality validation failed: {str(e)}"
        )


@render_router.get("/quality-report/{video_path:path}")
async def get_quality_report(video_path: str):
    """Get formatted quality report for video"""
    try:
        validator = QualityValidator()
        metrics = await validator.validate_video(video_path)
        report = validator.format_quality_report(metrics)
        
        return {
            "video_path": video_path,
            "report": report,
            "metrics": metrics.__dict__,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Quality report generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality report generation failed: {str(e)}"
        )


@render_router.get("/presets")
async def get_render_presets():
    """Get available render configuration presets"""
    presets = {
        "high_quality": {
            "description": "High quality output for professional use",
            "config": {
                "video_codec": "libx264",
                "crf": 18,
                "preset": "slow",
                "pixel_format": "yuv420p",
                "audio_bitrate": "320k",
                "two_pass": True
            }
        },
        "web_optimized": {
            "description": "Optimized for web streaming",
            "config": {
                "video_codec": "libx264",
                "crf": 23,
                "preset": "medium",
                "pixel_format": "yuv420p",
                "audio_bitrate": "128k",
                "movflags": "faststart"
            }
        },
        "mobile_optimized": {
            "description": "Optimized for mobile devices",
            "config": {
                "video_codec": "libx264",
                "crf": 26,
                "preset": "fast",
                "resolution": [1280, 720],
                "audio_bitrate": "96k"
            }
        },
        "fast_encode": {
            "description": "Fast encoding for previews",
            "config": {
                "video_codec": "libx264",
                "crf": 28,
                "preset": "ultrafast",
                "audio_bitrate": "96k"
            }
        },
        "hardware_accelerated": {
            "description": "Hardware accelerated encoding (NVIDIA)",
            "config": {
                "video_codec": "h264_nvenc",
                "preset": "medium",
                "hardware_accel": "cuda",
                "audio_bitrate": "128k"
            }
        }
    }
    
    return {
        "presets": presets,
        "total_presets": len(presets),
        "timestamp": datetime.now()
    }


@render_router.get("/capabilities")
async def get_render_capabilities():
    """Get render engine capabilities and system information"""
    try:
        # Test FFmpeg availability
        renderer = FFmpegRenderer()
        ffmpeg_available = True
        ffmpeg_error = None
        
        try:
            # This will validate FFmpeg during initialization
            pass
        except Exception as e:
            ffmpeg_available = False
            ffmpeg_error = str(e)
        
        # Get render manager stats
        render_manager = get_render_manager()
        queue_status = await render_manager.get_queue_status()
        
        capabilities = {
            "render_engine": {
                "ffmpeg_available": ffmpeg_available,
                "ffmpeg_error": ffmpeg_error,
                "supported_codecs": {
                    "video": ["libx264", "libx265", "h264_nvenc", "h265_nvenc"],
                    "audio": ["aac", "mp3", "opus", "flac"]
                },
                "supported_formats": ["mp4", "mov", "avi", "mkv", "webm"],
                "hardware_acceleration": ["cuda", "videotoolbox", "qsv", "vaapi"]
            },
            "render_manager": {
                "max_concurrent_renders": render_manager.max_concurrent_renders,
                "queue_status": queue_status
            },
            "quality_validator": {
                "available": True,
                "supported_formats": ["mp4", "mov", "avi", "mkv", "webm", "flv"]
            },
            "system_info": {
                "timestamp": datetime.now()
            }
        }
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Failed to get capabilities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get capabilities: {str(e)}"
        )


@render_router.post("/test-render")
async def test_render():
    """Test render functionality with a simple composition"""
    try:
        # Create simple test composition
        test_composition = {
            "layers": [
                {
                    "layer_type": "video",
                    "color": "#1a1a1a",  # Dark background
                    "duration": 5.0,
                    "start_time": 0.0,
                    "opacity": 1.0
                }
            ],
            "duration": 5.0,
            "resolution": (1280, 720),
            "fps": 30
        }
        
        render_manager = get_render_manager()
        
        task_id = await render_manager.submit_render_task(
            composition_data=test_composition,
            output_path="/tmp/aura_render_outputs/test_render.mp4",
            priority=10  # High priority for test
        )
        
        return {
            "task_id": task_id,
            "status": "submitted",
            "message": "Test render task submitted",
            "composition": test_composition,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Test render failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test render failed: {str(e)}"
        )