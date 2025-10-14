"""
Batch Processing API Endpoints

REST API endpoints for batch job management and bulk operations.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from task_queue.batch_processor import (
    get_batch_processor,
    BatchJobConfig,
    BatchType,
    BatchStatus,
    BatchJob
)

router = APIRouter(prefix="/api/batch", tags=["batch"])


class BatchJobConfigRequest(BaseModel):
    """Request model for batch job configuration"""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field("", max_length=500)
    batch_type: BatchType
    priority: int = Field(default=5, ge=1, le=10)
    max_concurrent: int = Field(default=3, ge=1, le=10)
    retry_limit: int = Field(default=3, ge=0, le=10)


class VideoGenerationRequest(BaseModel):
    """Request model for bulk video generation"""
    user_id: str = Field(..., min_length=1)
    items: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100)
    priority: int = Field(default=5, ge=1, le=10)
    max_concurrent: int = Field(default=2, ge=1, le=5)


class FormatConversionRequest(BaseModel):
    """Request model for bulk format conversion"""
    user_id: str = Field(..., min_length=1)
    items: List[Dict[str, Any]] = Field(..., min_items=1, max_items=200)
    priority: int = Field(default=3, ge=1, le=10)
    max_concurrent: int = Field(default=5, ge=1, le=10)
    timeout_seconds: int = Field(default=3600, ge=60)
    notify_on_completion: bool = True
    auto_cleanup: bool = True
    cleanup_after_days: int = Field(default=7, ge=1)


class CreateBatchJobRequest(BaseModel):
    """Request model for creating batch jobs"""
    user_id: str
    config: BatchJobConfigRequest
    items: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000)


class BatchJobResponse(BaseModel):
    """Response model for batch jobs"""
    id: str
    user_id: str
    config: Dict[str, Any]
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_items: int
    completed_items: int
    failed_items: int
    progress_percentage: float
    estimated_duration: Optional[float]
    actual_duration: Optional[float]


class ProgressResponse(BaseModel):
    """Response model for job progress"""
    job_id: str
    status: str
    progress_percentage: float
    total_items: int
    completed_items: int
    failed_items: int
    remaining_items: int
    estimated_time_remaining: Optional[float]
    started_at: Optional[datetime]
    estimated_completion: Optional[datetime]


@router.post("/", response_model=BatchJobResponse)
async def create_batch_job(request: CreateBatchJobRequest):
    """Create a new batch processing job"""
    try:
        batch_processor = get_batch_processor()
        
        # Convert request to batch job config
        config = BatchJobConfig(
            name=request.config.name,
            description=request.config.description,
            batch_type=request.config.batch_type,
            priority=request.config.priority,
            max_concurrent=request.config.max_concurrent,
            retry_limit=request.config.retry_limit,
            timeout_seconds=request.config.timeout_seconds,
            notify_on_completion=request.config.notify_on_completion,
            auto_cleanup=request.config.auto_cleanup,
            cleanup_after_days=request.config.cleanup_after_days
        )
        
        # Create the batch job
        job = await batch_processor.create_batch_job(
            user_id=request.user_id,
            config=config,
            items=request.items
        )
        
        return BatchJobResponse(
            id=job.id,
            user_id=job.user_id,
            config=job.config.dict(),
            status=job.status.value,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            total_items=job.total_items,
            completed_items=job.completed_items,
            failed_items=job.failed_items,
            progress_percentage=job.completed_items / job.total_items * 100 if job.total_items > 0 else 0,
            estimated_duration=job.estimated_duration,
            actual_duration=job.actual_duration
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create batch job: {str(e)}"
        )


@router.post("/{job_id}/start")
async def start_batch_job(job_id: str):
    """Start processing a batch job"""
    try:
        batch_processor = get_batch_processor()
        
        success = await batch_processor.start_batch_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job cannot be started (not found or invalid status)"
            )
        
        return {
            "success": True,
            "message": f"Batch job {job_id} started successfully",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start batch job: {str(e)}"
        )


@router.get("/{job_id}", response_model=BatchJobResponse)
async def get_batch_job(job_id: str):
    """Get batch job details"""
    try:
        batch_processor = get_batch_processor()
        
        job = await batch_processor.get_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Batch job not found"
            )
        
        return BatchJobResponse(
            id=job.id,
            user_id=job.user_id,
            config=job.config.dict(),
            status=job.status.value,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            total_items=job.total_items,
            completed_items=job.completed_items,
            failed_items=job.failed_items,
            progress_percentage=job.completed_items / job.total_items * 100 if job.total_items > 0 else 0,
            estimated_duration=job.estimated_duration,
            actual_duration=job.actual_duration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batch job: {str(e)}"
        )


@router.get("/{job_id}/progress", response_model=ProgressResponse)
async def get_job_progress(job_id: str):
    """Get detailed progress information for a job"""
    try:
        batch_processor = get_batch_processor()
        
        progress = await batch_processor.get_job_progress(job_id)
        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Batch job not found"
            )
        
        return ProgressResponse(**progress)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job progress: {str(e)}"
        )


@router.post("/{job_id}/cancel")
async def cancel_batch_job(job_id: str):
    """Cancel a batch job"""
    try:
        batch_processor = get_batch_processor()
        
        success = await batch_processor.cancel_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job cannot be cancelled (not found or invalid status)"
            )
        
        return {
            "success": True,
            "message": f"Batch job {job_id} cancelled successfully",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel batch job: {str(e)}"
        )


@router.get("/user/{user_id}", response_model=List[BatchJobResponse])
async def get_user_batch_jobs(user_id: str):
    """Get all batch jobs for a user"""
    try:
        batch_processor = get_batch_processor()
        
        jobs = await batch_processor.get_jobs_by_user(user_id)
        
        return [
            BatchJobResponse(
                id=job.id,
                user_id=job.user_id,
                config=job.config.dict() if hasattr(job.config, 'dict') else job.config,
                status=job.status.value,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                total_items=job.total_items,
                completed_items=job.completed_items,
                failed_items=job.failed_items,
                progress_percentage=job.completed_items / job.total_items * 100 if job.total_items > 0 else 0,
                estimated_duration=job.estimated_duration,
                actual_duration=job.actual_duration
            )
            for job in jobs
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user batch jobs: {str(e)}"
        )


@router.get("/types/list")
async def get_batch_types():
    """Get all available batch processing types"""
    return {
        "batch_types": [
            {
                "value": batch_type.value,
                "label": batch_type.value.replace("_", " ").title(),
                "description": f"Batch processing for {batch_type.value.replace('_', ' ')}"
            }
            for batch_type in BatchType
        ]
    }


@router.get("/status/list") 
async def get_batch_statuses():
    """Get all possible batch job statuses"""
    return {
        "statuses": [
            {
                "value": status.value,
                "label": status.value.replace("_", " ").title()
            }
            for status in BatchStatus
        ]
    }


@router.post("/cleanup")
async def cleanup_old_jobs(days: int = 7):
    """Clean up old completed batch jobs"""
    try:
        if days < 1 or days > 365:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Days must be between 1 and 365"
            )
        
        batch_processor = get_batch_processor()
        cleaned_count = await batch_processor.cleanup_old_jobs(days)
        
        return {
            "success": True,
            "message": f"Cleaned up {cleaned_count} old batch jobs",
            "cleaned_count": cleaned_count,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup old jobs: {str(e)}"
        )


# Bulk operation shortcuts

@router.post("/video-generation")
async def create_bulk_video_generation(request: VideoGenerationRequest):
    """Create a bulk video generation batch job"""
    try:
        batch_processor = get_batch_processor()
        
        config = BatchJobConfig(
            name="Bulk Video Generation",
            description=f"Generate {len(request.items)} videos in batch",
            batch_type=BatchType.VIDEO_GENERATION,
            priority=request.priority,
            max_concurrent=request.max_concurrent
        )
        
        job = await batch_processor.create_batch_job(
            user_id=request.user_id,
            config=config,
            items=request.items
        )
        
        # Auto-start the job
        await batch_processor.start_batch_job(job.id)
        
        return {
            "success": True,
            "job_id": job.id,
            "message": f"Bulk video generation started for {len(request.items)} videos",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create bulk video generation: {str(e)}"
        )


@router.post("/format-conversion")
async def create_bulk_format_conversion(request: FormatConversionRequest):
    """Create a bulk format conversion batch job"""
    try:
        batch_processor = get_batch_processor()
        
        config = BatchJobConfig(
            name="Bulk Format Conversion",
            description=f"Convert {len(request.items)} files in batch",
            batch_type=BatchType.FORMAT_CONVERSION,
            priority=request.priority,
            max_concurrent=request.max_concurrent,
            timeout_seconds=request.timeout_seconds,
            notify_on_completion=request.notify_on_completion,
            auto_cleanup=request.auto_cleanup,
            cleanup_after_days=request.cleanup_after_days
        )
        
        job = await batch_processor.create_batch_job(
            user_id=request.user_id,
            config=config,
            items=request.items
        )
        
        # Auto-start the job
        await batch_processor.start_batch_job(job.id)
        
        return {
            "success": True,
            "job_id": job.id,
            "message": f"Bulk format conversion started for {len(request.items)} files",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create bulk format conversion: {str(e)}"
        )