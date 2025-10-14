"""
Batch Processing System

Handles bulk video generation, batch operations, and large-scale processing
tasks with queue management, progress tracking, and resource optimization.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

from pydantic import BaseModel, Field
from celery import Celery

from .celery_app import get_celery_app
from analytics import get_metrics_collector, get_video_analytics


class BatchStatus(str, Enum):
    """Status of batch processing jobs"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class BatchType(str, Enum):
    """Types of batch operations"""
    VIDEO_GENERATION = "video_generation"
    TEMPLATE_APPLICATION = "template_application"
    FORMAT_CONVERSION = "format_conversion"
    BULK_EXPORT = "bulk_export"
    QUALITY_ENHANCEMENT = "quality_enhancement"
    THUMBNAIL_GENERATION = "thumbnail_generation"


@dataclass
class BatchItem:
    """Individual item in a batch operation"""
    id: str
    input_data: Dict[str, Any]
    output_path: Optional[str] = None
    status: BatchStatus = BatchStatus.PENDING
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


class BatchJobConfig(BaseModel):
    """Configuration for batch processing jobs"""
    name: str
    description: str = ""
    batch_type: BatchType
    priority: int = Field(default=5, ge=1, le=10)
    max_concurrent: int = Field(default=3, ge=1, le=10)
    retry_limit: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=3600, ge=60)
    notify_on_completion: bool = True
    auto_cleanup: bool = True
    cleanup_after_days: int = Field(default=7, ge=1)


class BatchJob(BaseModel):
    """Complete batch processing job"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    config: BatchJobConfig
    items: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Status tracking
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress tracking
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    
    # Results
    results: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Dict[str, str]] = Field(default_factory=list)
    
    # Metadata
    estimated_duration: Optional[float] = None
    actual_duration: Optional[float] = None


class BatchProcessor:
    """Core batch processing system"""
    
    def __init__(self, storage_dir: str = "batch_jobs"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.celery_app = get_celery_app()
        self.metrics = get_metrics_collector()
        self.video_analytics = get_video_analytics()
        
        # Active batch jobs
        self.active_jobs: Dict[str, BatchJob] = {}
        
        # Processing callbacks for different batch types
        self.processors: Dict[BatchType, Callable] = {
            BatchType.VIDEO_GENERATION: self._process_video_generation,
            BatchType.TEMPLATE_APPLICATION: self._process_template_application,
            BatchType.FORMAT_CONVERSION: self._process_format_conversion,
            BatchType.BULK_EXPORT: self._process_bulk_export,
            BatchType.QUALITY_ENHANCEMENT: self._process_quality_enhancement,
            BatchType.THUMBNAIL_GENERATION: self._process_thumbnail_generation
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def create_batch_job(
        self,
        user_id: str,
        config: BatchJobConfig,
        items: List[Dict[str, Any]]
    ) -> BatchJob:
        """Create a new batch processing job"""
        job = BatchJob(
            user_id=user_id,
            config=config,
            items=items,
            total_items=len(items)
        )
        
        # Save job to storage
        await self._save_job(job)
        
        # Track metrics
        await self.metrics.increment_counter("batch.jobs_created")
        await self.metrics.increment_counter(
            f"batch.{config.batch_type.value}_jobs_created"
        )
        
        self.logger.info(f"Created batch job {job.id} with {len(items)} items")
        
        return job
    
    async def start_batch_job(self, job_id: str) -> bool:
        """Start processing a batch job"""
        job = await self.get_job(job_id)
        if not job:
            return False
        
        if job.status != BatchStatus.PENDING:
            self.logger.warning(f"Job {job_id} is not in pending status: {job.status}")
            return False
        
        # Update job status
        job.status = BatchStatus.PROCESSING
        job.started_at = datetime.now()
        
        self.active_jobs[job_id] = job
        
        # Submit to Celery for processing
        self.celery_app.send_task(
            'task_queue.batch_processor.process_batch_job_task',
            args=[job_id],
            queue='batch_processing',
            priority=job.config.priority
        )
        
        await self._save_job(job)
        
        # Track metrics
        await self.metrics.increment_counter("batch.jobs_started")
        
        self.logger.info(f"Started batch job {job_id}")
        
        return True
    
    async def process_batch_job(self, job_id: str):
        """Process a batch job - main processing logic"""
        job = await self.get_job(job_id)
        if not job:
            self.logger.error(f"Job {job_id} not found")
            return
        
        processor = self.processors.get(job.config.batch_type)
        if not processor:
            self.logger.error(f"No processor for batch type: {job.config.batch_type}")
            job.status = BatchStatus.FAILED
            job.errors.append({
                "error": f"No processor available for {job.config.batch_type}",
                "timestamp": datetime.now().isoformat()
            })
            await self._save_job(job)
            return
        
        try:
            # Process items with concurrency control
            semaphore = asyncio.Semaphore(job.config.max_concurrent)
            tasks = []
            
            for i, item_data in enumerate(job.items):
                item = BatchItem(
                    id=f"{job_id}_{i}",
                    input_data=item_data
                )
                
                task = self._process_batch_item(semaphore, job, item, processor)
                tasks.append(task)
            
            # Wait for all items to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update job completion
            job.completed_at = datetime.now()
            job.actual_duration = (job.completed_at - job.started_at).total_seconds()
            
            # Determine final status
            if job.failed_items == 0:
                job.status = BatchStatus.COMPLETED
            elif job.completed_items > 0:
                job.status = BatchStatus.COMPLETED  # Partial success still counts as completed
            else:
                job.status = BatchStatus.FAILED
            
            # Clean up active job
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            await self._save_job(job)
            
            # Track completion metrics
            await self.metrics.increment_counter("batch.jobs_completed")
            await self.metrics.record_timer("batch.job_duration", job.actual_duration)
            
            self.logger.info(
                f"Completed batch job {job_id}: "
                f"{job.completed_items} successful, {job.failed_items} failed"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing batch job {job_id}: {e}")
            job.status = BatchStatus.FAILED
            job.completed_at = datetime.now()
            job.errors.append({
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            await self._save_job(job)
            await self.metrics.increment_counter("batch.jobs_failed")
    
    async def _process_batch_item(
        self,
        semaphore: asyncio.Semaphore,
        job: BatchJob,
        item: BatchItem,
        processor: Callable
    ):
        """Process a single batch item with concurrency control"""
        async with semaphore:
            try:
                item.started_at = datetime.now()
                item.status = BatchStatus.PROCESSING
                
                # Call the appropriate processor
                result = await processor(job, item)
                
                item.completed_at = datetime.now()
                item.status = BatchStatus.COMPLETED
                
                # Store result
                job.results[item.id] = result
                job.completed_items += 1
                
                # Track progress
                await self.metrics.set_gauge(
                    f"batch.job.{job.id}.progress",
                    job.completed_items / job.total_items * 100
                )
                
            except Exception as e:
                item.completed_at = datetime.now()
                item.status = BatchStatus.FAILED
                item.error_message = str(e)
                item.retry_count += 1
                
                job.failed_items += 1
                job.errors.append({
                    "item_id": item.id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
                self.logger.error(f"Error processing item {item.id}: {e}")
                
                # Retry logic
                if item.retry_count < job.config.retry_limit:
                    self.logger.info(f"Retrying item {item.id} ({item.retry_count}/{job.config.retry_limit})")
                    await asyncio.sleep(2 ** item.retry_count)  # Exponential backoff
                    return await self._process_batch_item(semaphore, job, item, processor)
            
            finally:
                # Update job progress periodically
                if (job.completed_items + job.failed_items) % 10 == 0:
                    await self._save_job(job)
    
    async def _process_video_generation(self, job: BatchJob, item: BatchItem) -> Dict[str, Any]:
        """Process video generation batch item"""
        input_data = item.input_data
        
        # Track video generation start
        await self.video_analytics.track_video_generation_start(
            user_id=job.user_id,
            template_id=input_data.get('template_id'),
            duration=input_data.get('duration')
        )
        
        # Simulate video generation (replace with actual implementation)
        start_time = datetime.now()
        
        # Here would be the actual video generation logic
        # For now, we'll simulate processing time
        await asyncio.sleep(2)
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Generate output path
        output_path = f"batch_output/{job.id}/{item.id}_video.mp4"
        item.output_path = output_path
        
        # Track successful completion
        await self.video_analytics.track_video_generation_complete(
            user_id=job.user_id,
            generation_time=generation_time,
            video_duration=input_data.get('duration', 30.0),
            template_id=input_data.get('template_id'),
            file_size=1024*1024  # Placeholder file size
        )
        
        return {
            "output_path": output_path,
            "generation_time": generation_time,
            "status": "success"
        }
    
    async def _process_template_application(self, job: BatchJob, item: BatchItem) -> Dict[str, Any]:
        """Process template application batch item"""
        input_data = item.input_data
        template_id = input_data.get('template_id')
        customizations = input_data.get('customizations', {})
        
        # Apply template (placeholder implementation)
        await asyncio.sleep(1)
        
        output_path = f"batch_output/{job.id}/{item.id}_templated.json"
        item.output_path = output_path
        
        return {
            "output_path": output_path,
            "template_id": template_id,
            "customizations_applied": len(customizations),
            "status": "success"
        }
    
    async def _process_format_conversion(self, job: BatchJob, item: BatchItem) -> Dict[str, Any]:
        """Process format conversion batch item"""
        input_data = item.input_data
        input_path = input_data.get('input_path')
        target_format = input_data.get('target_format')
        
        # Format conversion (placeholder implementation)
        await asyncio.sleep(3)
        
        output_path = f"batch_output/{job.id}/{item.id}.{target_format}"
        item.output_path = output_path
        
        return {
            "input_path": input_path,
            "output_path": output_path,
            "target_format": target_format,
            "status": "success"
        }
    
    async def _process_bulk_export(self, job: BatchJob, item: BatchItem) -> Dict[str, Any]:
        """Process bulk export batch item"""
        input_data = item.input_data
        export_format = input_data.get('export_format', 'mp4')
        quality = input_data.get('quality', 'high')
        
        # Bulk export (placeholder implementation)
        await asyncio.sleep(2.5)
        
        output_path = f"batch_output/{job.id}/{item.id}_export.{export_format}"
        item.output_path = output_path
        
        return {
            "output_path": output_path,
            "export_format": export_format,
            "quality": quality,
            "status": "success"
        }
    
    async def _process_quality_enhancement(self, job: BatchJob, item: BatchItem) -> Dict[str, Any]:
        """Process quality enhancement batch item"""
        input_data = item.input_data
        input_path = input_data.get('input_path')
        enhancement_type = input_data.get('enhancement_type', 'upscale')
        
        # Quality enhancement (placeholder implementation)
        await asyncio.sleep(5)
        
        output_path = f"batch_output/{job.id}/{item.id}_enhanced.mp4"
        item.output_path = output_path
        
        return {
            "input_path": input_path,
            "output_path": output_path,
            "enhancement_type": enhancement_type,
            "status": "success"
        }
    
    async def _process_thumbnail_generation(self, job: BatchJob, item: BatchItem) -> Dict[str, Any]:
        """Process thumbnail generation batch item"""
        input_data = item.input_data
        video_path = input_data.get('video_path')
        timestamp = input_data.get('timestamp', 5.0)
        
        # Thumbnail generation (placeholder implementation)
        await asyncio.sleep(1)
        
        output_path = f"batch_output/{job.id}/{item.id}_thumb.jpg"
        item.output_path = output_path
        
        return {
            "video_path": video_path,
            "output_path": output_path,
            "timestamp": timestamp,
            "status": "success"
        }
    
    async def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get a batch job by ID"""
        # Check active jobs first
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Load from storage
        return await self._load_job(job_id)
    
    async def get_jobs_by_user(self, user_id: str) -> List[BatchJob]:
        """Get all batch jobs for a user"""
        jobs = []
        
        # Check active jobs
        for job in self.active_jobs.values():
            if job.user_id == user_id:
                jobs.append(job)
        
        # Load from storage
        for job_file in self.storage_dir.glob(f"*_{user_id}_*.json"):
            try:
                job = await self._load_job_from_file(job_file)
                if job and job.id not in self.active_jobs:
                    jobs.append(job)
            except Exception as e:
                self.logger.error(f"Error loading job from {job_file}: {e}")
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job"""
        job = await self.get_job(job_id)
        if not job:
            return False
        
        if job.status not in [BatchStatus.PENDING, BatchStatus.PROCESSING]:
            return False
        
        job.status = BatchStatus.CANCELLED
        job.completed_at = datetime.now()
        
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        
        await self._save_job(job)
        await self.metrics.increment_counter("batch.jobs_cancelled")
        
        self.logger.info(f"Cancelled batch job {job_id}")
        
        return True
    
    async def get_job_progress(self, job_id: str) -> Dict[str, Any]:
        """Get detailed progress information for a job"""
        job = await self.get_job(job_id)
        if not job:
            return {}
        
        progress_percentage = 0
        if job.total_items > 0:
            progress_percentage = (job.completed_items + job.failed_items) / job.total_items * 100
        
        estimated_time_remaining = None
        if job.started_at and job.completed_items > 0:
            elapsed = (datetime.now() - job.started_at).total_seconds()
            avg_time_per_item = elapsed / (job.completed_items + job.failed_items)
            remaining_items = job.total_items - job.completed_items - job.failed_items
            estimated_time_remaining = avg_time_per_item * remaining_items
        
        return {
            "job_id": job_id,
            "status": job.status,
            "progress_percentage": progress_percentage,
            "total_items": job.total_items,
            "completed_items": job.completed_items,
            "failed_items": job.failed_items,
            "remaining_items": job.total_items - job.completed_items - job.failed_items,
            "estimated_time_remaining": estimated_time_remaining,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "estimated_completion": (datetime.now() + timedelta(seconds=estimated_time_remaining)).isoformat() if estimated_time_remaining else None
        }
    
    async def cleanup_old_jobs(self, days: int = 7):
        """Clean up old completed jobs"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        for job_file in self.storage_dir.glob("*.json"):
            try:
                job = await self._load_job_from_file(job_file)
                if job and job.completed_at and job.completed_at < cutoff_date:
                    if job.config.auto_cleanup:
                        job_file.unlink()
                        cleaned_count += 1
                        
            except Exception as e:
                self.logger.error(f"Error during cleanup of {job_file}: {e}")
        
        self.logger.info(f"Cleaned up {cleaned_count} old batch jobs")
        return cleaned_count
    
    async def _save_job(self, job: BatchJob):
        """Save job to storage"""
        job_file = self.storage_dir / f"batch_{job.user_id}_{job.id}.json"
        
        try:
            with open(job_file, 'w', encoding='utf-8') as f:
                json.dump(job.dict(), f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self.logger.error(f"Error saving job {job.id}: {e}")
    
    async def _load_job(self, job_id: str) -> Optional[BatchJob]:
        """Load job from storage"""
        for job_file in self.storage_dir.glob(f"*_{job_id}.json"):
            return await self._load_job_from_file(job_file)
        return None
    
    async def _load_job_from_file(self, job_file: Path) -> Optional[BatchJob]:
        """Load job from a specific file"""
        try:
            with open(job_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return BatchJob(**data)
        except Exception as e:
            self.logger.error(f"Error loading job from {job_file}: {e}")
            return None


# Global instance
_batch_processor = None

def get_batch_processor() -> BatchProcessor:
    """Get the global batch processor instance"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor