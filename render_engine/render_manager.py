"""
Render Manager

High-level render task management with queue handling,
progress tracking, and resource management.
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path

from .ffmpeg_renderer import FFmpegRenderer, RenderConfig


class RenderStatus(Enum):
    """Render task status"""
    QUEUED = "queued"
    PREPARING = "preparing"
    RENDERING = "rendering"
    POST_PROCESSING = "post_processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RenderTask:
    """Render task specification"""
    task_id: str
    composition_data: Dict[str, Any]
    output_path: str
    render_config: RenderConfig
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: RenderStatus = RenderStatus.QUEUED
    progress: float = 0.0
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    
    # Callbacks
    progress_callback: Optional[Callable[[float], None]] = None
    completion_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    
    @property
    def processing_time(self) -> Optional[float]:
        """Get processing time in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None


class RenderManager:
    """High-level render task manager"""
    
    def __init__(self, max_concurrent_renders: int = 2):
        self.max_concurrent_renders = max_concurrent_renders
        self.active_tasks: Dict[str, RenderTask] = {}
        self.render_queue: List[RenderTask] = []
        self.completed_tasks: Dict[str, RenderTask] = {}
        self.render_semaphore = asyncio.Semaphore(max_concurrent_renders)
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'average_render_time': 0.0,
            'total_render_time': 0.0
        }
        
        # Start queue processor
        self._queue_processor_task = None
        self._start_queue_processor()
    
    def _start_queue_processor(self):
        """Start the queue processor task"""
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(self._process_queue())
    
    async def submit_render_task(
        self,
        composition_data: Dict[str, Any],
        output_path: str,
        render_config: RenderConfig = None,
        priority: int = 0,
        progress_callback: Optional[Callable[[float], None]] = None,
        completion_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> str:
        """
        Submit a new render task
        
        Args:
            composition_data: Video composition data
            output_path: Output file path
            render_config: Render configuration
            priority: Task priority (higher = more priority)
            progress_callback: Progress update callback
            completion_callback: Completion callback
            
        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())
        
        task = RenderTask(
            task_id=task_id,
            composition_data=composition_data,
            output_path=output_path,
            render_config=render_config or RenderConfig(),
            priority=priority,
            progress_callback=progress_callback,
            completion_callback=completion_callback
        )
        
        # Add to queue
        self.render_queue.append(task)
        self.render_queue.sort(key=lambda t: t.priority, reverse=True)
        
        self.stats['total_tasks'] += 1
        
        self.logger.info(f"Render task submitted: {task_id}")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get render task status"""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
        # Check completed tasks
        elif task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
        # Check queued tasks
        else:
            task = next((t for t in self.render_queue if t.task_id == task_id), None)
        
        if task is None:
            return None
        
        return {
            'task_id': task.task_id,
            'status': task.status.value,
            'progress': task.progress,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'processing_time': task.processing_time,
            'error_message': task.error_message,
            'output_path': task.output_path,
            'result': task.result
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a render task"""
        # Remove from queue if not started
        for i, task in enumerate(self.render_queue):
            if task.task_id == task_id:
                task.status = RenderStatus.CANCELLED
                self.render_queue.pop(i)
                self.completed_tasks[task_id] = task
                self.stats['cancelled_tasks'] += 1
                self.logger.info(f"Render task cancelled: {task_id}")
                return True
        
        # Mark active task for cancellation
        if task_id in self.active_tasks:
            self.active_tasks[task_id].status = RenderStatus.CANCELLED
            self.logger.info(f"Active render task marked for cancellation: {task_id}")
            return True
        
        return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get render queue status"""
        return {
            'queued_tasks': len(self.render_queue),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'max_concurrent': self.max_concurrent_renders,
            'statistics': self.stats.copy(),
            'queue_details': [
                {
                    'task_id': task.task_id,
                    'priority': task.priority,
                    'created_at': task.created_at.isoformat(),
                    'output_path': Path(task.output_path).name
                }
                for task in self.render_queue[:10]  # Show first 10
            ],
            'active_details': [
                {
                    'task_id': task.task_id,
                    'progress': task.progress,
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'processing_time': task.processing_time,
                    'output_path': Path(task.output_path).name
                }
                for task in self.active_tasks.values()
            ]
        }
    
    async def _process_queue(self):
        """Process render queue continuously"""
        while True:
            try:
                # Check if we can start a new task
                if self.render_queue and len(self.active_tasks) < self.max_concurrent_renders:
                    # Get highest priority task
                    task = self.render_queue.pop(0)
                    
                    # Check if task was cancelled
                    if task.status == RenderStatus.CANCELLED:
                        continue
                    
                    # Start rendering task
                    asyncio.create_task(self._execute_render_task(task))
                
                # Wait before checking again
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    async def _execute_render_task(self, task: RenderTask):
        """Execute a single render task"""
        async with self.render_semaphore:
            try:
                # Mark as active
                self.active_tasks[task.task_id] = task
                task.status = RenderStatus.PREPARING
                task.started_at = datetime.now()
                
                self.logger.info(f"Starting render task: {task.task_id}")
                
                # Create renderer with task's config
                renderer = FFmpegRenderer(task.render_config)
                
                # Wrapper for progress callback
                def progress_wrapper(progress: float):
                    task.progress = progress
                    if task.progress_callback:
                        try:
                            task.progress_callback(progress)
                        except Exception as e:
                            self.logger.error(f"Progress callback error: {e}")
                
                # Check for cancellation before starting
                if task.status == RenderStatus.CANCELLED:
                    raise asyncio.CancelledError("Task was cancelled")
                
                task.status = RenderStatus.RENDERING
                
                # Execute rendering
                result = await renderer.render_video(
                    composition_data=task.composition_data,
                    output_path=task.output_path,
                    progress_callback=progress_wrapper
                )
                
                # Check for cancellation after rendering
                if task.status == RenderStatus.CANCELLED:
                    # Clean up output file if created
                    if os.path.exists(task.output_path):
                        os.remove(task.output_path)
                    raise asyncio.CancelledError("Task was cancelled")
                
                task.status = RenderStatus.POST_PROCESSING
                task.progress = 95.0
                
                # Post-processing (if needed)
                await self._post_process_render(task, result)
                
                # Mark as completed
                task.status = RenderStatus.COMPLETED
                task.progress = 100.0
                task.completed_at = datetime.now()
                task.result = result
                
                # Update statistics
                self._update_statistics(task)
                
                # Call completion callback
                if task.completion_callback:
                    try:
                        task.completion_callback(result)
                    except Exception as e:
                        self.logger.error(f"Completion callback error: {e}")
                
                self.logger.info(f"Render task completed: {task.task_id}")
                
            except asyncio.CancelledError:
                task.status = RenderStatus.CANCELLED
                task.completed_at = datetime.now()
                task.error_message = "Task was cancelled"
                self.stats['cancelled_tasks'] += 1
                self.logger.info(f"Render task cancelled: {task.task_id}")
                
            except Exception as e:
                task.status = RenderStatus.FAILED
                task.completed_at = datetime.now()
                task.error_message = str(e)
                self.stats['failed_tasks'] += 1
                self.logger.error(f"Render task failed: {task.task_id} - {e}")
                
            finally:
                # Move from active to completed
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks[task.task_id] = task
                
                # Cleanup old completed tasks (keep last 100)
                if len(self.completed_tasks) > 100:
                    oldest_tasks = sorted(
                        self.completed_tasks.values(),
                        key=lambda t: t.completed_at or datetime.min
                    )
                    for old_task in oldest_tasks[:-100]:
                        del self.completed_tasks[old_task.task_id]
    
    async def _post_process_render(self, task: RenderTask, result: Dict[str, Any]):
        """Post-process rendered video"""
        if not result.get('success', False):
            return
        
        # Verify output file exists and is valid
        output_path = task.output_path
        if not Path(output_path).exists():
            raise RuntimeError("Output file was not created")
        
        file_size = Path(output_path).stat().st_size
        if file_size == 0:
            raise RuntimeError("Output file is empty")
        
        # Update result with file size
        result['file_size'] = file_size
        
        # Additional post-processing could be added here:
        # - Thumbnail generation
        # - Upload to storage
        # - Metadata extraction
        # - Quality analysis
    
    def _update_statistics(self, task: RenderTask):
        """Update render statistics"""
        if task.status == RenderStatus.COMPLETED:
            self.stats['completed_tasks'] += 1
            
            if task.processing_time:
                self.stats['total_render_time'] += task.processing_time
                self.stats['average_render_time'] = (
                    self.stats['total_render_time'] / self.stats['completed_tasks']
                )
    
    async def cleanup(self):
        """Cleanup render manager resources"""
        # Cancel queue processor
        if self._queue_processor_task and not self._queue_processor_task.done():
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active tasks
        for task_id in list(self.active_tasks.keys()):
            await self.cancel_task(task_id)
        
        self.logger.info("Render manager cleanup completed")


# Global render manager instance
_render_manager: Optional[RenderManager] = None


def get_render_manager() -> RenderManager:
    """Get the global render manager instance"""
    global _render_manager
    if _render_manager is None:
        _render_manager = RenderManager()
    return _render_manager