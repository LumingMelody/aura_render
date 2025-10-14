"""
Task Manager for Coordinating Celery Tasks

Provides high-level task management interface for:
- Task submission and tracking
- Priority queue management
- Resource allocation
- Progress monitoring
- Error recovery
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from celery import Celery
from celery.result import AsyncResult

from config import Settings
from database.service_manager import get_service_manager
from database.cache_manager import get_cache_manager
from monitoring.error_handler import get_error_handler, ErrorCategory, ErrorSeverity
from monitoring.metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    URGENT = 10

class TaskStatus(Enum):
    """Task status states"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"

@dataclass
class TaskInfo:
    """Task information container"""
    task_id: str
    task_type: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    updated_at: datetime
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    celery_task_id: Optional[str] = None
    estimated_duration: Optional[int] = None
    actual_duration: Optional[int] = None

class TaskManager:
    """Central task management system"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.service_manager = get_service_manager()
        self.cache_manager = get_cache_manager()
        self.error_handler = get_error_handler()
        self.metrics = get_metrics_collector()
        self.logger = logging.getLogger(__name__)
        
        # Import Celery app here to avoid circular imports
        from .celery_app import get_celery_app
        self.celery_app = get_celery_app()
        
        # Task tracking
        self.active_tasks: Dict[str, TaskInfo] = {}
        self.task_limits = {
            TaskPriority.URGENT: 2,
            TaskPriority.HIGH: 5,
            TaskPriority.NORMAL: 10,
            TaskPriority.LOW: 20
        }
        
    async def submit_video_generation_task(
        self,
        user_input: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit a video generation task
        
        Args:
            user_input: User input data
            priority: Task priority level
            config: Optional generation configuration
            
        Returns:
            Task ID for tracking
        """
        
        try:
            # Check resource limits
            if not await self._check_resource_limits(priority):
                raise RuntimeError("Resource limits exceeded for priority level")
            
            # Create task ID
            task_id = f"video_{int(time.time())}_{hash(str(user_input))}"
            
            # Create task info
            task_info = TaskInfo(
                task_id=task_id,
                task_type="video_generation",
                status=TaskStatus.PENDING,
                priority=priority,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                estimated_duration=self._estimate_task_duration(user_input)
            )
            
            # Store in database
            await self.service_manager.create_task(
                task_id=task_id,
                task_type="video_generation",
                status=TaskStatus.QUEUED.value,
                user_input=user_input,
                priority=priority.value,
                config=config
            )
            
            # Submit to Celery
            from .tasks import process_video_generation_task
            
            celery_task = process_video_generation_task.apply_async(
                args=[task_id, user_input, config],
                priority=priority.value,
                queue=self._get_queue_for_priority(priority)
            )
            
            task_info.celery_task_id = celery_task.id
            task_info.status = TaskStatus.QUEUED
            
            # Update database with Celery task ID
            await self.service_manager.update_task(
                task_id=task_id,
                status=TaskStatus.QUEUED.value,
                celery_task_id=celery_task.id
            )
            
            # Cache task info
            await self.cache_manager.set(
                f"task_info:{task_id}",
                task_info.__dict__,
                ttl=86400
            )
            
            # Track locally
            self.active_tasks[task_id] = task_info
            
            # Record metrics
            self.metrics.record_task_submission(
                task_type="video_generation",
                priority=priority.value
            )
            
            self.logger.info(f"Submitted video generation task {task_id} with priority {priority.name}")
            return task_id
            
        except Exception as e:
            await self.error_handler.handle_error(
                exception=e,
                category=ErrorCategory.TASK_MANAGEMENT,
                severity=ErrorSeverity.MEDIUM,
                context={
                    "operation": "submit_video_generation_task",
                    "priority": priority.name,
                    "user_input_size": len(str(user_input))
                }
            )
            raise e
    
    async def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get current status of a task"""
        
        try:
            # Check local cache first
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
            else:
                # Try cache
                cached_info = await self.cache_manager.get(f"task_info:{task_id}")
                if cached_info:
                    task_info = TaskInfo(**cached_info)
                else:
                    # Get from database
                    db_task = await self.service_manager.get_task(task_id)
                    if not db_task:
                        return None
                    
                    task_info = TaskInfo(
                        task_id=db_task['task_id'],
                        task_type=db_task['task_type'],
                        status=TaskStatus(db_task['status']),
                        priority=TaskPriority(db_task.get('priority', 5)),
                        created_at=db_task['created_at'],
                        updated_at=db_task['updated_at'],
                        progress=db_task.get('progress', 0.0),
                        message=db_task.get('message', ''),
                        result=db_task.get('result'),
                        error=db_task.get('error'),
                        celery_task_id=db_task.get('celery_task_id')
                    )
            
            # Update status from Celery if available
            if task_info.celery_task_id:
                celery_result = AsyncResult(task_info.celery_task_id, app=self.celery_app)
                
                if celery_result.state == 'PENDING':
                    task_info.status = TaskStatus.QUEUED
                elif celery_result.state == 'PROCESSING':
                    task_info.status = TaskStatus.PROCESSING
                    if celery_result.info and isinstance(celery_result.info, dict):
                        task_info.progress = celery_result.info.get('progress', task_info.progress)
                        task_info.message = celery_result.info.get('message', task_info.message)
                elif celery_result.state == 'SUCCESS':
                    task_info.status = TaskStatus.COMPLETED
                    task_info.progress = 100.0
                    task_info.result = celery_result.result
                elif celery_result.state == 'FAILURE':
                    task_info.status = TaskStatus.FAILED
                    task_info.error = str(celery_result.info)
                elif celery_result.state == 'RETRY':
                    task_info.status = TaskStatus.RETRY
            
            return task_info
            
        except Exception as e:
            self.logger.error(f"Error getting task status for {task_id}: {str(e)}")
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        
        try:
            task_info = await self.get_task_status(task_id)
            if not task_info:
                return False
            
            if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
            
            # Cancel Celery task
            if task_info.celery_task_id:
                self.celery_app.control.revoke(task_info.celery_task_id, terminate=True)
            
            # Update status
            task_info.status = TaskStatus.CANCELLED
            task_info.updated_at = datetime.utcnow()
            
            # Update database
            await self.service_manager.update_task(
                task_id=task_id,
                status=TaskStatus.CANCELLED.value,
                updated_at=datetime.utcnow()
            )
            
            # Update cache
            await self.cache_manager.set(
                f"task_info:{task_id}",
                task_info.__dict__,
                ttl=86400
            )
            
            # Remove from active tasks
            self.active_tasks.pop(task_id, None)
            
            self.logger.info(f"Cancelled task {task_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling task {task_id}: {str(e)}")
            return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get status of all task queues"""
        
        try:
            # Get Celery queue stats
            inspect = self.celery_app.control.inspect()
            active_queues = inspect.active_queues() or {}
            
            # Get task counts by priority
            task_counts = {}
            for priority in TaskPriority:
                count = await self.service_manager.get_task_count_by_priority(priority.value)
                task_counts[priority.name] = count
            
            # Get active workers
            worker_stats = inspect.stats() or {}
            
            queue_status = {
                'timestamp': datetime.utcnow().isoformat(),
                'active_queues': active_queues,
                'task_counts_by_priority': task_counts,
                'worker_count': len(worker_stats),
                'worker_stats': worker_stats,
                'total_active_tasks': sum(task_counts.values())
            }
            
            return queue_status
            
        except Exception as e:
            self.logger.error(f"Error getting queue status: {str(e)}")
            return {'error': str(e)}
    
    async def get_task_history(
        self,
        limit: int = 50,
        task_type: Optional[str] = None,
        status: Optional[TaskStatus] = None
    ) -> List[TaskInfo]:
        """Get task execution history"""
        
        try:
            db_tasks = await self.service_manager.get_task_history(
                limit=limit,
                task_type=task_type,
                status=status.value if status else None
            )
            
            task_history = []
            for db_task in db_tasks:
                task_info = TaskInfo(
                    task_id=db_task['task_id'],
                    task_type=db_task['task_type'],
                    status=TaskStatus(db_task['status']),
                    priority=TaskPriority(db_task.get('priority', 5)),
                    created_at=db_task['created_at'],
                    updated_at=db_task['updated_at'],
                    progress=db_task.get('progress', 0.0),
                    message=db_task.get('message', ''),
                    result=db_task.get('result'),
                    error=db_task.get('error'),
                    actual_duration=db_task.get('duration')
                )
                task_history.append(task_info)
            
            return task_history
            
        except Exception as e:
            self.logger.error(f"Error getting task history: {str(e)}")
            return []
    
    async def cleanup_completed_tasks(self, older_than_hours: int = 24) -> int:
        """Clean up old completed tasks"""
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
            
            # Get completed tasks older than cutoff
            old_tasks = await self.service_manager.get_completed_tasks_before(cutoff_time)
            
            cleaned_count = 0
            for task in old_tasks:
                try:
                    # Remove from cache
                    await self.cache_manager.delete(f"task_info:{task['task_id']}")
                    await self.cache_manager.delete(f"task_result:{task['task_id']}")
                    
                    # Remove from database (or mark as archived)
                    await self.service_manager.archive_task(task['task_id'])
                    
                    cleaned_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error cleaning up task {task['task_id']}: {str(e)}")
            
            self.logger.info(f"Cleaned up {cleaned_count} old completed tasks")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error during task cleanup: {str(e)}")
            return 0
    
    # Private helper methods
    
    async def _check_resource_limits(self, priority: TaskPriority) -> bool:
        """Check if we can submit a task with given priority"""
        
        try:
            # Get current task count for this priority
            current_count = await self.service_manager.get_active_task_count_by_priority(priority.value)
            
            # Check against limits
            limit = self.task_limits.get(priority, 10)
            return current_count < limit
            
        except Exception:
            return True  # Default to allowing task if check fails
    
    def _estimate_task_duration(self, user_input: Dict[str, Any]) -> int:
        """Estimate task duration in seconds based on input complexity"""
        
        try:
            # Simple heuristic based on content length and complexity
            content_length = len(str(user_input.get('content', '')))
            
            base_time = 30  # Base 30 seconds
            content_factor = min(content_length / 100, 10)  # Up to 10x multiplier
            
            # Additional factors
            if user_input.get('include_audio', True):
                base_time += 15
            if user_input.get('high_quality', False):
                base_time *= 1.5
            
            return int(base_time + content_factor * 5)
            
        except Exception:
            return 60  # Default 1 minute
    
    def _get_queue_for_priority(self, priority: TaskPriority) -> str:
        """Get Celery queue name for priority level"""
        
        if priority == TaskPriority.URGENT:
            return "video_generation"
        elif priority == TaskPriority.HIGH:
            return "video_processing"
        else:
            return "default"

# Global task manager instance
_task_manager: Optional[TaskManager] = None

def get_task_manager(settings: Optional[Settings] = None) -> TaskManager:
    """Get global task manager instance"""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager(settings)
    return _task_manager