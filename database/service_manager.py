"""
Database Service Manager

High-level database operations with caching integration.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from .models import Task, Project, NodeExecution, TaskStatus, NodeStatus
from .cache_manager import get_cache_manager
from config import Settings
import logging

logger = logging.getLogger(__name__)


class DatabaseServiceManager:
    """High-level database service with caching"""
    
    def __init__(self, db_session: Session, settings: Optional[Settings] = None):
        self.db = db_session
        self.settings = settings or Settings()
        self.cache = get_cache_manager(settings)
        
    async def initialize(self):
        """Initialize cache manager"""
        await self.cache.initialize()
        
    # Project operations
    async def get_or_create_project(self, name: str, user_id: str = "default") -> Project:
        """Get existing project or create new one"""
        
        # Try cache first
        cache_key = f"project:{name}:{user_id}"
        cached_project = await self.cache.get(cache_key)
        
        if cached_project:
            # Convert back to Project object
            project = self.db.query(Project).filter(
                Project.id == cached_project['id']
            ).first()
            if project:
                return project
                
        # Query database
        project = self.db.query(Project).filter(
            and_(Project.name == name, Project.user_id == user_id)
        ).first()
        
        if not project:
            project = Project(
                name=name,
                description=f"Auto-created project for {name}",
                user_id=user_id
            )
            self.db.add(project)
            self.db.commit()
            self.db.refresh(project)
            
        # Cache for future use
        await self.cache.set(
            cache_key, 
            {
                'id': project.id,
                'name': project.name,
                'user_id': project.user_id,
                'created_at': project.created_at.isoformat()
            }, 
            ttl=86400  # 24 hours
        )
        
        return project
        
    # Task operations
    async def create_task(self, task_data: Dict[str, Any], project_name: str = "Default") -> Task:
        """Create new task"""
        
        # Get or create project
        project = await self.get_or_create_project(project_name)
        
        # Create task
        task = Task(
            theme=task_data.get('theme_id', task_data.get('theme', '')),
            keywords=task_data.get('keywords_id', task_data.get('keywords', [])),
            target_duration=task_data.get('target_duration_id', task_data.get('target_duration', 30)),
            user_description=task_data.get('user_description_id', task_data.get('user_description', '')),
            config=task_data.get('config', {}),
            priority=task_data.get('priority', 0),
            project_id=project.id
        )
        
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        
        # Cache task
        await self.cache.set(
            f"task:{task.task_id}",
            task.to_dict(),
            ttl=7200  # 2 hours
        )
        
        logger.info(f"Created task {task.task_id} in project {project.name}")
        return task
        
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        
        # Try cache first
        cached_task = await self.cache.get(f"task:{task_id}")
        if cached_task:
            # Verify task still exists in DB
            task = self.db.query(Task).filter(Task.task_id == task_id).first()
            if task:
                return task
                
        # Query database
        task = self.db.query(Task).filter(Task.task_id == task_id).first()
        if task:
            # Update cache
            await self.cache.set(
                f"task:{task_id}",
                task.to_dict(),
                ttl=7200
            )
            
        return task
        
    async def update_task_status(
        self, 
        task_id: str, 
        status: TaskStatus, 
        progress: float = None, 
        message: str = None,
        result: Dict[str, Any] = None,
        error_message: str = None
    ) -> bool:
        """Update task status and progress"""
        
        task = await self.get_task(task_id)
        if not task:
            return False
            
        # Update fields
        task.status = status
        if progress is not None:
            task.progress = progress
        if message is not None:
            task.message = message
        if result is not None:
            task.result = result
        if error_message is not None:
            task.error_message = error_message
            
        self.db.commit()
        
        # Update cache
        await self.cache.set(
            f"task:{task_id}",
            task.to_dict(),
            ttl=7200
        )
        
        # Cache progress for real-time updates
        if progress is not None:
            await self.cache.cache_render_progress(task_id, progress, message or "")
            
        logger.info(f"Updated task {task_id} status to {status.value}")
        return True
        
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status with cached progress"""
        
        task = await self.get_task(task_id)
        if not task:
            return None
            
        status_data = task.to_dict()
        
        # Check for real-time progress updates
        progress_update = await self.cache.get_render_progress(task_id)
        if progress_update:
            status_data.update({
                'progress': progress_update.get('progress', status_data['progress']),
                'message': progress_update.get('message', status_data['message']),
                'last_update': progress_update.get('timestamp')
            })
            
        return status_data
        
    async def get_recent_tasks(
        self, 
        limit: int = 50, 
        project_name: str = None,
        status: TaskStatus = None
    ) -> List[Dict[str, Any]]:
        """Get recent tasks with optional filtering"""
        
        query = self.db.query(Task)
        
        if project_name:
            project = await self.get_or_create_project(project_name)
            query = query.filter(Task.project_id == project.id)
            
        if status:
            query = query.filter(Task.status == status)
            
        tasks = query.order_by(desc(Task.created_at)).limit(limit).all()
        
        # Convert to dict and enhance with cache data
        results = []
        for task in tasks:
            task_data = task.to_dict()
            
            # Add real-time progress if available
            progress_update = await self.cache.get_render_progress(task.task_id)
            if progress_update:
                task_data.update({
                    'progress': progress_update.get('progress', task_data['progress']),
                    'message': progress_update.get('message', task_data['message']),
                    'last_update': progress_update.get('timestamp')
                })
                
            results.append(task_data)
            
        return results
        
    # Node execution tracking
    async def create_node_execution(
        self, 
        task_id: str, 
        node_name: str, 
        input_data: Dict[str, Any] = None
    ) -> Optional[int]:
        """Create node execution record"""
        
        task = await self.get_task(task_id)
        if not task:
            return None
            
        execution = NodeExecution(
            task_id=task.id,
            node_name=node_name,
            status=NodeStatus.RUNNING,
            input_data=input_data or {},
            started_at=datetime.utcnow()
        )
        
        self.db.add(execution)
        self.db.commit()
        self.db.refresh(execution)
        
        logger.info(f"Started node execution {execution.id} for {node_name}")
        return execution.id
        
    async def update_node_execution(
        self,
        execution_id: int,
        status: NodeStatus,
        output_data: Dict[str, Any] = None,
        error_message: str = None
    ) -> bool:
        """Update node execution status"""
        
        execution = self.db.query(NodeExecution).filter(
            NodeExecution.id == execution_id
        ).first()
        
        if not execution:
            return False
            
        execution.status = status
        if output_data is not None:
            execution.output_data = output_data
        if error_message is not None:
            execution.error_message = error_message
        if status in [NodeStatus.SUCCESS, NodeStatus.ERROR]:
            execution.completed_at = datetime.utcnow()
            if execution.started_at:
                delta = execution.completed_at - execution.started_at
                execution.processing_time = delta.total_seconds()
                
        self.db.commit()
        
        logger.info(f"Updated node execution {execution_id} status to {status.value}")
        return True
        
    # Analytics and statistics
    async def get_task_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get task statistics for the last N days"""
        
        # Try cache first
        cache_key = f"task_stats:{days}d"
        cached_stats = await self.cache.get(cache_key)
        if cached_stats:
            return cached_stats
            
        from datetime import datetime, timedelta
        since_date = datetime.utcnow() - timedelta(days=days)
        
        # Query task counts by status
        stats = {}
        for status in TaskStatus:
            count = self.db.query(Task).filter(
                and_(
                    Task.status == status,
                    Task.created_at >= since_date
                )
            ).count()
            stats[f"{status.value}_tasks"] = count
            
        # Calculate success rate
        total_completed = stats.get('completed_tasks', 0) + stats.get('failed_tasks', 0)
        if total_completed > 0:
            stats['success_rate'] = stats.get('completed_tasks', 0) / total_completed * 100
        else:
            stats['success_rate'] = 0.0
            
        # Average processing time for completed tasks
        completed_tasks = self.db.query(Task).filter(
            and_(
                Task.status == TaskStatus.COMPLETED,
                Task.created_at >= since_date,
                Task.processing_time.isnot(None)
            )
        ).all()
        
        if completed_tasks:
            avg_time = sum(t.processing_time for t in completed_tasks) / len(completed_tasks)
            stats['average_processing_time'] = avg_time
        else:
            stats['average_processing_time'] = 0.0
            
        stats['period_days'] = days
        stats['generated_at'] = datetime.utcnow().isoformat()
        
        # Cache for 1 hour
        await self.cache.set(cache_key, stats, ttl=3600)
        
        return stats
        
    # Maintenance operations
    async def cleanup_old_tasks(self, days: int = 30) -> int:
        """Clean up old completed tasks"""
        
        from datetime import datetime, timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Delete old completed/failed tasks
        deleted_count = self.db.query(Task).filter(
            and_(
                Task.status.in_([TaskStatus.COMPLETED, TaskStatus.FAILED]),
                Task.completed_at < cutoff_date
            )
        ).delete()
        
        self.db.commit()
        
        logger.info(f"Cleaned up {deleted_count} old tasks")
        return deleted_count
        
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        
        health_data = {
            'database_connected': True,
            'cache_connected': await self.cache.exists('health_check'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Database health
            pending_tasks = self.db.query(Task).filter(
                Task.status == TaskStatus.PROCESSING
            ).count()
            
            failed_tasks_24h = self.db.query(Task).filter(
                and_(
                    Task.status == TaskStatus.FAILED,
                    Task.created_at >= datetime.utcnow() - timedelta(hours=24)
                )
            ).count()
            
            health_data.update({
                'pending_tasks': pending_tasks,
                'failed_tasks_24h': failed_tasks_24h,
                'database_status': 'healthy'
            })
            
        except Exception as e:
            health_data.update({
                'database_connected': False,
                'database_status': 'error',
                'database_error': str(e)
            })
            
        # Cache health
        cache_stats = await self.cache.get_cache_stats()
        health_data['cache_stats'] = cache_stats
        
        return health_data


# Global service manager instance
_service_manager = None


def get_service_manager(db_session: Session = None, settings: Optional[Settings] = None) -> DatabaseServiceManager:
    """Get or create service manager instance"""
    global _service_manager
    
    if _service_manager is None and db_session is not None:
        _service_manager = DatabaseServiceManager(db_session, settings)
        
    return _service_manager


def initialize_service_manager(db_session: Session, settings: Optional[Settings] = None) -> DatabaseServiceManager:
    """Initialize service manager with database session"""
    global _service_manager
    _service_manager = DatabaseServiceManager(db_session, settings)
    return _service_manager