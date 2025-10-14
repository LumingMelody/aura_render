"""
Task Queue Package

Provides distributed task processing capabilities using Celery for:
- Asynchronous video generation
- Background processing
- Task scheduling and prioritization
- Distributed worker management
"""

from .celery_app import app as celery_app
from .tasks import (
    process_video_generation_task,
    generate_video_async,
    cleanup_expired_tasks,
    health_check_task
)
from .task_manager import TaskManager, get_task_manager
from .workers import start_worker, stop_worker

__all__ = [
    'celery_app',
    'process_video_generation_task',
    'generate_video_async',
    'cleanup_expired_tasks',
    'health_check_task',
    'TaskManager',
    'get_task_manager',
    'start_worker',
    'stop_worker'
]