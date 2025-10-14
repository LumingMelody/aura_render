"""
Celery Application Configuration

Central configuration for distributed task processing with:
- Redis broker for message passing
- Task routing and prioritization
- Monitoring and health checks
- Auto-scaling worker management
"""

import os
from typing import Dict, Any, Optional
from celery import Celery
from kombu import Queue
from config import Settings
import logging

logger = logging.getLogger(__name__)

def create_celery_app(settings: Optional[Settings] = None) -> Celery:
    """Create and configure Celery application"""

    # Create Celery app
    app = Celery('aura_render')

    # Try to get settings and configure broker
    try:
        if settings is None:
            settings = Settings()
        broker_url = settings.redis_url
        result_backend = broker_url
    except Exception as e:
        # Use minimal fallback configuration
        logger.warning(f"Could not load settings, using fallback Redis config: {e}")
        broker_url = "redis://localhost:6379/0"
        result_backend = broker_url
    
    # Celery configuration
    app.conf.update(
        # Broker settings
        broker_url=broker_url,
        result_backend=result_backend,
        broker_connection_retry_on_startup=True,
        
        # Task settings
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        
        # Task execution
        task_always_eager=False,  # Set to True for testing
        task_eager_propagates=True,
        task_ignore_result=False,
        task_store_eager_result=True,
        
        # Task routing with priority queues
        task_routes={
            'task_queue.tasks.process_video_generation_task': {'queue': 'video_generation'},
            'task_queue.tasks.generate_video_async': {'queue': 'video_processing'},
            'task_queue.tasks.cleanup_expired_tasks': {'queue': 'maintenance'},
            'task_queue.tasks.health_check_task': {'queue': 'monitoring'},
        },
        
        # Define queues with different priorities
        task_default_queue='default',
        task_queues=(
            Queue('video_generation', routing_key='video_generation', priority=10),
            Queue('video_processing', routing_key='video_processing', priority=8),
            Queue('audio_processing', routing_key='audio_processing', priority=6),
            Queue('image_processing', routing_key='image_processing', priority=5),
            Queue('maintenance', routing_key='maintenance', priority=2),
            Queue('monitoring', routing_key='monitoring', priority=1),
            Queue('default', routing_key='default', priority=4),
        ),
        
        # Worker settings
        worker_max_tasks_per_child=100,  # Prevent memory leaks
        worker_prefetch_multiplier=2,    # Tasks to prefetch
        worker_max_memory_per_child=500000,  # 500MB limit
        
        # Concurrency
        worker_concurrency=4,  # Number of worker processes
        
        # Task result expiration
        result_expires=3600,  # 1 hour
        
        # Task time limits
        task_soft_time_limit=300,  # 5 minutes soft limit
        task_time_limit=600,       # 10 minutes hard limit
        
        # Monitoring
        worker_send_task_events=True,
        task_send_sent_event=True,
        
        # Beat scheduler (for periodic tasks)
        beat_schedule={
            'cleanup-expired-tasks': {
                'task': 'task_queue.tasks.cleanup_expired_tasks',
                'schedule': 3600.0,  # Every hour
                'options': {'queue': 'maintenance'}
            },
            'health-check': {
                'task': 'task_queue.tasks.health_check_task',
                'schedule': 60.0,  # Every minute
                'options': {'queue': 'monitoring'}
            },
        },
    )
    
    # Auto-discover tasks
    app.autodiscover_tasks(['task_queue'])
    
    logger.info(f"Celery app configured with broker: {broker_url}")
    
    return app

# Create the global app instance
app = None

def get_or_create_celery_app() -> Celery:
    """Get or create the global Celery app instance"""
    global app
    if app is None:
        try:
            from config import get_settings
            settings = get_settings()
            app = create_celery_app(settings)
        except Exception as e:
            # Fallback with minimal settings
            logger.warning(f"Failed to load settings, using defaults: {e}")
            app = create_celery_app(None)
    return app

# Initialize app lazily
app = get_or_create_celery_app()

@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup additional periodic tasks"""
    logger.info("Setting up periodic tasks")
    
    # Add any additional periodic tasks here
    # sender.add_periodic_task(30.0, health_check_task.s(), name='health check every 30s')

class CeleryConfig:
    """Celery configuration class for different environments"""
    
    @staticmethod
    def get_config(environment: str = "development") -> Dict[str, Any]:
        """Get environment-specific Celery configuration"""
        
        base_config = {
            'broker_connection_retry_on_startup': True,
            'task_serializer': 'json',
            'accept_content': ['json'],
            'result_serializer': 'json',
            'timezone': 'UTC',
            'enable_utc': True,
        }
        
        if environment == "development":
            base_config.update({
                'task_always_eager': False,
                'worker_concurrency': 2,
                'task_soft_time_limit': 120,
                'task_time_limit': 300,
            })
        elif environment == "production":
            base_config.update({
                'task_always_eager': False,
                'worker_concurrency': 8,
                'task_soft_time_limit': 600,
                'task_time_limit': 1200,
                'worker_max_memory_per_child': 1000000,  # 1GB
            })
        elif environment == "testing":
            base_config.update({
                'task_always_eager': True,
                'task_eager_propagates': True,
            })
            
        return base_config

def get_celery_app() -> Celery:
    """Get the configured Celery application instance"""
    return app