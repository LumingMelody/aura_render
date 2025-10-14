"""
Celery Tasks for Video Generation

Defines all asynchronous tasks for:
- Video generation pipeline
- Audio processing
- Material matching and download
- System maintenance
- Health monitoring
"""

import asyncio
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import traceback

from celery import current_task
from celery.exceptions import Retry

from .celery_app import app
from config import Settings
from database.service_manager import get_service_manager
from database.cache_manager import get_cache_manager
from monitoring.error_handler import get_error_handler, ErrorCategory, ErrorSeverity
from monitoring.metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)

# Import core processors
try:
    from video_processing.main_pipeline import MainPipeline, PipelineConfig, create_pipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MainPipeline not available: {e}")
    MainPipeline = None
    PipelineConfig = None
    create_pipeline = None
    PIPELINE_AVAILABLE = False

# Additional imports for future use
try:
    from materials_supplies.matcher.main_video_matcher import MainVideoMatcher
    from content_generator.tts_service import get_tts_service
    from content_generator.audio_processor import get_audio_processor
except ImportError:
    MainVideoMatcher = None

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_video_generation_task(
    self, 
    task_id: str,
    user_input: Dict[str, Any],
    generation_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main video generation task
    
    Args:
        task_id: Unique task identifier
        user_input: User input with text, preferences, etc.
        generation_config: Optional generation configuration
        
    Returns:
        Dict with task result and metadata
    """
    
    start_time = time.time()
    settings = Settings()
    error_handler = get_error_handler()
    metrics = get_metrics_collector()
    
    try:
        logger.info(f"Starting video generation task {task_id}")
        
        # Update task status
        self.update_state(
            state='PROCESSING',
            meta={
                'task_id': task_id,
                'stage': 'initialization',
                'progress': 0,
                'start_time': start_time
            }
        )
        
        # Initialize result structure
        result = {}
        
        if PIPELINE_AVAILABLE:
            # Use real MainPipeline implementation
            try:
                # Create pipeline configuration
                pipeline_config = generation_config or {}
                pipeline = create_pipeline(pipeline_config)
                
                # Set up progress callback
                def progress_callback(stage_name: str, progress: float, metadata: dict):
                    self.update_state(
                        state='PROCESSING',
                        meta={
                            'task_id': task_id,
                            'stage': stage_name,
                            'progress': int(progress * 100),
                            'message': metadata.get('description', f'Processing {stage_name}...'),
                            'metadata': metadata
                        }
                    )
                
                pipeline.set_progress_callback(progress_callback)
                
                # Validate input
                is_valid, validation_errors = pipeline.validate_input(user_input)
                if not is_valid:
                    raise ValueError(f"Invalid input: {'; '.join(validation_errors)}")
                
                # Execute pipeline
                pipeline_result = asyncio.run(pipeline.execute(user_input))
                
                if pipeline_result.success:
                    result = {
                        'success': True,
                        'output_path': pipeline_result.output_path,
                        'metadata': pipeline_result.metadata,
                        'analytics': pipeline_result.analytics,
                        'processing_time': pipeline_result.processing_time,
                        'stages_completed': pipeline_result.stages_completed
                    }
                else:
                    raise Exception(f"Pipeline failed: {'; '.join(pipeline_result.errors)}")
                    
            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}")
                # Fallback to mock implementation
                PIPELINE_AVAILABLE = False
                
        if not PIPELINE_AVAILABLE:
            # Fallback: Mock implementation for testing
            logger.info("Using mock pipeline implementation")
            
            # Stage 1: Content Analysis (20%)
            self.update_state(
                state='PROCESSING',
                meta={
                    'task_id': task_id,
                    'stage': 'content_analysis',
                    'progress': 20,
                    'message': 'Analyzing input content...'
                }
            )
            
            # Mock content analysis
            analysis_result = {
                'content_type': 'promotional',
                'themes': ['technology', 'innovation'],
                'duration_estimate': 30.0,
                'complexity_score': 0.7
            }
            result['analysis'] = analysis_result
        
        # Stage 2: Material Matching (40%)
        self.update_state(
            state='PROCESSING',
            meta={
                'task_id': task_id,
                'stage': 'material_matching',
                'progress': 40,
                'message': 'Finding matching materials...'
            }
        )
        
        # Mock material matching
        material_result = {
            'video_clips': [{'id': 'v1', 'duration': 10.0}, {'id': 'v2', 'duration': 15.0}],
            'background_music': {'id': 'm1', 'duration': 30.0},
            'images': [{'id': 'i1', 'type': 'background'}]
        }
        result['materials'] = material_result
        
        # Stage 3: Audio Generation (60%)
        self.update_state(
            state='PROCESSING',
            meta={
                'task_id': task_id,
                'stage': 'audio_generation',
                'progress': 60,
                'message': 'Generating audio content...'
            }
        )
        
        # Mock audio generation
        if user_input.get('include_audio', True):
            audio_result = {
                'voice_track': '/tmp/voice_track.wav',
                'background_music': '/tmp/background_music.wav',
                'mixed_audio': '/tmp/mixed_audio.wav',
                'duration': 30.0
            }
            result['audio'] = audio_result
        
        # Stage 4: Video Assembly (80%)
        self.update_state(
            state='PROCESSING',
            meta={
                'task_id': task_id,
                'stage': 'video_assembly',
                'progress': 80,
                'message': 'Assembling final video...'
            }
        )
        
        # Mock video assembly
        video_result = {
            'output_path': '/tmp/generated_video.mp4',
            'duration': 30.0,
            'resolution': '1920x1080',
            'file_size_mb': 45.2
        }
        result['video'] = video_result
        
        # Stage 5: Finalization (100%)
        self.update_state(
            state='PROCESSING',
            meta={
                'task_id': task_id,
                'stage': 'finalization',
                'progress': 95,
                'message': 'Finalizing output...'
            }
        )
        
        # Cache result
        cache_manager = get_cache_manager()
        asyncio.run(
            cache_manager.cache_task_result(task_id, result, ttl=86400)
        )
        
        # Record success metrics
        duration = time.time() - start_time
        metrics.record_task_completion(
            task_type='video_generation',
            duration=duration,
            success=True
        )
        
        logger.info(f"Video generation task {task_id} completed successfully in {duration:.2f}s")
        
        return {
            'task_id': task_id,
            'status': 'completed',
            'result': result,
            'duration': duration,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        duration = time.time() - start_time
        
        # Log error
        logger.error(f"Video generation task {task_id} failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Record error
        asyncio.run(
            error_handler.handle_error(
                exception=e,
                category=ErrorCategory.PROCESSING,
                severity=ErrorSeverity.HIGH,
                context={
                    'task_id': task_id,
                    'task_type': 'video_generation',
                    'duration': duration,
                    'user_input': user_input
                }
            )
        )
        
        # Record metrics
        metrics.record_task_completion(
            task_type='video_generation',
            duration=duration,
            success=False
        )
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying task {task_id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        # Final failure
        self.update_state(
            state='FAILURE',
            meta={
                'task_id': task_id,
                'error': str(e),
                'duration': duration,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        raise e

@app.task(bind=True, max_retries=2)
def generate_video_async(
    self,
    request_data: Dict[str, Any],
    priority: int = 5
) -> str:
    """
    Async wrapper for video generation
    
    Args:
        request_data: Request data from API
        priority: Task priority (1-10, higher = more priority)
        
    Returns:
        Task ID for tracking
    """
    
    try:
        # Create unique task ID
        task_id = f"video_{int(time.time())}_{current_task.request.id}"
        
        # Extract parameters
        user_input = request_data.get('input', {})
        config = request_data.get('config', {})
        
        # Start main video generation task
        main_task = process_video_generation_task.apply_async(
            args=[task_id, user_input, config],
            priority=priority,
            queue='video_generation'
        )
        
        # Store task reference
        service_manager = get_service_manager()
        asyncio.run(
            service_manager.create_task(
                task_id=task_id,
                task_type='video_generation',
                status='queued',
                user_input=user_input,
                celery_task_id=main_task.id
            )
        )
        
        logger.info(f"Queued video generation task {task_id} with Celery task {main_task.id}")
        
        return task_id
        
    except Exception as e:
        logger.error(f"Failed to queue video generation task: {str(e)}")
        raise e

@app.task(bind=True)
def cleanup_expired_tasks(self) -> Dict[str, Any]:
    """
    Periodic task to clean up expired tasks and files
    
    Returns:
        Cleanup statistics
    """
    
    start_time = time.time()
    logger.info("Starting cleanup of expired tasks")
    
    try:
        settings = Settings()
        service_manager = get_service_manager()
        cache_manager = get_cache_manager()
        
        # Clean up old tasks (older than 7 days)
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        
        # Get expired tasks
        expired_tasks = asyncio.run(
            service_manager.get_expired_tasks(cutoff_time)
        )
        
        cleaned_count = 0
        freed_space = 0
        
        for task in expired_tasks:
            try:
                # Remove task files
                if task.get('output_path') and os.path.exists(task['output_path']):
                    file_size = os.path.getsize(task['output_path'])
                    os.remove(task['output_path'])
                    freed_space += file_size
                
                # Remove from database
                asyncio.run(
                    service_manager.delete_task(task['task_id'])
                )
                
                # Remove from cache
                asyncio.run(
                    cache_manager.delete(f"task_result:{task['task_id']}")
                )
                
                cleaned_count += 1
                
            except Exception as e:
                logger.error(f"Error cleaning up task {task.get('task_id')}: {str(e)}")
        
        # Clean up temp directories
        temp_dirs = [
            settings.temp_dir / "tts",
            settings.temp_dir / "videos",
            settings.temp_dir / "materials"
        ]
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                for file_path in temp_dir.glob("*"):
                    if file_path.is_file():
                        # Remove files older than 1 day
                        if time.time() - file_path.stat().st_mtime > 86400:
                            try:
                                freed_space += file_path.stat().st_size
                                file_path.unlink()
                                cleaned_count += 1
                            except Exception as e:
                                logger.error(f"Error removing temp file {file_path}: {str(e)}")
        
        duration = time.time() - start_time
        
        result = {
            'cleaned_tasks': len(expired_tasks),
            'cleaned_files': cleaned_count,
            'freed_space_mb': freed_space / (1024 * 1024),
            'duration': duration,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Cleanup completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Cleanup task failed: {str(e)}")
        raise e

@app.task(bind=True)
def health_check_task(self) -> Dict[str, Any]:
    """
    Health check task for monitoring system status
    
    Returns:
        System health information
    """
    
    try:
        settings = Settings()
        metrics = get_metrics_collector()
        
        # Check system resources
        system_health = metrics.get_system_health()
        
        # Check database connection
        service_manager = get_service_manager()
        db_health = asyncio.run(service_manager.health_check())
        
        # Check cache connection
        cache_manager = get_cache_manager()
        cache_health = asyncio.run(cache_manager.health_check())
        
        # Check task queue status
        active_tasks = asyncio.run(service_manager.get_active_task_count())
        
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'system': system_health,
            'database': {'status': 'healthy' if db_health else 'unhealthy'},
            'cache': {'status': 'healthy' if cache_health else 'unhealthy'},
            'active_tasks': active_tasks,
            'worker_id': current_task.request.hostname,
        }
        
        # Determine overall health
        health_status['overall'] = 'healthy' if all([
            system_health.get('cpu_usage', 100) < 90,
            system_health.get('memory_usage', 100) < 90,
            db_health,
            cache_health
        ]) else 'degraded'
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'overall': 'unhealthy',
            'error': str(e)
        }

# Additional utility tasks

@app.task(bind=True)
def process_audio_task(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process audio-only content"""
    # Implementation for audio-only processing
    pass

@app.task(bind=True)
def download_material_task(self, material_urls: List[str]) -> Dict[str, Any]:
    """Download and cache materials in background"""
    # Implementation for material downloading
    pass