"""
Celery Worker Management

Provides utilities for:
- Starting and stopping workers
- Worker health monitoring
- Dynamic scaling
- Resource management
"""

import os
import sys
import subprocess
import signal
import time
import psutil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import threading

from config import Settings
from monitoring.metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)

class WorkerStatus(Enum):
    """Worker status states"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class WorkerInfo:
    """Worker information"""
    worker_id: str
    pid: Optional[int]
    status: WorkerStatus
    queue: str
    concurrency: int
    started_at: Optional[float]
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_tasks: int = 0

class WorkerManager:
    """Manages Celery workers"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.logger = logging.getLogger(__name__)
        self.metrics = get_metrics_collector()
        
        # Worker tracking
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_processes: Dict[str, subprocess.Popen] = {}
        
        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        
    def start_worker(
        self,
        worker_id: str,
        queue: str = "default",
        concurrency: int = 4,
        loglevel: str = "info"
    ) -> bool:
        """Start a new Celery worker"""
        
        try:
            if worker_id in self.workers:
                if self.workers[worker_id].status == WorkerStatus.RUNNING:
                    self.logger.warning(f"Worker {worker_id} is already running")
                    return True
            
            # Create worker info
            worker_info = WorkerInfo(
                worker_id=worker_id,
                pid=None,
                status=WorkerStatus.STARTING,
                queue=queue,
                concurrency=concurrency,
                started_at=time.time()
            )
            
            # Build Celery command
            cmd = [
                sys.executable,
                "-m", "celery",
                "worker",
                "-A", "task_queue.celery_app:app",
                "-n", f"{worker_id}@%h",
                "-Q", queue,
                "-c", str(concurrency),
                "--loglevel", loglevel,
                "--without-gossip",
                "--without-mingle",
                "--without-heartbeat"
            ]
            
            # Add worker-specific settings
            if self.settings.development_mode:
                cmd.extend(["--pool", "solo"])
            
            # Start process
            self.logger.info(f"Starting worker {worker_id} with command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(os.environ, PYTHONPATH=os.getcwd())
            )
            
            # Wait a bit to check if it started successfully
            time.sleep(2)
            
            if process.poll() is None:
                # Process is running
                worker_info.pid = process.pid
                worker_info.status = WorkerStatus.RUNNING
                
                self.workers[worker_id] = worker_info
                self.worker_processes[worker_id] = process
                
                self.logger.info(f"Worker {worker_id} started successfully with PID {process.pid}")
                
                # Start monitoring if not already running
                if not self.monitoring_active:
                    self.start_monitoring()
                
                return True
            else:
                # Process failed to start
                stdout, stderr = process.communicate()
                worker_info.status = WorkerStatus.ERROR
                
                self.logger.error(f"Failed to start worker {worker_id}:")
                self.logger.error(f"STDOUT: {stdout.decode()}")
                self.logger.error(f"STDERR: {stderr.decode()}")
                
                return False
                
        except Exception as e:
            self.logger.error(f"Exception starting worker {worker_id}: {str(e)}")
            return False
    
    def stop_worker(self, worker_id: str, timeout: int = 30) -> bool:
        """Stop a Celery worker gracefully"""
        
        try:
            if worker_id not in self.workers:
                self.logger.warning(f"Worker {worker_id} not found")
                return False
            
            worker_info = self.workers[worker_id]
            
            if worker_info.status != WorkerStatus.RUNNING:
                self.logger.warning(f"Worker {worker_id} is not running")
                return True
            
            # Update status
            worker_info.status = WorkerStatus.STOPPING
            
            # Get process
            process = self.worker_processes.get(worker_id)
            if not process:
                self.logger.error(f"Process for worker {worker_id} not found")
                return False
            
            self.logger.info(f"Stopping worker {worker_id} (PID {process.pid})")
            
            try:
                # Send SIGTERM for graceful shutdown
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=timeout)
                    self.logger.info(f"Worker {worker_id} stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't stop gracefully
                    self.logger.warning(f"Worker {worker_id} didn't stop gracefully, force killing")
                    process.kill()
                    process.wait()
                
                # Update status
                worker_info.status = WorkerStatus.STOPPED
                worker_info.pid = None
                
                # Clean up
                self.worker_processes.pop(worker_id, None)
                
                return True
                
            except ProcessLookupError:
                # Process already stopped
                worker_info.status = WorkerStatus.STOPPED
                worker_info.pid = None
                self.worker_processes.pop(worker_id, None)
                return True
                
        except Exception as e:
            self.logger.error(f"Exception stopping worker {worker_id}: {str(e)}")
            return False
    
    def restart_worker(self, worker_id: str) -> bool:
        """Restart a worker"""
        
        if worker_id not in self.workers:
            self.logger.error(f"Worker {worker_id} not found")
            return False
        
        worker_info = self.workers[worker_id]
        
        # Stop the worker
        if worker_info.status == WorkerStatus.RUNNING:
            if not self.stop_worker(worker_id):
                return False
        
        # Wait a moment
        time.sleep(1)
        
        # Start it again with same configuration
        return self.start_worker(
            worker_id=worker_id,
            queue=worker_info.queue,
            concurrency=worker_info.concurrency
        )
    
    def get_worker_status(self, worker_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of workers"""
        
        if worker_id:
            if worker_id in self.workers:
                worker_info = self.workers[worker_id]
                return {
                    'worker_id': worker_info.worker_id,
                    'status': worker_info.status.value,
                    'pid': worker_info.pid,
                    'queue': worker_info.queue,
                    'concurrency': worker_info.concurrency,
                    'cpu_usage': worker_info.cpu_usage,
                    'memory_usage': worker_info.memory_usage,
                    'active_tasks': worker_info.active_tasks,
                    'uptime': time.time() - worker_info.started_at if worker_info.started_at else 0
                }
            else:
                return {'error': f'Worker {worker_id} not found'}
        else:
            # Return all workers
            return {
                worker_id: {
                    'worker_id': info.worker_id,
                    'status': info.status.value,
                    'pid': info.pid,
                    'queue': info.queue,
                    'concurrency': info.concurrency,
                    'cpu_usage': info.cpu_usage,
                    'memory_usage': info.memory_usage,
                    'active_tasks': info.active_tasks,
                    'uptime': time.time() - info.started_at if info.started_at else 0
                }
                for worker_id, info in self.workers.items()
            }
    
    def start_monitoring(self):
        """Start background monitoring of workers"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_workers, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started worker monitoring thread")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped worker monitoring thread")
    
    def stop_all_workers(self):
        """Stop all workers"""
        
        self.logger.info("Stopping all workers...")
        
        for worker_id in list(self.workers.keys()):
            self.stop_worker(worker_id)
        
        # Stop monitoring
        self.stop_monitoring()
        
        self.logger.info("All workers stopped")
    
    def auto_scale(self, target_workers: int, queue: str = "default") -> List[str]:
        """Auto-scale workers based on queue load"""
        
        try:
            # Get current workers for this queue
            current_workers = [
                worker_id for worker_id, info in self.workers.items()
                if info.queue == queue and info.status == WorkerStatus.RUNNING
            ]
            
            current_count = len(current_workers)
            started_workers = []
            
            if target_workers > current_count:
                # Scale up
                for i in range(target_workers - current_count):
                    worker_id = f"{queue}_worker_{len(self.workers) + i + 1}"
                    
                    if self.start_worker(
                        worker_id=worker_id,
                        queue=queue,
                        concurrency=self.settings.celery_worker_concurrency
                    ):
                        started_workers.append(worker_id)
                        
                self.logger.info(f"Scaled up {queue} queue: added {len(started_workers)} workers")
                
            elif target_workers < current_count:
                # Scale down
                workers_to_stop = current_workers[target_workers:]
                
                for worker_id in workers_to_stop:
                    if self.stop_worker(worker_id):
                        started_workers.append(f"stopped_{worker_id}")
                        
                self.logger.info(f"Scaled down {queue} queue: stopped {len(workers_to_stop)} workers")
            
            return started_workers
            
        except Exception as e:
            self.logger.error(f"Error during auto-scaling: {str(e)}")
            return []
    
    def _monitor_workers(self):
        """Background monitoring of worker health and performance"""
        
        while self.monitoring_active:
            try:
                for worker_id, worker_info in self.workers.items():
                    if worker_info.status == WorkerStatus.RUNNING and worker_info.pid:
                        try:
                            # Get process info
                            process = psutil.Process(worker_info.pid)
                            
                            # Update resource usage
                            worker_info.cpu_usage = process.cpu_percent()
                            worker_info.memory_usage = process.memory_percent()
                            
                            # Check if process is still alive
                            if not process.is_running():
                                self.logger.error(f"Worker {worker_id} process died")
                                worker_info.status = WorkerStatus.ERROR
                                worker_info.pid = None
                                self.worker_processes.pop(worker_id, None)
                            
                        except psutil.NoSuchProcess:
                            # Process no longer exists
                            self.logger.error(f"Worker {worker_id} process no longer exists")
                            worker_info.status = WorkerStatus.ERROR
                            worker_info.pid = None
                            self.worker_processes.pop(worker_id, None)
                            
                        except Exception as e:
                            self.logger.error(f"Error monitoring worker {worker_id}: {str(e)}")
                
                # Record metrics
                running_workers = sum(1 for info in self.workers.values() 
                                    if info.status == WorkerStatus.RUNNING)
                
                self.metrics.record_worker_status(
                    total_workers=len(self.workers),
                    running_workers=running_workers,
                    avg_cpu=sum(info.cpu_usage for info in self.workers.values()) / len(self.workers) if self.workers else 0,
                    avg_memory=sum(info.memory_usage for info in self.workers.values()) / len(self.workers) if self.workers else 0
                )
                
            except Exception as e:
                self.logger.error(f"Error in worker monitoring: {str(e)}")
            
            # Sleep for monitoring interval
            time.sleep(10)  # Monitor every 10 seconds

# Global worker manager instance
_worker_manager: Optional[WorkerManager] = None

def get_worker_manager(settings: Optional[Settings] = None) -> WorkerManager:
    """Get global worker manager instance"""
    global _worker_manager
    if _worker_manager is None:
        _worker_manager = WorkerManager(settings)
    return _worker_manager

def start_worker(
    worker_id: str,
    queue: str = "default",
    concurrency: int = 4,
    loglevel: str = "info"
) -> bool:
    """Start a Celery worker"""
    manager = get_worker_manager()
    return manager.start_worker(worker_id, queue, concurrency, loglevel)

def stop_worker(worker_id: str, timeout: int = 30) -> bool:
    """Stop a Celery worker"""
    manager = get_worker_manager()
    return manager.stop_worker(worker_id, timeout)