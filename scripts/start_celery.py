#!/usr/bin/env python3
"""
Celery Worker and Beat Scheduler Startup Script

Provides convenient commands to start Celery components:
- Workers for different queues
- Beat scheduler for periodic tasks
- Monitoring with Flower
"""

import os
import sys
import subprocess
import signal
import time
import argparse
from pathlib import Path
from typing import List, Optional
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings
import logging

logger = logging.getLogger(__name__)

class CeleryManager:
    """Manages Celery processes"""
    
    def __init__(self):
        self.settings = Settings()
        self.processes: List[subprocess.Popen] = []
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n📡 Received signal {signum}, shutting down Celery processes...")
        self.stop_all()
        sys.exit(0)
    
    def start_worker(
        self,
        worker_name: str,
        queue: str = "default",
        concurrency: int = 4,
        loglevel: str = "info",
        max_tasks_per_child: int = 100
    ) -> subprocess.Popen:
        """Start a Celery worker"""
        
        cmd = [
            sys.executable,
            "-m", "celery",
            "worker",
            "-A", "task_queue.celery_app:app",
            "-n", f"{worker_name}@%h",
            "-Q", queue,
            "-c", str(concurrency),
            "--loglevel", loglevel,
            "--max-tasks-per-child", str(max_tasks_per_child),
            "--without-gossip",
            "--without-mingle",
            "--without-heartbeat"
        ]
        
        # Add development-specific options
        if self.settings.development_mode:
            cmd.extend(["--pool", "solo"])
        
        print(f"🚀 Starting worker {worker_name} for queue {queue}")
        print(f"📋 Command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT)),
                cwd=PROJECT_ROOT
            )
            
            self.processes.append(process)
            return process
            
        except Exception as e:
            print(f"❌ Failed to start worker {worker_name}: {e}")
            return None
    
    def start_beat(self, loglevel: str = "info") -> subprocess.Popen:
        """Start Celery beat scheduler"""
        
        cmd = [
            sys.executable,
            "-m", "celery",
            "beat",
            "-A", "task_queue.celery_app:app",
            "--loglevel", loglevel,
            "--schedule", str(PROJECT_ROOT / "celerybeat-schedule")
        ]
        
        print("⏰ Starting Celery beat scheduler")
        print(f"📋 Command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT)),
                cwd=PROJECT_ROOT
            )
            
            self.processes.append(process)
            return process
            
        except Exception as e:
            print(f"❌ Failed to start beat scheduler: {e}")
            return None
    
    def start_flower(self, port: int = 5555) -> subprocess.Popen:
        """Start Flower monitoring"""
        
        cmd = [
            sys.executable,
            "-m", "flower",
            "-A", "task_queue.celery_app:app",
            "--port", str(port),
            "--broker", self.settings.redis_url
        ]
        
        print(f"🌺 Starting Flower monitoring on port {port}")
        print(f"📋 Command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT)),
                cwd=PROJECT_ROOT
            )
            
            self.processes.append(process)
            print(f"🌐 Flower available at: http://localhost:{port}")
            return process
            
        except Exception as e:
            print(f"❌ Failed to start Flower: {e}")
            return None
    
    def start_default_setup(self):
        """Start default Celery setup with multiple workers"""
        
        print("🚀 Starting Aura Render Celery Setup")
        print("=" * 50)
        
        # Start different workers for different queues
        workers_config = [
            {"name": "video_worker", "queue": "video_generation", "concurrency": 2},
            {"name": "audio_worker", "queue": "audio_processing", "concurrency": 2},
            {"name": "general_worker", "queue": "default", "concurrency": 4},
            {"name": "maintenance_worker", "queue": "maintenance", "concurrency": 1}
        ]
        
        for config in workers_config:
            worker_process = self.start_worker(**config)
            if worker_process:
                time.sleep(1)  # Stagger startup
        
        # Start beat scheduler
        beat_process = self.start_beat()
        if beat_process:
            time.sleep(1)
        
        # Start Flower monitoring
        flower_process = self.start_flower()
        
        print("\n✅ Celery setup completed!")
        print("📊 Services running:")
        print(f"   • {len([p for p in self.processes if p and p.poll() is None])} active processes")
        print("   • Beat scheduler for periodic tasks")
        print("   • Flower monitoring at http://localhost:5555")
        print("\nPress Ctrl+C to stop all services")
        
        # Monitor processes
        self.monitor_processes()
    
    def monitor_processes(self):
        """Monitor running processes"""
        
        try:
            while True:
                # Check process status
                running_count = 0
                for i, process in enumerate(self.processes):
                    if process and process.poll() is None:
                        running_count += 1
                    elif process and process.poll() is not None:
                        print(f"⚠️ Process {i} exited with code {process.poll()}")
                
                if running_count == 0:
                    print("❌ All processes have stopped")
                    break
                
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print("\n📡 Shutdown signal received")
    
    def stop_all(self):
        """Stop all Celery processes"""
        
        print("🛑 Stopping all Celery processes...")
        
        for i, process in enumerate(self.processes):
            if process and process.poll() is None:
                try:
                    print(f"📋 Stopping process {i} (PID: {process.pid})")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=10)
                        print(f"✅ Process {i} stopped gracefully")
                    except subprocess.TimeoutExpired:
                        print(f"⚠️ Process {i} didn't stop gracefully, force killing")
                        process.kill()
                        process.wait()
                        
                except Exception as e:
                    print(f"❌ Error stopping process {i}: {e}")
        
        print("✅ All processes stopped")
    
    def show_status(self):
        """Show status of Celery components"""
        
        try:
            # Use Celery inspect to get status
            cmd = [
                sys.executable,
                "-m", "celery",
                "inspect", "active",
                "-A", "task_queue.celery_app:app"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT)),
                cwd=PROJECT_ROOT
            )
            
            if result.returncode == 0:
                print("📊 Celery Status:")
                print(result.stdout)
            else:
                print("❌ Failed to get Celery status:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Error getting status: {e}")

def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(description="Aura Render Celery Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start Celery services")
    start_parser.add_argument("--worker-name", default="aura_worker", help="Worker name")
    start_parser.add_argument("--queue", default="default", help="Queue name")
    start_parser.add_argument("--concurrency", type=int, default=4, help="Worker concurrency")
    start_parser.add_argument("--loglevel", default="info", help="Log level")
    start_parser.add_argument("--beat", action="store_true", help="Start beat scheduler")
    start_parser.add_argument("--flower", action="store_true", help="Start Flower monitoring")
    start_parser.add_argument("--flower-port", type=int, default=5555, help="Flower port")
    
    # Default setup command
    default_parser = subparsers.add_parser("default", help="Start default Celery setup")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show Celery status")
    
    # Worker command
    worker_parser = subparsers.add_parser("worker", help="Start single worker")
    worker_parser.add_argument("name", help="Worker name")
    worker_parser.add_argument("--queue", default="default", help="Queue name")
    worker_parser.add_argument("--concurrency", type=int, default=4, help="Concurrency")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = CeleryManager()
    
    if args.command == "start":
        # Start individual components
        processes = []
        
        if not (args.beat or args.flower):
            # Default: start worker
            worker_process = manager.start_worker(
                worker_name=args.worker_name,
                queue=args.queue,
                concurrency=args.concurrency,
                loglevel=args.loglevel
            )
            if worker_process:
                processes.append(worker_process)
        
        if args.beat:
            beat_process = manager.start_beat(args.loglevel)
            if beat_process:
                processes.append(beat_process)
        
        if args.flower:
            flower_process = manager.start_flower(args.flower_port)
            if flower_process:
                processes.append(flower_process)
        
        if processes:
            print("\nPress Ctrl+C to stop")
            manager.monitor_processes()
    
    elif args.command == "default":
        manager.start_default_setup()
    
    elif args.command == "status":
        manager.show_status()
    
    elif args.command == "worker":
        worker_process = manager.start_worker(
            worker_name=args.name,
            queue=args.queue,
            concurrency=args.concurrency
        )
        if worker_process:
            print("\nPress Ctrl+C to stop")
            manager.monitor_processes()

if __name__ == "__main__":
    main()