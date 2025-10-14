"""
Health Checks and System Monitoring

Comprehensive system health checks, dependency monitoring,
and service availability verification.
"""

import asyncio
import time
import psutil
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import httpx

from pydantic import BaseModel
from celery import Celery

from analytics import get_metrics_collector


class HealthStatus(str, Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ServiceType(str, Enum):
    """Types of services to monitor"""
    DATABASE = "database"
    REDIS = "redis"
    CELERY = "celery"
    HTTP_API = "http_api"
    WEBSOCKET = "websocket"
    FILESYSTEM = "filesystem"
    EXTERNAL_API = "external_api"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    service_name: str
    service_type: ServiceType
    status: HealthStatus
    response_time_ms: float
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "service_type": self.service_type.value,
            "status": self.status.value,
            "response_time_ms": self.response_time_ms,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class SystemMetrics(BaseModel):
    """Current system metrics"""
    cpu_usage: float
    memory_usage: float
    memory_available_gb: float
    disk_usage: float
    disk_free_gb: float
    network_io: Dict[str, int]
    boot_time: datetime
    uptime_seconds: float
    load_average: List[float]
    timestamp: datetime


class HealthChecker:
    """Comprehensive system health checker"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = get_metrics_collector()
        
        # Health check configurations
        self.checks: Dict[str, Dict[str, Any]] = {
            "system_resources": {
                "function": self._check_system_resources,
                "interval": 30,
                "timeout": 5,
                "critical_thresholds": {
                    "cpu_usage": 90.0,
                    "memory_usage": 85.0,
                    "disk_usage": 95.0
                },
                "warning_thresholds": {
                    "cpu_usage": 70.0,
                    "memory_usage": 70.0,
                    "disk_usage": 80.0
                }
            },
            "database": {
                "function": self._check_database,
                "interval": 60,
                "timeout": 10
            },
            "redis": {
                "function": self._check_redis,
                "interval": 30,
                "timeout": 5
            },
            "celery": {
                "function": self._check_celery,
                "interval": 45,
                "timeout": 10
            },
            "filesystem": {
                "function": self._check_filesystem,
                "interval": 120,
                "timeout": 5
            }
        }
        
        # Health check results cache
        self.results: Dict[str, HealthCheckResult] = {}
        self.system_metrics: Optional[SystemMetrics] = None
        
        # Health check scheduler
        self.running = False
        self.tasks: List[asyncio.Task] = []
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("🏥 Starting health monitoring...")
        
        # Start health check tasks
        for check_name, config in self.checks.items():
            task = asyncio.create_task(
                self._run_periodic_check(check_name, config)
            )
            self.tasks.append(task)
        
        # Start system metrics collection
        metrics_task = asyncio.create_task(self._collect_system_metrics())
        self.tasks.append(metrics_task)
        
        self.logger.info("✅ Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        
        self.logger.info("🛑 Health monitoring stopped")
    
    async def _run_periodic_check(self, check_name: str, config: Dict[str, Any]):
        """Run a health check periodically"""
        check_function = config["function"]
        interval = config.get("interval", 60)
        timeout = config.get("timeout", 10)
        
        while self.running:
            try:
                start_time = time.time()
                
                # Run health check with timeout
                result = await asyncio.wait_for(
                    check_function(),
                    timeout=timeout
                )
                
                # Calculate response time
                response_time = (time.time() - start_time) * 1000
                result.response_time_ms = response_time
                
                # Store result
                self.results[check_name] = result
                
                # Record metrics
                await self.metrics.record_timer(
                    f"health_check.{check_name}.response_time",
                    response_time
                )
                
                await self.metrics.increment_counter(
                    f"health_check.{check_name}.{result.status.value}"
                )
                
                self.logger.debug(
                    f"Health check {check_name}: {result.status.value} "
                    f"({response_time:.1f}ms)"
                )
                
            except asyncio.TimeoutError:
                result = HealthCheckResult(
                    service_name=check_name,
                    service_type=ServiceType.HTTP_API,  # Default type
                    status=HealthStatus.CRITICAL,
                    response_time_ms=timeout * 1000,
                    message=f"Health check timed out after {timeout}s",
                    details={},
                    timestamp=datetime.now()
                )
                
                self.results[check_name] = result
                self.logger.error(f"Health check {check_name} timed out")
                
            except Exception as e:
                result = HealthCheckResult(
                    service_name=check_name,
                    service_type=ServiceType.HTTP_API,  # Default type
                    status=HealthStatus.CRITICAL,
                    response_time_ms=0.0,
                    message=f"Health check failed: {str(e)}",
                    details={"error": str(e)},
                    timestamp=datetime.now()
                )
                
                self.results[check_name] = result
                self.logger.error(f"Health check {check_name} failed: {e}")
            
            # Wait for next check
            await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while self.running:
            try:
                # CPU usage
                cpu_usage = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                
                # Disk usage
                disk = psutil.disk_usage('/')
                
                # Network I/O
                network = psutil.net_io_counters()
                
                # System info
                boot_time = datetime.fromtimestamp(psutil.boot_time())
                uptime = time.time() - psutil.boot_time()
                load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
                
                self.system_metrics = SystemMetrics(
                    cpu_usage=cpu_usage,
                    memory_usage=memory.percent,
                    memory_available_gb=memory.available / (1024**3),
                    disk_usage=disk.percent,
                    disk_free_gb=disk.free / (1024**3),
                    network_io={
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv,
                        "packets_sent": network.packets_sent,
                        "packets_recv": network.packets_recv
                    },
                    boot_time=boot_time,
                    uptime_seconds=uptime,
                    load_average=list(load_avg),
                    timestamp=datetime.now()
                )
                
                # Record metrics
                await self.metrics.set_gauge("system.cpu_usage", cpu_usage)
                await self.metrics.set_gauge("system.memory_usage", memory.percent)
                await self.metrics.set_gauge("system.memory_available_gb", memory.available / (1024**3))
                await self.metrics.set_gauge("system.disk_usage", disk.percent)
                await self.metrics.set_gauge("system.disk_free_gb", disk.free / (1024**3))
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
            
            await asyncio.sleep(30)
    
    async def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get thresholds
            config = self.checks["system_resources"]
            critical_thresholds = config["critical_thresholds"]
            warning_thresholds = config["warning_thresholds"]
            
            # Determine status
            status = HealthStatus.HEALTHY
            issues = []
            
            if (cpu_usage > critical_thresholds["cpu_usage"] or
                memory.percent > critical_thresholds["memory_usage"] or
                disk.percent > critical_thresholds["disk_usage"]):
                status = HealthStatus.CRITICAL
                
                if cpu_usage > critical_thresholds["cpu_usage"]:
                    issues.append(f"CPU usage critical: {cpu_usage:.1f}%")
                if memory.percent > critical_thresholds["memory_usage"]:
                    issues.append(f"Memory usage critical: {memory.percent:.1f}%")
                if disk.percent > critical_thresholds["disk_usage"]:
                    issues.append(f"Disk usage critical: {disk.percent:.1f}%")
                    
            elif (cpu_usage > warning_thresholds["cpu_usage"] or
                  memory.percent > warning_thresholds["memory_usage"] or
                  disk.percent > warning_thresholds["disk_usage"]):
                status = HealthStatus.WARNING
                
                if cpu_usage > warning_thresholds["cpu_usage"]:
                    issues.append(f"CPU usage high: {cpu_usage:.1f}%")
                if memory.percent > warning_thresholds["memory_usage"]:
                    issues.append(f"Memory usage high: {memory.percent:.1f}%")
                if disk.percent > warning_thresholds["disk_usage"]:
                    issues.append(f"Disk usage high: {disk.percent:.1f}%")
            
            message = "; ".join(issues) if issues else "System resources within normal limits"
            
            return HealthCheckResult(
                service_name="system_resources",
                service_type=ServiceType.FILESYSTEM,
                status=status,
                response_time_ms=0.0,
                message=message,
                details={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_usage": disk.percent,
                    "disk_free_gb": disk.free / (1024**3)
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name="system_resources",
                service_type=ServiceType.FILESYSTEM,
                status=HealthStatus.CRITICAL,
                response_time_ms=0.0,
                message=f"Failed to check system resources: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def _check_database(self) -> HealthCheckResult:
        """Check database connectivity and performance"""
        try:
            from database.base import SessionLocal, engine
            
            start_time = time.time()
            
            # Test database connection
            with SessionLocal() as db:
                result = db.execute("SELECT 1").scalar()
                if result != 1:
                    raise Exception("Database query returned unexpected result")
            
            # Test connection pool
            pool = engine.pool
            pool_info = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "status": "healthy"
            }
            
            response_time = (time.time() - start_time) * 1000
            
            # Check for potential issues
            status = HealthStatus.HEALTHY
            message = "Database connection healthy"
            
            if pool.checkedout() / pool.size() > 0.8:
                status = HealthStatus.WARNING
                message = "Database connection pool usage high"
            
            return HealthCheckResult(
                service_name="database",
                service_type=ServiceType.DATABASE,
                status=status,
                response_time_ms=response_time,
                message=message,
                details=pool_info,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name="database",
                service_type=ServiceType.DATABASE,
                status=HealthStatus.CRITICAL,
                response_time_ms=0.0,
                message=f"Database check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def _check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity and performance"""
        try:
            start_time = time.time()
            
            # Connect to Redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            # Test basic operations
            test_key = f"health_check:{int(time.time())}"
            r.set(test_key, "test_value", ex=60)
            value = r.get(test_key)
            r.delete(test_key)
            
            if value != "test_value":
                raise Exception("Redis test operation failed")
            
            # Get Redis info
            info = r.info()
            
            response_time = (time.time() - start_time) * 1000
            
            # Check Redis health metrics
            status = HealthStatus.HEALTHY
            message = "Redis connection healthy"
            
            memory_usage_ratio = info.get('used_memory', 0) / info.get('maxmemory', 1) if info.get('maxmemory', 0) > 0 else 0
            
            if memory_usage_ratio > 0.9:
                status = HealthStatus.CRITICAL
                message = f"Redis memory usage critical: {memory_usage_ratio:.1%}"
            elif memory_usage_ratio > 0.7:
                status = HealthStatus.WARNING
                message = f"Redis memory usage high: {memory_usage_ratio:.1%}"
            
            return HealthCheckResult(
                service_name="redis",
                service_type=ServiceType.REDIS,
                status=status,
                response_time_ms=response_time,
                message=message,
                details={
                    "version": info.get('redis_version'),
                    "connected_clients": info.get('connected_clients'),
                    "used_memory_human": info.get('used_memory_human'),
                    "memory_usage_ratio": memory_usage_ratio,
                    "keyspace_hits": info.get('keyspace_hits', 0),
                    "keyspace_misses": info.get('keyspace_misses', 0)
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name="redis",
                service_type=ServiceType.REDIS,
                status=HealthStatus.CRITICAL,
                response_time_ms=0.0,
                message=f"Redis check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def _check_celery(self) -> HealthCheckResult:
        """Check Celery worker status"""
        try:
            from task_queue.celery_app import get_celery_app
            
            start_time = time.time()
            app = get_celery_app()
            
            # Check active workers
            inspect = app.control.inspect()
            active_workers = inspect.active()
            stats = inspect.stats()
            
            response_time = (time.time() - start_time) * 1000
            
            if not active_workers:
                return HealthCheckResult(
                    service_name="celery",
                    service_type=ServiceType.CELERY,
                    status=HealthStatus.CRITICAL,
                    response_time_ms=response_time,
                    message="No active Celery workers found",
                    details={"active_workers": 0},
                    timestamp=datetime.now()
                )
            
            # Calculate worker statistics
            total_workers = len(active_workers)
            total_active_tasks = sum(len(tasks) for tasks in active_workers.values())
            
            status = HealthStatus.HEALTHY
            message = f"Celery healthy: {total_workers} workers, {total_active_tasks} active tasks"
            
            return HealthCheckResult(
                service_name="celery",
                service_type=ServiceType.CELERY,
                status=status,
                response_time_ms=response_time,
                message=message,
                details={
                    "active_workers": total_workers,
                    "active_tasks": total_active_tasks,
                    "worker_stats": stats
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name="celery",
                service_type=ServiceType.CELERY,
                status=HealthStatus.CRITICAL,
                response_time_ms=0.0,
                message=f"Celery check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def _check_filesystem(self) -> HealthCheckResult:
        """Check filesystem health and available space"""
        try:
            import tempfile
            import os
            
            start_time = time.time()
            
            # Test write/read operations
            test_content = f"health_check_{int(time.time())}"
            
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
                f.write(test_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'r') as f:
                    read_content = f.read()
                
                if read_content != test_content:
                    raise Exception("Filesystem read/write test failed")
                
            finally:
                os.unlink(temp_file)
            
            response_time = (time.time() - start_time) * 1000
            
            # Check disk space
            disk_usage = psutil.disk_usage('/')
            free_space_gb = disk_usage.free / (1024**3)
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            status = HealthStatus.HEALTHY
            message = f"Filesystem healthy: {free_space_gb:.1f}GB free ({usage_percent:.1f}% used)"
            
            if usage_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Disk space critical: {usage_percent:.1f}% used"
            elif usage_percent > 85:
                status = HealthStatus.WARNING
                message = f"Disk space warning: {usage_percent:.1f}% used"
            
            return HealthCheckResult(
                service_name="filesystem",
                service_type=ServiceType.FILESYSTEM,
                status=status,
                response_time_ms=response_time,
                message=message,
                details={
                    "total_gb": disk_usage.total / (1024**3),
                    "used_gb": disk_usage.used / (1024**3),
                    "free_gb": free_space_gb,
                    "usage_percent": usage_percent
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name="filesystem",
                service_type=ServiceType.FILESYSTEM,
                status=HealthStatus.CRITICAL,
                response_time_ms=0.0,
                message=f"Filesystem check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        overall_status = HealthStatus.HEALTHY
        
        # Determine overall status from individual checks
        for result in self.results.values():
            if result.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif result.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.WARNING
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {name: result.to_dict() for name, result in self.results.items()},
            "system_metrics": self.system_metrics.dict() if self.system_metrics else None,
            "summary": {
                "total_checks": len(self.results),
                "healthy": sum(1 for r in self.results.values() if r.status == HealthStatus.HEALTHY),
                "warning": sum(1 for r in self.results.values() if r.status == HealthStatus.WARNING),
                "critical": sum(1 for r in self.results.values() if r.status == HealthStatus.CRITICAL),
                "unknown": sum(1 for r in self.results.values() if r.status == HealthStatus.UNKNOWN)
            }
        }
    
    async def run_manual_check(self, check_name: Optional[str] = None) -> Dict[str, Any]:
        """Run health checks manually"""
        if check_name:
            if check_name not in self.checks:
                raise ValueError(f"Unknown health check: {check_name}")
            
            config = self.checks[check_name]
            result = await config["function"]()
            self.results[check_name] = result
            
            return {check_name: result.to_dict()}
        else:
            # Run all checks
            results = {}
            for name, config in self.checks.items():
                try:
                    result = await config["function"]()
                    self.results[name] = result
                    results[name] = result.to_dict()
                except Exception as e:
                    self.logger.error(f"Manual check {name} failed: {e}")
                    
            return results


# Global health checker instance
_health_checker = None

def get_health_checker() -> HealthChecker:
    """Get the global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker