"""
Health Check System

Comprehensive health monitoring for all system components including:
- Database connectivity
- Redis connectivity
- External services
- System resources
- AI model availability
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import aioredis
import asyncpg
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    message: str
    response_time: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealth:
    status: HealthStatus
    timestamp: float
    components: List[ComponentHealth]
    overall_response_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "overall_response_time": self.overall_response_time,
            "components": {
                component.name: {
                    "status": component.status.value,
                    "message": component.message,
                    "response_time": component.response_time,
                    "details": component.details or {}
                }
                for component in self.components
            }
        }


class HealthChecker:
    """Comprehensive system health checker"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checks = {
            "database": self._check_database,
            "redis": self._check_redis,
            "disk_space": self._check_disk_space,
            "memory": self._check_memory,
            "cpu": self._check_cpu,
            "ffmpeg": self._check_ffmpeg,
            "ai_models": self._check_ai_models,
            "task_queue": self._check_task_queue,
            "external_apis": self._check_external_apis
        }
    
    async def check_health(self) -> SystemHealth:
        """Perform comprehensive health check"""
        start_time = time.time()
        components = []
        
        # Run all health checks concurrently
        tasks = []
        for name, check_func in self.checks.items():
            task = asyncio.create_task(self._run_check(name, check_func))
            tasks.append(task)
        
        # Wait for all checks to complete
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in check_results:
            if isinstance(result, ComponentHealth):
                components.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Health check failed with exception: {result}")
                components.append(ComponentHealth(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(result)}",
                    response_time=0.0
                ))
        
        # Determine overall health
        overall_status = self._determine_overall_status(components)
        overall_response_time = time.time() - start_time
        
        return SystemHealth(
            status=overall_status,
            timestamp=time.time(),
            components=components,
            overall_response_time=overall_response_time
        )
    
    async def _run_check(self, name: str, check_func) -> ComponentHealth:
        """Run a single health check with timeout"""
        start_time = time.time()
        
        try:
            # Set timeout for each check
            timeout = self.config.get('check_timeout', 5.0)
            result = await asyncio.wait_for(check_func(), timeout=timeout)
            response_time = time.time() - start_time
            
            if result is True:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="OK",
                    response_time=response_time
                )
            elif isinstance(result, dict):
                return ComponentHealth(
                    name=name,
                    status=result.get('status', HealthStatus.HEALTHY),
                    message=result.get('message', 'OK'),
                    response_time=response_time,
                    details=result.get('details')
                )
            else:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(result),
                    response_time=response_time
                )
                
        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {timeout}s",
                response_time=time.time() - start_time
            )
        except Exception as e:
            logger.exception(f"Health check {name} failed: {e}")
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def _determine_overall_status(self, components: List[ComponentHealth]) -> HealthStatus:
        """Determine overall system health from component statuses"""
        if not components:
            return HealthStatus.UNKNOWN
        
        unhealthy_count = sum(1 for c in components if c.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for c in components if c.status == HealthStatus.DEGRADED)
        
        if unhealthy_count > 0:
            # Any unhealthy critical component makes system unhealthy
            critical_components = ['database', 'redis', 'disk_space', 'memory']
            for component in components:
                if component.name in critical_components and component.status == HealthStatus.UNHEALTHY:
                    return HealthStatus.UNHEALTHY
            
            # Non-critical unhealthy components degrade system
            if unhealthy_count >= len(components) // 2:
                return HealthStatus.UNHEALTHY
            else:
                return HealthStatus.DEGRADED
        
        if degraded_count > 0:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            db_config = self.config.get('database', {})
            dsn = f"postgresql://{db_config.get('user', 'postgres')}:{db_config.get('password', 'password')}@{db_config.get('host', 'localhost')}:{db_config.get('port', 5432)}/{db_config.get('database', 'aura_render')}"
            
            start_time = time.time()
            conn = await asyncpg.connect(dsn, timeout=3.0)
            
            # Test simple query
            result = await conn.fetchval("SELECT 1")
            
            # Check connection count
            connection_count = await conn.fetchval(
                "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
            )
            
            # Check database size
            db_size = await conn.fetchval(
                "SELECT pg_size_pretty(pg_database_size(current_database()))"
            )
            
            await conn.close()
            query_time = time.time() - start_time
            
            status = HealthStatus.HEALTHY
            message = "Database connection successful"
            
            # Check if connection count is high
            max_connections = db_config.get('max_connections', 100)
            if connection_count > max_connections * 0.8:
                status = HealthStatus.DEGRADED
                message = f"High connection count: {connection_count}"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'query_time': query_time,
                    'connection_count': connection_count,
                    'database_size': db_size,
                    'test_query_result': result
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Database check failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance"""
        try:
            redis_config = self.config.get('redis', {})
            redis_url = f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}"
            
            start_time = time.time()
            redis = await aioredis.from_url(redis_url, timeout=3.0)
            
            # Test ping
            pong = await redis.ping()
            
            # Get Redis info
            info = await redis.info()
            
            # Test set/get
            test_key = "health_check_test"
            await redis.set(test_key, "test_value", ex=60)
            test_value = await redis.get(test_key)
            await redis.delete(test_key)
            
            await redis.close()
            response_time = time.time() - start_time
            
            # Check memory usage
            memory_usage = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            memory_percent = (memory_usage / max_memory * 100) if max_memory > 0 else 0
            
            status = HealthStatus.HEALTHY
            message = "Redis connection successful"
            
            if memory_percent > 90:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {memory_percent:.1f}%"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'response_time': response_time,
                    'memory_usage_mb': memory_usage / 1024 / 1024,
                    'memory_percent': memory_percent,
                    'connected_clients': info.get('connected_clients', 0),
                    'ping_result': pong,
                    'test_operation': test_value == b'test_value'
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Redis check failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability"""
        try:
            disk_usage = psutil.disk_usage('/')
            used_percent = (disk_usage.used / disk_usage.total) * 100
            free_gb = disk_usage.free / 1024**3
            
            status = HealthStatus.HEALTHY
            message = f"Disk usage: {used_percent:.1f}%"
            
            if used_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk usage: {used_percent:.1f}%"
            elif used_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {used_percent:.1f}%"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'used_percent': used_percent,
                    'free_gb': free_gb,
                    'total_gb': disk_usage.total / 1024**3,
                    'used_gb': disk_usage.used / 1024**3
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Disk check failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            used_percent = memory.percent
            available_gb = memory.available / 1024**3
            
            status = HealthStatus.HEALTHY
            message = f"Memory usage: {used_percent:.1f}%"
            
            if used_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage: {used_percent:.1f}%"
            elif used_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {used_percent:.1f}%"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'used_percent': used_percent,
                    'available_gb': available_gb,
                    'total_gb': memory.total / 1024**3,
                    'used_gb': memory.used / 1024**3
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Memory check failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage"""
        try:
            # Get CPU usage over 1 second interval
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            status = HealthStatus.HEALTHY
            message = f"CPU usage: {cpu_percent:.1f}%"
            
            if cpu_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"High CPU usage: {cpu_percent:.1f}%"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'load_avg_1m': load_avg[0],
                    'load_avg_5m': load_avg[1],
                    'load_avg_15m': load_avg[2]
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"CPU check failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def _check_ffmpeg(self) -> Dict[str, Any]:
        """Check FFmpeg availability"""
        try:
            import subprocess
            
            # Check if FFmpeg is installed
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                return {
                    'status': HealthStatus.HEALTHY,
                    'message': "FFmpeg available",
                    'details': {
                        'version': version_line,
                        'available': True
                    }
                }
            else:
                return {
                    'status': HealthStatus.UNHEALTHY,
                    'message': "FFmpeg not working",
                    'details': {
                        'error': result.stderr,
                        'available': False
                    }
                }
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"FFmpeg check failed: {str(e)}",
                'details': {
                    'error': str(e),
                    'available': False
                }
            }
        except Exception as e:
            return {
                'status': HealthStatus.DEGRADED,
                'message': f"FFmpeg check error: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def _check_ai_models(self) -> Dict[str, Any]:
        """Check AI model availability"""
        try:
            model_paths = self.config.get('ai_models', {})
            model_status = {}
            
            for model_name, model_path in model_paths.items():
                if isinstance(model_path, str):
                    model_file = Path(model_path)
                    model_status[model_name] = model_file.exists()
                else:
                    model_status[model_name] = False
            
            available_models = sum(model_status.values())
            total_models = len(model_status)
            
            if available_models == total_models:
                status = HealthStatus.HEALTHY
                message = f"All {total_models} AI models available"
            elif available_models > total_models // 2:
                status = HealthStatus.DEGRADED
                message = f"{available_models}/{total_models} AI models available"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Only {available_models}/{total_models} AI models available"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'models': model_status,
                    'available_count': available_models,
                    'total_count': total_models
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.DEGRADED,
                'message': f"AI models check error: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def _check_task_queue(self) -> Dict[str, Any]:
        """Check task queue status"""
        try:
            # This would need to be implemented based on your task queue system
            # For now, return a basic check
            return {
                'status': HealthStatus.HEALTHY,
                'message': "Task queue check not implemented",
                'details': {
                    'implemented': False
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.DEGRADED,
                'message': f"Task queue check error: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API connectivity"""
        try:
            # This would check external APIs your system depends on
            # For now, return a basic check
            return {
                'status': HealthStatus.HEALTHY,
                'message': "External API checks not implemented",
                'details': {
                    'implemented': False
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.DEGRADED,
                'message': f"External API check error: {str(e)}",
                'details': {'error': str(e)}
            }


# Global health checker instance
health_checker: Optional[HealthChecker] = None


def initialize_health_checker(config: Dict[str, Any]):
    """Initialize the global health checker"""
    global health_checker
    health_checker = HealthChecker(config)


async def get_system_health() -> SystemHealth:
    """Get current system health"""
    if health_checker is None:
        raise RuntimeError("Health checker not initialized")
    
    return await health_checker.check_health()


async def get_component_health(component_name: str) -> ComponentHealth:
    """Get health of a specific component"""
    if health_checker is None:
        raise RuntimeError("Health checker not initialized")
    
    if component_name not in health_checker.checks:
        return ComponentHealth(
            name=component_name,
            status=HealthStatus.UNKNOWN,
            message=f"Unknown component: {component_name}",
            response_time=0.0
        )
    
    return await health_checker._run_check(component_name, health_checker.checks[component_name])