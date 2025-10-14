#!/usr/bin/env python3
"""
Health Check Script for Aura Render

Performs comprehensive health checks on the running application
and provides detailed status reports.
"""

import asyncio
import json
import sys
import time
import httpx
from pathlib import Path
from typing import Dict, List, Optional
import psutil
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import settings
except ImportError:
    print("❌ Cannot import config - is the project properly set up?")
    sys.exit(1)


class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.results: Dict[str, Dict] = {}
        self.start_time = time.time()
    
    async def check_api_endpoint(self, endpoint: str, expected_status: int = 200) -> Dict:
        """Check if API endpoint is responding"""
        try:
            base_url = f"http://{settings.server.host}:{settings.server.port}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                start_time = time.time()
                response = await client.get(f"{base_url}{endpoint}")
                response_time = (time.time() - start_time) * 1000
                
                return {
                    "status": "healthy" if response.status_code == expected_status else "unhealthy",
                    "status_code": response.status_code,
                    "response_time_ms": round(response_time, 2),
                    "content_length": len(response.content) if response.content else 0
                }
        except httpx.TimeoutException:
            return {
                "status": "timeout",
                "error": "Request timed out after 10 seconds"
            }
        except httpx.ConnectError:
            return {
                "status": "connection_error",
                "error": "Cannot connect to server - is it running?"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def check_api_health(self) -> Dict:
        """Check API endpoints health"""
        endpoints_to_check = [
            ("/", 200),
            ("/health", 200),
            ("/docs", 200),
            ("/openapi.json", 200)
        ]
        
        results = {}
        for endpoint, expected_status in endpoints_to_check:
            results[endpoint] = await self.check_api_endpoint(endpoint, expected_status)
        
        # Overall API health
        healthy_endpoints = sum(1 for r in results.values() if r.get("status") == "healthy")
        total_endpoints = len(endpoints_to_check)
        
        return {
            "overall_status": "healthy" if healthy_endpoints == total_endpoints else "degraded",
            "healthy_endpoints": healthy_endpoints,
            "total_endpoints": total_endpoints,
            "endpoints": results,
            "avg_response_time": round(
                sum(r.get("response_time_ms", 0) for r in results.values() if "response_time_ms" in r) / 
                max(1, sum(1 for r in results.values() if "response_time_ms" in r)), 2
            )
        }
    
    def check_system_resources(self) -> Dict:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = (memory.total - memory.available) / 1024 / 1024
            memory_total_mb = memory.total / 1024 / 1024
            
            # Disk usage
            disk = psutil.disk_usage(PROJECT_ROOT)
            disk_usage_gb = (disk.total - disk.free) / 1024 / 1024 / 1024
            disk_total_gb = disk.total / 1024 / 1024 / 1024
            
            # Process info
            current_process = psutil.Process()
            process_memory_mb = current_process.memory_info().rss / 1024 / 1024
            process_cpu_percent = current_process.cpu_percent()
            
            return {
                "status": "healthy",
                "system": {
                    "cpu_percent": cpu_percent,
                    "cpu_count": cpu_count,
                    "cpu_status": "healthy" if cpu_percent < 80 else "warning" if cpu_percent < 90 else "critical",
                    "memory_usage_mb": round(memory_usage_mb, 1),
                    "memory_total_mb": round(memory_total_mb, 1),
                    "memory_percent": memory.percent,
                    "memory_status": "healthy" if memory.percent < 80 else "warning" if memory.percent < 90 else "critical",
                    "disk_usage_gb": round(disk_usage_gb, 1),
                    "disk_total_gb": round(disk_total_gb, 1),
                    "disk_percent": round((disk_usage_gb / disk_total_gb) * 100, 1),
                    "disk_status": "healthy" if disk_usage_gb / disk_total_gb < 0.8 else "warning" if disk_usage_gb / disk_total_gb < 0.9 else "critical"
                },
                "process": {
                    "pid": current_process.pid,
                    "memory_mb": round(process_memory_mb, 1),
                    "cpu_percent": process_cpu_percent,
                    "num_threads": current_process.num_threads(),
                    "status": current_process.status(),
                    "create_time": current_process.create_time()
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_file_system(self) -> Dict:
        """Check file system health and permissions"""
        try:
            checks = {}
            
            # Check required directories
            required_dirs = [
                settings.storage.upload_dir,
                settings.storage.output_dir,
                settings.storage.temp_dir,
                PROJECT_ROOT / "logs"
            ]
            
            for dir_path in required_dirs:
                dir_name = dir_path.name
                try:
                    # Check if directory exists
                    exists = dir_path.exists()
                    
                    # Check if writable (create test file)
                    if exists:
                        test_file = dir_path / "health_check_test.tmp"
                        test_file.write_text("test")
                        test_file.unlink()
                        writable = True
                    else:
                        writable = False
                    
                    checks[dir_name] = {
                        "status": "healthy" if exists and writable else "error",
                        "exists": exists,
                        "writable": writable,
                        "path": str(dir_path)
                    }
                except Exception as e:
                    checks[dir_name] = {
                        "status": "error",
                        "error": str(e),
                        "path": str(dir_path)
                    }
            
            # Overall file system status
            all_healthy = all(check.get("status") == "healthy" for check in checks.values())
            
            return {
                "status": "healthy" if all_healthy else "error",
                "directories": checks
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_external_services(self) -> Dict:
        """Check external service availability"""
        try:
            checks = {}
            
            # Check AI service (if configured)
            if hasattr(settings, 'ai') and settings.ai.dashscope_api_key:
                try:
                    # This is a basic check - in production you'd make actual API calls
                    checks["dashscope"] = {
                        "status": "configured",
                        "api_key_present": bool(settings.ai.dashscope_api_key),
                        "model": settings.ai.qwen_model_name
                    }
                except Exception as e:
                    checks["dashscope"] = {
                        "status": "error",
                        "error": str(e)
                    }
            else:
                checks["dashscope"] = {
                    "status": "not_configured",
                    "message": "AI service not configured"
                }
            
            return {
                "status": "healthy",
                "services": checks
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def run_all_checks(self) -> Dict:
        """Run all health checks"""
        print("🏥 Running comprehensive health checks...")
        
        # API health check
        print("🔍 Checking API endpoints...")
        self.results["api"] = await self.check_api_health()
        
        # System resources
        print("📊 Checking system resources...")
        self.results["system"] = self.check_system_resources()
        
        # File system
        print("📁 Checking file system...")
        self.results["filesystem"] = self.check_file_system()
        
        # External services
        print("🌐 Checking external services...")
        self.results["external_services"] = self.check_external_services()
        
        # Calculate overall health
        component_statuses = []
        for component, result in self.results.items():
            status = result.get("status", "unknown")
            component_statuses.append(status)
        
        if all(status in ["healthy", "configured"] for status in component_statuses):
            overall_status = "healthy"
        elif any(status == "error" for status in component_statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        # Add metadata
        check_duration = time.time() - self.start_time
        
        return {
            "overall_status": overall_status,
            "timestamp": time.time(),
            "check_duration_seconds": round(check_duration, 2),
            "components": self.results
        }


def print_health_report(health_data: Dict):
    """Print formatted health report"""
    overall_status = health_data["overall_status"]
    
    # Status colors
    status_colors = {
        "healthy": "🟢",
        "degraded": "🟡", 
        "unhealthy": "🔴",
        "error": "🔴",
        "warning": "🟡",
        "configured": "🟢",
        "not_configured": "🟡"
    }
    
    print("\n" + "="*60)
    print(f"🏥 AURA RENDER HEALTH REPORT")
    print("="*60)
    
    status_icon = status_colors.get(overall_status, "❓")
    print(f"\n{status_icon} Overall Status: {overall_status.upper()}")
    print(f"⏱️  Check Duration: {health_data['check_duration_seconds']}s")
    print(f"📅 Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(health_data['timestamp']))}")
    
    # Component details
    for component, data in health_data["components"].items():
        status_icon = status_colors.get(data.get("status"), "❓")
        print(f"\n{status_icon} {component.upper()}: {data.get('status', 'unknown')}")
        
        if component == "api" and "endpoints" in data:
            for endpoint, endpoint_data in data["endpoints"].items():
                endpoint_icon = status_colors.get(endpoint_data.get("status"), "❓")
                response_time = endpoint_data.get("response_time_ms", 0)
                print(f"  {endpoint_icon} {endpoint}: {response_time}ms")
        
        elif component == "system":
            sys_data = data.get("system", {})
            print(f"  💻 CPU: {sys_data.get('cpu_percent', 0)}%")
            print(f"  🧠 Memory: {sys_data.get('memory_percent', 0)}%")
            print(f"  💾 Disk: {sys_data.get('disk_percent', 0)}%")
        
        elif component == "filesystem" and "directories" in data:
            for dir_name, dir_data in data["directories"].items():
                dir_icon = status_colors.get(dir_data.get("status"), "❓")
                print(f"  {dir_icon} {dir_name}: {'✓' if dir_data.get('writable') else '✗'}")
    
    print("\n" + "="*60)


async def main():
    """Main health check function"""
    try:
        checker = HealthChecker()
        health_data = await checker.run_all_checks()
        
        print_health_report(health_data)
        
        # Save to file
        health_file = PROJECT_ROOT / "logs" / "health_check.json"
        health_file.parent.mkdir(exist_ok=True)
        
        with open(health_file, 'w') as f:
            json.dump(health_data, f, indent=2)
        
        print(f"📄 Detailed report saved to: {health_file}")
        
        # Exit code based on health
        if health_data["overall_status"] == "healthy":
            sys.exit(0)
        elif health_data["overall_status"] == "degraded":
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())