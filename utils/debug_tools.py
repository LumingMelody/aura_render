#!/usr/bin/env python3
"""
Advanced Debug and Monitoring Tools for Aura Render

Provides comprehensive debugging utilities, performance monitoring,
memory profiling, and system diagnostics.
"""

import asyncio
import functools
import gc
import inspect
import json
import os
import psutil
import sys
import threading
import time
import traceback
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import cProfile
import pstats
from io import StringIO
from collections import defaultdict, deque
import resource
import linecache

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    import objgraph
    OBJGRAPH_AVAILABLE = True
except ImportError:
    OBJGRAPH_AVAILABLE = False


@dataclass
class DebugSession:
    """Debug session information"""
    session_id: str
    start_time: datetime
    process_id: int
    thread_id: int
    module: str
    function: str
    args: Tuple
    kwargs: Dict
    stack_trace: List[str] = field(default_factory=list)
    end_time: Optional[datetime] = None
    exception: Optional[Exception] = None
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class PerformanceSnapshot:
    """System performance snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_usage: int  # bytes
    memory_percent: float
    disk_usage: Dict[str, Tuple[int, int, int]]  # total, used, free
    network_io: Dict[str, int]
    open_files: int
    thread_count: int
    process_count: int


@dataclass 
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: datetime
    total_memory: int
    available_memory: int
    process_memory: int
    gc_stats: Dict[int, Dict[str, int]]
    top_objects: List[Tuple[str, int]]
    memory_leaks: List[str] = field(default_factory=list)


class DebugProfiler:
    """Advanced debugging and profiling utility"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.sessions: Dict[str, DebugSession] = {}
        self.performance_history: deque = deque(maxlen=1000)
        self.memory_history: deque = deque(maxlen=100)
        self.function_calls: defaultdict = defaultdict(int)
        self.slow_calls: List[Tuple[str, float]] = []
        self.error_counts: defaultdict = defaultdict(int)
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._tracked_objects: Set[int] = set()
        
    def start_monitoring(self, interval: float = 5.0):
        """Start background performance monitoring"""
        if self._monitoring_active:
            return
            
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._background_monitor,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _background_monitor(self, interval: float):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                snapshot = self._capture_performance_snapshot()
                self.performance_history.append(snapshot)
                
                # Capture memory snapshot less frequently
                if len(self.performance_history) % 10 == 0:
                    memory_snapshot = self._capture_memory_snapshot()
                    self.memory_history.append(memory_snapshot)
                
                time.sleep(interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _capture_performance_snapshot(self) -> PerformanceSnapshot:
        """Capture current performance metrics"""
        process = psutil.Process()
        
        # CPU and memory
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # Disk usage for main directories
        disk_usage = {}
        for path in ["/", "/tmp", str(Path.cwd())]:
            try:
                disk_stat = psutil.disk_usage(path)
                disk_usage[path] = (disk_stat.total, disk_stat.used, disk_stat.free)
            except:
                continue
        
        # Network IO
        try:
            net_io = psutil.net_io_counters()._asdict()
        except:
            net_io = {}
        
        # File handles and threads
        try:
            open_files = len(process.open_files())
        except:
            open_files = 0
            
        thread_count = process.num_threads()
        
        # Total process count
        try:
            process_count = len(psutil.pids())
        except:
            process_count = 0
        
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_usage=memory_info.rss,
            memory_percent=memory_percent,
            disk_usage=disk_usage,
            network_io=net_io,
            open_files=open_files,
            thread_count=thread_count,
            process_count=process_count
        )
    
    def _capture_memory_snapshot(self) -> MemorySnapshot:
        """Capture detailed memory information"""
        # System memory
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        # GC statistics
        gc_stats = {}
        for generation in range(3):
            stats = gc.get_stats()[generation]
            gc_stats[generation] = dict(stats)
        
        # Top objects by count
        top_objects = []
        if OBJGRAPH_AVAILABLE:
            try:
                # Get most common object types
                most_common = objgraph.most_common_types(limit=20)
                top_objects = [(name, count) for name, count in most_common]
            except:
                pass
        
        # Memory leak detection
        memory_leaks = []
        try:
            # Simple leak detection - look for growing object counts
            current_objects = gc.get_objects()
            object_types = defaultdict(int)
            for obj in current_objects:
                obj_type = type(obj).__name__
                object_types[obj_type] += 1
            
            # This is a simplified approach - in production you'd want more sophisticated leak detection
            if hasattr(self, '_last_object_counts'):
                for obj_type, count in object_types.items():
                    last_count = self._last_object_counts.get(obj_type, 0)
                    if count > last_count * 1.5 and count > 1000:  # 50% increase and substantial count
                        memory_leaks.append(f"Potential leak in {obj_type}: {count} objects (+{count-last_count})")
            
            self._last_object_counts = dict(object_types)
        except:
            pass
        
        return MemorySnapshot(
            timestamp=datetime.now(),
            total_memory=memory.total,
            available_memory=memory.available,
            process_memory=process.memory_info().rss,
            gc_stats=gc_stats,
            top_objects=top_objects,
            memory_leaks=memory_leaks
        )
    
    def debug_function(self, include_args: bool = True, include_result: bool = False, 
                      profile: bool = False, memory_profile: bool = False):
        """Decorator for debugging function calls"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._debug_call(func, args, kwargs, include_args, include_result, profile, memory_profile)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(self._debug_call(func, args, kwargs, include_args, include_result, profile, memory_profile))
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        return decorator
    
    async def _debug_call(self, func: Callable, args: Tuple, kwargs: Dict,
                         include_args: bool, include_result: bool, 
                         profile: bool, memory_profile: bool):
        """Execute function with debugging"""
        if not self.enabled:
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        
        # Create debug session
        session_id = f"{func.__module__}.{func.__name__}_{id(func)}_{time.time()}"
        session = DebugSession(
            session_id=session_id,
            start_time=datetime.now(),
            process_id=os.getpid(),
            thread_id=threading.get_ident(),
            module=func.__module__,
            function=func.__name__,
            args=args if include_args else (),
            kwargs=kwargs if include_args else {},
            stack_trace=traceback.format_stack()[:-1]
        )
        
        self.sessions[session_id] = session
        self.function_calls[f"{func.__module__}.{func.__name__}"] += 1
        
        # Performance profiling setup
        profiler = None
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()
        
        # Memory profiling setup
        initial_memory = None
        if memory_profile and MEMORY_PROFILER_AVAILABLE:
            initial_memory = memory_profiler.memory_usage()[0]
        
        try:
            # Execute function
            start_time = time.time()
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Track slow calls
            if execution_time > 1.0:  # Calls taking more than 1 second
                self.slow_calls.append((f"{func.__module__}.{func.__name__}", execution_time))
                if len(self.slow_calls) > 100:  # Keep only last 100
                    self.slow_calls = self.slow_calls[-100:]
            
            session.end_time = datetime.now()
            
            # Log debug information
            debug_info = {
                "function": f"{func.__module__}.{func.__name__}",
                "execution_time": execution_time,
                "success": True
            }
            
            if include_result:
                debug_info["result"] = str(result)[:1000]  # Truncate large results
            
            print(f"DEBUG: {json.dumps(debug_info, default=str, indent=2)}")
            
            return result
            
        except Exception as e:
            session.exception = e
            session.end_time = datetime.now()
            
            # Track errors
            error_key = f"{func.__module__}.{func.__name__}:{type(e).__name__}"
            self.error_counts[error_key] += 1
            
            # Log error
            error_info = {
                "function": f"{func.__module__}.{func.__name__}",
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
            print(f"ERROR: {json.dumps(error_info, default=str, indent=2)}")
            
            raise
        finally:
            # Performance profiling cleanup
            if profiler:
                profiler.disable()
                s = StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(20)  # Top 20 functions
                print(f"PROFILE: {func.__module__}.{func.__name__}")
                print(s.getvalue())
            
            # Memory profiling cleanup
            if memory_profile and MEMORY_PROFILER_AVAILABLE and initial_memory:
                final_memory = memory_profiler.memory_usage()[0]
                memory_delta = final_memory - initial_memory
                print(f"MEMORY: {func.__module__}.{func.__name__} used {memory_delta:.2f} MB")
    
    def get_debug_stats(self) -> Dict[str, Any]:
        """Get comprehensive debug statistics"""
        return {
            "function_calls": dict(self.function_calls),
            "slow_calls": self.slow_calls[-20:],  # Last 20 slow calls
            "error_counts": dict(self.error_counts),
            "active_sessions": len([s for s in self.sessions.values() if s.end_time is None]),
            "total_sessions": len(self.sessions),
            "memory_snapshots": len(self.memory_history),
            "performance_snapshots": len(self.performance_history),
            "monitoring_active": self._monitoring_active
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from recent snapshots"""
        if not self.performance_history:
            return {}
        
        recent_snapshots = list(self.performance_history)[-60:]  # Last 5 minutes at 5s intervals
        
        # Calculate averages
        avg_cpu = sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots)
        avg_memory = sum(s.memory_usage for s in recent_snapshots) / len(recent_snapshots)
        avg_memory_percent = sum(s.memory_percent for s in recent_snapshots) / len(recent_snapshots)
        
        # Find peaks
        max_cpu = max(s.cpu_percent for s in recent_snapshots)
        max_memory = max(s.memory_usage for s in recent_snapshots)
        
        return {
            "summary_period_minutes": len(recent_snapshots) * 5 / 60,
            "average_cpu_percent": round(avg_cpu, 2),
            "maximum_cpu_percent": round(max_cpu, 2),
            "average_memory_mb": round(avg_memory / 1024 / 1024, 1),
            "maximum_memory_mb": round(max_memory / 1024 / 1024, 1),
            "average_memory_percent": round(avg_memory_percent, 2),
            "current_thread_count": recent_snapshots[-1].thread_count if recent_snapshots else 0,
            "current_open_files": recent_snapshots[-1].open_files if recent_snapshots else 0
        }
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get detailed memory analysis report"""
        if not self.memory_history:
            return {}
        
        latest = self.memory_history[-1]
        
        # Memory trend (if we have multiple snapshots)
        trend = "stable"
        if len(self.memory_history) >= 3:
            recent_memory = [s.process_memory for s in list(self.memory_history)[-3:]]
            if recent_memory[-1] > recent_memory[0] * 1.1:
                trend = "increasing"
            elif recent_memory[-1] < recent_memory[0] * 0.9:
                trend = "decreasing"
        
        return {
            "timestamp": latest.timestamp.isoformat(),
            "total_system_memory_gb": round(latest.total_memory / 1024 / 1024 / 1024, 2),
            "available_system_memory_gb": round(latest.available_memory / 1024 / 1024 / 1024, 2),
            "process_memory_mb": round(latest.process_memory / 1024 / 1024, 1),
            "memory_trend": trend,
            "gc_stats": latest.gc_stats,
            "top_object_types": latest.top_objects[:10],
            "potential_memory_leaks": latest.memory_leaks,
            "snapshot_count": len(self.memory_history)
        }
    
    def export_debug_report(self, filepath: Path):
        """Export comprehensive debug report"""
        report = {
            "export_time": datetime.now().isoformat(),
            "debug_stats": self.get_debug_stats(),
            "performance_summary": self.get_performance_summary(),
            "memory_report": self.get_memory_report(),
            "recent_sessions": [
                {
                    "session_id": s.session_id,
                    "function": f"{s.module}.{s.function}",
                    "start_time": s.start_time.isoformat(),
                    "duration": s.duration.total_seconds() if s.duration else None,
                    "success": s.exception is None,
                    "error": str(s.exception) if s.exception else None
                }
                for s in list(self.sessions.values())[-50:]  # Last 50 sessions
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def clear_history(self):
        """Clear debugging history to free memory"""
        self.sessions.clear()
        self.performance_history.clear()
        self.memory_history.clear()
        self.slow_calls.clear()
        gc.collect()


# Global debug profiler instance
_debug_profiler = None


def get_debug_profiler() -> DebugProfiler:
    """Get global debug profiler instance"""
    global _debug_profiler
    if _debug_profiler is None:
        _debug_profiler = DebugProfiler()
    return _debug_profiler


def debug_function(**kwargs):
    """Convenience decorator for debugging functions"""
    profiler = get_debug_profiler()
    return profiler.debug_function(**kwargs)


@contextmanager
def debug_context(operation_name: str):
    """Context manager for debugging code blocks"""
    profiler = get_debug_profiler()
    start_time = time.time()
    
    print(f"DEBUG: Starting {operation_name}")
    try:
        yield
        duration = time.time() - start_time
        print(f"DEBUG: Completed {operation_name} in {duration:.3f}s")
    except Exception as e:
        duration = time.time() - start_time
        print(f"ERROR: {operation_name} failed after {duration:.3f}s: {e}")
        raise


def memory_usage_tracker():
    """Decorator to track memory usage of functions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not MEMORY_PROFILER_AVAILABLE:
                return func(*args, **kwargs)
            
            # Measure memory before and after
            mem_before = memory_profiler.memory_usage()[0]
            result = func(*args, **kwargs)
            mem_after = memory_profiler.memory_usage()[0]
            
            mem_diff = mem_after - mem_before
            if abs(mem_diff) > 1.0:  # Only log if significant change
                print(f"MEMORY: {func.__module__}.{func.__name__} changed memory by {mem_diff:.2f} MB")
            
            return result
        return wrapper
    return decorator


def trace_calls(depth: int = 5):
    """Decorator to trace function calls with limited depth"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            stack = inspect.stack()
            current_depth = len([frame for frame in stack if frame.function == func.__name__])
            
            if current_depth <= depth:
                indent = "  " * (current_depth - 1)
                print(f"TRACE: {indent}→ {func.__module__}.{func.__name__}")
                
                try:
                    result = func(*args, **kwargs)
                    print(f"TRACE: {indent}← {func.__module__}.{func.__name__} (success)")
                    return result
                except Exception as e:
                    print(f"TRACE: {indent}← {func.__module__}.{func.__name__} (error: {e})")
                    raise
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def setup_debug_monitoring(enable: bool = True, interval: float = 5.0):
    """Setup global debug monitoring"""
    if enable:
        profiler = get_debug_profiler()
        profiler.start_monitoring(interval)
        return profiler
    return None


def get_debug_report() -> Dict[str, Any]:
    """Get comprehensive debug report"""
    profiler = get_debug_profiler()
    return {
        "debug_stats": profiler.get_debug_stats(),
        "performance_summary": profiler.get_performance_summary(),
        "memory_report": profiler.get_memory_report()
    }


def print_debug_summary():
    """Print a human-readable debug summary"""
    report = get_debug_report()
    
    print("\n" + "="*60)
    print("🔍 AURA RENDER DEBUG SUMMARY")
    print("="*60)
    
    # Function call stats
    debug_stats = report["debug_stats"]
    print(f"\n📊 Function Call Statistics:")
    print(f"  • Total function calls tracked: {sum(debug_stats['function_calls'].values())}")
    print(f"  • Active debug sessions: {debug_stats['active_sessions']}")
    print(f"  • Total debug sessions: {debug_stats['total_sessions']}")
    
    if debug_stats['slow_calls']:
        print(f"\n⏰ Recent Slow Calls:")
        for func_name, duration in debug_stats['slow_calls'][-5:]:
            print(f"  • {func_name}: {duration:.3f}s")
    
    if debug_stats['error_counts']:
        print(f"\n❌ Error Counts:")
        for error, count in list(debug_stats['error_counts'].items())[-5:]:
            print(f"  • {error}: {count} times")
    
    # Performance summary
    perf_summary = report["performance_summary"]
    if perf_summary:
        print(f"\n💻 Performance Summary:")
        print(f"  • Average CPU usage: {perf_summary['average_cpu_percent']}%")
        print(f"  • Maximum CPU usage: {perf_summary['maximum_cpu_percent']}%")
        print(f"  • Average memory usage: {perf_summary['average_memory_mb']} MB")
        print(f"  • Maximum memory usage: {perf_summary['maximum_memory_mb']} MB")
        print(f"  • Current threads: {perf_summary['current_thread_count']}")
        print(f"  • Open file handles: {perf_summary['current_open_files']}")
    
    # Memory report
    memory_report = report["memory_report"]
    if memory_report:
        print(f"\n🧠 Memory Report:")
        print(f"  • Process memory usage: {memory_report['process_memory_mb']} MB")
        print(f"  • Memory trend: {memory_report['memory_trend']}")
        print(f"  • System memory available: {memory_report['available_system_memory_gb']} GB")
        
        if memory_report['potential_memory_leaks']:
            print(f"  ⚠️  Potential memory leaks detected:")
            for leak in memory_report['potential_memory_leaks'][:3]:
                print(f"    • {leak}")
    
    print("="*60)
    print(f"📅 Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)