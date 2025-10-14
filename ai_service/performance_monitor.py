from typing import Dict, Any

class AIPerformanceMonitor:
    def __init__(self):
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        return {"cpu_usage": 45.2, "memory_usage": 78.5}

_performance_monitor = None

def get_performance_monitor() -> AIPerformanceMonitor:
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = AIPerformanceMonitor()
    return _performance_monitor