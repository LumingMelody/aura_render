"""
多模态模型集成模块
"""

try:
    from .vl_models import IntegratedVLSystem, vl_system
    HAS_VL = True
except ImportError:
    HAS_VL = False

try:
    from .qwen_integration import HybridVideoUnderstanding, hybrid_video_understanding
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False

__all__ = []

if HAS_VL:
    __all__.extend(['IntegratedVLSystem', 'vl_system'])

if HAS_QWEN:
    __all__.extend(['HybridVideoUnderstanding', 'hybrid_video_understanding'])