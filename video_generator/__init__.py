"""
视频生成器模块
"""

# Use enhanced generator instead of simple generator
from .enhanced_video_generator import EnhancedVideoGenerator as RealVideoGenerator, get_enhanced_video_generator as get_video_generator

# try:
#     from .real_video_generator import RealVideoGenerator, get_video_generator
# except ImportError:
#     # 如果复杂版本导入失败，使用简化版本
#     from .simple_video_generator import SimpleVideoGenerator as RealVideoGenerator, get_simple_video_generator as get_video_generator

__all__ = ['RealVideoGenerator', 'get_video_generator']