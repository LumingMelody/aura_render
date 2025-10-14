"""
AI-Powered Video Optimization Package

Advanced AI algorithms for intelligent video optimization including:
- Content-aware quality enhancement
- Intelligent bitrate optimization
- Smart scene detection and transitions
- AI-powered audio enhancement
- Automatic color correction and grading
- Performance optimization based on content analysis
"""

from .optimizer_engine import (
    VideoOptimizer,
    OptimizationConfig,
    OptimizationResult,
    OptimizationLevel,
    OptimizationType,
    get_video_optimizer
)

from .quality_enhancer import (
    QualityEnhancer,
    EnhancementType,
    EnhancementConfig,
    QualityLevel,
    QualityMetrics,
    get_quality_enhancer
)

from .scene_analyzer import (
    SceneAnalyzer,
    SceneType,
    SceneTransition,
    get_scene_analyzer
)

from .content_optimizer import (
    ContentOptimizer,
    ContentAnalysis,
    OptimizationStrategy,
    TargetPlatform,
    ContentCategory,
    AudienceSegment,
    get_content_optimizer
)

__all__ = [
    'VideoOptimizer',
    'OptimizationConfig', 
    'OptimizationResult',
    'OptimizationLevel',
    'OptimizationType',
    'get_video_optimizer',
    'QualityEnhancer',
    'EnhancementType',
    'EnhancementConfig',
    'QualityLevel',
    'QualityMetrics',
    'get_quality_enhancer',
    'SceneAnalyzer',
    'SceneType',
    'SceneTransition',
    'get_scene_analyzer',
    'ContentOptimizer',
    'ContentAnalysis',
    'OptimizationStrategy',
    'TargetPlatform',
    'ContentCategory',
    'AudienceSegment',
    'get_content_optimizer'
]