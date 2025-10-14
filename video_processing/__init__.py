"""
Video Processing Package

Provides comprehensive video processing capabilities including:
- Advanced FFmpeg rendering engine
- Video editing and compositing
- Professional codec support
- Hardware acceleration
- 4K/8K video processing
"""

from .advanced_ffmpeg_engine import (
    FFmpegEngine,
    get_ffmpeg_engine,
    VideoCodec,
    AudioCodec,
    VideoFormat,
    VideoQuality,
    Resolution,
    VideoTrack,
    AudioTrack,
    SubtitleTrack,
    RenderSettings
)
from .composition_engine import (
    UnifiedCompositionEngine,
    get_composition_engine,
    VideoComposition,
    CompositionLayer
)
from .effects_processor import (
    AdvancedEffectsProcessor,
    get_effects_processor,
    VideoEffect,
    EffectParameter,
    EffectCategory,
    EffectComplexity
)

__all__ = [
    'FFmpegEngine',
    'get_ffmpeg_engine',
    'VideoCodec',
    'AudioCodec', 
    'VideoFormat',
    'VideoQuality',
    'Resolution',
    'VideoTrack',
    'AudioTrack',
    'SubtitleTrack',
    'RenderSettings',
    'UnifiedCompositionEngine',
    'get_composition_engine',
    'VideoComposition',
    'CompositionLayer',
    'AdvancedEffectsProcessor',
    'get_effects_processor',
    'VideoEffect',
    'EffectParameter',
    'EffectCategory',
    'EffectComplexity'
]