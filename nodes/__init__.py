"""
Video Generation Processing Nodes

Core processing nodes for the video generation pipeline, including audio processing,
subtitle generation, effects, transitions, and rendering capabilities.
"""

from .base_node import BaseNode, NodeConfig, NodeResult, ProcessingContext
from .audio_processor import AudioProcessingNode, AudioProcessingConfig
from .subtitle_generator import SubtitleGeneratorNode, SubtitleConfig
from .effects_processor import EffectsProcessorNode, EffectsConfig
from .transitions_processor import TransitionsProcessorNode, TransitionsConfig
from .render_compositor import RenderCompositorNode, NodeRenderConfig as RenderConfig

__all__ = [
    'BaseNode',
    'NodeConfig', 
    'NodeResult',
    'ProcessingContext',
    'AudioProcessingNode',
    'AudioProcessingConfig',
    'SubtitleGeneratorNode',
    'SubtitleConfig',
    'EffectsProcessorNode',
    'EffectsConfig',
    'TransitionsProcessorNode',
    'TransitionsConfig',
    'RenderCompositorNode',
    'RenderConfig'
]