"""
千问API集成模块
"""

from .tts_generator import (
    QwenTTSGenerator,
    get_qwen_tts_generator,
    generate_speech_from_text
)

__all__ = [
    'QwenTTSGenerator',
    'get_qwen_tts_generator',
    'generate_speech_from_text'
]
