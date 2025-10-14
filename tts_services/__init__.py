"""
TTS Services Package - 统一的文本转语音服务
"""

from .base_tts_client import (
    BaseTTSClient,
    TTSRequest,
    TTSResponse,
    VoiceProfile,
    VoiceGender,
    VoiceStyle,
    EmotionMapping
)

from .azure_tts_client import AzureTTSClient
from .openai_tts_client import OpenAITTSClient
from .edge_tts_client import EdgeTTSClient

from .tts_service_manager import (
    TTSServiceManager,
    TTSProvider,
    TTSConfig,
    VoiceSelectionCriteria,
    create_default_tts_manager
)

__all__ = [
    # Base classes
    "BaseTTSClient",
    "TTSRequest",
    "TTSResponse",
    "VoiceProfile",
    "VoiceGender",
    "VoiceStyle",
    "EmotionMapping",

    # Client implementations
    "AzureTTSClient",
    "OpenAITTSClient",
    "EdgeTTSClient",

    # Service manager
    "TTSServiceManager",
    "TTSProvider",
    "TTSConfig",
    "VoiceSelectionCriteria",
    "create_default_tts_manager"
]

__version__ = "1.0.0"