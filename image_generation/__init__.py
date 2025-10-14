"""
Image Generation Services Package - 统一的AI图像生成服务
"""

from .base_image_client import (
    BaseImageGenerationClient,
    ImageGenerationRequest,
    ImageGenerationResponse,
    GeneratedImage,
    ImageStyle,
    ImageQuality,
    AspectRatio,
    ModelCapabilities
)

from .openai_dalle_client import OpenAIDALLEClient
from .stability_ai_client import StabilityAIClient
from .midjourney_client import MidjourneyClient

from .image_generation_manager import (
    ImageGenerationManager,
    ImageProvider,
    ImageProviderConfig,
    GenerationCriteria,
    create_default_image_manager
)

__all__ = [
    # Base classes
    "BaseImageGenerationClient",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "GeneratedImage",
    "ImageStyle",
    "ImageQuality",
    "AspectRatio",
    "ModelCapabilities",

    # Client implementations
    "OpenAIDALLEClient",
    "StabilityAIClient",
    "MidjourneyClient",

    # Service manager
    "ImageGenerationManager",
    "ImageProvider",
    "ImageProviderConfig",
    "GenerationCriteria",
    "create_default_image_manager"
]

__version__ = "1.0.0"