"""
Material Providers Package

Integrates with multiple material providers including:
- External material service
- Pexels (videos and images)
- Pixabay (videos and images)
- Unsplash (images)
- Freesound (audio)
"""

from .base_provider import BaseMaterialProvider, MaterialSearchResult, MaterialType
from .provider_manager import MaterialProviderManager, get_provider_manager
from .external_provider import ExternalMaterialProvider
from .pexels_provider import PexelsProvider
from .pixabay_provider import PixabayProvider
from .unsplash_provider import UnsplashProvider
from .freesound_provider import FreesoundProvider

__all__ = [
    'BaseMaterialProvider',
    'MaterialSearchResult',
    'MaterialType',
    'MaterialProviderManager',
    'get_provider_manager',
    'ExternalMaterialProvider',
    'PexelsProvider',
    'PixabayProvider',
    'UnsplashProvider',
    'FreesoundProvider'
]