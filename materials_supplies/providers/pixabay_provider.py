"""Pixabay Material Provider - Placeholder"""

from .pexels_provider import PexelsProvider

class PixabayProvider(PexelsProvider):
    """Pixabay provider placeholder - inherits from Pexels for now"""
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.name = "pixabay"