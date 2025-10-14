"""Unsplash Material Provider - Placeholder"""

from .base_provider import BaseMaterialProvider, MaterialType
from typing import List, Optional, Dict, Any

class UnsplashProvider(BaseMaterialProvider):
    """Unsplash provider placeholder"""
    def __init__(self, api_key=None):
        super().__init__("unsplash", api_key)
    
    async def _initialize(self):
        pass
    
    async def search(self, query, material_type, limit=10, offset=0, filters=None):
        return []
    
    async def get_material(self, material_id):
        return None
    
    async def download(self, material_id, destination):
        return False
    
    def supports_type(self, material_type):
        return material_type == MaterialType.IMAGE