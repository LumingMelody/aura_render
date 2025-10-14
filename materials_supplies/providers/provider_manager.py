"""
Material Provider Manager

Manages multiple material providers and routes requests to appropriate providers.
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from .base_provider import BaseMaterialProvider, MaterialSearchResult, MaterialType
import logging
from config import Settings


class MaterialProviderManager:
    """Manages and coordinates multiple material providers"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.providers: Dict[str, BaseMaterialProvider] = {}
        self.logger = logging.getLogger(__name__)
        self._initialized = False
        
    async def initialize(self):
        """Initialize all configured providers"""
        if self._initialized:
            return
            
        # Initialize Pexels provider
        if self.settings.materials.pexels_api_key:
            from .pexels_provider import PexelsProvider
            provider = PexelsProvider(api_key=self.settings.materials.pexels_api_key)
            await provider.initialize()
            if provider.is_available():
                self.providers["pexels"] = provider
                self.logger.info("Pexels provider initialized")
                
        # Initialize Pixabay provider
        if self.settings.materials.pixabay_api_key:
            from .pixabay_provider import PixabayProvider
            provider = PixabayProvider(api_key=self.settings.materials.pixabay_api_key)
            await provider.initialize()
            if provider.is_available():
                self.providers["pixabay"] = provider
                self.logger.info("Pixabay provider initialized")
                
        # Initialize Unsplash provider
        if self.settings.materials.unsplash_access_key:
            from .unsplash_provider import UnsplashProvider
            provider = UnsplashProvider(api_key=self.settings.materials.unsplash_access_key)
            await provider.initialize()
            if provider.is_available():
                self.providers["unsplash"] = provider
                self.logger.info("Unsplash provider initialized")
                
        # Initialize Freesound provider
        if self.settings.materials.freesound_api_key:
            from .freesound_provider import FreesoundProvider
            provider = FreesoundProvider(api_key=self.settings.materials.freesound_api_key)
            await provider.initialize()
            if provider.is_available():
                self.providers["freesound"] = provider
                self.logger.info("Freesound provider initialized")
                
        # Initialize external provider
        if self.settings.materials.external_api_key:
            from .external_provider import ExternalMaterialProvider
            provider = ExternalMaterialProvider(
                base_url=self.settings.materials.external_base_url,
                api_key=self.settings.materials.external_api_key
            )
            await provider.initialize()
            if provider.is_available():
                self.providers["external"] = provider
                self.logger.info("External provider initialized")
                
        self._initialized = True
        self.logger.info(f"Initialized {len(self.providers)} material providers")
        
    async def search(
        self,
        query: str,
        material_type: MaterialType,
        limit: int = 10,
        providers: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        aggregate: bool = True
    ) -> List[MaterialSearchResult]:
        """
        Search for materials across providers
        
        Args:
            query: Search query
            material_type: Type of material
            limit: Maximum results per provider
            providers: Specific providers to use (None = all)
            filters: Search filters
            aggregate: Whether to aggregate results from all providers
            
        Returns:
            List of search results
        """
        
        if not self._initialized:
            await self.initialize()
            
        # Determine which providers to use
        target_providers = self._get_target_providers(material_type, providers)
        
        if not target_providers:
            self.logger.warning(f"No providers available for {material_type}")
            return []
            
        # Search across providers
        if aggregate and len(target_providers) > 1:
            results = await self._search_aggregate(
                query, material_type, limit, target_providers, filters
            )
        else:
            # Use single provider
            provider_name = list(target_providers.keys())[0]
            provider = target_providers[provider_name]
            results = await provider.search(query, material_type, limit, 0, filters)
            
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:limit] if aggregate else results
        
    async def _search_aggregate(
        self,
        query: str,
        material_type: MaterialType,
        limit: int,
        providers: Dict[str, BaseMaterialProvider],
        filters: Optional[Dict[str, Any]]
    ) -> List[MaterialSearchResult]:
        """Aggregate search results from multiple providers"""
        
        # Calculate per-provider limit
        per_provider_limit = max(5, limit // len(providers))
        
        # Create search tasks
        tasks = []
        for name, provider in providers.items():
            task = asyncio.create_task(
                provider.search(query, material_type, per_provider_limit, 0, filters)
            )
            tasks.append((name, task))
            
        # Gather results
        all_results = []
        for name, task in tasks:
            try:
                results = await task
                all_results.extend(results)
                self.logger.debug(f"Got {len(results)} results from {name}")
            except Exception as e:
                self.logger.error(f"Search error in {name}: {e}")
                
        return all_results
        
    def _get_target_providers(
        self,
        material_type: MaterialType,
        provider_names: Optional[List[str]] = None
    ) -> Dict[str, BaseMaterialProvider]:
        """Get providers that support the material type"""
        
        target_providers = {}
        
        # Get preferred providers for this material type
        if provider_names is None:
            if material_type == MaterialType.VIDEO:
                provider_names = self.settings.materials.preferred_video_providers
            elif material_type == MaterialType.AUDIO:
                provider_names = self.settings.materials.preferred_audio_providers
            elif material_type == MaterialType.IMAGE:
                provider_names = self.settings.materials.preferred_image_providers
            else:
                provider_names = list(self.providers.keys())
                
        # Filter providers
        for name in provider_names:
            if name in self.providers:
                provider = self.providers[name]
                if provider.supports_type(material_type):
                    target_providers[name] = provider
                    
        return target_providers
        
    async def get_material(self, material_id: str) -> Optional[MaterialSearchResult]:
        """Get material by ID from any provider"""
        
        if not self._initialized:
            await self.initialize()
            
        # Extract provider from material ID
        provider_name = None
        for name in self.providers:
            if material_id.startswith(f"{name}_"):
                provider_name = name
                break
                
        if not provider_name or provider_name not in self.providers:
            self.logger.warning(f"No provider found for material {material_id}")
            return None
            
        provider = self.providers[provider_name]
        return await provider.get_material(material_id)
        
    async def download(self, material_id: str, destination: str) -> bool:
        """Download material from appropriate provider"""
        
        if not self._initialized:
            await self.initialize()
            
        # Extract provider from material ID
        provider_name = None
        for name in self.providers:
            if material_id.startswith(f"{name}_"):
                provider_name = name
                break
                
        if not provider_name or provider_name not in self.providers:
            self.logger.error(f"No provider found for material {material_id}")
            return False
            
        provider = self.providers[provider_name]
        return await provider.download(material_id, destination)
        
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.providers.keys())
        
    def get_supported_types(self, provider_name: str) -> Set[MaterialType]:
        """Get supported material types for a provider"""
        
        if provider_name not in self.providers:
            return set()
            
        provider = self.providers[provider_name]
        supported = set()
        
        for material_type in MaterialType:
            if provider.supports_type(material_type):
                supported.add(material_type)
                
        return supported
        
    async def validate_providers(self) -> Dict[str, bool]:
        """Validate all provider API keys"""
        
        if not self._initialized:
            await self.initialize()
            
        results = {}
        
        for name, provider in self.providers.items():
            try:
                is_valid = await provider.validate_api_key()
                results[name] = is_valid
                self.logger.info(f"Provider {name} validation: {is_valid}")
            except Exception as e:
                results[name] = False
                self.logger.error(f"Provider {name} validation error: {e}")
                
        return results
        
    async def close(self):
        """Close all provider connections"""
        
        for provider in self.providers.values():
            if hasattr(provider, '__aexit__'):
                await provider.__aexit__(None, None, None)
                
        self.providers.clear()
        self._initialized = False


# Global provider manager instance
_provider_manager: Optional[MaterialProviderManager] = None


def get_provider_manager(settings: Optional[Settings] = None) -> MaterialProviderManager:
    """Get global provider manager instance"""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = MaterialProviderManager(settings)
    return _provider_manager