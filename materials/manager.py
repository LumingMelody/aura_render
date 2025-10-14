"""
Material Manager

Central manager for all material providers, handles multi-provider search,
caching, and material selection logic.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import asdict

from .base import (
    MaterialProvider, MaterialSearchQuery, MaterialSearchResponse,
    MaterialSearchResult, MaterialType, MockMaterialProvider
)
from .video import PexelsVideoProvider, PixabayVideoProvider
from .audio import FreeSoundProvider, MockAudioProvider
from .image import UnsplashImageProvider, PexelsImageProvider, MockImageProvider
from .external import ExternalMaterialProvider, MockExternalProvider


class MaterialManager:
    """
    Central manager for material providers
    
    Handles multi-provider searches, result aggregation, and caching.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        self.providers: Dict[str, MaterialProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available material providers"""
        provider_configs = self.config.get("providers", {})
        
        # External provider (prioritized)
        try:
            external_config = provider_configs.get("external", {})
            if external_config.get("base_url") or external_config.get("api_key"):
                external_provider = ExternalMaterialProvider(external_config)
                if external_provider.is_available():
                    self.providers["external"] = external_provider
                    self.logger.info("✅ External Material provider initialized")
                else:
                    self.logger.info("⚠️ External Material provider not available")
            else:
                # Use mock external provider for development
                mock_external = MockExternalProvider(external_config)
                self.providers["external"] = mock_external
                self.logger.info("✅ Mock External Material provider initialized (development mode)")
        except Exception as e:
            self.logger.error(f"Failed to initialize External provider: {e}")
        
        # Video providers
        try:
            pexels_config = provider_configs.get("pexels", {})
            pexels_video = PexelsVideoProvider(pexels_config)
            if pexels_video.is_available():
                self.providers["pexels_video"] = pexels_video
                self.logger.info("✅ Pexels Video provider initialized")
            else:
                self.logger.info("⚠️ Pexels Video provider not available (missing API key)")
        except Exception as e:
            self.logger.error(f"Failed to initialize Pexels Video provider: {e}")
        
        try:
            pixabay_config = provider_configs.get("pixabay", {})
            pixabay_video = PixabayVideoProvider(pixabay_config)
            if pixabay_video.is_available():
                self.providers["pixabay_video"] = pixabay_video
                self.logger.info("✅ Pixabay Video provider initialized")
            else:
                self.logger.info("⚠️ Pixabay Video provider not available (missing API key)")
        except Exception as e:
            self.logger.error(f"Failed to initialize Pixabay Video provider: {e}")
        
        # Audio providers
        try:
            freesound_config = provider_configs.get("freesound", {})
            freesound = FreeSoundProvider(freesound_config)
            if freesound.is_available():
                self.providers["freesound"] = freesound
                self.logger.info("✅ Freesound provider initialized")
            else:
                self.logger.info("⚠️ Freesound provider not available (missing API key)")
        except Exception as e:
            self.logger.error(f"Failed to initialize Freesound provider: {e}")
        
        # Image providers
        try:
            unsplash_config = provider_configs.get("unsplash", {})
            unsplash = UnsplashImageProvider(unsplash_config)
            if unsplash.is_available():
                self.providers["unsplash"] = unsplash
                self.logger.info("✅ Unsplash provider initialized")
            else:
                self.logger.info("⚠️ Unsplash provider not available (missing API key)")
        except Exception as e:
            self.logger.error(f"Failed to initialize Unsplash provider: {e}")
        
        try:
            pexels_img_config = provider_configs.get("pexels", {})
            pexels_images = PexelsImageProvider(pexels_img_config)
            if pexels_images.is_available():
                self.providers["pexels_images"] = pexels_images
                self.logger.info("✅ Pexels Images provider initialized")
            else:
                self.logger.info("⚠️ Pexels Images provider not available (missing API key)")
        except Exception as e:
            self.logger.error(f"Failed to initialize Pexels Images provider: {e}")
        
        # Always add mock providers as fallback
        self.providers["mock_video"] = MockMaterialProvider()
        self.providers["mock_audio"] = MockAudioProvider()
        self.providers["mock_image"] = MockImageProvider()
        
        self.logger.info(f"📊 Initialized {len(self.providers)} material providers")
    
    def get_available_providers(self, material_type: Optional[MaterialType] = None) -> List[str]:
        """Get list of available provider names, optionally filtered by material type"""
        if material_type is None:
            return list(self.providers.keys())
        
        available = []
        for name, provider in self.providers.items():
            if material_type in provider.get_supported_types():
                available.append(name)
        
        return available
    
    def get_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all providers"""
        info = {}
        for name, provider in self.providers.items():
            info[name] = {
                "provider_name": provider.provider_name,
                "is_available": provider.is_available(),
                "supported_types": [t.value for t in provider.get_supported_types()],
                "rate_limits": provider.get_rate_limit_info()
            }
        return info
    
    async def search_materials(
        self,
        query: MaterialSearchQuery,
        providers: Optional[List[str]] = None,
        max_concurrent: int = 3
    ) -> List[MaterialSearchResponse]:
        """
        Search for materials across multiple providers
        
        Args:
            query: Search query
            providers: List of provider names to use (None = auto-select)
            max_concurrent: Maximum concurrent provider requests
            
        Returns:
            List of responses from each provider
        """
        if providers is None:
            providers = self.get_available_providers(query.material_type)
        
        # Filter to available providers
        available_providers = []
        for provider_name in providers:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                if query.material_type is None or query.material_type in provider.get_supported_types():
                    available_providers.append(provider_name)
        
        if not available_providers:
            self.logger.warning(f"No available providers for material type: {query.material_type}")
            return []
        
        self.logger.info(f"🔍 Searching {len(available_providers)} providers: {available_providers}")
        
        # Create semaphore for concurrent request limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def search_provider(provider_name: str) -> MaterialSearchResponse:
            async with semaphore:
                try:
                    provider = self.providers[provider_name]
                    self.logger.debug(f"🔍 Searching {provider_name}...")
                    response = await provider.search(query)
                    self.logger.debug(f"✅ {provider_name}: {len(response.results)} results")
                    return response
                except Exception as e:
                    self.logger.error(f"❌ {provider_name} search failed: {e}")
                    return MaterialSearchResponse(
                        query=query,
                        results=[],
                        total_count=0,
                        search_time=0.0,
                        provider=provider_name,
                        provider_info={"error": str(e)}
                    )
        
        # Execute searches concurrently
        tasks = [search_provider(name) for name in available_providers]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_responses = []
        for response in responses:
            if isinstance(response, MaterialSearchResponse):
                valid_responses.append(response)
            else:
                self.logger.error(f"Provider search exception: {response}")
        
        return valid_responses
    
    async def search_and_aggregate(
        self,
        query: MaterialSearchQuery,
        providers: Optional[List[str]] = None,
        max_concurrent: int = 3,
        sort_by: str = "relevance"
    ) -> MaterialSearchResponse:
        """
        Search multiple providers and aggregate results into single response
        
        Args:
            query: Search query
            providers: Provider names to use
            max_concurrent: Max concurrent requests
            sort_by: How to sort results ("relevance", "quality", "popularity")
            
        Returns:
            Aggregated search response
        """
        import time
        start_time = time.time()
        
        # Get responses from all providers
        responses = await self.search_materials(query, providers, max_concurrent)
        
        if not responses:
            return MaterialSearchResponse(
                query=query,
                results=[],
                total_count=0,
                search_time=time.time() - start_time,
                provider="aggregate",
                provider_info={"providers": [], "errors": ["No providers available"]}
            )
        
        # Aggregate results
        all_results = []
        provider_info = {"providers": [], "errors": []}
        
        for response in responses:
            provider_info["providers"].append({
                "name": response.provider,
                "result_count": len(response.results),
                "search_time": response.search_time,
                "total_count": response.total_count
            })
            
            if "error" in response.provider_info:
                provider_info["errors"].append(f"{response.provider}: {response.provider_info['error']}")
            
            all_results.extend(response.results)
        
        # Sort results
        if sort_by == "relevance":
            all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        elif sort_by == "quality":
            all_results.sort(key=lambda x: x.quality_score, reverse=True)
        elif sort_by == "popularity":
            all_results.sort(key=lambda x: x.popularity_score, reverse=True)
        elif sort_by == "combined":
            # Combined score: relevance * 0.4 + quality * 0.3 + popularity * 0.3
            all_results.sort(
                key=lambda x: (x.relevance_score * 0.4 + x.quality_score * 0.3 + x.popularity_score * 0.3),
                reverse=True
            )
        
        # Apply limit and offset
        start_idx = query.offset
        end_idx = start_idx + query.limit
        limited_results = all_results[start_idx:end_idx]
        
        search_time = time.time() - start_time
        
        return MaterialSearchResponse(
            query=query,
            results=limited_results,
            total_count=len(all_results),
            search_time=search_time,
            provider="aggregate",
            provider_info=provider_info
        )
    
    async def get_material_info(self, material_id: str) -> Optional[MaterialSearchResult]:
        """Get detailed information about a specific material"""
        # Extract provider from material ID
        if "_" in material_id:
            provider_prefix = material_id.split("_")[0]
            
            # Find matching provider
            for name, provider in self.providers.items():
                if provider.provider_name == provider_prefix or name.startswith(provider_prefix):
                    return await provider.get_material_info(material_id)
        
        # Try all providers if no specific match
        for provider in self.providers.values():
            result = await provider.get_material_info(material_id)
            if result:
                return result
        
        return None
    
    async def download_material(
        self, 
        material: MaterialSearchResult, 
        local_path: str
    ) -> bool:
        """Download a material to local path"""
        # Extract provider from material ID  
        provider_name = None
        if "_" in material.material_id:
            provider_prefix = material.material_id.split("_")[0]
            
            for name, provider in self.providers.items():
                if provider.provider_name == provider_prefix or name.startswith(provider_prefix):
                    provider_name = name
                    break
        
        if provider_name and provider_name in self.providers:
            provider = self.providers[provider_name]
            return await provider.download_material(material, local_path)
        
        self.logger.error(f"No provider found for material: {material.material_id}")
        return False
    
    async def smart_search(
        self,
        keywords: List[str],
        material_type: MaterialType,
        context: Optional[Dict[str, Any]] = None,
        max_results: int = 10
    ) -> List[MaterialSearchResult]:
        """
        Smart material search with enhanced filtering and selection
        
        Args:
            keywords: Search keywords
            material_type: Type of material to search for
            context: Additional context (emotion, theme, duration, etc.)
            max_results: Maximum results to return
            
        Returns:
            List of best matching materials
        """
        # Build search query
        query = MaterialSearchQuery(
            keywords=keywords,
            material_type=material_type,
            limit=max_results * 2,  # Get more results for better filtering
            sort_by="combined"
        )
        
        # Apply context filters
        if context:
            if "max_duration" in context:
                query.max_duration = context["max_duration"]
            if "min_duration" in context:
                query.min_duration = context["min_duration"]
            if "aspect_ratio" in context:
                query.preferred_aspect_ratio = context["aspect_ratio"]
            if "min_quality" in context:
                query.min_quality = context["min_quality"]
        
        # Search and aggregate
        response = await self.search_and_aggregate(query, sort_by="combined")
        
        # Apply smart filtering
        filtered_results = self._apply_smart_filters(response.results, context)
        
        return filtered_results[:max_results]
    
    def _apply_smart_filters(
        self,
        results: List[MaterialSearchResult],
        context: Optional[Dict[str, Any]]
    ) -> List[MaterialSearchResult]:
        """Apply intelligent filtering based on context"""
        if not context:
            return results
        
        filtered = results.copy()
        
        # Filter by emotion/mood if specified
        if "emotion" in context:
            emotion = context["emotion"].lower()
            emotion_keywords = {
                "happy": ["bright", "colorful", "joy", "celebration", "smile"],
                "sad": ["dark", "rain", "lonely", "melancholy", "blue"],
                "excited": ["action", "fast", "energy", "dynamic", "movement"],
                "calm": ["peaceful", "serene", "quiet", "gentle", "soft"],
                "professional": ["clean", "business", "formal", "corporate", "modern"]
            }
            
            if emotion in emotion_keywords:
                mood_keywords = emotion_keywords[emotion]
                for result in filtered:
                    # Boost relevance if tags match mood
                    for tag in result.metadata.tags:
                        if any(mood_word in tag.lower() for mood_word in mood_keywords):
                            result.relevance_score = min(1.0, result.relevance_score + 0.1)
        
        # Re-sort after filtering
        filtered.sort(
            key=lambda x: (x.relevance_score * 0.4 + x.quality_score * 0.3 + x.popularity_score * 0.3),
            reverse=True
        )
        
        return filtered