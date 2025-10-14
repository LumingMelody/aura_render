"""
Video Material Providers

Implementations for various video material providers like Pexels, Pixabay, etc.
"""

import asyncio
import httpx
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from .base import (
    MaterialProvider, MaterialSearchQuery, MaterialSearchResponse, 
    MaterialSearchResult, MaterialMetadata, MaterialType, MaterialFormat,
    MaterialLicense
)


class PexelsVideoProvider(MaterialProvider):
    """
    Pexels API video provider
    
    Requires PEXELS_API_KEY environment variable or in config
    Free API: 200 requests per hour
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("pexels", config)
        self.api_key = self.config.get("api_key") or self.config.get("PEXELS_API_KEY")
        self.base_url = "https://api.pexels.com/videos"
        
        if not self.api_key:
            self.logger.warning("Pexels API key not configured")
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_supported_types(self) -> List[MaterialType]:
        return [MaterialType.VIDEO]
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        return {
            "requests_per_hour": 200,
            "requests_per_day": 20000,
            "concurrent_requests": 3
        }
    
    async def search(self, query: MaterialSearchQuery) -> MaterialSearchResponse:
        """Search Pexels for videos"""
        import time
        start_time = time.time()
        
        if not self.is_available():
            self.logger.error("Pexels provider not available - missing API key")
            return MaterialSearchResponse(
                query=query,
                results=[],
                total_count=0,
                search_time=0.0,
                provider="pexels",
                provider_info={"error": "API key not configured"}
            )
        
        # Build search parameters
        search_term = " ".join(query.keywords)
        params = {
            "query": search_term,
            "per_page": min(query.limit, 80),  # Pexels max is 80
            "page": (query.offset // query.limit) + 1 if query.limit > 0 else 1
        }
        
        # Add orientation filter if aspect ratio is specified
        if query.preferred_aspect_ratio:
            if query.preferred_aspect_ratio in ["16:9", "landscape"]:
                params["orientation"] = "landscape"
            elif query.preferred_aspect_ratio in ["9:16", "portrait"]:
                params["orientation"] = "portrait"
            elif query.preferred_aspect_ratio in ["1:1", "square"]:
                params["orientation"] = "square"
        
        headers = {
            "Authorization": self.api_key,
            "User-Agent": "Aura-Render/1.0"
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/search",
                    params=params,
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
                
                # Convert Pexels response to our format
                results = []
                for video_data in data.get("videos", []):
                    result = self._convert_pexels_video(video_data, query.keywords)
                    if result:
                        results.append(result)
                
                search_time = time.time() - start_time
                
                return MaterialSearchResponse(
                    query=query,
                    results=results,
                    total_count=data.get("total_results", len(results)),
                    search_time=search_time,
                    provider="pexels",
                    provider_info={
                        "page": data.get("page", 1),
                        "per_page": data.get("per_page", query.limit),
                        "next_page": data.get("next_page")
                    }
                )
        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Pexels API error {e.response.status_code}: {e.response.text}")
            error_msg = f"HTTP {e.response.status_code}"
            if e.response.status_code == 429:
                error_msg = "Rate limit exceeded"
            elif e.response.status_code == 401:
                error_msg = "Invalid API key"
        except Exception as e:
            self.logger.error(f"Pexels search failed: {e}")
            error_msg = str(e)
        
        return MaterialSearchResponse(
            query=query,
            results=[],
            total_count=0,
            search_time=time.time() - start_time,
            provider="pexels",
            provider_info={"error": error_msg}
        )
    
    def _convert_pexels_video(self, video_data: Dict[str, Any], search_keywords: List[str]) -> Optional[MaterialSearchResult]:
        """Convert Pexels video data to MaterialSearchResult"""
        try:
            video_id = str(video_data.get("id"))
            
            # Get best quality video file
            video_files = video_data.get("video_files", [])
            if not video_files:
                return None
            
            # Sort by quality (prefer HD)
            video_files.sort(key=lambda x: x.get("width", 0) * x.get("height", 0), reverse=True)
            best_video = video_files[0]
            
            # Create metadata
            metadata = MaterialMetadata(
                title=f"Pexels Video {video_id}",
                description=f"Video from Pexels related to: {', '.join(search_keywords)}",
                tags=search_keywords[:5],
                duration=float(video_data.get("duration", 0)),
                width=best_video.get("width"),
                height=best_video.get("height"),
                format=MaterialFormat.MP4,
                resolution=f"{best_video.get('width')}x{best_video.get('height')}",
                license=MaterialLicense.FREE,
                license_url="https://www.pexels.com/license/",
                source="Pexels",
                author=video_data.get("user", {}).get("name", "Unknown")
            )
            
            # Calculate relevance score (simplified)
            relevance_score = min(1.0, len(search_keywords) * 0.3)
            
            # Get thumbnail
            thumbnail_url = video_data.get("image")
            
            return MaterialSearchResult(
                material_id=f"pexels_{video_id}",
                material_type=MaterialType.VIDEO,
                url=best_video.get("link"),
                thumbnail_url=thumbnail_url,
                metadata=metadata,
                relevance_score=relevance_score,
                matching_tags=search_keywords,
                quality_score=min(1.0, (best_video.get("width", 0) / 1920) * 0.8 + 0.2),
                popularity_score=0.7  # Pexels videos are generally popular
            )
        
        except Exception as e:
            self.logger.error(f"Failed to convert Pexels video data: {e}")
            return None
    
    async def get_material_info(self, material_id: str) -> Optional[MaterialSearchResult]:
        """Get detailed info about a Pexels video"""
        if not material_id.startswith("pexels_"):
            return None
        
        video_id = material_id.replace("pexels_", "")
        
        if not self.is_available():
            return None
        
        headers = {
            "Authorization": self.api_key,
            "User-Agent": "Aura-Render/1.0"
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/videos/{video_id}",
                    headers=headers
                )
                response.raise_for_status()
                video_data = response.json()
                
                return self._convert_pexels_video(video_data, ["video"])
        
        except Exception as e:
            self.logger.error(f"Failed to get Pexels video info: {e}")
            return None
    
    async def download_material(self, material: MaterialSearchResult, local_path: str) -> bool:
        """Download video from Pexels"""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get(material.url)
                response.raise_for_status()
                
                with open(local_path, "wb") as f:
                    async for chunk in response.aiter_bytes(8192):
                        f.write(chunk)
                
                return True
        
        except Exception as e:
            self.logger.error(f"Failed to download material: {e}")
            return False
    
    async def get_download_url(self, material: MaterialSearchResult) -> str:
        """Get direct download URL"""
        return material.url


class PixabayVideoProvider(MaterialProvider):
    """
    Pixabay API video provider
    
    Requires PIXABAY_API_KEY environment variable or in config
    Free API: 5000 requests per hour
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("pixabay", config)
        self.api_key = self.config.get("api_key") or self.config.get("PIXABAY_API_KEY")
        self.base_url = "https://pixabay.com/api/videos/"
        
        if not self.api_key:
            self.logger.warning("Pixabay API key not configured")
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_supported_types(self) -> List[MaterialType]:
        return [MaterialType.VIDEO]
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        return {
            "requests_per_hour": 5000,
            "requests_per_day": 50000,
            "concurrent_requests": 5
        }
    
    async def search(self, query: MaterialSearchQuery) -> MaterialSearchResponse:
        """Search Pixabay for videos"""
        import time
        start_time = time.time()
        
        if not self.is_available():
            self.logger.error("Pixabay provider not available - missing API key")
            return MaterialSearchResponse(
                query=query,
                results=[],
                total_count=0,
                search_time=0.0,
                provider="pixabay",
                provider_info={"error": "API key not configured"}
            )
        
        # Build search parameters
        search_term = " ".join(query.keywords)
        params = {
            "key": self.api_key,
            "q": search_term,
            "per_page": min(query.limit, 200),  # Pixabay max is 200
            "page": (query.offset // query.limit) + 1 if query.limit > 0 else 1,
            "video_type": "all",
            "safesearch": "true"
        }
        
        # Add duration filter
        if query.min_duration or query.max_duration:
            if query.min_duration and query.min_duration <= 20:
                params["min_duration"] = int(query.min_duration)
            if query.max_duration and query.max_duration >= 20:
                params["max_duration"] = int(query.max_duration)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Convert Pixabay response to our format
                results = []
                for video_data in data.get("hits", []):
                    result = self._convert_pixabay_video(video_data, query.keywords)
                    if result:
                        results.append(result)
                
                search_time = time.time() - start_time
                
                return MaterialSearchResponse(
                    query=query,
                    results=results,
                    total_count=data.get("total", len(results)),
                    search_time=search_time,
                    provider="pixabay",
                    provider_info={
                        "total_hits": data.get("totalHits", 0)
                    }
                )
        
        except Exception as e:
            self.logger.error(f"Pixabay search failed: {e}")
            return MaterialSearchResponse(
                query=query,
                results=[],
                total_count=0,
                search_time=time.time() - start_time,
                provider="pixabay",
                provider_info={"error": str(e)}
            )
    
    def _convert_pixabay_video(self, video_data: Dict[str, Any], search_keywords: List[str]) -> Optional[MaterialSearchResult]:
        """Convert Pixabay video data to MaterialSearchResult"""
        try:
            video_id = str(video_data.get("id"))
            
            # Get video URL (try different sizes)
            video_url = None
            video_sizes = ["large", "medium", "small"]
            for size in video_sizes:
                size_data = video_data.get("videos", {}).get(size, {})
                if size_data and size_data.get("url"):
                    video_url = size_data.get("url")
                    width = size_data.get("width", 0)
                    height = size_data.get("height", 0)
                    break
            
            if not video_url:
                return None
            
            # Parse tags
            tags = video_data.get("tags", "").split(", ") if video_data.get("tags") else search_keywords
            
            metadata = MaterialMetadata(
                title=f"Pixabay Video {video_id}",
                description=f"Video from Pixabay: {tags[0] if tags else 'Video'}",
                tags=tags[:10],
                duration=float(video_data.get("duration", 0)),
                width=width,
                height=height,
                format=MaterialFormat.MP4,
                resolution=f"{width}x{height}",
                license=MaterialLicense.FREE,
                license_url="https://pixabay.com/service/license/",
                source="Pixabay",
                author=video_data.get("user", "Unknown")
            )
            
            # Calculate scores
            relevance_score = min(1.0, video_data.get("views", 0) / 10000 * 0.3 + 0.4)
            quality_score = min(1.0, (width / 1920) * 0.8 + 0.2)
            popularity_score = min(1.0, video_data.get("downloads", 0) / 1000 * 0.5 + 0.3)
            
            return MaterialSearchResult(
                material_id=f"pixabay_{video_id}",
                material_type=MaterialType.VIDEO,
                url=video_url,
                thumbnail_url=video_data.get("picture"),
                metadata=metadata,
                relevance_score=relevance_score,
                matching_tags=tags[:5],
                quality_score=quality_score,
                popularity_score=popularity_score
            )
        
        except Exception as e:
            self.logger.error(f"Failed to convert Pixabay video data: {e}")
            return None
    
    async def get_material_info(self, material_id: str) -> Optional[MaterialSearchResult]:
        """Get detailed info about a Pixabay video"""
        # Pixabay doesn't have a single video endpoint, so we'd need to search by ID
        # This is a limitation of their API
        return None
    
    async def download_material(self, material: MaterialSearchResult, local_path: str) -> bool:
        """Download video from Pixabay"""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get(material.url)
                response.raise_for_status()
                
                with open(local_path, "wb") as f:
                    async for chunk in response.aiter_bytes(8192):
                        f.write(chunk)
                
                return True
        
        except Exception as e:
            self.logger.error(f"Failed to download material: {e}")
            return False
    
    async def get_download_url(self, material: MaterialSearchResult) -> str:
        """Get direct download URL"""
        return material.url


# Alias for the primary video provider
VideoMaterialProvider = PexelsVideoProvider