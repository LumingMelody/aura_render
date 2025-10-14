"""
Pexels Material Provider

Integrates with Pexels API for videos and images.
API Documentation: https://www.pexels.com/api/documentation/
"""

import httpx
from typing import List, Dict, Any, Optional
from .base_provider import BaseMaterialProvider, MaterialSearchResult, MaterialType
import logging


class PexelsProvider(BaseMaterialProvider):
    """Pexels API provider for videos and images"""
    
    BASE_URL = "https://api.pexels.com"
    VIDEOS_BASE_URL = "https://api.pexels.com/videos"
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(name="pexels", api_key=api_key)
        self.client = None
        
    async def _initialize(self):
        """Initialize HTTP client"""
        if not self.api_key:
            self.logger.warning("Pexels API key not provided")
            return
            
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": self.api_key
            },
            timeout=30.0
        )
        
    async def search(
        self,
        query: str,
        material_type: MaterialType,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MaterialSearchResult]:
        """Search for materials on Pexels"""
        
        if not self.client:
            return []
            
        results = []
        page = (offset // limit) + 1
        per_page = min(limit, 80)  # Pexels max is 80
        
        try:
            if material_type == MaterialType.VIDEO:
                results = await self._search_videos(query, per_page, page, filters)
            elif material_type == MaterialType.IMAGE:
                results = await self._search_images(query, per_page, page, filters)
            else:
                self.logger.warning(f"Unsupported material type: {material_type}")
                
        except Exception as e:
            self.logger.error(f"Pexels search error: {e}")
            
        return results
        
    async def _search_videos(
        self,
        query: str,
        per_page: int,
        page: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MaterialSearchResult]:
        """Search for videos"""
        
        params = {
            "query": query,
            "per_page": per_page,
            "page": page
        }
        
        # Add filters
        if filters:
            if "orientation" in filters:
                params["orientation"] = filters["orientation"]
            if "size" in filters:
                params["size"] = filters["size"]
            if "min_width" in filters:
                params["min_width"] = filters["min_width"]
            if "min_height" in filters:
                params["min_height"] = filters["min_height"]
            if "min_duration" in filters:
                params["min_duration"] = filters["min_duration"]
            if "max_duration" in filters:
                params["max_duration"] = filters["max_duration"]
                
        response = await self.client.get(
            f"{self.VIDEOS_BASE_URL}/search",
            params=params
        )
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for video in data.get("videos", []):
            # Find best quality video file
            video_file = self._get_best_video_file(video.get("video_files", []))
            if not video_file:
                continue
                
            results.append(MaterialSearchResult(
                material_id=f"pexels_video_{video['id']}",
                provider="pexels",
                type=MaterialType.VIDEO,
                url=video_file["link"],
                thumbnail_url=video.get("image"),
                preview_url=video.get("video_pictures", [{}])[0].get("picture") if video.get("video_pictures") else None,
                title=f"Pexels Video {video['id']}",
                description=video.get("url", ""),
                tags=query.split(),  # Use query words as tags
                duration=float(video.get("duration", 0)),
                width=video_file.get("width"),
                height=video_file.get("height"),
                file_size=video_file.get("file_size"),
                format=video_file.get("file_type", "mp4"),
                author=video.get("user", {}).get("name"),
                author_url=video.get("user", {}).get("url"),
                source_url=video.get("url"),
                relevance_score=1.0,
                metadata={
                    "fps": video_file.get("fps"),
                    "quality": video_file.get("quality")
                }
            ))
            
        return results
        
    async def _search_images(
        self,
        query: str,
        per_page: int,
        page: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MaterialSearchResult]:
        """Search for images"""
        
        params = {
            "query": query,
            "per_page": per_page,
            "page": page
        }
        
        # Add filters
        if filters:
            if "orientation" in filters:
                params["orientation"] = filters["orientation"]
            if "size" in filters:
                params["size"] = filters["size"]
            if "color" in filters:
                params["color"] = filters["color"]
                
        response = await self.client.get(
            f"{self.BASE_URL}/v1/search",
            params=params
        )
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for photo in data.get("photos", []):
            results.append(MaterialSearchResult(
                material_id=f"pexels_image_{photo['id']}",
                provider="pexels",
                type=MaterialType.IMAGE,
                url=photo.get("src", {}).get("original", ""),
                thumbnail_url=photo.get("src", {}).get("small"),
                preview_url=photo.get("src", {}).get("medium"),
                title=photo.get("alt", f"Pexels Image {photo['id']}"),
                description=photo.get("alt", ""),
                tags=query.split(),
                width=photo.get("width"),
                height=photo.get("height"),
                format="jpeg",
                author=photo.get("photographer"),
                author_url=photo.get("photographer_url"),
                source_url=photo.get("url"),
                relevance_score=1.0,
                metadata={
                    "avg_color": photo.get("avg_color"),
                    "liked": photo.get("liked", False)
                }
            ))
            
        return results
        
    def _get_best_video_file(self, video_files: List[Dict]) -> Optional[Dict]:
        """Get the best quality video file"""
        if not video_files:
            return None
            
        # Sort by quality (HD > SD) and width
        sorted_files = sorted(
            video_files,
            key=lambda x: (
                x.get("quality") == "hd",
                x.get("width", 0)
            ),
            reverse=True
        )
        
        return sorted_files[0] if sorted_files else None
        
    async def get_material(self, material_id: str) -> Optional[MaterialSearchResult]:
        """Get material by ID"""
        
        if not self.client or not material_id.startswith("pexels_"):
            return None
            
        parts = material_id.split("_")
        if len(parts) < 3:
            return None
            
        material_type = parts[1]
        pexels_id = parts[2]
        
        try:
            if material_type == "video":
                return await self._get_video(pexels_id)
            elif material_type == "image":
                return await self._get_image(pexels_id)
        except Exception as e:
            self.logger.error(f"Error getting material {material_id}: {e}")
            
        return None
        
    async def _get_video(self, video_id: str) -> Optional[MaterialSearchResult]:
        """Get video by ID"""
        
        response = await self.client.get(f"{self.VIDEOS_BASE_URL}/videos/{video_id}")
        response.raise_for_status()
        
        video = response.json()
        video_file = self._get_best_video_file(video.get("video_files", []))
        
        if not video_file:
            return None
            
        return MaterialSearchResult(
            material_id=f"pexels_video_{video['id']}",
            provider="pexels",
            type=MaterialType.VIDEO,
            url=video_file["link"],
            thumbnail_url=video.get("image"),
            title=f"Pexels Video {video['id']}",
            duration=float(video.get("duration", 0)),
            width=video_file.get("width"),
            height=video_file.get("height"),
            file_size=video_file.get("file_size"),
            format=video_file.get("file_type", "mp4"),
            author=video.get("user", {}).get("name"),
            source_url=video.get("url"),
            relevance_score=1.0
        )
        
    async def _get_image(self, photo_id: str) -> Optional[MaterialSearchResult]:
        """Get image by ID"""
        
        response = await self.client.get(f"{self.BASE_URL}/v1/photos/{photo_id}")
        response.raise_for_status()
        
        photo = response.json()
        
        return MaterialSearchResult(
            material_id=f"pexels_image_{photo['id']}",
            provider="pexels",
            type=MaterialType.IMAGE,
            url=photo.get("src", {}).get("original", ""),
            thumbnail_url=photo.get("src", {}).get("small"),
            title=photo.get("alt", f"Pexels Image {photo['id']}"),
            width=photo.get("width"),
            height=photo.get("height"),
            format="jpeg",
            author=photo.get("photographer"),
            source_url=photo.get("url"),
            relevance_score=1.0
        )
        
    async def download(self, material_id: str, destination: str) -> bool:
        """Download material to local storage"""
        
        if not self.client:
            return False
            
        try:
            material = await self.get_material(material_id)
            if not material:
                return False
                
            # Download the file
            response = await self.client.get(material.url, follow_redirects=True)
            response.raise_for_status()
            
            # Save to destination
            with open(destination, "wb") as f:
                f.write(response.content)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Download error for {material_id}: {e}")
            return False
            
    def supports_type(self, material_type: MaterialType) -> bool:
        """Check if provider supports material type"""
        return material_type in [MaterialType.VIDEO, MaterialType.IMAGE]
        
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()