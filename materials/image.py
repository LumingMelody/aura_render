"""
Image Material Providers

Implementations for various image material providers like Unsplash, Pexels, etc.
"""

import httpx
from typing import List, Dict, Any, Optional

from .base import (
    MaterialProvider, MaterialSearchQuery, MaterialSearchResponse, 
    MaterialSearchResult, MaterialMetadata, MaterialType, MaterialFormat,
    MaterialLicense
)


class UnsplashImageProvider(MaterialProvider):
    """
    Unsplash API image provider
    
    Requires UNSPLASH_ACCESS_KEY environment variable or in config
    Free API: 50 requests per hour
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("unsplash", config)
        self.access_key = self.config.get("access_key") or self.config.get("UNSPLASH_ACCESS_KEY")
        self.base_url = "https://api.unsplash.com"
        
        if not self.access_key:
            self.logger.warning("Unsplash access key not configured")
    
    def is_available(self) -> bool:
        return bool(self.access_key)
    
    def get_supported_types(self) -> List[MaterialType]:
        return [MaterialType.IMAGE]
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        return {
            "requests_per_hour": 50,
            "requests_per_day": 1000,
            "concurrent_requests": 3
        }
    
    async def search(self, query: MaterialSearchQuery) -> MaterialSearchResponse:
        """Search Unsplash for images"""
        import time
        start_time = time.time()
        
        if not self.is_available():
            return MaterialSearchResponse(
                query=query,
                results=[],
                total_count=0,
                search_time=0.0,
                provider="unsplash",
                provider_info={"error": "Access key not configured"}
            )
        
        search_term = " ".join(query.keywords)
        params = {
            "query": search_term,
            "per_page": min(query.limit, 30),  # Unsplash max is 30
            "page": (query.offset // query.limit) + 1 if query.limit > 0 else 1,
            "client_id": self.access_key
        }
        
        # Add orientation filter
        if query.preferred_aspect_ratio:
            if query.preferred_aspect_ratio in ["16:9", "landscape"]:
                params["orientation"] = "landscape"
            elif query.preferred_aspect_ratio in ["9:16", "portrait"]:
                params["orientation"] = "portrait"
            elif query.preferred_aspect_ratio in ["1:1", "square"]:
                params["orientation"] = "squarish"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/search/photos",
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                
                results = []
                for photo_data in data.get("results", []):
                    result = self._convert_unsplash_photo(photo_data, query.keywords)
                    if result:
                        results.append(result)
                
                search_time = time.time() - start_time
                
                return MaterialSearchResponse(
                    query=query,
                    results=results,
                    total_count=data.get("total", len(results)),
                    search_time=search_time,
                    provider="unsplash",
                    provider_info={
                        "total_pages": data.get("total_pages", 1)
                    }
                )
        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Unsplash API error {e.response.status_code}: {e.response.text}")
            error_msg = f"HTTP {e.response.status_code}"
            if e.response.status_code == 429:
                error_msg = "Rate limit exceeded"
            elif e.response.status_code == 401:
                error_msg = "Invalid access key"
        except Exception as e:
            self.logger.error(f"Unsplash search failed: {e}")
            error_msg = str(e)
        
        return MaterialSearchResponse(
            query=query,
            results=[],
            total_count=0,
            search_time=time.time() - start_time,
            provider="unsplash",
            provider_info={"error": error_msg}
        )
    
    def _convert_unsplash_photo(self, photo_data: Dict[str, Any], search_keywords: List[str]) -> Optional[MaterialSearchResult]:
        """Convert Unsplash photo data to MaterialSearchResult"""
        try:
            photo_id = photo_data.get("id")
            
            # Get image URLs
            urls = photo_data.get("urls", {})
            full_url = urls.get("full") or urls.get("regular")
            thumbnail_url = urls.get("thumb") or urls.get("small")
            
            if not full_url:
                return None
            
            # Get image dimensions
            width = photo_data.get("width", 0)
            height = photo_data.get("height", 0)
            
            # Get user info
            user = photo_data.get("user", {})
            author = user.get("name", "Unknown")
            user_profile = user.get("links", {}).get("html", "")
            
            # Parse description/alt_description
            description = photo_data.get("description") or photo_data.get("alt_description", "")
            
            # Generate tags from description and search keywords
            tags = search_keywords.copy()
            if description:
                # Simple tag extraction from description
                words = description.lower().split()
                tags.extend([word.strip('.,!?') for word in words if len(word) > 3])
            tags = list(set(tags[:10]))  # Remove duplicates and limit
            
            metadata = MaterialMetadata(
                title=description[:100] if description else f"Unsplash Photo {photo_id}",
                description=description[:500],
                tags=tags,
                width=width,
                height=height,
                format=MaterialFormat.JPG,
                resolution=f"{width}x{height}",
                license=MaterialLicense.FREE,
                license_url="https://unsplash.com/license",
                attribution=f"Photo by {author} on Unsplash",
                source="Unsplash",
                author=author,
                created_at=photo_data.get("created_at")
            )
            
            # Calculate relevance and quality scores
            likes = photo_data.get("likes", 0)
            downloads = photo_data.get("downloads", 0)
            
            relevance_score = min(1.0, len(tags) * 0.1 + 0.3)
            quality_score = min(1.0, (width * height) / (1920 * 1080) * 0.6 + 0.4)
            popularity_score = min(1.0, (likes / 1000) * 0.5 + (downloads / 10000) * 0.3 + 0.2)
            
            return MaterialSearchResult(
                material_id=f"unsplash_{photo_id}",
                material_type=MaterialType.IMAGE,
                url=full_url,
                thumbnail_url=thumbnail_url,
                metadata=metadata,
                relevance_score=relevance_score,
                matching_tags=search_keywords,
                quality_score=quality_score,
                popularity_score=popularity_score
            )
        
        except Exception as e:
            self.logger.error(f"Failed to convert Unsplash photo data: {e}")
            return None
    
    async def get_material_info(self, material_id: str) -> Optional[MaterialSearchResult]:
        """Get detailed info about an Unsplash photo"""
        if not material_id.startswith("unsplash_"):
            return None
        
        photo_id = material_id.replace("unsplash_", "")
        
        if not self.is_available():
            return None
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/photos/{photo_id}",
                    params={"client_id": self.access_key}
                )
                response.raise_for_status()
                photo_data = response.json()
                
                return self._convert_unsplash_photo(photo_data, ["image"])
        
        except Exception as e:
            self.logger.error(f"Failed to get Unsplash photo info: {e}")
            return None
    
    async def download_material(self, material: MaterialSearchResult, local_path: str) -> bool:
        """Download image from Unsplash"""
        try:
            # For Unsplash, we should trigger download tracking
            photo_id = material.material_id.replace("unsplash_", "")
            
            # Trigger download tracking (best practice)
            if self.is_available():
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        await client.get(
                            f"{self.base_url}/photos/{photo_id}/download",
                            params={"client_id": self.access_key}
                        )
                except:
                    pass  # Don't fail download if tracking fails
            
            # Download the actual image
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get(material.url)
                response.raise_for_status()
                
                with open(local_path, "wb") as f:
                    async for chunk in response.aiter_bytes(8192):
                        f.write(chunk)
                
                return True
        
        except Exception as e:
            self.logger.error(f"Failed to download image: {e}")
            return False
    
    async def get_download_url(self, material: MaterialSearchResult) -> str:
        """Get direct download URL"""
        return material.url


class PexelsImageProvider(MaterialProvider):
    """
    Pexels API image provider
    
    Requires PEXELS_API_KEY environment variable or in config
    Free API: 200 requests per hour
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("pexels_images", config)
        self.api_key = self.config.get("api_key") or self.config.get("PEXELS_API_KEY")
        self.base_url = "https://api.pexels.com/v1"
        
        if not self.api_key:
            self.logger.warning("Pexels API key not configured")
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_supported_types(self) -> List[MaterialType]:
        return [MaterialType.IMAGE]
    
    async def search(self, query: MaterialSearchQuery) -> MaterialSearchResponse:
        """Search Pexels for images"""
        import time
        start_time = time.time()
        
        if not self.is_available():
            return MaterialSearchResponse(
                query=query,
                results=[],
                total_count=0,
                search_time=0.0,
                provider="pexels_images",
                provider_info={"error": "API key not configured"}
            )
        
        search_term = " ".join(query.keywords)
        params = {
            "query": search_term,
            "per_page": min(query.limit, 80),  # Pexels max is 80
            "page": (query.offset // query.limit) + 1 if query.limit > 0 else 1
        }
        
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
                
                results = []
                for photo_data in data.get("photos", []):
                    result = self._convert_pexels_photo(photo_data, query.keywords)
                    if result:
                        results.append(result)
                
                search_time = time.time() - start_time
                
                return MaterialSearchResponse(
                    query=query,
                    results=results,
                    total_count=data.get("total_results", len(results)),
                    search_time=search_time,
                    provider="pexels_images",
                    provider_info={
                        "page": data.get("page", 1),
                        "per_page": data.get("per_page", query.limit),
                        "next_page": data.get("next_page")
                    }
                )
        
        except Exception as e:
            self.logger.error(f"Pexels image search failed: {e}")
            return MaterialSearchResponse(
                query=query,
                results=[],
                total_count=0,
                search_time=time.time() - start_time,
                provider="pexels_images",
                provider_info={"error": str(e)}
            )
    
    def _convert_pexels_photo(self, photo_data: Dict[str, Any], search_keywords: List[str]) -> Optional[MaterialSearchResult]:
        """Convert Pexels photo data to MaterialSearchResult"""
        try:
            photo_id = str(photo_data.get("id"))
            
            # Get image URLs
            src = photo_data.get("src", {})
            original_url = src.get("original")
            thumbnail_url = src.get("medium") or src.get("small")
            
            if not original_url:
                return None
            
            # Get dimensions
            width = photo_data.get("width", 0)
            height = photo_data.get("height", 0)
            
            # Get photographer info
            photographer = photo_data.get("photographer", "Unknown")
            photographer_url = photo_data.get("photographer_url", "")
            
            metadata = MaterialMetadata(
                title=f"Pexels Photo {photo_id}",
                description=f"Photo by {photographer} from Pexels",
                tags=search_keywords[:10],
                width=width,
                height=height,
                format=MaterialFormat.JPG,
                resolution=f"{width}x{height}",
                license=MaterialLicense.FREE,
                license_url="https://www.pexels.com/license/",
                attribution=f"Photo by {photographer} from Pexels",
                source="Pexels",
                author=photographer
            )
            
            # Calculate scores
            quality_score = min(1.0, (width * height) / (1920 * 1080) * 0.7 + 0.3)
            relevance_score = min(1.0, len(search_keywords) * 0.2 + 0.5)
            
            return MaterialSearchResult(
                material_id=f"pexels_img_{photo_id}",
                material_type=MaterialType.IMAGE,
                url=original_url,
                thumbnail_url=thumbnail_url,
                metadata=metadata,
                relevance_score=relevance_score,
                matching_tags=search_keywords,
                quality_score=quality_score,
                popularity_score=0.7  # Pexels photos are generally high quality
            )
        
        except Exception as e:
            self.logger.error(f"Failed to convert Pexels photo data: {e}")
            return None
    
    async def get_material_info(self, material_id: str) -> Optional[MaterialSearchResult]:
        """Get detailed info about a Pexels photo"""
        # Pexels doesn't have a single photo endpoint in the free API
        return None
    
    async def download_material(self, material: MaterialSearchResult, local_path: str) -> bool:
        """Download image from Pexels"""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get(material.url)
                response.raise_for_status()
                
                with open(local_path, "wb") as f:
                    async for chunk in response.aiter_bytes(8192):
                        f.write(chunk)
                
                return True
        
        except Exception as e:
            self.logger.error(f"Failed to download image: {e}")
            return False
    
    async def get_download_url(self, material: MaterialSearchResult) -> str:
        """Get direct download URL"""
        return material.url


class MockImageProvider(MaterialProvider):
    """Mock image provider for development"""
    
    def __init__(self):
        super().__init__("mock_image")
    
    async def search(self, query: MaterialSearchQuery) -> MaterialSearchResponse:
        """Return mock image results"""
        import time
        start_time = time.time()
        
        results = []
        image_styles = ["photo", "illustration", "abstract", "nature", "technology"]
        
        for i in range(min(query.limit, 4)):
            style = image_styles[i % len(image_styles)]
            material_id = f"mock_image_{style}_{i}"
            
            # Generate different sizes
            sizes = [(1920, 1080), (1280, 720), (800, 600), (640, 480)]
            width, height = sizes[i % len(sizes)]
            
            metadata = MaterialMetadata(
                title=f"Mock {style.title()}: {' '.join(query.keywords[:2])}",
                description=f"Mock {style} image for {', '.join(query.keywords)}",
                tags=query.keywords + [style],
                width=width,
                height=height,
                format=MaterialFormat.JPG,
                resolution=f"{width}x{height}",
                license=MaterialLicense.FREE,
                source="Mock Image Provider"
            )
            
            result = MaterialSearchResult(
                material_id=material_id,
                material_type=MaterialType.IMAGE,
                url=f"https://mock-images.com/{material_id}.jpg",
                thumbnail_url=f"https://mock-images.com/thumb/{material_id}.jpg",
                metadata=metadata,
                relevance_score=max(0.4, 1.0 - i * 0.1),
                matching_tags=query.keywords[:3],
                quality_score=min(1.0, (width * height) / (1920 * 1080) * 0.8 + 0.2),
                popularity_score=0.6
            )
            results.append(result)
        
        return MaterialSearchResponse(
            query=query,
            results=results,
            total_count=len(results),
            search_time=time.time() - start_time,
            provider="mock_image",
            provider_info={"mock": True}
        )
    
    async def get_material_info(self, material_id: str) -> Optional[MaterialSearchResult]:
        """Return mock image info"""
        if not material_id.startswith("mock_image_"):
            return None
        
        metadata = MaterialMetadata(
            title=f"Mock Image {material_id}",
            description="Mock image for testing",
            tags=["mock", "test", "image"],
            width=1920,
            height=1080,
            format=MaterialFormat.JPG,
            license=MaterialLicense.FREE
        )
        
        return MaterialSearchResult(
            material_id=material_id,
            material_type=MaterialType.IMAGE,
            url=f"https://mock-images.com/{material_id}.jpg",
            thumbnail_url=f"https://mock-images.com/thumb/{material_id}.jpg",
            metadata=metadata,
            relevance_score=0.8,
            quality_score=0.9,
            popularity_score=0.6
        )
    
    async def download_material(self, material: MaterialSearchResult, local_path: str) -> bool:
        """Mock download"""
        try:
            with open(local_path, 'w') as f:
                f.write(f"Mock image content for {material.material_id}")
            return True
        except Exception as e:
            self.logger.error(f"Mock image download failed: {e}")
            return False
    
    async def get_download_url(self, material: MaterialSearchResult) -> str:
        """Return the mock URL"""
        return material.url


# Alias for the primary image provider
ImageMaterialProvider = MockImageProvider  # Switch to UnsplashImageProvider when API key is available