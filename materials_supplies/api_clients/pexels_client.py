"""
Pexels API客户端 - 视频和图片素材
"""
from typing import Dict, List, Any, Optional
import asyncio
from .base_client import (
    BaseMaterialClient,
    MaterialSearchRequest,
    MaterialMetadata,
    SearchResponse
)


class PexelsClient(BaseMaterialClient):
    """Pexels API客户端"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.pexels.com/v1",
            timeout=30
        )
        self.rate_limit_delay = 1.0  # Pexels: 1 request per second

    def _get_auth_headers(self) -> Dict[str, str]:
        return {"Authorization": self.api_key}

    async def search_materials(self, request: MaterialSearchRequest) -> SearchResponse:
        """搜索Pexels素材"""
        if request.content_type == "video":
            return await self._search_videos(request)
        elif request.content_type == "image":
            return await self._search_photos(request)
        else:
            # 默认搜索图片
            return await self._search_photos(request)

    async def _search_videos(self, request: MaterialSearchRequest) -> SearchResponse:
        """搜索视频素材"""
        params = {
            "query": request.query,
            "per_page": min(request.limit, 80),  # Pexels max 80
            "page": (request.offset // request.limit) + 1
        }

        if request.resolution:
            params["size"] = self._map_resolution(request.resolution)

        start_time = asyncio.get_event_loop().time()
        response = await self._make_request("GET", "/videos/search", params=params)
        search_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

        materials = []
        for video in response.get("videos", []):
            # Find the best quality video file
            best_file = self._get_best_video_file(video.get("video_files", []))

            material = MaterialMetadata(
                id=str(video["id"]),
                title=video.get("tags", "Pexels Video"),
                description=video.get("tags", ""),
                url=best_file["link"] if best_file else "",
                thumbnail_url=video.get("image", ""),
                duration=float(video.get("duration", 0)),
                resolution=f"{best_file.get('width', 0)}x{best_file.get('height', 0)}" if best_file else "",
                format="mp4",
                tags=video.get("tags", "").split(", ") if video.get("tags") else [],
                author=video.get("user", {}).get("name", "Pexels"),
                quality_score=self._calculate_quality_score(video, best_file)
            )
            materials.append(material)

        return SearchResponse(
            materials=materials,
            total_count=response.get("total_results", 0),
            page_size=response.get("per_page", request.limit),
            current_page=response.get("page", 1),
            has_more=response.get("next_page") is not None,
            search_time_ms=search_time
        )

    async def _search_photos(self, request: MaterialSearchRequest) -> SearchResponse:
        """搜索图片素材"""
        params = {
            "query": request.query,
            "per_page": min(request.limit, 80),
            "page": (request.offset // request.limit) + 1
        }

        if request.resolution:
            params["size"] = self._map_photo_size(request.resolution)

        start_time = asyncio.get_event_loop().time()
        response = await self._make_request("GET", "/search", params=params)
        search_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

        materials = []
        for photo in response.get("photos", []):
            # Get the original or large size
            src = photo.get("src", {})
            photo_url = src.get("original", src.get("large", ""))

            material = MaterialMetadata(
                id=str(photo["id"]),
                title=photo.get("alt", "Pexels Photo"),
                description=photo.get("alt", ""),
                url=photo_url,
                thumbnail_url=src.get("medium", src.get("small", "")),
                resolution=f"{photo.get('width', 0)}x{photo.get('height', 0)}",
                format="jpg",
                tags=[photo.get("alt", "")] if photo.get("alt") else [],
                author=photo.get("photographer", "Pexels"),
                quality_score=self._calculate_photo_quality_score(photo)
            )
            materials.append(material)

        return SearchResponse(
            materials=materials,
            total_count=response.get("total_results", 0),
            page_size=response.get("per_page", request.limit),
            current_page=response.get("page", 1),
            has_more=response.get("next_page") is not None,
            search_time_ms=search_time
        )

    async def get_material_details(self, material_id: str) -> MaterialMetadata:
        """获取素材详情"""
        # Pexels doesn't have a separate detail endpoint
        # Return basic info or make a search to find it
        search_request = MaterialSearchRequest(
            query="",  # Empty query to get popular content
            limit=1
        )
        # This is a simplified implementation
        # In practice, you might need to cache or use different approach
        return None

    def _get_best_video_file(self, video_files: List[Dict]) -> Optional[Dict]:
        """选择最佳质量的视频文件"""
        if not video_files:
            return None

        # Prefer HD quality
        for file in video_files:
            if file.get("quality") == "hd":
                return file

        # Fallback to first available
        return video_files[0]

    def _map_resolution(self, resolution: str) -> str:
        """映射分辨率到Pexels参数"""
        resolution_map = {
            "1920x1080": "large",
            "1280x720": "medium",
            "640x360": "small"
        }
        return resolution_map.get(resolution, "medium")

    def _map_photo_size(self, resolution: str) -> str:
        """映射图片分辨率"""
        if "1920" in resolution or "1080" in resolution:
            return "large"
        elif "1280" in resolution or "720" in resolution:
            return "medium"
        else:
            return "small"

    def _calculate_quality_score(self, video: Dict, video_file: Dict) -> float:
        """计算视频质量分数"""
        score = 0.5  # Base score

        # Quality bonus
        if video_file and video_file.get("quality") == "hd":
            score += 0.3

        # Duration bonus (prefer 3-30 seconds for most use cases)
        duration = video.get("duration", 0)
        if 3 <= duration <= 30:
            score += 0.2
        elif duration > 30:
            score += 0.1

        return min(score, 1.0)

    def _calculate_photo_quality_score(self, photo: Dict) -> float:
        """计算图片质量分数"""
        score = 0.5  # Base score

        # Resolution bonus
        width = photo.get("width", 0)
        height = photo.get("height", 0)

        if width >= 1920 and height >= 1080:
            score += 0.3
        elif width >= 1280 and height >= 720:
            score += 0.2

        # Alt text bonus (better metadata)
        if photo.get("alt"):
            score += 0.2

        return min(score, 1.0)

    async def get_popular_videos(self, limit: int = 15) -> List[MaterialMetadata]:
        """获取热门视频"""
        params = {"per_page": min(limit, 80)}

        response = await self._make_request("GET", "/videos/popular", params=params)

        materials = []
        for video in response.get("videos", []):
            best_file = self._get_best_video_file(video.get("video_files", []))

            material = MaterialMetadata(
                id=str(video["id"]),
                title=video.get("tags", "Popular Pexels Video"),
                description=video.get("tags", ""),
                url=best_file["link"] if best_file else "",
                thumbnail_url=video.get("image", ""),
                duration=float(video.get("duration", 0)),
                resolution=f"{best_file.get('width', 0)}x{best_file.get('height', 0)}" if best_file else "",
                format="mp4",
                tags=video.get("tags", "").split(", ") if video.get("tags") else [],
                author=video.get("user", {}).get("name", "Pexels"),
                quality_score=self._calculate_quality_score(video, best_file)
            )
            materials.append(material)

        return materials

    async def get_curated_photos(self, limit: int = 15) -> List[MaterialMetadata]:
        """获取精选图片"""
        params = {"per_page": min(limit, 80)}

        response = await self._make_request("GET", "/curated", params=params)

        materials = []
        for photo in response.get("photos", []):
            src = photo.get("src", {})
            photo_url = src.get("original", src.get("large", ""))

            material = MaterialMetadata(
                id=str(photo["id"]),
                title=photo.get("alt", "Curated Pexels Photo"),
                description=photo.get("alt", ""),
                url=photo_url,
                thumbnail_url=src.get("medium", src.get("small", "")),
                resolution=f"{photo.get('width', 0)}x{photo.get('height', 0)}",
                format="jpg",
                tags=[photo.get("alt", "")] if photo.get("alt") else [],
                author=photo.get("photographer", "Pexels"),
                quality_score=self._calculate_photo_quality_score(photo)
            )
            materials.append(material)

        return materials