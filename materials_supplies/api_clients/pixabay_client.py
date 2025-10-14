"""
Pixabay API客户端 - 视频、图片、音乐素材
"""
from typing import Dict, List, Any, Optional
import asyncio
from .base_client import (
    BaseMaterialClient,
    MaterialSearchRequest,
    MaterialMetadata,
    SearchResponse
)


class PixabayClient(BaseMaterialClient):
    """Pixabay API客户端"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://pixabay.com/api",
            timeout=30
        )
        self.rate_limit_delay = 0.2  # 5 requests per second

    def _get_auth_headers(self) -> Dict[str, str]:
        # Pixabay uses API key as query parameter, not header
        return {}

    async def search_materials(self, request: MaterialSearchRequest) -> SearchResponse:
        """搜索Pixabay素材"""
        if request.content_type == "video":
            return await self._search_videos(request)
        elif request.content_type == "audio":
            return await self._search_music(request)
        elif request.content_type == "image":
            return await self._search_images(request)
        else:
            # 默认搜索图片
            return await self._search_images(request)

    async def _search_videos(self, request: MaterialSearchRequest) -> SearchResponse:
        """搜索视频素材"""
        params = {
            "key": self.api_key,
            "q": request.query,
            "video_type": "all",
            "per_page": min(request.limit, 200),  # Pixabay max 200
            "page": (request.offset // request.limit) + 1,
            "safesearch": "true"
        }

        if request.duration_range:
            params["min_duration"] = int(request.duration_range[0])
            params["max_duration"] = int(request.duration_range[1])

        if request.tags:
            # Combine query with tags
            params["q"] = f"{request.query} {' '.join(request.tags)}"

        start_time = asyncio.get_event_loop().time()
        response = await self._make_request("GET", "/videos/", params=params)
        search_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

        materials = []
        for video in response.get("hits", []):
            # Get the best quality video
            videos = video.get("videos", {})
            best_video = self._get_best_pixabay_video(videos)

            material = MaterialMetadata(
                id=str(video["id"]),
                title=video.get("tags", f"Pixabay Video {video['id']}"),
                description=video.get("tags", ""),
                url=best_video["url"] if best_video else "",
                thumbnail_url=video.get("webformatURL", ""),
                duration=float(video.get("duration", 0)),
                resolution=f"{best_video.get('width', 0)}x{best_video.get('height', 0)}" if best_video else "",
                format="mp4",
                size_bytes=best_video.get("size", 0) if best_video else 0,
                tags=video.get("tags", "").split(", ") if video.get("tags") else [],
                author=video.get("user", "Pixabay"),
                download_count=video.get("downloads", 0),
                quality_score=self._calculate_video_quality_score(video, best_video)
            )
            materials.append(material)

        return SearchResponse(
            materials=materials,
            total_count=response.get("total", 0),
            page_size=len(materials),
            current_page=params["page"],
            has_more=len(materials) == params["per_page"],
            search_time_ms=search_time
        )

    async def _search_images(self, request: MaterialSearchRequest) -> SearchResponse:
        """搜索图片素材"""
        params = {
            "key": self.api_key,
            "q": request.query,
            "image_type": "photo",
            "per_page": min(request.limit, 200),
            "page": (request.offset // request.limit) + 1,
            "safesearch": "true"
        }

        if request.resolution:
            params["min_width"], params["min_height"] = self._parse_resolution(request.resolution)

        if request.tags:
            params["q"] = f"{request.query} {' '.join(request.tags)}"

        start_time = asyncio.get_event_loop().time()
        response = await self._make_request("GET", "/", params=params)
        search_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

        materials = []
        for image in response.get("hits", []):
            material = MaterialMetadata(
                id=str(image["id"]),
                title=image.get("tags", f"Pixabay Image {image['id']}"),
                description=image.get("tags", ""),
                url=image.get("fullHDURL", image.get("webformatURL", "")),
                thumbnail_url=image.get("webformatURL", ""),
                resolution=f"{image.get('imageWidth', 0)}x{image.get('imageHeight', 0)}",
                format="jpg",
                size_bytes=image.get("imageSize", 0),
                tags=image.get("tags", "").split(", ") if image.get("tags") else [],
                author=image.get("user", "Pixabay"),
                download_count=image.get("downloads", 0),
                quality_score=self._calculate_image_quality_score(image)
            )
            materials.append(material)

        return SearchResponse(
            materials=materials,
            total_count=response.get("total", 0),
            page_size=len(materials),
            current_page=params["page"],
            has_more=len(materials) == params["per_page"],
            search_time_ms=search_time
        )

    async def _search_music(self, request: MaterialSearchRequest) -> SearchResponse:
        """搜索音乐素材"""
        # Note: Pixabay music API might have different endpoint
        params = {
            "key": self.api_key,
            "q": request.query,
            "per_page": min(request.limit, 200),
            "page": (request.offset // request.limit) + 1,
            "safesearch": "true"
        }

        if request.duration_range:
            params["min_duration"] = int(request.duration_range[0])
            params["max_duration"] = int(request.duration_range[1])

        start_time = asyncio.get_event_loop().time()

        try:
            # Try music endpoint (may not exist in all Pixabay plans)
            response = await self._make_request("GET", "/music/", params=params)
        except Exception:
            # Fallback: return empty response
            return SearchResponse(
                materials=[],
                total_count=0,
                page_size=0,
                current_page=1,
                has_more=False,
                search_time_ms=0
            )

        search_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

        materials = []
        for music in response.get("hits", []):
            material = MaterialMetadata(
                id=str(music["id"]),
                title=music.get("tags", f"Pixabay Music {music['id']}"),
                description=music.get("tags", ""),
                url=music.get("download_url", ""),
                thumbnail_url="",  # Music doesn't have thumbnails
                duration=float(music.get("duration", 0)),
                format="mp3",
                tags=music.get("tags", "").split(", ") if music.get("tags") else [],
                author=music.get("user", "Pixabay"),
                download_count=music.get("downloads", 0),
                quality_score=0.7  # Default score for music
            )
            materials.append(material)

        return SearchResponse(
            materials=materials,
            total_count=response.get("total", 0),
            page_size=len(materials),
            current_page=params["page"],
            has_more=len(materials) == params["per_page"],
            search_time_ms=search_time
        )

    async def get_material_details(self, material_id: str) -> MaterialMetadata:
        """获取素材详情 - Pixabay没有单独的详情接口"""
        return None

    def _get_best_pixabay_video(self, videos: Dict) -> Optional[Dict]:
        """选择最佳质量的Pixabay视频"""
        # Priority order for video quality
        quality_order = ["large", "medium", "small", "tiny"]

        for quality in quality_order:
            if quality in videos:
                return videos[quality]

        return None

    def _parse_resolution(self, resolution: str) -> tuple:
        """解析分辨率字符串"""
        try:
            width, height = resolution.split('x')
            return int(width), int(height)
        except:
            return 1920, 1080  # Default

    def _calculate_video_quality_score(self, video: Dict, video_file: Optional[Dict]) -> float:
        """计算视频质量分数"""
        score = 0.5

        # Download count bonus
        downloads = video.get("downloads", 0)
        if downloads > 1000:
            score += 0.2
        elif downloads > 100:
            score += 0.1

        # Duration preference
        duration = video.get("duration", 0)
        if 3 <= duration <= 30:
            score += 0.2

        # File size indicates quality
        if video_file and video_file.get("size", 0) > 1024 * 1024:  # > 1MB
            score += 0.1

        return min(score, 1.0)

    def _calculate_image_quality_score(self, image: Dict) -> float:
        """计算图片质量分数"""
        score = 0.5

        # Resolution bonus
        width = image.get("imageWidth", 0)
        height = image.get("imageHeight", 0)

        if width >= 1920 and height >= 1080:
            score += 0.3
        elif width >= 1280 and height >= 720:
            score += 0.2

        # Download count bonus
        downloads = image.get("downloads", 0)
        if downloads > 10000:
            score += 0.2
        elif downloads > 1000:
            score += 0.1

        return min(score, 1.0)

    async def get_popular_content(self, content_type: str = "image", limit: int = 20) -> List[MaterialMetadata]:
        """获取热门内容"""
        request = MaterialSearchRequest(
            query="",  # Empty query for popular content
            content_type=content_type,
            limit=limit
        )

        if content_type == "video":
            params = {
                "key": self.api_key,
                "per_page": min(limit, 200),
                "order": "popular",
                "safesearch": "true"
            }
            response = await self._make_request("GET", "/videos/", params=params)
            return self._parse_video_response(response)
        else:
            params = {
                "key": self.api_key,
                "per_page": min(limit, 200),
                "order": "popular",
                "safesearch": "true"
            }
            response = await self._make_request("GET", "/", params=params)
            return self._parse_image_response(response)

    def _parse_video_response(self, response: Dict) -> List[MaterialMetadata]:
        """解析视频响应"""
        materials = []
        for video in response.get("hits", []):
            videos = video.get("videos", {})
            best_video = self._get_best_pixabay_video(videos)

            material = MaterialMetadata(
                id=str(video["id"]),
                title=video.get("tags", f"Popular Video {video['id']}"),
                description=video.get("tags", ""),
                url=best_video["url"] if best_video else "",
                thumbnail_url=video.get("webformatURL", ""),
                duration=float(video.get("duration", 0)),
                resolution=f"{best_video.get('width', 0)}x{best_video.get('height', 0)}" if best_video else "",
                format="mp4",
                tags=video.get("tags", "").split(", ") if video.get("tags") else [],
                author=video.get("user", "Pixabay"),
                quality_score=self._calculate_video_quality_score(video, best_video)
            )
            materials.append(material)

        return materials

    def _parse_image_response(self, response: Dict) -> List[MaterialMetadata]:
        """解析图片响应"""
        materials = []
        for image in response.get("hits", []):
            material = MaterialMetadata(
                id=str(image["id"]),
                title=image.get("tags", f"Popular Image {image['id']}"),
                description=image.get("tags", ""),
                url=image.get("fullHDURL", image.get("webformatURL", "")),
                thumbnail_url=image.get("webformatURL", ""),
                resolution=f"{image.get('imageWidth', 0)}x{image.get('imageHeight', 0)}",
                format="jpg",
                tags=image.get("tags", "").split(", ") if image.get("tags") else [],
                author=image.get("user", "Pixabay"),
                quality_score=self._calculate_image_quality_score(image)
            )
            materials.append(material)

        return materials