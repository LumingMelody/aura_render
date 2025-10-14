"""
素材API客户端基类
"""
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import asyncio
import httpx
import time
from dataclasses import dataclass


@dataclass
class MaterialSearchRequest:
    """素材搜索请求"""
    query: str
    tags: List[str] = None
    duration_range: tuple = None  # (min, max) seconds
    resolution: str = None  # "1920x1080", "1280x720", etc.
    format: str = None  # "mp4", "jpg", "mp3", etc.
    style: str = None
    limit: int = 20
    offset: int = 0
    language: str = None
    content_type: str = None  # "video", "audio", "image"


@dataclass
class MaterialMetadata:
    """素材元数据"""
    id: str
    title: str
    description: str
    url: str
    thumbnail_url: str
    duration: float = 0.0
    resolution: str = ""
    format: str = ""
    size_bytes: int = 0
    tags: List[str] = None
    style: str = ""
    license_type: str = ""
    author: str = ""
    created_at: str = ""
    quality_score: float = 0.0
    download_count: int = 0


@dataclass
class SearchResponse:
    """搜索响应"""
    materials: List[MaterialMetadata]
    total_count: int
    page_size: int
    current_page: int
    has_more: bool
    search_time_ms: int


class BaseMaterialClient(ABC):
    """素材API客户端基类"""

    def __init__(self, api_key: str, base_url: str, timeout: int = 30):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        data: Dict = None,
        headers: Dict = None
    ) -> Dict:
        """发起API请求"""

        # Rate limiting
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)

        # Prepare headers
        request_headers = {
            "User-Agent": "AuraRender-MaterialClient/1.0",
            "Accept": "application/json"
        }
        if headers:
            request_headers.update(headers)

        # Add API key to headers
        request_headers.update(self._get_auth_headers())

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=request_headers
                )

                self.last_request_time = time.time()

                response.raise_for_status()
                return response.json()

            except httpx.RequestError as e:
                raise ConnectionError(f"Network error: {e}")
            except httpx.HTTPStatusError as e:
                raise ValueError(f"API error {e.response.status_code}: {e.response.text}")

    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """获取认证头"""
        pass

    @abstractmethod
    async def search_materials(self, request: MaterialSearchRequest) -> SearchResponse:
        """搜索素材"""
        pass

    @abstractmethod
    async def get_material_details(self, material_id: str) -> MaterialMetadata:
        """获取素材详情"""
        pass

    async def download_material(self, material: MaterialMetadata, save_path: str) -> bool:
        """下载素材文件"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(material.url, timeout=60)
                response.raise_for_status()

                with open(save_path, 'wb') as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

                return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def _build_search_params(self, request: MaterialSearchRequest) -> Dict:
        """构建搜索参数"""
        params = {
            "q": request.query,
            "limit": request.limit,
            "offset": request.offset
        }

        if request.tags:
            params["tags"] = ",".join(request.tags)

        if request.duration_range:
            params["min_duration"] = request.duration_range[0]
            params["max_duration"] = request.duration_range[1]

        if request.resolution:
            params["resolution"] = request.resolution

        if request.format:
            params["format"] = request.format

        if request.style:
            params["style"] = request.style

        if request.language:
            params["language"] = request.language

        if request.content_type:
            params["type"] = request.content_type

        return params


class MaterialClientError(Exception):
    """素材客户端异常"""
    pass


class RateLimitError(MaterialClientError):
    """请求限流异常"""
    pass


class QuotaExceededError(MaterialClientError):
    """配额超出异常"""
    pass