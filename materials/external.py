"""
External Material Provider

Integration with external material API service.
"""

import httpx
from typing import List, Dict, Any, Optional

from .base import (
    MaterialProvider, MaterialSearchQuery, MaterialSearchResponse,
    MaterialSearchResult, MaterialMetadata, MaterialType, MaterialFormat,
    MaterialLicense
)


class ExternalMaterialProvider(MaterialProvider):
    """
    External material API provider
    
    Connects to external material service API
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("external", config)
        self.base_url = self.config.get("base_url", "https://api.materials-service.com/v1")
        self.api_key = self.config.get("api_key", "")
        self.timeout = self.config.get("timeout", 30)
        
        if not self.base_url:
            self.logger.warning("External material service base URL not configured")
    
    def is_available(self) -> bool:
        return bool(self.base_url)
    
    def get_supported_types(self) -> List[MaterialType]:
        return [MaterialType.VIDEO, MaterialType.AUDIO, MaterialType.IMAGE]
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        return {
            "requests_per_minute": 100,
            "requests_per_hour": 5000,
            "concurrent_requests": 5
        }
    
    async def search(self, query: MaterialSearchQuery) -> MaterialSearchResponse:
        """Search external material service"""
        import time
        start_time = time.time()
        
        if not self.is_available():
            return MaterialSearchResponse(
                query=query,
                results=[],
                total_count=0,
                search_time=0.0,
                provider="external",
                provider_info={"error": "Base URL not configured"}
            )
        
        # 构建请求参数
        params = {
            "tags": ",".join(query.keywords),  # 将keywords转换为tags参数
            "limit": query.limit,
            "offset": query.offset
        }
        
        # 添加材料类型过滤
        if query.material_type:
            params["type"] = query.material_type.value
        
        # 添加时长过滤
        if query.min_duration:
            params["min_duration"] = query.min_duration
        if query.max_duration:
            params["max_duration"] = query.max_duration
        
        # 添加质量过滤
        if query.min_quality > 0:
            params["min_quality"] = query.min_quality
        
        # 准备请求头
        headers = {"User-Agent": "Aura-Render/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 发送请求到外部API
                response = await client.get(
                    f"{self.base_url}/materials/search",
                    params=params,
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
                
                # 转换外部API响应格式
                results = []
                for item in data.get("results", []):
                    result = self._convert_external_item(item, query.keywords)
                    if result:
                        results.append(result)
                
                search_time = time.time() - start_time
                
                return MaterialSearchResponse(
                    query=query,
                    results=results,
                    total_count=data.get("total_count", len(results)),
                    search_time=search_time,
                    provider="external",
                    provider_info={
                        "api_version": data.get("version", "1.0"),
                        "service": "external-materials"
                    }
                )
        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"External API error {e.response.status_code}: {e.response.text}")
            error_msg = f"HTTP {e.response.status_code}"
            if e.response.status_code == 429:
                error_msg = "Rate limit exceeded"
            elif e.response.status_code == 401:
                error_msg = "Invalid API key"
        except httpx.TimeoutException:
            error_msg = "Request timeout"
            self.logger.error("External API request timeout")
        except Exception as e:
            self.logger.error(f"External API search failed: {e}")
            error_msg = str(e)
        
        return MaterialSearchResponse(
            query=query,
            results=[],
            total_count=0,
            search_time=time.time() - start_time,
            provider="external",
            provider_info={"error": error_msg}
        )
    
    def _convert_external_item(self, item: Dict[str, Any], search_keywords: List[str]) -> Optional[MaterialSearchResult]:
        """Convert external API item to MaterialSearchResult"""
        try:
            # 预期的外部API响应格式
            material_id = str(item.get("id", ""))
            if not material_id:
                return None
            
            # 获取材料类型
            item_type = item.get("type", "video").lower()
            if item_type == "video":
                material_type = MaterialType.VIDEO
            elif item_type == "audio":
                material_type = MaterialType.AUDIO  
            elif item_type == "image":
                material_type = MaterialType.IMAGE
            else:
                material_type = MaterialType.VIDEO  # 默认
            
            # 获取格式
            file_format = item.get("format", "")
            format_enum = None
            if file_format:
                try:
                    format_enum = MaterialFormat(file_format.lower())
                except ValueError:
                    pass
            
            # 创建元数据
            metadata = MaterialMetadata(
                title=item.get("title", f"Material {material_id}"),
                description=item.get("description", ""),
                tags=item.get("tags", search_keywords[:5]),
                duration=item.get("duration"),
                width=item.get("width"),
                height=item.get("height"),
                format=format_enum,
                file_size=item.get("file_size"),
                license=MaterialLicense.FREE,  # 可以根据item.get("license")调整
                source="External Service",
                author=item.get("author", "Unknown")
            )
            
            return MaterialSearchResult(
                material_id=f"ext_{material_id}",
                material_type=material_type,
                url=item.get("url", ""),
                thumbnail_url=item.get("thumbnail"),
                preview_url=item.get("preview"),
                metadata=metadata,
                relevance_score=float(item.get("relevance", 0.8)),
                quality_score=float(item.get("quality", 0.8)),
                popularity_score=float(item.get("popularity", 0.6))
            )
        
        except Exception as e:
            self.logger.error(f"Failed to convert external item: {e}")
            return None
    
    async def get_material_info(self, material_id: str) -> Optional[MaterialSearchResult]:
        """Get detailed info about a material from external service"""
        if not material_id.startswith("ext_"):
            return None
        
        # 提取实际ID
        actual_id = material_id.replace("ext_", "")
        
        if not self.is_available():
            return None
        
        headers = {"User-Agent": "Aura-Render/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/materials/{actual_id}",
                    headers=headers
                )
                response.raise_for_status()
                item = response.json()
                
                return self._convert_external_item(item, ["material"])
        
        except Exception as e:
            self.logger.error(f"Failed to get external material info: {e}")
            return None
    
    async def download_material(self, material: MaterialSearchResult, local_path: str) -> bool:
        """Download material from external service"""
        try:
            headers = {"User-Agent": "Aura-Render/1.0"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get(material.url, headers=headers)
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


# Mock external provider for testing (模拟外部服务的响应格式)
class MockExternalProvider(ExternalMaterialProvider):
    """Mock external provider for testing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.base_url = "https://mock-external-api.com/v1"  # 模拟URL
    
    async def search(self, query: MaterialSearchQuery) -> MaterialSearchResponse:
        """Return mock search results in external API format"""
        import time
        start_time = time.time()
        
        # 模拟外部API返回的数据格式
        mock_results = []
        for i in range(min(query.limit, 4)):
            mock_item = {
                "id": f"ext_mock_{i}",
                "type": query.material_type.value if query.material_type else "video",
                "title": f"External Material: {' '.join(query.keywords[:2])}",
                "description": f"Mock external material for {', '.join(query.keywords)}",
                "url": f"https://external-service.com/materials/ext_mock_{i}.mp4",
                "thumbnail": f"https://external-service.com/thumbs/ext_mock_{i}.jpg",
                "tags": query.keywords + ["external", "mock"],
                "duration": 30.0,
                "width": 1920,
                "height": 1080,
                "format": "mp4",
                "quality": 0.85,
                "relevance": max(0.6, 1.0 - i * 0.1),
                "popularity": 0.7,
                "author": "External Provider"
            }
            
            result = self._convert_external_item(mock_item, query.keywords)
            if result:
                mock_results.append(result)
        
        return MaterialSearchResponse(
            query=query,
            results=mock_results,
            total_count=len(mock_results),
            search_time=time.time() - start_time,
            provider="external",
            provider_info={
                "mock": True,
                "api_version": "1.0",
                "service": "mock-external-materials"
            }
        )