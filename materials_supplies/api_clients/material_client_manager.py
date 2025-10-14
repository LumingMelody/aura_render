"""
素材客户端管理器 - 统一管理多个素材API客户端
"""
from typing import Dict, List, Any, Optional, Union
import asyncio
from dataclasses import dataclass

from .base_client import (
    MaterialSearchRequest,
    MaterialMetadata,
    SearchResponse,
    BaseMaterialClient
)
from .pexels_client import PexelsClient
from .pixabay_client import PixabayClient
from .freesound_client import FreesoundClient


@dataclass
class ClientConfig:
    """客户端配置"""
    client_type: str
    api_key: str
    enabled: bool = True
    priority: int = 1  # 1=highest priority
    rate_limit: float = 1.0  # seconds between requests


class MaterialClientManager:
    """素材客户端管理器"""

    def __init__(self):
        self.clients: Dict[str, BaseMaterialClient] = {}
        self.client_priorities: Dict[str, int] = {}
        self.client_configs: Dict[str, ClientConfig] = {}

    def register_client(self, name: str, config: ClientConfig):
        """注册素材客户端"""
        if not config.enabled:
            return

        try:
            if config.client_type.lower() == "pexels":
                client = PexelsClient(config.api_key)
            elif config.client_type.lower() == "pixabay":
                client = PixabayClient(config.api_key)
            elif config.client_type.lower() == "freesound":
                client = FreesoundClient(config.api_key)
            else:
                raise ValueError(f"Unsupported client type: {config.client_type}")

            self.clients[name] = client
            self.client_priorities[name] = config.priority
            self.client_configs[name] = config

            print(f"✅ Registered {config.client_type} client: {name}")

        except Exception as e:
            print(f"❌ Failed to register {name}: {e}")

    def get_clients_for_content_type(self, content_type: str) -> List[str]:
        """获取支持特定内容类型的客户端"""
        suitable_clients = []

        for name, client in self.clients.items():
            # Check if client supports the content type
            if content_type == "video":
                if isinstance(client, (PexelsClient, PixabayClient)):
                    suitable_clients.append(name)
            elif content_type == "image":
                if isinstance(client, (PexelsClient, PixabayClient)):
                    suitable_clients.append(name)
            elif content_type == "audio":
                if isinstance(client, (PixabayClient, FreesoundClient)):
                    suitable_clients.append(name)

        # Sort by priority
        suitable_clients.sort(key=lambda x: self.client_priorities.get(x, 999))
        return suitable_clients

    async def search_materials(
        self,
        request: MaterialSearchRequest,
        client_names: Optional[List[str]] = None,
        max_results_per_client: Optional[int] = None
    ) -> SearchResponse:
        """搜索素材，支持多客户端并发"""

        if client_names is None:
            client_names = self.get_clients_for_content_type(
                request.content_type or "image"
            )

        if not client_names:
            return SearchResponse(
                materials=[],
                total_count=0,
                page_size=0,
                current_page=1,
                has_more=False,
                search_time_ms=0
            )

        # Adjust request limit for multiple clients
        if max_results_per_client:
            original_limit = request.limit
            request.limit = min(request.limit, max_results_per_client)

        # Create search tasks for each client
        tasks = []
        for client_name in client_names:
            if client_name in self.clients:
                client = self.clients[client_name]
                task = self._search_with_client(client, client_name, request)
                tasks.append(task)

        # Execute searches concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_search_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

        # Combine results
        all_materials = []
        total_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Search failed for {client_names[i]}: {result}")
                continue

            if isinstance(result, SearchResponse):
                # Add source information to materials
                for material in result.materials:
                    material.description = f"[{client_names[i]}] {material.description}"

                all_materials.extend(result.materials)
                total_count += result.total_count

        # Sort by quality score
        all_materials.sort(key=lambda x: x.quality_score, reverse=True)

        # Restore original limit if it was adjusted
        if max_results_per_client:
            request.limit = original_limit

        # Limit results
        if request.limit:
            all_materials = all_materials[:request.limit]

        return SearchResponse(
            materials=all_materials,
            total_count=total_count,
            page_size=len(all_materials),
            current_page=1,
            has_more=len(all_materials) == request.limit,
            search_time_ms=total_search_time
        )

    async def _search_with_client(
        self,
        client: BaseMaterialClient,
        client_name: str,
        request: MaterialSearchRequest
    ) -> SearchResponse:
        """使用单个客户端搜索"""
        try:
            return await client.search_materials(request)
        except Exception as e:
            print(f"Client {client_name} search error: {e}")
            return SearchResponse(
                materials=[],
                total_count=0,
                page_size=0,
                current_page=1,
                has_more=False,
                search_time_ms=0
            )

    async def get_material_details(
        self,
        material_id: str,
        client_name: str
    ) -> Optional[MaterialMetadata]:
        """获取素材详情"""
        if client_name not in self.clients:
            return None

        try:
            client = self.clients[client_name]
            return await client.get_material_details(material_id)
        except Exception as e:
            print(f"Failed to get material details: {e}")
            return None

    async def download_material(
        self,
        material: MaterialMetadata,
        save_path: str,
        client_name: str
    ) -> bool:
        """下载素材"""
        if client_name not in self.clients:
            return False

        try:
            client = self.clients[client_name]
            return await client.download_material(material, save_path)
        except Exception as e:
            print(f"Failed to download material: {e}")
            return False

    async def search_by_category(
        self,
        category: str,
        content_type: str = "image",
        limit: int = 20
    ) -> List[MaterialMetadata]:
        """按类别搜索素材"""

        # Create category-specific search request
        category_queries = {
            # Video/Image categories
            "nature": "nature landscape outdoor forest mountain ocean",
            "business": "business office meeting corporate professional",
            "technology": "technology computer digital innovation tech",
            "people": "people person human lifestyle portrait",
            "travel": "travel vacation destination city landmark",
            "food": "food cooking restaurant meal culinary",
            "sports": "sports fitness exercise athletic activity",
            "abstract": "abstract pattern texture geometric design",

            # Audio categories
            "music": "music background instrumental ambient",
            "sfx": "sound effect noise impact transition",
            "voice": "voice speech narration announcement",
            "nature_audio": "nature ambient birds water wind rain"
        }

        query = category_queries.get(category.lower(), category)

        request = MaterialSearchRequest(
            query=query,
            content_type=content_type,
            limit=limit
        )

        response = await self.search_materials(request)
        return response.materials

    async def get_popular_materials(
        self,
        content_type: str = "image",
        limit: int = 20
    ) -> List[MaterialMetadata]:
        """获取热门素材"""

        client_names = self.get_clients_for_content_type(content_type)
        all_materials = []

        # Get popular content from each client
        for client_name in client_names[:2]:  # Limit to top 2 clients
            client = self.clients[client_name]

            try:
                if hasattr(client, 'get_popular_content'):
                    materials = await client.get_popular_content(content_type, limit // 2)
                elif hasattr(client, 'get_popular_videos') and content_type == "video":
                    materials = await client.get_popular_videos(limit // 2)
                elif hasattr(client, 'get_curated_photos') and content_type == "image":
                    materials = await client.get_curated_photos(limit // 2)
                else:
                    # Fallback: search with empty query to get popular results
                    request = MaterialSearchRequest(
                        query="",
                        content_type=content_type,
                        limit=limit // 2
                    )
                    response = await client.search_materials(request)
                    materials = response.materials

                # Add source info
                for material in materials:
                    material.description = f"[{client_name}] {material.description}"

                all_materials.extend(materials)

            except Exception as e:
                print(f"Failed to get popular content from {client_name}: {e}")

        # Sort by quality and limit
        all_materials.sort(key=lambda x: x.quality_score, reverse=True)
        return all_materials[:limit]

    def get_client_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        status = {}

        for name, client in self.clients.items():
            config = self.client_configs[name]
            status[name] = {
                "type": config.client_type,
                "enabled": config.enabled,
                "priority": config.priority,
                "rate_limit": config.rate_limit,
                "last_request_time": getattr(client, 'last_request_time', 0)
            }

        return status

    async def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        health_status = {}

        for name, client in self.clients.items():
            try:
                # Try a simple search to test the client
                test_request = MaterialSearchRequest(
                    query="test",
                    limit=1
                )
                result = await client.search_materials(test_request)
                health_status[name] = True
            except Exception as e:
                print(f"Health check failed for {name}: {e}")
                health_status[name] = False

        return health_status

    def get_usage_statistics(self) -> Dict[str, Any]:
        """获取使用统计"""
        # This could be enhanced to track actual usage
        return {
            "total_clients": len(self.clients),
            "active_clients": len([c for c in self.client_configs.values() if c.enabled]),
            "client_types": {
                config.client_type: sum(1 for c in self.client_configs.values()
                                       if c.client_type == config.client_type and c.enabled)
                for config in self.client_configs.values()
            }
        }


# 预设配置
def create_default_manager(
    pexels_api_key: Optional[str] = None,
    pixabay_api_key: Optional[str] = None,
    freesound_api_key: Optional[str] = None
) -> MaterialClientManager:
    """创建默认的素材客户端管理器"""

    manager = MaterialClientManager()

    # Register Pexels client
    if pexels_api_key:
        manager.register_client("pexels", ClientConfig(
            client_type="pexels",
            api_key=pexels_api_key,
            enabled=True,
            priority=1,  # Highest priority for video
            rate_limit=1.0
        ))

    # Register Pixabay client
    if pixabay_api_key:
        manager.register_client("pixabay", ClientConfig(
            client_type="pixabay",
            api_key=pixabay_api_key,
            enabled=True,
            priority=2,
            rate_limit=0.2
        ))

    # Register Freesound client
    if freesound_api_key:
        manager.register_client("freesound", ClientConfig(
            client_type="freesound",
            api_key=freesound_api_key,
            enabled=True,
            priority=1,  # Highest priority for audio
            rate_limit=1.0
        ))

    return manager