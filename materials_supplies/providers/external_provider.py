"""
External Material Provider

Integrates with external/custom material service API.
"""

import httpx
from typing import List, Dict, Any, Optional
from .base_provider import BaseMaterialProvider, MaterialSearchResult, MaterialType
import logging


class ExternalMaterialProvider(BaseMaterialProvider):
    """External material service provider"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        super().__init__(name="external", api_key=api_key)
        self.base_url = base_url.rstrip('/')
        self.client = None
        
    async def _initialize(self):
        """Initialize HTTP client"""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
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
        """Search for materials in external service"""
        
        if not self.client:
            return []
            
        try:
            response = await self.client.post(
                "/search",
                json={
                    "query": query,
                    "type": material_type.value,
                    "limit": limit,
                    "offset": offset,
                    "filters": filters or {}
                }
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("results", []):
                results.append(MaterialSearchResult(
                    material_id=f"external_{item['id']}",
                    provider="external",
                    type=MaterialType(item.get("type", material_type.value)),
                    url=item["url"],
                    thumbnail_url=item.get("thumbnail"),
                    title=item.get("title", f"Material {item['id']}"),
                    description=item.get("description"),
                    tags=item.get("tags", []),
                    duration=item.get("duration"),
                    width=item.get("width"),
                    height=item.get("height"),
                    relevance_score=item.get("score", 1.0),
                    metadata=item.get("metadata", {})
                ))
                
            return results
            
        except Exception as e:
            self.logger.error(f"External search error: {e}")
            return []
            
    async def get_material(self, material_id: str) -> Optional[MaterialSearchResult]:
        """Get material by ID"""
        
        if not self.client or not material_id.startswith("external_"):
            return None
            
        external_id = material_id.replace("external_", "")
        
        try:
            response = await self.client.get(f"/materials/{external_id}")
            response.raise_for_status()
            
            item = response.json()
            
            return MaterialSearchResult(
                material_id=material_id,
                provider="external",
                type=MaterialType(item["type"]),
                url=item["url"],
                thumbnail_url=item.get("thumbnail"),
                title=item.get("title"),
                description=item.get("description"),
                tags=item.get("tags", []),
                duration=item.get("duration"),
                relevance_score=1.0
            )
            
        except Exception as e:
            self.logger.error(f"Get material error: {e}")
            return None
            
    async def download(self, material_id: str, destination: str) -> bool:
        """Download material"""
        
        material = await self.get_material(material_id)
        if not material:
            return False
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(material.url, follow_redirects=True)
                response.raise_for_status()
                
                with open(destination, "wb") as f:
                    f.write(response.content)
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Download error: {e}")
            return False
            
    def supports_type(self, material_type: MaterialType) -> bool:
        """Check if provider supports material type"""
        return True  # External provider supports all types