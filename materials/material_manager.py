import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from .material_types import MaterialType
from .providers import MaterialProvider, MaterialSearchResult, MockProvider

@dataclass
class MaterialSearchQuery:
    keywords: List[str]
    material_type: Optional[MaterialType] = None
    max_duration: Optional[float] = None
    min_duration: Optional[float] = None
    preferred_aspect_ratio: Optional[str] = None
    min_quality: float = 0.0
    limit: int = 10
    offset: int = 0

@dataclass
class MaterialSearchResponse:
    results: List[MaterialSearchResult]
    total_count: int
    search_time: float
    provider_info: Dict[str, Any] = field(default_factory=dict)

class MaterialManager:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.providers: Dict[str, MaterialProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        # Initialize with mock provider for now
        self.providers["mock"] = MockProvider()
    
    async def search_and_aggregate(self, query: MaterialSearchQuery, 
                                 providers: Optional[List[str]] = None,
                                 max_concurrent: int = 3,
                                 sort_by: str = "relevance") -> MaterialSearchResponse:
        start_time = datetime.now()
        
        # Use all providers if none specified
        if providers is None:
            providers = list(self.providers.keys())
        
        # Filter available providers
        available_providers = [p for p in providers if p in self.providers]
        
        # Search all providers concurrently
        tasks = []
        for provider_name in available_providers:
            provider = self.providers[provider_name]
            task = provider.search(
                keywords=query.keywords,
                material_type=query.material_type or MaterialType.VIDEO,
                max_duration=query.max_duration,
                min_duration=query.min_duration
            )
            tasks.append(task)
        
        # Execute searches
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        all_results = []
        for results in results_lists:
            if isinstance(results, list):
                all_results.extend(results)
        
        # Filter by quality threshold
        filtered_results = [r for r in all_results if r.quality_score >= query.min_quality]
        
        # Sort results
        if sort_by == "relevance":
            filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
        elif sort_by == "quality":
            filtered_results.sort(key=lambda x: x.quality_score, reverse=True)
        elif sort_by == "combined":
            filtered_results.sort(key=lambda x: (x.relevance_score + x.quality_score) / 2, reverse=True)
        
        # Apply pagination
        paginated_results = filtered_results[query.offset:query.offset + query.limit]
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        return MaterialSearchResponse(
            results=paginated_results,
            total_count=len(filtered_results),
            search_time=search_time,
            provider_info={"providers": [{"name": p} for p in available_providers]}
        )
    
    async def smart_search(self, keywords: List[str], material_type: MaterialType,
                         context: Optional[Dict] = None, max_results: int = 10) -> List[MaterialSearchResult]:
        query = MaterialSearchQuery(
            keywords=keywords,
            material_type=material_type,
            limit=max_results
        )
        
        response = await self.search_and_aggregate(query)
        return response.results
    
    async def get_material_info(self, material_id: str) -> Optional[MaterialSearchResult]:
        # For mock implementation, create a fake result
        if material_id.startswith("mock_"):
            parts = material_id.split("_")
            if len(parts) >= 3:
                material_type = MaterialType(parts[1])
                keyword = parts[2]
                
                # Use mock provider to generate info
                mock_provider = self.providers.get("mock")
                if mock_provider:
                    results = await mock_provider.search([keyword], material_type)
                    return results[0] if results else None
        return None
    
    def get_provider_info(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "name": name,
                "is_available": provider.is_available(),
                "supported_types": ["video", "audio", "image"]
            }
            for name, provider in self.providers.items()
        }
    
    def get_available_providers(self, material_type: MaterialType) -> List[str]:
        return [name for name, provider in self.providers.items() if provider.is_available()]