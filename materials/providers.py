from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from .material_types import MaterialType, MaterialMetadata, LicenseType
from dataclasses import dataclass

@dataclass
class MaterialSearchResult:
    material_id: str
    material_type: MaterialType
    url: str
    thumbnail_url: Optional[str]
    metadata: MaterialMetadata
    relevance_score: float = 0.8
    quality_score: float = 0.8
    popularity_score: float = 0.8

class MaterialProvider(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
    
    @abstractmethod
    async def search(self, keywords: List[str], material_type: MaterialType, **kwargs) -> List[MaterialSearchResult]:
        pass
    
    def is_available(self) -> bool:
        return True

class MockProvider(MaterialProvider):
    def __init__(self, name: str = "mock", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
    
    async def search(self, keywords: List[str], material_type: MaterialType, **kwargs) -> List[MaterialSearchResult]:
        results = []
        for i, keyword in enumerate(keywords[:3]):  # Limit to 3 results per keyword
            result = MaterialSearchResult(
                material_id=f"mock_{material_type.value}_{keyword}_{i}",
                material_type=material_type,
                url=f"https://example.com/{material_type.value}/{keyword}_{i}.{self._get_extension(material_type)}",
                thumbnail_url=f"https://example.com/thumb/{keyword}_{i}.jpg",
                metadata=MaterialMetadata(
                    title=f"{keyword.title()} {material_type.value.title()}",
                    description=f"Mock {material_type.value} about {keyword}",
                    tags=[keyword, material_type.value, "mock"],
                    duration=30.0 if material_type in [MaterialType.VIDEO, MaterialType.AUDIO] else None,
                    width=1920 if material_type in [MaterialType.VIDEO, MaterialType.IMAGE] else None,
                    height=1080 if material_type in [MaterialType.VIDEO, MaterialType.IMAGE] else None,
                    license=LicenseType.FREE,
                    source="Mock Provider",
                    author="Mock Author"
                ),
                relevance_score=0.9 - i * 0.1,
                quality_score=0.8,
                popularity_score=0.7
            )
            results.append(result)
        return results
    
    def _get_extension(self, material_type: MaterialType) -> str:
        if material_type == MaterialType.VIDEO:
            return "mp4"
        elif material_type == MaterialType.AUDIO:
            return "mp3"
        elif material_type == MaterialType.IMAGE:
            return "jpg"
        return "unknown"