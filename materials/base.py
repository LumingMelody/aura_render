"""
Base Material Provider Interface

Defines the common interface for all material providers (video, audio, image).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MaterialType(Enum):
    """Material type enumeration"""
    VIDEO = "video"
    AUDIO = "audio" 
    IMAGE = "image"
    FONT = "font"


class MaterialFormat(Enum):
    """Material format enumeration"""
    # Video formats
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"
    
    # Audio formats  
    MP3 = "mp3"
    WAV = "wav"
    AAC = "aac"
    OGG = "ogg"
    
    # Image formats
    JPG = "jpg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"


class MaterialLicense(Enum):
    """Material license type"""
    FREE = "free"
    ROYALTY_FREE = "royalty_free"
    CREATIVE_COMMONS = "creative_commons"
    PREMIUM = "premium"
    CUSTOM = "custom"


@dataclass
class MaterialMetadata:
    """Material metadata information"""
    # Basic info
    title: str
    description: Optional[str] = None
    tags: List[str] = None
    
    # Technical info
    duration: Optional[float] = None  # seconds for video/audio
    width: Optional[int] = None       # pixels for video/image
    height: Optional[int] = None      # pixels for video/image
    format: Optional[MaterialFormat] = None
    file_size: Optional[int] = None   # bytes
    
    # Quality metrics
    resolution: Optional[str] = None  # "1920x1080", "4K", etc.
    bitrate: Optional[int] = None     # kbps
    fps: Optional[float] = None       # frames per second
    
    # Licensing
    license: MaterialLicense = MaterialLicense.FREE
    license_url: Optional[str] = None
    attribution: Optional[str] = None
    
    # Provenance
    source: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass 
class MaterialSearchResult:
    """Search result for a single material"""
    # Identity
    material_id: str
    material_type: MaterialType
    
    # URLs
    url: str                          # Direct download/stream URL
    thumbnail_url: Optional[str] = None
    preview_url: Optional[str] = None # Low quality preview
    
    # Metadata
    metadata: MaterialMetadata = None
    
    # Search relevance
    relevance_score: float = 0.0      # 0.0 - 1.0
    matching_tags: List[str] = None   # Which search tags matched
    
    # Quality assessment
    quality_score: float = 0.0        # 0.0 - 1.0, AI assessed quality
    popularity_score: float = 0.0     # 0.0 - 1.0, usage popularity
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = MaterialMetadata(title=f"Material {self.material_id}")
        if self.matching_tags is None:
            self.matching_tags = []


@dataclass
class MaterialSearchQuery:
    """Search query for materials"""
    # Search terms
    keywords: List[str]
    
    # Filters
    material_type: Optional[MaterialType] = None
    max_duration: Optional[float] = None
    min_duration: Optional[float] = None
    formats: Optional[List[MaterialFormat]] = None
    licenses: Optional[List[MaterialLicense]] = None
    min_quality: float = 0.0
    
    # Preferences  
    preferred_resolution: Optional[str] = None
    preferred_aspect_ratio: Optional[str] = None  # "16:9", "4:3", "1:1"
    
    # Result control
    limit: int = 20
    offset: int = 0
    sort_by: str = "relevance"  # "relevance", "quality", "popularity", "date"
    
    def __post_init__(self):
        # Normalize keywords
        self.keywords = [kw.strip().lower() for kw in self.keywords if kw.strip()]


@dataclass
class MaterialSearchResponse:
    """Response from a material search"""
    query: MaterialSearchQuery
    results: List[MaterialSearchResult]
    total_count: int
    search_time: float  # seconds
    provider: str       # Which provider returned these results
    
    # Provider specific info
    provider_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.provider_info is None:
            self.provider_info = {}


class MaterialProvider(ABC):
    """Abstract base class for material providers"""
    
    def __init__(self, provider_name: str, config: Dict[str, Any] = None):
        self.provider_name = provider_name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{provider_name}")
    
    @abstractmethod
    async def search(self, query: MaterialSearchQuery) -> MaterialSearchResponse:
        """Search for materials matching the query"""
        pass
    
    @abstractmethod
    async def get_material_info(self, material_id: str) -> Optional[MaterialSearchResult]:
        """Get detailed information about a specific material"""
        pass
    
    @abstractmethod 
    async def download_material(self, material: MaterialSearchResult, local_path: str) -> bool:
        """Download material to local path"""
        pass
    
    @abstractmethod
    async def get_download_url(self, material: MaterialSearchResult) -> str:
        """Get direct download URL for the material"""
        pass
    
    def is_available(self) -> bool:
        """Check if the provider is available/configured"""
        return True
    
    def get_supported_types(self) -> List[MaterialType]:
        """Get list of material types supported by this provider"""
        return [MaterialType.VIDEO, MaterialType.AUDIO, MaterialType.IMAGE]
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limiting information"""
        return {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "concurrent_requests": 5
        }


class MockMaterialProvider(MaterialProvider):
    """Mock provider for testing and development"""
    
    def __init__(self):
        super().__init__("mock")
        
    async def search(self, query: MaterialSearchQuery) -> MaterialSearchResponse:
        """Return mock search results"""
        import time
        start_time = time.time()
        
        # Generate mock results based on query
        results = []
        for i in range(min(query.limit, 5)):  # Return up to 5 mock results
            material_id = f"mock_{query.material_type.value if query.material_type else 'video'}_{i}"
            
            # Create mock metadata based on material type
            if query.material_type == MaterialType.VIDEO or query.material_type is None:
                metadata = MaterialMetadata(
                    title=f"Mock Video: {' '.join(query.keywords[:2])}",
                    description=f"Mock video content related to {', '.join(query.keywords)}",
                    tags=query.keywords[:3],
                    duration=30.0,
                    width=1920,
                    height=1080,
                    format=MaterialFormat.MP4,
                    resolution="1920x1080",
                    fps=30.0,
                    license=MaterialLicense.FREE,
                    source="Mock Provider"
                )
                
                result = MaterialSearchResult(
                    material_id=material_id,
                    material_type=MaterialType.VIDEO,
                    url=f"https://mock-provider.com/video/{material_id}.mp4",
                    thumbnail_url=f"https://mock-provider.com/thumbs/{material_id}.jpg",
                    metadata=metadata,
                    relevance_score=max(0.5, 1.0 - i * 0.1),
                    matching_tags=query.keywords[:2],
                    quality_score=0.8,
                    popularity_score=0.6
                )
                results.append(result)
        
        search_time = time.time() - start_time
        
        return MaterialSearchResponse(
            query=query,
            results=results,
            total_count=len(results),
            search_time=search_time,
            provider="mock",
            provider_info={"mock": True}
        )
    
    async def get_material_info(self, material_id: str) -> Optional[MaterialSearchResult]:
        """Return mock material info"""
        if not material_id.startswith("mock_"):
            return None
            
        metadata = MaterialMetadata(
            title=f"Mock Material {material_id}",
            description="Mock material for testing",
            tags=["mock", "test"],
            duration=30.0,
            width=1920,
            height=1080,
            format=MaterialFormat.MP4,
            license=MaterialLicense.FREE
        )
        
        return MaterialSearchResult(
            material_id=material_id,
            material_type=MaterialType.VIDEO,
            url=f"https://mock-provider.com/video/{material_id}.mp4",
            thumbnail_url=f"https://mock-provider.com/thumbs/{material_id}.jpg",
            metadata=metadata,
            relevance_score=0.8,
            quality_score=0.8,
            popularity_score=0.6
        )
    
    async def download_material(self, material: MaterialSearchResult, local_path: str) -> bool:
        """Mock download - just create an empty file"""
        try:
            with open(local_path, 'w') as f:
                f.write(f"Mock material content for {material.material_id}")
            return True
        except Exception as e:
            self.logger.error(f"Mock download failed: {e}")
            return False
    
    async def get_download_url(self, material: MaterialSearchResult) -> str:
        """Return the mock URL"""
        return material.url