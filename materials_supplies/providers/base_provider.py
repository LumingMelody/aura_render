"""
Base Material Provider

Abstract base class for all material providers.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging


class MaterialType(str, Enum):
    """Material type enumeration"""
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"
    ANIMATION = "animation"


class MaterialSearchResult(BaseModel):
    """Material search result model"""
    material_id: str = Field(..., description="Unique material ID")
    provider: str = Field(..., description="Provider name")
    type: MaterialType = Field(..., description="Material type")
    url: str = Field(..., description="Material URL")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    preview_url: Optional[str] = Field(None, description="Preview URL")
    title: str = Field(..., description="Material title")
    description: Optional[str] = Field(None, description="Material description")
    tags: List[str] = Field(default_factory=list, description="Material tags")
    duration: Optional[float] = Field(None, description="Duration in seconds (for video/audio)")
    width: Optional[int] = Field(None, description="Width in pixels (for video/image)")
    height: Optional[int] = Field(None, description="Height in pixels (for video/image)")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    format: Optional[str] = Field(None, description="File format")
    license: Optional[str] = Field(None, description="License type")
    author: Optional[str] = Field(None, description="Author/creator name")
    author_url: Optional[str] = Field(None, description="Author profile URL")
    source_url: Optional[str] = Field(None, description="Original source URL")
    created_at: Optional[datetime] = Field(None, description="Creation date")
    relevance_score: float = Field(1.0, description="Relevance score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseMaterialProvider(ABC):
    """Base class for material providers"""
    
    def __init__(self, name: str, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.api_key = api_key
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._initialized = False
        
    async def initialize(self):
        """Initialize the provider"""
        if not self._initialized:
            await self._initialize()
            self._initialized = True
            
    @abstractmethod
    async def _initialize(self):
        """Provider-specific initialization"""
        pass
        
    @abstractmethod
    async def search(
        self,
        query: str,
        material_type: MaterialType,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MaterialSearchResult]:
        """
        Search for materials
        
        Args:
            query: Search query
            material_type: Type of material to search
            limit: Maximum number of results
            offset: Result offset for pagination
            filters: Additional search filters
            
        Returns:
            List of search results
        """
        pass
        
    @abstractmethod
    async def get_material(self, material_id: str) -> Optional[MaterialSearchResult]:
        """
        Get material by ID
        
        Args:
            material_id: Material ID
            
        Returns:
            Material details or None if not found
        """
        pass
        
    @abstractmethod
    async def download(self, material_id: str, destination: str) -> bool:
        """
        Download material to local storage
        
        Args:
            material_id: Material ID
            destination: Local file path
            
        Returns:
            True if successful
        """
        pass
        
    @abstractmethod
    def supports_type(self, material_type: MaterialType) -> bool:
        """
        Check if provider supports material type
        
        Args:
            material_type: Material type to check
            
        Returns:
            True if supported
        """
        pass
        
    async def validate_api_key(self) -> bool:
        """
        Validate API key
        
        Returns:
            True if API key is valid
        """
        try:
            # Try a simple search to validate
            await self.search("test", MaterialType.IMAGE, limit=1)
            return True
        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            return False
            
    def is_available(self) -> bool:
        """
        Check if provider is available
        
        Returns:
            True if provider is available
        """
        return bool(self.api_key) or not self._requires_api_key()
        
    def _requires_api_key(self) -> bool:
        """
        Check if provider requires API key
        
        Returns:
            True if API key is required
        """
        return True
        
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', available={self.is_available()})"