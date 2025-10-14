"""
Material Models

Models for managing media materials, sources, libraries, and usage tracking.
"""

from sqlalchemy import Column, String, Boolean, Integer, DateTime, Text, Float, ForeignKey, Enum, JSON, Index, BigInteger
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime, timezone
from enum import Enum as PyEnum
from .base import EnhancedBaseModel

class MaterialType(PyEnum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"

class MaterialStatus(PyEnum):
    AVAILABLE = "available"
    PROCESSING = "processing"
    UNAVAILABLE = "unavailable"

class LicenseType(PyEnum):
    FREE = "free"
    CREATIVE_COMMONS = "creative_commons"
    PAID = "paid"
    ROYALTY_FREE = "royalty_free"

class Material(EnhancedBaseModel):
    """Material assets model"""
    __tablename__ = "materials"
    
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    material_type = Column(Enum(MaterialType), nullable=False)
    
    # File information
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    file_format = Column(String(20), nullable=False)
    
    # Media properties
    duration = Column(Float, nullable=True)  # for video/audio
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    
    # Metadata
    tags = Column(ARRAY(String), nullable=True)
    license = Column(Enum(LicenseType), default=LicenseType.FREE, nullable=False)
    source = Column(String(255), nullable=True)
    author = Column(String(255), nullable=True)
    
    # Status and availability
    status = Column(Enum(MaterialStatus), default=MaterialStatus.AVAILABLE, nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)
    
    # Usage tracking
    download_count = Column(Integer, default=0, nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)
    
    __table_args__ = (Index('idx_material_type_status', 'material_type', 'status'),)

class MaterialSource(EnhancedBaseModel):
    """External material sources"""
    __tablename__ = "material_sources"
    
    name = Column(String(255), nullable=False)
    base_url = Column(String(500), nullable=False)
    api_key = Column(String(255), nullable=True)
    source_type = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

class MaterialTag(EnhancedBaseModel):
    """Material tagging system"""
    __tablename__ = "material_tags"
    
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)

class MaterialLibrary(EnhancedBaseModel):
    """User material libraries"""
    __tablename__ = "material_libraries"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    is_public = Column(Boolean, default=False, nullable=False)
    material_count = Column(Integer, default=0, nullable=False)

class MaterialUsage(EnhancedBaseModel):
    """Material usage tracking"""
    __tablename__ = "material_usage"
    
    material_id = Column(UUID(as_uuid=True), ForeignKey('materials.id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    composition_id = Column(UUID(as_uuid=True), ForeignKey('video_compositions.id'), nullable=True)
    usage_type = Column(String(50), nullable=False)  # download, preview, embed

class MaterialDownload(EnhancedBaseModel):
    """Material download history"""
    __tablename__ = "material_downloads"
    
    material_id = Column(UUID(as_uuid=True), ForeignKey('materials.id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    download_url = Column(String(500), nullable=True)
    file_size = Column(BigInteger, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)