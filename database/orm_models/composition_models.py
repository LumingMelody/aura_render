"""
Video Composition Models

Models for video compositions, layers, effects, and related metadata
supporting the video generation and editing pipeline.
"""

from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime, Text, Float,
    ForeignKey, Enum, JSON, Index, BigInteger
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import Dict, Any, List

from .base import EnhancedBaseModel

class CompositionStatus(PyEnum):
    """Video composition status"""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    READY = "ready"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

class LayerType(PyEnum):
    """Composition layer types"""
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"
    SHAPE = "shape"
    EFFECT = "effect"
    BACKGROUND = "background"

class EffectCategory(PyEnum):
    """Effect categories"""
    COLOR_GRADING = "color_grading"
    TRANSITIONS = "transitions"
    MOTION = "motion"
    FILTERS = "filters"
    TEXT_EFFECTS = "text_effects"
    AUDIO_EFFECTS = "audio_effects"
    AI_ENHANCED = "ai_enhanced"
    CINEMATIC = "cinematic"

class CompositionTemplate(PyEnum):
    """Composition templates"""
    STORYTELLING = "storytelling"
    PRODUCT_SHOWCASE = "product_showcase"
    SOCIAL_MEDIA = "social_media"
    EDUCATIONAL = "educational"
    CORPORATE = "corporate"
    MUSIC_VIDEO = "music_video"
    ADVERTISEMENT = "advertisement"

class VideoComposition(EnhancedBaseModel):
    """Main video composition model"""
    __tablename__ = "video_compositions"
    
    # Basic information
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Ownership
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Composition settings
    duration = Column(Float, nullable=False)  # in seconds
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    framerate = Column(Float, default=30.0, nullable=False)
    
    # Status and template
    status = Column(Enum(CompositionStatus), default=CompositionStatus.DRAFT, nullable=False)
    template = Column(Enum(CompositionTemplate), nullable=True)
    
    # AI generation data
    generation_prompt = Column(Text, nullable=True)
    ai_parameters = Column(JSON, default=dict, nullable=False)
    style_preferences = Column(JSON, default=dict, nullable=False)
    
    # Timeline settings
    timeline_scale = Column(Float, default=1.0, nullable=False)
    timeline_position = Column(Float, default=0.0, nullable=False)
    
    # Quality settings
    quality_preset = Column(String(50), default='standard', nullable=False)
    bitrate = Column(Integer, nullable=True)  # kbps
    
    # Export settings
    output_format = Column(String(10), default='mp4', nullable=False)
    output_codec = Column(String(20), default='h264', nullable=False)
    
    # Collaboration
    is_public = Column(Boolean, default=False, nullable=False)
    allow_collaboration = Column(Boolean, default=False, nullable=False)
    
    # Statistics
    view_count = Column(Integer, default=0, nullable=False)
    like_count = Column(Integer, default=0, nullable=False)
    share_count = Column(Integer, default=0, nullable=False)
    
    # File information
    thumbnail_url = Column(String(500), nullable=True)
    preview_url = Column(String(500), nullable=True)
    final_video_url = Column(String(500), nullable=True)
    file_size = Column(BigInteger, nullable=True)  # bytes
    
    # Relationships
    project = relationship("Project", back_populates="compositions")
    user = relationship("User")
    layers = relationship("CompositionLayer", back_populates="composition", cascade="all, delete-orphan")
    effects = relationship("CompositionEffect", back_populates="composition", cascade="all, delete-orphan")
    revisions = relationship("CompositionRevision", back_populates="composition", cascade="all, delete-orphan")
    metadata_entries = relationship("CompositionMetadata", back_populates="composition", cascade="all, delete-orphan")
    render_tasks = relationship("RenderTask", back_populates="composition")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_composition_user_status', 'user_id', 'status'),
        Index('idx_composition_project', 'project_id'),
        Index('idx_composition_template', 'template'),
        Index('idx_composition_created', 'created_at'),
    )
    
    @property
    def aspect_ratio(self) -> float:
        """Get composition aspect ratio"""
        return self.width / self.height if self.height > 0 else 16/9
    
    @property
    def resolution(self) -> str:
        """Get resolution string"""
        return f"{self.width}x{self.height}"
    
    @property
    def layer_count(self) -> int:
        """Get number of layers"""
        return len(self.layers)
    
    @property
    def effect_count(self) -> int:
        """Get number of effects"""
        return len(self.effects)
    
    def can_edit(self, user_id: str) -> bool:
        """Check if user can edit composition"""
        # Owner can always edit
        if str(self.user_id) == str(user_id):
            return True
        
        # Check project permissions
        if self.project and self.project.can_edit(user_id):
            return True
        
        return False

class CompositionLayer(EnhancedBaseModel):
    """Individual layer within a composition"""
    __tablename__ = "composition_layers"
    
    composition_id = Column(UUID(as_uuid=True), ForeignKey('video_compositions.id'), nullable=False)
    
    # Layer properties
    name = Column(String(255), nullable=False)
    layer_type = Column(Enum(LayerType), nullable=False)
    layer_index = Column(Integer, nullable=False)  # z-order
    
    # Content source
    source_path = Column(String(500), nullable=True)  # file path or URL
    source_type = Column(String(50), nullable=True)  # local, url, generated
    content = Column(Text, nullable=True)  # for text layers or generated content
    
    # Timeline properties
    start_time = Column(Float, nullable=False)  # seconds
    duration = Column(Float, nullable=True)  # seconds, null = auto
    end_time = Column(Float, nullable=True)  # calculated or explicit
    
    # Transform properties
    position_x = Column(Float, default=0.0, nullable=False)
    position_y = Column(Float, default=0.0, nullable=False)
    scale_x = Column(Float, default=1.0, nullable=False)
    scale_y = Column(Float, default=1.0, nullable=False)
    rotation = Column(Float, default=0.0, nullable=False)  # degrees
    
    # Visual properties
    opacity = Column(Float, default=1.0, nullable=False)
    blend_mode = Column(String(20), default='normal', nullable=False)
    
    # Audio properties (for audio/video layers)
    volume = Column(Float, default=1.0, nullable=False)
    muted = Column(Boolean, default=False, nullable=False)
    
    # Layer settings
    visible = Column(Boolean, default=True, nullable=False)
    locked = Column(Boolean, default=False, nullable=False)
    
    # Style properties (for text, shapes, etc.)
    style_properties = Column(JSON, default=dict, nullable=False)
    
    # AI generation metadata
    ai_generated = Column(Boolean, default=False, nullable=False)
    ai_prompt = Column(Text, nullable=True)
    ai_parameters = Column(JSON, default=dict, nullable=False)
    
    # Performance metadata
    estimated_render_time = Column(Float, nullable=True)  # seconds
    memory_usage = Column(Integer, nullable=True)  # MB
    
    # Relationships
    composition = relationship("VideoComposition", back_populates="layers")
    
    # Indexes
    __table_args__ = (
        Index('idx_layer_composition_index', 'composition_id', 'layer_index'),
        Index('idx_layer_type_time', 'layer_type', 'start_time'),
    )
    
    def __repr__(self):
        return f"<CompositionLayer(name={self.name}, type={self.layer_type}, index={self.layer_index})>"
    
    @property
    def actual_end_time(self) -> float:
        """Get actual end time (calculated or explicit)"""
        if self.end_time is not None:
            return self.end_time
        if self.duration is not None:
            return self.start_time + self.duration
        return self.start_time  # Fallback for instantaneous layers

class CompositionEffect(EnhancedBaseModel):
    """Effects applied to compositions or layers"""
    __tablename__ = "composition_effects"
    
    composition_id = Column(UUID(as_uuid=True), ForeignKey('video_compositions.id'), nullable=False)
    
    # Effect properties
    name = Column(String(255), nullable=False)
    effect_type = Column(String(100), nullable=False)
    category = Column(Enum(EffectCategory), nullable=False)
    
    # Target specification
    target_layer_ids = Column(ARRAY(String), nullable=True)  # PostgreSQL array, JSON for others
    target_all_layers = Column(Boolean, default=False, nullable=False)
    
    # Timeline properties
    start_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=True)
    end_time = Column(Float, nullable=True)
    
    # Effect parameters
    parameters = Column(JSON, default=dict, nullable=False)
    keyframes = Column(JSON, default=list, nullable=False)
    
    # Effect settings
    intensity = Column(Float, default=1.0, nullable=False)
    enabled = Column(Boolean, default=True, nullable=False)
    
    # AI enhancement
    ai_enhanced = Column(Boolean, default=False, nullable=False)
    ai_prompt = Column(Text, nullable=True)
    ai_confidence = Column(Float, nullable=True)
    
    # Processing metadata
    complexity_level = Column(String(20), default='medium', nullable=False)
    estimated_processing_time = Column(Float, nullable=True)
    gpu_accelerated = Column(Boolean, default=True, nullable=False)
    
    # Preset reference
    preset_id = Column(UUID(as_uuid=True), ForeignKey('effect_presets.id'), nullable=True)
    
    # Relationships
    composition = relationship("VideoComposition", back_populates="effects")
    preset = relationship("EffectPreset")
    
    # Indexes
    __table_args__ = (
        Index('idx_effect_composition_time', 'composition_id', 'start_time'),
        Index('idx_effect_category_type', 'category', 'effect_type'),
    )

class EffectPreset(EnhancedBaseModel):
    """Reusable effect presets"""
    __tablename__ = "effect_presets"
    
    # Preset information
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(Enum(EffectCategory), nullable=False)
    
    # Creator
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    
    # Preset data
    effect_type = Column(String(100), nullable=False)
    parameters = Column(JSON, nullable=False)
    thumbnail_url = Column(String(500), nullable=True)
    
    # Sharing settings
    is_public = Column(Boolean, default=False, nullable=False)
    is_system_preset = Column(Boolean, default=False, nullable=False)
    
    # Usage statistics
    usage_count = Column(Integer, default=0, nullable=False)
    rating = Column(Float, default=0.0, nullable=False)
    rating_count = Column(Integer, default=0, nullable=False)
    
    # Tags for discovery
    tags = Column(ARRAY(String), nullable=True)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_preset_category_public', 'category', 'is_public'),
        Index('idx_preset_user', 'user_id'),
    )

class CompositionRevision(EnhancedBaseModel):
    """Version history for compositions"""
    __tablename__ = "composition_revisions"
    
    composition_id = Column(UUID(as_uuid=True), ForeignKey('video_compositions.id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Revision information
    revision_number = Column(Integer, nullable=False)
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    
    # Snapshot data
    composition_snapshot = Column(JSON, nullable=False)  # Full composition state
    layers_snapshot = Column(JSON, nullable=False)  # All layers state
    effects_snapshot = Column(JSON, nullable=False)  # All effects state
    
    # Change information
    changes_summary = Column(Text, nullable=True)
    changed_fields = Column(ARRAY(String), nullable=True)
    
    # File references
    thumbnail_url = Column(String(500), nullable=True)
    preview_url = Column(String(500), nullable=True)
    
    # Relationships
    composition = relationship("VideoComposition", back_populates="revisions")
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_revision_composition_number', 'composition_id', 'revision_number'),
    )

class CompositionMetadata(EnhancedBaseModel):
    """Additional metadata for compositions"""
    __tablename__ = "composition_metadata"
    
    composition_id = Column(UUID(as_uuid=True), ForeignKey('video_compositions.id'), nullable=False)
    
    # Metadata key-value
    key = Column(String(100), nullable=False)
    value = Column(Text, nullable=True)
    value_type = Column(String(20), default='string', nullable=False)  # string, number, boolean, json
    
    # Metadata properties
    is_system = Column(Boolean, default=False, nullable=False)
    is_searchable = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    composition = relationship("VideoComposition", back_populates="metadata_entries")
    
    # Indexes
    __table_args__ = (
        Index('idx_metadata_composition_key', 'composition_id', 'key'),
        Index('idx_metadata_searchable', 'is_searchable', 'key'),
    )