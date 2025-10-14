"""
Render Models

Models for managing render tasks, jobs, queues, and rendering infrastructure.
"""

from sqlalchemy import Column, String, Boolean, Integer, DateTime, Text, Float, ForeignKey, Enum, JSON, Index, BigInteger
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
from enum import Enum as PyEnum
from .base import EnhancedBaseModel

class RenderStatus(PyEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    RENDERING = "rendering"
    POST_PROCESSING = "post_processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RenderPriority(PyEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class RenderTask(EnhancedBaseModel):
    """Individual render tasks"""
    __tablename__ = "render_tasks"
    
    composition_id = Column(UUID(as_uuid=True), ForeignKey('video_compositions.id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Task properties
    status = Column(Enum(RenderStatus), default=RenderStatus.QUEUED, nullable=False)
    priority = Column(Enum(RenderPriority), default=RenderPriority.NORMAL, nullable=False)
    
    # Configuration
    render_config = Column(JSON, nullable=False)
    output_path = Column(String(500), nullable=False)
    
    # Progress tracking
    progress = Column(Float, default=0.0, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Results
    error_message = Column(Text, nullable=True)
    result_data = Column(JSON, nullable=True)
    
    # Relationships
    composition = relationship("VideoComposition", back_populates="render_tasks")
    user = relationship("User")

class RenderJob(EnhancedBaseModel):
    """Batch render jobs"""
    __tablename__ = "render_jobs"
    
    name = Column(String(255), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    total_tasks = Column(Integer, nullable=False)
    completed_tasks = Column(Integer, default=0, nullable=False)

class RenderQueue(EnhancedBaseModel):
    """Render queue management"""
    __tablename__ = "render_queues"
    
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    max_concurrent_renders = Column(Integer, default=2, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

class RenderNode(EnhancedBaseModel):
    """Render node infrastructure"""
    __tablename__ = "render_nodes"
    
    name = Column(String(255), nullable=False)
    hostname = Column(String(255), nullable=False)
    ip_address = Column(String(45), nullable=False)
    port = Column(Integer, default=8080, nullable=False)
    
    # Node capabilities
    cpu_cores = Column(Integer, nullable=False)
    memory_gb = Column(Integer, nullable=False)
    gpu_count = Column(Integer, default=0, nullable=False)
    storage_gb = Column(Integer, nullable=False)
    
    # Status
    is_online = Column(Boolean, default=False, nullable=False)
    current_load = Column(Float, default=0.0, nullable=False)
    last_heartbeat = Column(DateTime(timezone=True), nullable=True)

class RenderStatistics(EnhancedBaseModel):
    """Render performance statistics"""
    __tablename__ = "render_statistics"
    
    render_task_id = Column(UUID(as_uuid=True), ForeignKey('render_tasks.id'), nullable=False)
    
    # Performance metrics
    render_time_seconds = Column(Float, nullable=False)
    queue_time_seconds = Column(Float, nullable=False)
    cpu_usage_avg = Column(Float, nullable=True)
    memory_usage_peak = Column(Integer, nullable=True)  # MB
    gpu_usage_avg = Column(Float, nullable=True)

class RenderOutput(EnhancedBaseModel):
    """Render output files"""
    __tablename__ = "render_outputs"
    
    render_task_id = Column(UUID(as_uuid=True), ForeignKey('render_tasks.id'), nullable=False)
    
    # File information
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    file_format = Column(String(20), nullable=False)
    
    # Video properties
    duration = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    framerate = Column(Float, nullable=True)
    bitrate = Column(Integer, nullable=True)