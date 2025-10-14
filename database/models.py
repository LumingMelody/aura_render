"""
Database Models

ORM models for the Aura Render video generation system.
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, Dict, Any
import json
import uuid

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Enum, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from .base import Base


class TaskStatus(PyEnum):
    """Task status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class NodeStatus(PyEnum):
    """Node execution status"""
    WAITING = "waiting"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class MaterialType(PyEnum):
    """Material type enumeration"""
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"
    FONT = "font"
    EFFECT = "effect"


class Project(Base):
    """Project model - groups related tasks"""
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    user_id = Column(String(100), index=True)  # External user ID
    settings = Column(JSON, default={})  # Project-specific settings
    
    # Relationships
    tasks = relationship("Task", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}')>"


class Task(Base):
    """Task model - represents a video generation task"""
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False)
    
    # Task input
    theme = Column(String(200), nullable=False)
    keywords = Column(JSON, nullable=False)  # List of keywords
    target_duration = Column(Integer, nullable=False)  # In seconds
    user_description = Column(Text, nullable=False)
    
    # Task configuration
    config = Column(JSON, default={})  # Additional configuration
    priority = Column(Integer, default=0)  # Task priority (higher = more important)
    
    # Task status
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    progress = Column(Float, default=0.0)  # Progress percentage (0-100)
    message = Column(Text)  # Current status message
    
    # Task results
    result = Column(JSON)  # Final processing results
    output_url = Column(String(500))  # Final video URL
    error_message = Column(Text)  # Error details if failed
    retry_count = Column(Integer, default=0)
    
    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    processing_time = Column(Float)  # Total processing time in seconds
    
    # Relationships
    project_id = Column(Integer, ForeignKey("projects.id"))
    project = relationship("Project", back_populates="tasks")
    node_executions = relationship("NodeExecution", back_populates="task", cascade="all, delete-orphan")
    materials = relationship("TaskMaterial", back_populates="task", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_task_status_created', 'status', 'created_at'),
        Index('idx_task_project_status', 'project_id', 'status'),
    )
    
    @validates('status')
    def validate_status(self, key, value):
        """Update timestamps based on status changes"""
        if value == TaskStatus.PROCESSING and not self.started_at:
            self.started_at = datetime.utcnow()
        elif value in [TaskStatus.COMPLETED, TaskStatus.FAILED] and not self.completed_at:
            self.completed_at = datetime.utcnow()
            if self.started_at:
                delta = self.completed_at - self.started_at
                self.processing_time = delta.total_seconds()
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "task_id": self.task_id,
            "theme": self.theme,
            "keywords": self.keywords,
            "target_duration": self.target_duration,
            "user_description": self.user_description,
            "status": self.status.value if self.status else None,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "output_url": self.output_url,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time": self.processing_time
        }
    
    def __repr__(self):
        return f"<Task(id={self.id}, task_id='{self.task_id}', status={self.status})>"


class NodeExecution(Base):
    """Node execution history"""
    __tablename__ = "node_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Node information
    node_name = Column(String(100), nullable=False)
    node_type = Column(String(100))
    node_version = Column(String(20))
    
    # Execution details
    status = Column(Enum(NodeStatus), default=NodeStatus.WAITING, nullable=False)
    input_data = Column(JSON)  # Node input
    output_data = Column(JSON)  # Node output
    error_message = Column(Text)
    
    # Timing
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    execution_time = Column(Float)  # In seconds
    
    # Relationships
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    task = relationship("Task", back_populates="node_executions")
    
    # Indexes
    __table_args__ = (
        Index('idx_node_task_status', 'task_id', 'status'),
        Index('idx_node_name_status', 'node_name', 'status'),
    )
    
    def __repr__(self):
        return f"<NodeExecution(id={self.id}, node='{self.node_name}', status={self.status})>"


class Material(Base):
    """Material/Asset library"""
    __tablename__ = "materials"
    
    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False)
    
    # Material information
    name = Column(String(200), nullable=False)
    type = Column(Enum(MaterialType), nullable=False, index=True)
    url = Column(String(500), nullable=False)
    thumbnail_url = Column(String(500))
    
    # Metadata
    description = Column(Text)
    tags = Column(JSON, default=[])  # List of tags
    meta_info = Column(JSON, default={})  # Additional metadata (duration, resolution, etc.)
    
    # Quality and usage
    quality_score = Column(Float, default=0.0)  # Quality rating (0-1)
    usage_count = Column(Integer, default=0)  # How many times used
    
    # Source information
    source = Column(String(100))  # Where the material came from
    license = Column(String(100))  # License type
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    task_materials = relationship("TaskMaterial", back_populates="material")
    
    # Indexes
    __table_args__ = (
        Index('idx_material_type_quality', 'type', 'quality_score'),
        Index('idx_material_tags', 'tags', postgresql_using='gin'),  # GIN index for JSON search (PostgreSQL)
    )
    
    def __repr__(self):
        return f"<Material(id={self.id}, name='{self.name}', type={self.type})>"


class TaskMaterial(Base):
    """Association between tasks and materials used"""
    __tablename__ = "task_materials"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Usage details
    node_name = Column(String(100))  # Which node used this material
    usage_type = Column(String(50))  # How it was used (main, background, overlay, etc.)
    start_time = Column(Float)  # Start time in the video
    end_time = Column(Float)  # End time in the video
    
    # Timestamps
    used_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    material_id = Column(Integer, ForeignKey("materials.id"), nullable=False)
    
    task = relationship("Task", back_populates="materials")
    material = relationship("Material", back_populates="task_materials")
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('task_id', 'material_id', 'node_name', name='_task_material_node_uc'),
        Index('idx_task_material', 'task_id', 'material_id'),
    )
    
    def __repr__(self):
        return f"<TaskMaterial(task_id={self.task_id}, material_id={self.material_id})>"


# Optional: Add more models as needed
class UserPreference(Base):
    """User preferences and settings"""
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), unique=True, nullable=False)
    
    # Preferences
    default_duration = Column(Integer, default=60)
    preferred_style = Column(String(50))
    language = Column(String(10), default="zh")
    
    # Settings
    settings = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<UserPreference(user_id='{self.user_id}')>"