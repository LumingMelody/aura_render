"""
Project Models

Project management models for organizing video compositions,
collaboration, templates, and version control.
"""

from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime, Text, Float,
    ForeignKey, Enum, JSON, Index, BigInteger
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime, timezone
from enum import Enum as PyEnum

from .base import EnhancedBaseModel

class ProjectStatus(PyEnum):
    """Project status enumeration"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    DELETED = "deleted"

class ProjectVisibility(PyEnum):
    """Project visibility settings"""
    PRIVATE = "private"
    TEAM = "team"
    PUBLIC = "public"

class CollaboratorRole(PyEnum):
    """Collaborator role in projects"""
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"

class Project(EnhancedBaseModel):
    """Main project model for organizing compositions"""
    __tablename__ = "projects"
    
    # Basic information
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    slug = Column(String(100), nullable=True, index=True)
    
    # Ownership
    owner_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    team_id = Column(UUID(as_uuid=True), ForeignKey('teams.id'), nullable=True)
    
    # Project settings
    status = Column(Enum(ProjectStatus), default=ProjectStatus.ACTIVE, nullable=False)
    visibility = Column(Enum(ProjectVisibility), default=ProjectVisibility.PRIVATE, nullable=False)
    
    # Project metadata
    tags = Column(ARRAY(String), nullable=True)
    category = Column(String(50), nullable=True)
    thumbnail_url = Column(String(500), nullable=True)
    
    # Collaboration settings
    allow_public_view = Column(Boolean, default=False, nullable=False)
    allow_comments = Column(Boolean, default=True, nullable=False)
    allow_downloads = Column(Boolean, default=False, nullable=False)
    
    # Statistics
    view_count = Column(Integer, default=0, nullable=False)
    like_count = Column(Integer, default=0, nullable=False)
    star_count = Column(Integer, default=0, nullable=False)
    fork_count = Column(Integer, default=0, nullable=False)
    
    # File organization
    storage_path = Column(String(500), nullable=True)
    storage_used = Column(BigInteger, default=0, nullable=False)  # bytes
    file_count = Column(Integer, default=0, nullable=False)
    
    # Dates
    deadline = Column(DateTime(timezone=True), nullable=True)
    last_activity_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    team = relationship("Team", back_populates="projects")
    compositions = relationship("VideoComposition", back_populates="project", cascade="all, delete-orphan")
    collaborators = relationship("ProjectCollaborator", back_populates="project", cascade="all, delete-orphan")
    versions = relationship("ProjectVersion", back_populates="project", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_project_owner_status', 'owner_id', 'status'),
        Index('idx_project_team', 'team_id'),
        Index('idx_project_visibility', 'visibility'),
        Index('idx_project_category', 'category'),
    )
    
    @property
    def composition_count(self) -> int:
        """Get number of compositions in project"""
        return len(self.compositions)
    
    @property
    def collaborator_count(self) -> int:
        """Get number of collaborators"""
        return len(self.collaborators)
    
    @property
    def is_public(self) -> bool:
        """Check if project is publicly visible"""
        return self.visibility == ProjectVisibility.PUBLIC or self.allow_public_view
    
    def can_view(self, user_id: str) -> bool:
        """Check if user can view project"""
        # Owner can always view
        if str(self.owner_id) == str(user_id):
            return True
        
        # Public projects
        if self.is_public:
            return True
        
        # Check collaborator permissions
        for collaborator in self.collaborators:
            if str(collaborator.user_id) == str(user_id):
                return True
        
        # Team members (if project belongs to team)
        if self.team_id:
            for member in self.team.members:
                if str(member.user_id) == str(user_id):
                    return True
        
        return False
    
    def can_edit(self, user_id: str) -> bool:
        """Check if user can edit project"""
        # Owner can always edit
        if str(self.owner_id) == str(user_id):
            return True
        
        # Check collaborator permissions
        for collaborator in self.collaborators:
            if (str(collaborator.user_id) == str(user_id) and 
                collaborator.role in [CollaboratorRole.ADMIN, CollaboratorRole.EDITOR]):
                return True
        
        # Team admin permissions
        if self.team_id:
            for member in self.team.members:
                if (str(member.user_id) == str(user_id) and 
                    member.can_edit_projects):
                    return True
        
        return False
    
    def can_admin(self, user_id: str) -> bool:
        """Check if user can admin project"""
        # Owner can always admin
        if str(self.owner_id) == str(user_id):
            return True
        
        # Check admin collaborators
        for collaborator in self.collaborators:
            if (str(collaborator.user_id) == str(user_id) and 
                collaborator.role == CollaboratorRole.ADMIN):
                return True
        
        return False

class ProjectCollaborator(EnhancedBaseModel):
    """Project collaboration model"""
    __tablename__ = "project_collaborators"
    
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Collaborator role and permissions
    role = Column(Enum(CollaboratorRole), default=CollaboratorRole.VIEWER, nullable=False)
    
    # Specific permissions
    can_view = Column(Boolean, default=True, nullable=False)
    can_edit = Column(Boolean, default=False, nullable=False)
    can_comment = Column(Boolean, default=True, nullable=False)
    can_download = Column(Boolean, default=False, nullable=False)
    can_share = Column(Boolean, default=False, nullable=False)
    can_manage_collaborators = Column(Boolean, default=False, nullable=False)
    
    # Invitation details
    invited_by_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    invited_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    joined_at = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    status = Column(String(20), default='pending', nullable=False)  # pending, active, inactive
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="collaborators")
    user = relationship("User", foreign_keys=[user_id])
    invited_by = relationship("User", foreign_keys=[invited_by_id])
    
    # Indexes
    __table_args__ = (
        Index('idx_collaborator_project_user', 'project_id', 'user_id'),
        Index('idx_collaborator_user_status', 'user_id', 'status'),
    )

class ProjectTemplate(EnhancedBaseModel):
    """Project templates for quick setup"""
    __tablename__ = "project_templates"
    
    # Template information
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=False)
    
    # Creator
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    
    # Template data
    template_data = Column(JSON, nullable=False)  # Project structure and settings
    composition_templates = Column(JSON, default=list, nullable=False)  # Included compositions
    
    # Visual assets
    thumbnail_url = Column(String(500), nullable=True)
    preview_images = Column(ARRAY(String), nullable=True)
    
    # Sharing and usage
    is_public = Column(Boolean, default=False, nullable=False)
    is_system_template = Column(Boolean, default=False, nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)
    
    # Rating system
    rating = Column(Float, default=0.0, nullable=False)
    rating_count = Column(Integer, default=0, nullable=False)
    
    # Tags for discovery
    tags = Column(ARRAY(String), nullable=True)
    
    # Requirements
    required_features = Column(ARRAY(String), nullable=True)  # premium, pro, etc.
    min_subscription_level = Column(String(20), nullable=True)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_template_category_public', 'category', 'is_public'),
        Index('idx_template_user', 'user_id'),
    )

class ProjectVersion(EnhancedBaseModel):
    """Project version/release management"""
    __tablename__ = "project_versions"
    
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Version information
    version_number = Column(String(20), nullable=False)  # e.g., "1.0.0", "v2.1"
    version_name = Column(String(255), nullable=True)  # e.g., "Initial Release"
    description = Column(Text, nullable=True)
    
    # Version type
    is_major = Column(Boolean, default=False, nullable=False)
    is_published = Column(Boolean, default=False, nullable=False)
    is_draft = Column(Boolean, default=True, nullable=False)
    
    # Content snapshot
    project_snapshot = Column(JSON, nullable=False)
    compositions_count = Column(Integer, default=0, nullable=False)
    
    # Files and assets
    archive_path = Column(String(500), nullable=True)  # Backup/archive location
    archive_size = Column(BigInteger, nullable=True)  # bytes
    
    # Release notes
    changelog = Column(Text, nullable=True)
    breaking_changes = Column(Text, nullable=True)
    
    # Publishing details
    published_at = Column(DateTime(timezone=True), nullable=True)
    download_count = Column(Integer, default=0, nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="versions")
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_version_project_number', 'project_id', 'version_number'),
        Index('idx_version_published', 'is_published', 'published_at'),
    )