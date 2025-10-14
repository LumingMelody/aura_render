"""
Database Models

Comprehensive data models for Aura Render platform including
video compositions, projects, users, rendering tasks, and analytics.
"""

from .base import BaseModel, TimestampMixin, SoftDeleteMixin, get_db_engine, get_db_session
from .user_models import User, UserProfile, UserSubscription, Team, TeamMember
from .project_models import Project, ProjectCollaborator, ProjectTemplate, ProjectVersion
from .composition_models import (
    VideoComposition, CompositionLayer, CompositionEffect, 
    EffectPreset, CompositionRevision, CompositionMetadata
)
from .material_models import (
    Material, MaterialSource, MaterialTag, MaterialLibrary,
    MaterialUsage, MaterialDownload
)
from .render_models import (
    RenderTask, RenderJob, RenderQueue, RenderNode, 
    RenderStatistics, RenderOutput
)
from .analytics_models import (
    UserActivity, SystemMetrics, UsageStatistics, 
    ErrorLog, PerformanceMetrics, AuditLog
)

__all__ = [
    # Base models
    'BaseModel', 'TimestampMixin', 'SoftDeleteMixin',
    'get_db_engine', 'get_db_session',
    
    # User models
    'User', 'UserProfile', 'UserSubscription', 'Team', 'TeamMember',
    
    # Project models  
    'Project', 'ProjectCollaborator', 'ProjectTemplate', 'ProjectVersion',
    
    # Composition models
    'VideoComposition', 'CompositionLayer', 'CompositionEffect',
    'EffectPreset', 'CompositionRevision', 'CompositionMetadata',
    
    # Material models
    'Material', 'MaterialSource', 'MaterialTag', 'MaterialLibrary',
    'MaterialUsage', 'MaterialDownload',
    
    # Render models
    'RenderTask', 'RenderJob', 'RenderQueue', 'RenderNode',
    'RenderStatistics', 'RenderOutput',
    
    # Analytics models
    'UserActivity', 'SystemMetrics', 'UsageStatistics',
    'ErrorLog', 'PerformanceMetrics', 'AuditLog'
]