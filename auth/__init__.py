"""
Authentication and User Management Package

Provides comprehensive user authentication, project management, and access control
for the Aura Render platform.
"""

from .user_management import (
    User,
    Project,
    UserSession,
    UserManager,
    UserRole,
    ProjectStatus,
    SubscriptionPlan,
    UserPermissions,
    get_user_manager
)

__all__ = [
    'User',
    'Project',
    'UserSession', 
    'UserManager',
    'UserRole',
    'ProjectStatus',
    'SubscriptionPlan',
    'UserPermissions',
    'get_user_manager'
]