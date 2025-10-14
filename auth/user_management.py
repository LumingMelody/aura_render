"""
User Authentication and Project Management System

Provides comprehensive user management, authentication, project organization,
and access control for the Aura Render platform.
"""

import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from pydantic import BaseModel, Field, EmailStr
import bcrypt


class UserRole(str, Enum):
    """User roles with different permissions"""
    ADMIN = "admin"
    PREMIUM = "premium"
    STANDARD = "standard"
    TRIAL = "trial"


class ProjectStatus(str, Enum):
    """Project status types"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    TEMPLATE = "template"


class SubscriptionPlan(str, Enum):
    """Available subscription plans"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class UserPermissions:
    """User permission configuration"""
    max_projects: int = 5
    max_videos_per_month: int = 10
    max_video_duration: int = 60  # seconds
    can_use_premium_templates: bool = False
    can_export_4k: bool = False
    can_use_batch_processing: bool = False
    can_use_api: bool = False
    storage_limit_mb: int = 1000
    
    @classmethod
    def get_permissions_for_plan(cls, plan: SubscriptionPlan) -> 'UserPermissions':
        """Get permission configuration for subscription plan"""
        if plan == SubscriptionPlan.FREE:
            return cls()
        elif plan == SubscriptionPlan.BASIC:
            return cls(
                max_projects=15,
                max_videos_per_month=50,
                max_video_duration=300,
                can_use_premium_templates=True,
                storage_limit_mb=5000
            )
        elif plan == SubscriptionPlan.PRO:
            return cls(
                max_projects=50,
                max_videos_per_month=200,
                max_video_duration=1800,
                can_use_premium_templates=True,
                can_export_4k=True,
                can_use_batch_processing=True,
                can_use_api=True,
                storage_limit_mb=20000
            )
        elif plan == SubscriptionPlan.ENTERPRISE:
            return cls(
                max_projects=999,
                max_videos_per_month=9999,
                max_video_duration=3600,
                can_use_premium_templates=True,
                can_export_4k=True,
                can_use_batch_processing=True,
                can_use_api=True,
                storage_limit_mb=100000
            )


class User(BaseModel):
    """User account model"""
    id: str
    email: EmailStr
    username: str
    password_hash: str
    role: UserRole = UserRole.STANDARD
    subscription_plan: SubscriptionPlan = SubscriptionPlan.FREE
    
    # Profile information
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    
    # Account status
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    
    # Usage tracking
    videos_created_this_month: int = 0
    storage_used_mb: float = 0.0
    projects_count: int = 0
    
    # Subscription information
    subscription_started: Optional[datetime] = None
    subscription_expires: Optional[datetime] = None
    
    # API access
    api_key: Optional[str] = None
    api_calls_this_month: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_permissions(self) -> UserPermissions:
        """Get user permissions based on subscription plan"""
        return UserPermissions.get_permissions_for_plan(self.subscription_plan)
    
    def can_create_video(self) -> bool:
        """Check if user can create a new video"""
        permissions = self.get_permissions()
        return self.videos_created_this_month < permissions.max_videos_per_month
    
    def can_create_project(self) -> bool:
        """Check if user can create a new project"""
        permissions = self.get_permissions()
        return self.projects_count < permissions.max_projects
    
    def has_storage_space(self, required_mb: float) -> bool:
        """Check if user has enough storage space"""
        permissions = self.get_permissions()
        return (self.storage_used_mb + required_mb) <= permissions.storage_limit_mb


class Project(BaseModel):
    """Project model for organizing videos and resources"""
    id: str
    name: str
    description: str = ""
    user_id: str
    
    # Project settings
    status: ProjectStatus = ProjectStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Content organization
    video_ids: List[str] = Field(default_factory=list)
    template_ids: List[str] = Field(default_factory=list)
    asset_ids: List[str] = Field(default_factory=list)
    
    # Project metadata
    tags: List[str] = Field(default_factory=list)
    thumbnail_url: Optional[str] = None
    color_scheme: Optional[str] = None
    
    # Collaboration (future feature)
    collaborators: List[str] = Field(default_factory=list)
    is_public: bool = False
    
    # Statistics
    total_videos: int = 0
    total_duration: float = 0.0
    storage_used_mb: float = 0.0
    
    # Settings
    default_template_id: Optional[str] = None
    default_settings: Dict[str, Any] = Field(default_factory=dict)


class UserSession(BaseModel):
    """User session model"""
    id: str
    user_id: str
    token: str
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: datetime
    is_active: bool = True
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    last_activity: datetime = Field(default_factory=datetime.now)


class UserManager:
    """Core user management system"""
    
    def __init__(self, storage_dir: str = "user_data", secret_key: str = None):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.users_dir = self.storage_dir / "users"
        self.projects_dir = self.storage_dir / "projects"
        self.sessions_dir = self.storage_dir / "sessions"
        
        for dir_path in [self.users_dir, self.projects_dir, self.sessions_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        
        # In-memory caches
        self.user_cache: Dict[str, User] = {}
        self.session_cache: Dict[str, UserSession] = {}
        self.project_cache: Dict[str, Project] = {}
    
    # User Management
    
    async def create_user(
        self,
        email: str,
        username: str,
        password: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None
    ) -> User:
        """Create a new user account"""
        # Check if user already exists
        existing_user = await self.get_user_by_email(email)
        if existing_user:
            raise ValueError("User with this email already exists")
        
        existing_username = await self.get_user_by_username(username)
        if existing_username:
            raise ValueError("Username already taken")
        
        # Generate user ID
        user_id = secrets.token_urlsafe(16)
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Create user
        user = User(
            id=user_id,
            email=email,
            username=username,
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            api_key=secrets.token_urlsafe(32)
        )
        
        await self._save_user(user)
        
        # Create default project
        await self.create_project(user_id, "My First Project", "Welcome to Aura Render!")
        
        return user
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        user = await self.get_user_by_email(email)
        if not user or not user.is_active:
            return None
        
        if bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            # Update last login
            user.last_login = datetime.now()
            await self._save_user(user)
            return user
        
        return None
    
    async def create_session(self, user: User, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> UserSession:
        """Create a new user session"""
        session_id = secrets.token_urlsafe(24)
        
        # Generate JWT token
        payload = {
            'user_id': user.id,
            'session_id': session_id,
            'exp': datetime.utcnow() + timedelta(days=7)
        }
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        session = UserSession(
            id=session_id,
            user_id=user.id,
            token=token,
            expires_at=datetime.now() + timedelta(days=7),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        await self._save_session(session)
        return session
    
    async def verify_session(self, token: str) -> Optional[User]:
        """Verify session token and return user"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            session_id = payload.get('session_id')
            user_id = payload.get('user_id')
            
            # Get session
            session = await self.get_session(session_id)
            if not session or not session.is_active or session.expires_at < datetime.now():
                return None
            
            # Update last activity
            session.last_activity = datetime.now()
            await self._save_session(session)
            
            return await self.get_user(user_id)
            
        except jwt.InvalidTokenError:
            return None
    
    async def logout_user(self, session_id: str) -> bool:
        """Logout user by deactivating session"""
        session = await self.get_session(session_id)
        if session:
            session.is_active = False
            await self._save_session(session)
            return True
        return False
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        if user_id in self.user_cache:
            return self.user_cache[user_id]
        
        user_file = self.users_dir / f"{user_id}.json"
        if user_file.exists():
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    user = User(**data)
                    self.user_cache[user_id] = user
                    return user
            except Exception:
                pass
        
        return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email address"""
        for user_file in self.users_dir.glob("*.json"):
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('email') == email:
                        user = User(**data)
                        self.user_cache[user.id] = user
                        return user
            except Exception:
                continue
        return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        for user_file in self.users_dir.glob("*.json"):
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('username') == username:
                        user = User(**data)
                        self.user_cache[user.id] = user
                        return user
            except Exception:
                continue
        return None
    
    async def update_user(self, user: User) -> User:
        """Update user information"""
        user.updated_at = datetime.now()
        await self._save_user(user)
        return user
    
    # Project Management
    
    async def create_project(self, user_id: str, name: str, description: str = "") -> Project:
        """Create a new project for a user"""
        user = await self.get_user(user_id)
        if not user or not user.can_create_project():
            raise ValueError("Cannot create project - limit reached or user not found")
        
        project_id = secrets.token_urlsafe(16)
        
        project = Project(
            id=project_id,
            name=name,
            description=description,
            user_id=user_id
        )
        
        await self._save_project(project)
        
        # Update user project count
        user.projects_count += 1
        await self._save_user(user)
        
        return project
    
    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        if project_id in self.project_cache:
            return self.project_cache[project_id]
        
        project_file = self.projects_dir / f"{project_id}.json"
        if project_file.exists():
            try:
                with open(project_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    project = Project(**data)
                    self.project_cache[project_id] = project
                    return project
            except Exception:
                pass
        
        return None
    
    async def get_user_projects(self, user_id: str, status: Optional[ProjectStatus] = None) -> List[Project]:
        """Get all projects for a user"""
        projects = []
        
        for project_file in self.projects_dir.glob("*.json"):
            try:
                with open(project_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('user_id') == user_id:
                        if status is None or data.get('status') == status.value:
                            project = Project(**data)
                            projects.append(project)
            except Exception:
                continue
        
        return sorted(projects, key=lambda p: p.updated_at, reverse=True)
    
    async def update_project(self, project: Project) -> Project:
        """Update project information"""
        project.updated_at = datetime.now()
        await self._save_project(project)
        return project
    
    async def delete_project(self, project_id: str, user_id: str) -> bool:
        """Delete a project (soft delete)"""
        project = await self.get_project(project_id)
        if not project or project.user_id != user_id:
            return False
        
        project.status = ProjectStatus.DELETED
        await self._save_project(project)
        
        # Update user project count
        user = await self.get_user(user_id)
        if user:
            user.projects_count = max(0, user.projects_count - 1)
            await self._save_user(user)
        
        return True
    
    # Session Management
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        if session_id in self.session_cache:
            return self.session_cache[session_id]
        
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    session = UserSession(**data)
                    self.session_cache[session_id] = session
                    return session
            except Exception:
                pass
        
        return None
    
    async def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        cleaned_count = 0
        
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    expires_at = datetime.fromisoformat(data.get('expires_at'))
                    if expires_at < current_time:
                        session_file.unlink()
                        cleaned_count += 1
                        
                        # Remove from cache
                        session_id = data.get('id')
                        if session_id in self.session_cache:
                            del self.session_cache[session_id]
                            
            except Exception:
                continue
        
        return cleaned_count
    
    # Storage operations
    
    async def _save_user(self, user: User):
        """Save user to disk"""
        user_file = self.users_dir / f"{user.id}.json"
        
        try:
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(user.dict(), f, indent=2, ensure_ascii=False, default=str)
            
            # Update cache
            self.user_cache[user.id] = user
            
        except Exception as e:
            raise ValueError(f"Failed to save user: {e}")
    
    async def _save_project(self, project: Project):
        """Save project to disk"""
        project_file = self.projects_dir / f"{project.id}.json"
        
        try:
            with open(project_file, 'w', encoding='utf-8') as f:
                json.dump(project.dict(), f, indent=2, ensure_ascii=False, default=str)
            
            # Update cache
            self.project_cache[project.id] = project
            
        except Exception as e:
            raise ValueError(f"Failed to save project: {e}")
    
    async def _save_session(self, session: UserSession):
        """Save session to disk"""
        session_file = self.sessions_dir / f"{session.id}.json"
        
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.dict(), f, indent=2, ensure_ascii=False, default=str)
            
            # Update cache
            self.session_cache[session.id] = session
            
        except Exception as e:
            raise ValueError(f"Failed to save session: {e}")


# Global instance
_user_manager = None

def get_user_manager() -> UserManager:
    """Get the global user manager instance"""
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
    return _user_manager