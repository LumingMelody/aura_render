"""
Authentication API Endpoints

REST API endpoints for user management, authentication, and project management.
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime

from auth import (
    get_user_manager,
    User,
    Project,
    UserSession,
    UserRole,
    SubscriptionPlan,
    ProjectStatus
)

router = APIRouter(prefix="/api/auth", tags=["authentication"])
security = HTTPBearer()


class RegisterRequest(BaseModel):
    """Request model for user registration"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    first_name: Optional[str] = Field(None, max_length=50)
    last_name: Optional[str] = Field(None, max_length=50)


class LoginRequest(BaseModel):
    """Request model for user login"""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Response model for user data"""
    id: str
    email: str
    username: str
    role: str
    subscription_plan: str
    first_name: Optional[str]
    last_name: Optional[str]
    avatar_url: Optional[str]
    bio: Optional[str]
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]
    videos_created_this_month: int
    storage_used_mb: float
    projects_count: int


class LoginResponse(BaseModel):
    """Response model for login"""
    success: bool
    token: str
    user: UserResponse
    session_id: str
    expires_at: datetime


class ProjectRequest(BaseModel):
    """Request model for creating projects"""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field("", max_length=500)


class ProjectResponse(BaseModel):
    """Response model for project data"""
    id: str
    name: str
    description: str
    user_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    total_videos: int
    total_duration: float
    storage_used_mb: float
    tags: List[str]


# Dependency to get current user from token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    try:
        user_manager = get_user_manager()
        user = await user_manager.verify_session(credentials.credentials)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


# Optional user dependency (doesn't fail if no token)
async def get_current_user_optional(request: Request):
    """Get current user if token is provided, otherwise return None"""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        user_manager = get_user_manager()
        user = await user_manager.verify_session(token)
        
        return user
    except Exception:
        return None


@router.post("/register", response_model=UserResponse)
async def register_user(request: RegisterRequest):
    """Register a new user account"""
    try:
        user_manager = get_user_manager()
        
        user = await user_manager.create_user(
            email=request.email,
            username=request.username,
            password=request.password,
            first_name=request.first_name,
            last_name=request.last_name
        )
        
        return UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            role=user.role.value,
            subscription_plan=user.subscription_plan.value,
            first_name=user.first_name,
            last_name=user.last_name,
            avatar_url=user.avatar_url,
            bio=user.bio,
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at,
            last_login=user.last_login,
            videos_created_this_month=user.videos_created_this_month,
            storage_used_mb=user.storage_used_mb,
            projects_count=user.projects_count
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/login", response_model=LoginResponse)
async def login_user(request: LoginRequest, req: Request):
    """Authenticate user and create session"""
    try:
        user_manager = get_user_manager()
        
        user = await user_manager.authenticate_user(request.email, request.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create session
        session = await user_manager.create_session(
            user=user,
            ip_address=req.client.host,
            user_agent=req.headers.get("user-agent")
        )
        
        return LoginResponse(
            success=True,
            token=session.token,
            user=UserResponse(
                id=user.id,
                email=user.email,
                username=user.username,
                role=user.role.value,
                subscription_plan=user.subscription_plan.value,
                first_name=user.first_name,
                last_name=user.last_name,
                avatar_url=user.avatar_url,
                bio=user.bio,
                is_active=user.is_active,
                is_verified=user.is_verified,
                created_at=user.created_at,
                last_login=user.last_login,
                videos_created_this_month=user.videos_created_this_month,
                storage_used_mb=user.storage_used_mb,
                projects_count=user.projects_count
            ),
            session_id=session.id,
            expires_at=session.expires_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.post("/logout")
async def logout_user(current_user: User = Depends(get_current_user)):
    """Logout current user"""
    try:
        user_manager = get_user_manager()
        
        # We would need to track session ID to properly logout
        # For now, just return success (client should discard token)
        
        return {
            "success": True,
            "message": "Logged out successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Logout failed: {str(e)}"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile"""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        role=current_user.role.value,
        subscription_plan=current_user.subscription_plan.value,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        avatar_url=current_user.avatar_url,
        bio=current_user.bio,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at,
        last_login=current_user.last_login,
        videos_created_this_month=current_user.videos_created_this_month,
        storage_used_mb=current_user.storage_used_mb,
        projects_count=current_user.projects_count
    )


@router.put("/me")
async def update_user_profile(
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    bio: Optional[str] = None,
    avatar_url: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Update current user profile"""
    try:
        user_manager = get_user_manager()
        
        # Update fields if provided
        if first_name is not None:
            current_user.first_name = first_name
        if last_name is not None:
            current_user.last_name = last_name
        if bio is not None:
            current_user.bio = bio
        if avatar_url is not None:
            current_user.avatar_url = avatar_url
        
        updated_user = await user_manager.update_user(current_user)
        
        return {
            "success": True,
            "message": "Profile updated successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Profile update failed: {str(e)}"
        )


@router.get("/permissions")
async def get_user_permissions(current_user: User = Depends(get_current_user)):
    """Get current user's permissions"""
    permissions = current_user.get_permissions()
    
    return {
        "user_id": current_user.id,
        "subscription_plan": current_user.subscription_plan.value,
        "permissions": {
            "max_projects": permissions.max_projects,
            "max_videos_per_month": permissions.max_videos_per_month,
            "max_video_duration": permissions.max_video_duration,
            "can_use_premium_templates": permissions.can_use_premium_templates,
            "can_export_4k": permissions.can_export_4k,
            "can_use_batch_processing": permissions.can_use_batch_processing,
            "can_use_api": permissions.can_use_api,
            "storage_limit_mb": permissions.storage_limit_mb
        },
        "current_usage": {
            "videos_created_this_month": current_user.videos_created_this_month,
            "storage_used_mb": current_user.storage_used_mb,
            "projects_count": current_user.projects_count
        }
    }


# Project Management Endpoints

@router.post("/projects", response_model=ProjectResponse)
async def create_project(
    request: ProjectRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new project"""
    try:
        user_manager = get_user_manager()
        
        project = await user_manager.create_project(
            user_id=current_user.id,
            name=request.name,
            description=request.description
        )
        
        return ProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            user_id=project.user_id,
            status=project.status.value,
            created_at=project.created_at,
            updated_at=project.updated_at,
            total_videos=project.total_videos,
            total_duration=project.total_duration,
            storage_used_mb=project.storage_used_mb,
            tags=project.tags
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Project creation failed: {str(e)}"
        )


@router.get("/projects", response_model=List[ProjectResponse])
async def get_user_projects(
    status_filter: Optional[ProjectStatus] = None,
    current_user: User = Depends(get_current_user)
):
    """Get all projects for current user"""
    try:
        user_manager = get_user_manager()
        
        projects = await user_manager.get_user_projects(
            user_id=current_user.id,
            status=status_filter
        )
        
        return [
            ProjectResponse(
                id=project.id,
                name=project.name,
                description=project.description,
                user_id=project.user_id,
                status=project.status.value,
                created_at=project.created_at,
                updated_at=project.updated_at,
                total_videos=project.total_videos,
                total_duration=project.total_duration,
                storage_used_mb=project.storage_used_mb,
                tags=project.tags
            )
            for project in projects
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get projects: {str(e)}"
        )


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get a specific project"""
    try:
        user_manager = get_user_manager()
        
        project = await user_manager.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        # Check if user owns the project
        if project.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return ProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            user_id=project.user_id,
            status=project.status.value,
            created_at=project.created_at,
            updated_at=project.updated_at,
            total_videos=project.total_videos,
            total_duration=project.total_duration,
            storage_used_mb=project.storage_used_mb,
            tags=project.tags
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get project: {str(e)}"
        )


@router.put("/projects/{project_id}")
async def update_project(
    project_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user)
):
    """Update a project"""
    try:
        user_manager = get_user_manager()
        
        project = await user_manager.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        # Check if user owns the project
        if project.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Update fields if provided
        if name is not None:
            project.name = name
        if description is not None:
            project.description = description
        if tags is not None:
            project.tags = tags
        
        await user_manager.update_project(project)
        
        return {
            "success": True,
            "message": "Project updated successfully",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Project update failed: {str(e)}"
        )


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a project (soft delete)"""
    try:
        user_manager = get_user_manager()
        
        success = await user_manager.delete_project(project_id, current_user.id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found or access denied"
            )
        
        return {
            "success": True,
            "message": "Project deleted successfully",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Project deletion failed: {str(e)}"
        )


@router.get("/subscription-plans")
async def get_subscription_plans():
    """Get all available subscription plans"""
    return {
        "plans": [
            {
                "value": plan.value,
                "label": plan.value.replace("_", " ").title(),
                "features": {
                    "max_projects": 5 if plan == SubscriptionPlan.FREE else 
                                   15 if plan == SubscriptionPlan.BASIC else
                                   50 if plan == SubscriptionPlan.PRO else 999,
                    "max_videos_per_month": 10 if plan == SubscriptionPlan.FREE else
                                          50 if plan == SubscriptionPlan.BASIC else  
                                          200 if plan == SubscriptionPlan.PRO else 9999,
                    "storage_limit_mb": 1000 if plan == SubscriptionPlan.FREE else
                                       5000 if plan == SubscriptionPlan.BASIC else
                                       20000 if plan == SubscriptionPlan.PRO else 100000
                }
            }
            for plan in SubscriptionPlan
        ]
    }


@router.get("/user-roles")
async def get_user_roles():
    """Get all available user roles"""
    return {
        "roles": [
            {"value": role.value, "label": role.value.replace("_", " ").title()}
            for role in UserRole
        ]
    }