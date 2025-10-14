"""
Admin Management Interface

Comprehensive administration interface for managing users, monitoring system
health, configuring settings, and performing maintenance operations.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from fastapi import HTTPException, Depends, status, BackgroundTasks
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from auth.auth_manager import (
    UserRole, Permission, get_current_user, require_role, 
    require_permission, TokenPayload
)

logger = logging.getLogger(__name__)


class SystemStatus(Enum):
    """System status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"


class MaintenanceType(Enum):
    """Maintenance operation types"""
    SCHEDULED = "scheduled"
    EMERGENCY = "emergency"
    UPGRADE = "upgrade"
    BACKUP = "backup"


@dataclass
class SystemInfo:
    """System information summary"""
    version: str
    uptime: float
    status: SystemStatus
    active_users: int
    total_users: int
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    memory_usage: float
    cpu_usage: float
    disk_usage: float
    cache_hit_rate: float


class UserManagement(BaseModel):
    """User management data"""
    id: str
    email: EmailStr
    first_name: str
    last_name: str
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]
    login_count: int = 0
    total_videos: int = 0


class UserUpdate(BaseModel):
    """User update data"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None


class SystemSetting(BaseModel):
    """System setting configuration"""
    key: str
    value: Any
    category: str
    description: str
    data_type: str
    is_public: bool = False
    requires_restart: bool = False


class MaintenanceTask(BaseModel):
    """Maintenance task definition"""
    id: str
    type: MaintenanceType
    title: str
    description: str
    scheduled_at: datetime
    duration_minutes: int
    affected_services: List[str]
    created_by: str
    status: str = "pending"


class AdminInterface:
    """Main admin interface class"""
    
    def __init__(self):
        self.system_start_time = time.time()
        self.maintenance_mode = False
        self.maintenance_message = ""
        self.system_settings: Dict[str, SystemSetting] = {}
        self.pending_tasks: Dict[str, MaintenanceTask] = {}
        
        # Initialize default settings
        self._initialize_default_settings()
    
    def _initialize_default_settings(self):
        """Initialize default system settings"""
        default_settings = [
            SystemSetting(
                key="max_concurrent_generations",
                value=10,
                category="performance",
                description="Maximum concurrent video generations",
                data_type="integer"
            ),
            SystemSetting(
                key="default_video_quality",
                value="standard",
                category="video",
                description="Default video quality setting",
                data_type="string"
            ),
            SystemSetting(
                key="max_video_duration",
                value=300,
                category="video",
                description="Maximum video duration in seconds",
                data_type="integer"
            ),
            SystemSetting(
                key="enable_ai_optimization",
                value=True,
                category="ai",
                description="Enable AI optimization features",
                data_type="boolean"
            ),
            SystemSetting(
                key="cache_ttl_default",
                value=3600,
                category="cache",
                description="Default cache TTL in seconds",
                data_type="integer"
            ),
            SystemSetting(
                key="rate_limit_enabled",
                value=True,
                category="security",
                description="Enable API rate limiting",
                data_type="boolean"
            ),
            SystemSetting(
                key="max_file_size_mb",
                value=100,
                category="upload",
                description="Maximum file upload size in MB",
                data_type="integer"
            ),
            SystemSetting(
                key="notification_email",
                value="admin@aurarender.com",
                category="notifications",
                description="Admin notification email",
                data_type="string",
                is_public=True
            ),
        ]
        
        for setting in default_settings:
            self.system_settings[setting.key] = setting
    
    async def get_system_info(self, current_user: TokenPayload = Depends(require_role(UserRole.ADMIN))) -> SystemInfo:
        """Get comprehensive system information"""
        try:
            # Get system metrics (these would be real implementations)
            uptime = time.time() - self.system_start_time
            
            # Mock data - replace with real implementations
            system_info = SystemInfo(
                version="1.0.0",
                uptime=uptime,
                status=SystemStatus.MAINTENANCE if self.maintenance_mode else SystemStatus.HEALTHY,
                active_users=25,  # From user session tracking
                total_users=1500,  # From database
                active_tasks=8,   # From task queue
                completed_tasks=15420,  # From database
                failed_tasks=89,  # From database
                memory_usage=65.2,  # From system monitoring
                cpu_usage=23.1,   # From system monitoring
                disk_usage=45.8,  # From system monitoring
                cache_hit_rate=0.92  # From cache statistics
            )
            
            return system_info
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve system information"
            )
    
    async def get_users(self, 
                       skip: int = 0, 
                       limit: int = 100,
                       role: Optional[UserRole] = None,
                       is_active: Optional[bool] = None,
                       search: Optional[str] = None,
                       db: AsyncSession = Depends(),
                       current_user: TokenPayload = Depends(require_permission(Permission.USER_READ))) -> Dict[str, Any]:
        """Get users with filtering and pagination"""
        try:
            # Mock data - replace with real database queries
            users = [
                UserManagement(
                    id="user_1",
                    email="user1@example.com",
                    first_name="John",
                    last_name="Doe",
                    role=UserRole.USER,
                    is_active=True,
                    is_verified=True,
                    created_at=datetime.utcnow() - timedelta(days=30),
                    last_login=datetime.utcnow() - timedelta(hours=2),
                    login_count=45,
                    total_videos=12
                ),
                UserManagement(
                    id="user_2", 
                    email="premium@example.com",
                    first_name="Jane",
                    last_name="Smith",
                    role=UserRole.PREMIUM,
                    is_active=True,
                    is_verified=True,
                    created_at=datetime.utcnow() - timedelta(days=15),
                    last_login=datetime.utcnow() - timedelta(minutes=30),
                    login_count=78,
                    total_videos=25
                )
            ]
            
            # Apply filters
            if role:
                users = [u for u in users if u.role == role]
            if is_active is not None:
                users = [u for u in users if u.is_active == is_active]
            if search:
                search_lower = search.lower()
                users = [
                    u for u in users 
                    if search_lower in u.email.lower() 
                    or search_lower in u.first_name.lower()
                    or search_lower in u.last_name.lower()
                ]
            
            # Apply pagination
            total = len(users)
            users = users[skip:skip + limit]
            
            return {
                "users": users,
                "total": total,
                "page": skip // limit + 1,
                "pages": (total + limit - 1) // limit,
                "per_page": limit
            }
            
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve users"
            )
    
    async def update_user(self, 
                         user_id: str, 
                         update_data: UserUpdate,
                         db: AsyncSession = Depends(),
                         current_user: TokenPayload = Depends(require_permission(Permission.USER_WRITE))) -> UserManagement:
        """Update user information"""
        try:
            # Mock update - replace with real database update
            logger.info(f"User {current_user.user_id} updating user {user_id}: {update_data}")
            
            # Return mock updated user
            updated_user = UserManagement(
                id=user_id,
                email="updated@example.com",
                first_name=update_data.first_name or "Updated",
                last_name=update_data.last_name or "User",
                role=update_data.role or UserRole.USER,
                is_active=update_data.is_active if update_data.is_active is not None else True,
                is_verified=update_data.is_verified if update_data.is_verified is not None else True,
                created_at=datetime.utcnow() - timedelta(days=10),
                last_login=datetime.utcnow() - timedelta(hours=1)
            )
            
            return updated_user
            
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user"
            )
    
    async def delete_user(self, 
                         user_id: str,
                         db: AsyncSession = Depends(),
                         current_user: TokenPayload = Depends(require_permission(Permission.USER_DELETE))) -> Dict[str, str]:
        """Delete a user (soft delete)"""
        try:
            # Prevent self-deletion
            if user_id == current_user.user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete your own account"
                )
            
            # Mock deletion - replace with real database soft delete
            logger.info(f"User {current_user.user_id} deleting user {user_id}")
            
            return {"message": "User deleted successfully", "user_id": user_id}
            
        except Exception as e:
            logger.error(f"Error deleting user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user"
            )
    
    async def get_system_settings(self,
                                 category: Optional[str] = None,
                                 current_user: TokenPayload = Depends(require_permission(Permission.SYSTEM_READ))) -> List[SystemSetting]:
        """Get system settings"""
        try:
            settings = list(self.system_settings.values())
            
            if category:
                settings = [s for s in settings if s.category == category]
            
            # Filter sensitive settings for non-super-admin users
            if current_user.role != UserRole.SUPER_ADMIN:
                settings = [s for s in settings if s.is_public or s.category in ["video", "upload"]]
            
            return settings
            
        except Exception as e:
            logger.error(f"Error getting system settings: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve system settings"
            )
    
    async def update_system_setting(self,
                                   key: str,
                                   value: Any,
                                   current_user: TokenPayload = Depends(require_permission(Permission.SYSTEM_WRITE))) -> SystemSetting:
        """Update a system setting"""
        try:
            if key not in self.system_settings:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Setting not found"
                )
            
            setting = self.system_settings[key]
            
            # Validate value type
            if setting.data_type == "integer":
                value = int(value)
            elif setting.data_type == "float":
                value = float(value)
            elif setting.data_type == "boolean":
                value = bool(value)
            elif setting.data_type == "string":
                value = str(value)
            
            # Update setting
            setting.value = value
            self.system_settings[key] = setting
            
            logger.info(f"User {current_user.user_id} updated setting {key} to {value}")
            
            return setting
            
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid value type for setting {key}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error updating system setting {key}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update system setting"
            )
    
    async def enable_maintenance_mode(self,
                                     message: str = "System maintenance in progress",
                                     current_user: TokenPayload = Depends(require_role(UserRole.ADMIN))) -> Dict[str, str]:
        """Enable maintenance mode"""
        try:
            self.maintenance_mode = True
            self.maintenance_message = message
            
            logger.info(f"Maintenance mode enabled by user {current_user.user_id}")
            
            return {
                "message": "Maintenance mode enabled",
                "maintenance_message": message
            }
            
        except Exception as e:
            logger.error(f"Error enabling maintenance mode: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to enable maintenance mode"
            )
    
    async def disable_maintenance_mode(self,
                                      current_user: TokenPayload = Depends(require_role(UserRole.ADMIN))) -> Dict[str, str]:
        """Disable maintenance mode"""
        try:
            self.maintenance_mode = False
            self.maintenance_message = ""
            
            logger.info(f"Maintenance mode disabled by user {current_user.user_id}")
            
            return {"message": "Maintenance mode disabled"}
            
        except Exception as e:
            logger.error(f"Error disabling maintenance mode: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to disable maintenance mode"
            )
    
    async def get_system_logs(self,
                             lines: int = 100,
                             level: str = "INFO",
                             component: Optional[str] = None,
                             current_user: TokenPayload = Depends(require_permission(Permission.SYSTEM_ADMIN))) -> List[Dict[str, Any]]:
        """Get system logs"""
        try:
            # Mock log data - replace with real log retrieval
            logs = []
            
            for i in range(lines):
                log_entry = {
                    "timestamp": datetime.utcnow() - timedelta(minutes=i),
                    "level": level,
                    "component": component or "system",
                    "message": f"Sample log message {i}",
                    "details": {"request_id": f"req_{i}", "user_id": "user_123"}
                }
                logs.append(log_entry)
            
            return logs
            
        except Exception as e:
            logger.error(f"Error getting system logs: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve system logs"
            )
    
    async def create_backup(self,
                           include_data: bool = True,
                           include_media: bool = False,
                           background_tasks: BackgroundTasks = BackgroundTasks(),
                           current_user: TokenPayload = Depends(require_role(UserRole.ADMIN))) -> Dict[str, str]:
        """Create system backup"""
        try:
            backup_id = f"backup_{int(time.time())}"
            
            # Start backup task in background
            background_tasks.add_task(
                self._perform_backup,
                backup_id, include_data, include_media, current_user.user_id
            )
            
            return {
                "message": "Backup started",
                "backup_id": backup_id,
                "status": "in_progress"
            }
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create backup"
            )
    
    async def _perform_backup(self, backup_id: str, include_data: bool, 
                             include_media: bool, user_id: str):
        """Perform backup operation (background task)"""
        try:
            logger.info(f"Starting backup {backup_id} requested by user {user_id}")
            
            # Simulate backup process
            await asyncio.sleep(5)  # Database backup
            if include_media:
                await asyncio.sleep(10)  # Media file backup
            
            logger.info(f"Backup {backup_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Backup {backup_id} failed: {e}")
    
    async def get_analytics_data(self,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                metric: Optional[str] = None,
                                current_user: TokenPayload = Depends(require_permission(Permission.ANALYTICS_READ))) -> Dict[str, Any]:
        """Get analytics data"""
        try:
            # Set default date range
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Mock analytics data - replace with real analytics
            analytics = {
                "period": {
                    "start": start_date,
                    "end": end_date
                },
                "metrics": {
                    "total_videos_generated": 1250,
                    "active_users": 89,
                    "premium_users": 23,
                    "api_requests": 45670,
                    "average_generation_time": 23.5,
                    "success_rate": 0.97
                },
                "charts": {
                    "daily_generations": [
                        {"date": "2024-01-01", "count": 45},
                        {"date": "2024-01-02", "count": 52},
                        {"date": "2024-01-03", "count": 38}
                    ],
                    "user_growth": [
                        {"date": "2024-01-01", "total_users": 1420},
                        {"date": "2024-01-02", "total_users": 1435},
                        {"date": "2024-01-03", "total_users": 1450}
                    ]
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting analytics data: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve analytics data"
            )
    
    async def clear_cache(self,
                         pattern: Optional[str] = None,
                         current_user: TokenPayload = Depends(require_permission(Permission.SYSTEM_ADMIN))) -> Dict[str, Any]:
        """Clear system cache"""
        try:
            # Mock cache clearing - replace with real cache operations
            cleared_keys = 0
            
            if pattern:
                # Clear specific pattern
                cleared_keys = 25  # Mock number
                logger.info(f"User {current_user.user_id} cleared cache pattern: {pattern}")
            else:
                # Clear all cache
                cleared_keys = 150  # Mock number
                logger.info(f"User {current_user.user_id} cleared all cache")
            
            return {
                "message": "Cache cleared successfully",
                "cleared_keys": cleared_keys,
                "pattern": pattern
            }
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear cache"
            )
    
    async def restart_service(self,
                             service_name: str,
                             current_user: TokenPayload = Depends(require_role(UserRole.SUPER_ADMIN))) -> Dict[str, str]:
        """Restart a system service"""
        try:
            allowed_services = ["worker", "scheduler", "cache", "monitoring"]
            
            if service_name not in allowed_services:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Service {service_name} cannot be restarted"
                )
            
            logger.info(f"User {current_user.user_id} restarting service: {service_name}")
            
            # Mock service restart - replace with real service management
            return {
                "message": f"Service {service_name} restart initiated",
                "service": service_name,
                "status": "restarting"
            }
            
        except Exception as e:
            logger.error(f"Error restarting service {service_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to restart service {service_name}"
            )
    
    def is_maintenance_mode(self) -> bool:
        """Check if system is in maintenance mode"""
        return self.maintenance_mode
    
    def get_maintenance_message(self) -> str:
        """Get maintenance mode message"""
        return self.maintenance_message
    
    def get_system_setting(self, key: str, default: Any = None) -> Any:
        """Get a system setting value"""
        setting = self.system_settings.get(key)
        return setting.value if setting else default


# Global admin interface instance
admin_interface = AdminInterface()


def get_admin_interface() -> AdminInterface:
    """Get global admin interface"""
    return admin_interface


def maintenance_mode_check():
    """Dependency to check maintenance mode"""
    def check():
        if admin_interface.is_maintenance_mode():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=admin_interface.get_maintenance_message(),
                headers={"Retry-After": "300"}
            )
    return check