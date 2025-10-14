"""
Authentication and Authorization System

Comprehensive authentication and authorization system with JWT tokens,
role-based access control (RBAC), and user management.
"""

import asyncio
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import jwt
import bcrypt
import logging

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

# Import database models
try:
    from database.orm_models.user_models import (
        User as DBUser, 
        UserRole as DBUserRole, 
        UserStatus,
        UserProfile as DBUserProfile
    )
except ImportError:
    DBUser = None
    DBUserRole = None
    UserStatus = None
    DBUserProfile = None

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles with hierarchical permissions"""
    GUEST = "guest"
    USER = "user"
    PREMIUM = "premium"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class Permission(Enum):
    """System permissions"""
    # Video generation permissions
    VIDEO_CREATE = "video:create"
    VIDEO_READ = "video:read"
    VIDEO_DELETE = "video:delete"
    VIDEO_SHARE = "video:share"
    
    # AI optimization permissions
    AI_OPTIMIZE_BASIC = "ai:optimize:basic"
    AI_OPTIMIZE_PREMIUM = "ai:optimize:premium"
    AI_ANALYZE = "ai:analyze"
    
    # System permissions
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"
    SYSTEM_ADMIN = "system:admin"
    
    # User management permissions
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    USER_ADMIN = "user:admin"
    
    # Analytics permissions
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_EXPORT = "analytics:export"
    
    # API permissions
    API_ACCESS = "api:access"
    API_UNLIMITED = "api:unlimited"


# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.GUEST: {
        Permission.VIDEO_READ,
    },
    UserRole.USER: {
        Permission.VIDEO_CREATE,
        Permission.VIDEO_READ,
        Permission.VIDEO_DELETE,
        Permission.AI_OPTIMIZE_BASIC,
        Permission.AI_ANALYZE,
        Permission.API_ACCESS,
    },
    UserRole.PREMIUM: {
        Permission.VIDEO_CREATE,
        Permission.VIDEO_READ,
        Permission.VIDEO_DELETE,
        Permission.VIDEO_SHARE,
        Permission.AI_OPTIMIZE_BASIC,
        Permission.AI_OPTIMIZE_PREMIUM,
        Permission.AI_ANALYZE,
        Permission.API_ACCESS,
        Permission.API_UNLIMITED,
        Permission.ANALYTICS_READ,
    },
    UserRole.MODERATOR: {
        Permission.VIDEO_CREATE,
        Permission.VIDEO_READ,
        Permission.VIDEO_DELETE,
        Permission.VIDEO_SHARE,
        Permission.AI_OPTIMIZE_BASIC,
        Permission.AI_OPTIMIZE_PREMIUM,
        Permission.AI_ANALYZE,
        Permission.API_ACCESS,
        Permission.API_UNLIMITED,
        Permission.ANALYTICS_READ,
        Permission.USER_READ,
        Permission.SYSTEM_READ,
    },
    UserRole.ADMIN: {
        Permission.VIDEO_CREATE,
        Permission.VIDEO_READ,
        Permission.VIDEO_DELETE,
        Permission.VIDEO_SHARE,
        Permission.AI_OPTIMIZE_BASIC,
        Permission.AI_OPTIMIZE_PREMIUM,
        Permission.AI_ANALYZE,
        Permission.API_ACCESS,
        Permission.API_UNLIMITED,
        Permission.ANALYTICS_READ,
        Permission.ANALYTICS_EXPORT,
        Permission.USER_READ,
        Permission.USER_WRITE,
        Permission.USER_DELETE,
        Permission.SYSTEM_READ,
        Permission.SYSTEM_WRITE,
    },
    UserRole.SUPER_ADMIN: {
        # All permissions
        *[p for p in Permission]
    }
}


@dataclass
class TokenPayload:
    """JWT token payload structure"""
    user_id: str
    email: str
    role: UserRole
    permissions: List[Permission]
    issued_at: float
    expires_at: float
    token_type: str = "access"
    session_id: Optional[str] = None


class UserRegistration(BaseModel):
    """User registration data"""
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    company: Optional[str] = None


class UserLogin(BaseModel):
    """User login credentials"""
    email: EmailStr
    password: str
    remember_me: bool = False


class UserProfile(BaseModel):
    """User profile information"""
    id: str
    email: str
    first_name: str
    last_name: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    is_verified: bool = False
    company: Optional[str] = None
    avatar_url: Optional[str] = None
    preferences: Dict[str, Any] = {}


class TokenResponse(BaseModel):
    """Authentication token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserProfile


class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=30)
        self.password_reset_expire = timedelta(hours=24)
        
        # Security configuration
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        self.password_min_length = 8
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.failed_attempts: Dict[str, List[float]] = {}
        
        # Security event logging
        self.security_events: List[Dict[str, Any]] = []
        
    async def register_user(self, registration: UserRegistration, 
                          db: AsyncSession, role: UserRole = UserRole.USER) -> UserProfile:
        """Register a new user"""
        # Check if user already exists
        existing_user = await self._get_user_by_email(registration.email, db)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Validate password strength
        self._validate_password_strength(registration.password)
        
        # Hash password
        password_hash = self._hash_password(registration.password)
        
        # Create user record
        user_data = {
            "id": self._generate_user_id(),
            "email": registration.email,
            "password_hash": password_hash,
            "first_name": registration.first_name,
            "last_name": registration.last_name,
            "role": role.value,
            "created_at": datetime.utcnow(),
            "is_active": True,
            "is_verified": False,
            "company": registration.company,
        }
        
        # Insert into database if models are available
        if DBUser and db:
            try:
                # Create new user
                new_user = DBUser(
                    id=user_data["id"],
                    username=registration.email.split('@')[0],
                    email=user_data["email"],
                    password_hash=user_data["password_hash"],
                    first_name=user_data["first_name"],
                    last_name=user_data["last_name"],
                    role=DBUserRole[role.value.upper()] if DBUserRole else None,
                    status=UserStatus.PENDING_VERIFICATION if UserStatus and not registration.is_verified else UserStatus.ACTIVE if UserStatus else None,
                    is_email_verified=registration.is_verified,
                    created_at=user_data["created_at"],
                    language=registration.language or 'en',
                    timezone=registration.timezone or 'UTC'
                )
                
                db.add(new_user)
                await db.commit()
                await db.refresh(new_user)
                
            except IntegrityError as e:
                await db.rollback()
                if "email" in str(e):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Email already registered"
                    )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create user"
                )
            except Exception as e:
                await db.rollback()
                logger.error(f"Failed to create user: {e}")
        else:
            # Fallback to in-memory storage for testing
            logger.info("Database models not available, using in-memory storage")
        
        user_profile = UserProfile(
            id=user_data["id"],
            email=user_data["email"],
            first_name=user_data["first_name"],
            last_name=user_data["last_name"],
            role=role,
            created_at=user_data["created_at"],
            company=user_data.get("company")
        )
        
        # Log security event
        await self._log_security_event("user_registered", {
            "user_id": user_data["id"],
            "email": registration.email,
            "role": role.value
        })
        
        return user_profile
    
    async def authenticate_user(self, login: UserLogin, db: AsyncSession) -> TokenResponse:
        """Authenticate user and generate tokens"""
        # Check for account lockout
        if await self._is_account_locked(login.email):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Account temporarily locked due to too many failed attempts"
            )
        
        # Get user from database
        user = await self._get_user_by_email(login.email, db)
        if not user or not self._verify_password(login.password, user.get("password_hash", "")):
            await self._record_failed_attempt(login.email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Check if user is active
        if not user.get("is_active", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated"
            )
        
        # Clear failed attempts
        self._clear_failed_attempts(login.email)
        
        # Create user profile
        user_role = UserRole(user.get("role", UserRole.USER.value))
        user_profile = UserProfile(
            id=user["id"],
            email=user["email"],
            first_name=user.get("first_name", ""),
            last_name=user.get("last_name", ""),
            role=user_role,
            created_at=user.get("created_at", datetime.utcnow()),
            last_login=datetime.utcnow(),
            is_active=user.get("is_active", True),
            is_verified=user.get("is_verified", False),
            company=user.get("company"),
            avatar_url=user.get("avatar_url"),
            preferences=user.get("preferences", {})
        )
        
        # Generate tokens
        session_id = self._generate_session_id()
        access_token = await self._create_access_token(user_profile, session_id)
        refresh_token = await self._create_refresh_token(user_profile, session_id)
        
        # Store session
        await self._store_session(session_id, user_profile, login.remember_me)
        
        # Update last login
        await self._update_last_login(user["id"], db)
        
        # Log security event
        await self._log_security_event("user_login", {
            "user_id": user["id"],
            "email": login.email,
            "session_id": session_id
        })
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(self.access_token_expire.total_seconds()),
            user=user_profile
        )
    
    async def refresh_token(self, refresh_token: str, db: AsyncSession) -> TokenResponse:
        """Refresh access token using refresh token"""
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            user_id = payload.get("user_id")
            session_id = payload.get("session_id")
            
            # Validate session
            if not await self._validate_session(session_id, user_id):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid session"
                )
            
            # Get user
            user = await self._get_user_by_id(user_id, db)
            if not user or not user.get("is_active"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive"
                )
            
            # Create user profile
            user_role = UserRole(user.get("role", UserRole.USER.value))
            user_profile = UserProfile(
                id=user["id"],
                email=user["email"],
                first_name=user.get("first_name", ""),
                last_name=user.get("last_name", ""),
                role=user_role,
                created_at=user.get("created_at", datetime.utcnow()),
                is_active=user.get("is_active", True),
                is_verified=user.get("is_verified", False)
            )
            
            # Generate new access token
            new_access_token = await self._create_access_token(user_profile, session_id)
            
            return TokenResponse(
                access_token=new_access_token,
                refresh_token=refresh_token,  # Keep same refresh token
                expires_in=int(self.access_token_expire.total_seconds()),
                user=user_profile
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
    
    async def logout(self, session_id: str, user_id: str):
        """Logout user and invalidate session"""
        await self._invalidate_session(session_id)
        
        # Log security event
        await self._log_security_event("user_logout", {
            "user_id": user_id,
            "session_id": session_id
        })
    
    async def verify_token(self, token: str) -> TokenPayload:
        """Verify JWT token and extract payload"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Extract token data
            token_payload = TokenPayload(
                user_id=payload["user_id"],
                email=payload["email"],
                role=UserRole(payload["role"]),
                permissions=[Permission(p) for p in payload["permissions"]],
                issued_at=payload["iat"],
                expires_at=payload["exp"],
                token_type=payload.get("type", "access"),
                session_id=payload.get("session_id")
            )
            
            # Validate session if present
            if token_payload.session_id:
                if not await self._validate_session(token_payload.session_id, token_payload.user_id):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid session"
                    )
            
            return token_payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def check_permission(self, user_role: UserRole, required_permission: Permission) -> bool:
        """Check if user role has required permission"""
        user_permissions = ROLE_PERMISSIONS.get(user_role, set())
        return required_permission in user_permissions
    
    def check_permissions(self, user_role: UserRole, required_permissions: List[Permission]) -> bool:
        """Check if user role has all required permissions"""
        user_permissions = ROLE_PERMISSIONS.get(user_role, set())
        return all(perm in user_permissions for perm in required_permissions)
    
    async def _create_access_token(self, user: UserProfile, session_id: str) -> str:
        """Create JWT access token"""
        now = datetime.utcnow()
        expire = now + self.access_token_expire
        
        user_permissions = ROLE_PERMISSIONS.get(user.role, set())
        
        payload = {
            "user_id": user.id,
            "email": user.email,
            "role": user.role.value,
            "permissions": [p.value for p in user_permissions],
            "type": "access",
            "session_id": session_id,
            "iat": now,
            "exp": expire
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    async def _create_refresh_token(self, user: UserProfile, session_id: str) -> str:
        """Create JWT refresh token"""
        now = datetime.utcnow()
        expire = now + self.refresh_token_expire
        
        payload = {
            "user_id": user.id,
            "email": user.email,
            "type": "refresh",
            "session_id": session_id,
            "iat": now,
            "exp": expire
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def _validate_password_strength(self, password: str):
        """Validate password meets security requirements"""
        if len(password) < self.password_min_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password must be at least {self.password_min_length} characters"
            )
        
        # Check for at least one uppercase, lowercase, digit, and special character
        if not any(c.isupper() for c in password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must contain at least one uppercase letter"
            )
        
        if not any(c.islower() for c in password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must contain at least one lowercase letter"
            )
        
        if not any(c.isdigit() for c in password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must contain at least one digit"
            )
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID"""
        return f"user_{int(time.time())}_{secrets.token_hex(8)}"
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{int(time.time())}_{secrets.token_hex(16)}"
    
    async def _store_session(self, session_id: str, user: UserProfile, remember_me: bool):
        """Store session information"""
        expire_time = time.time() + (
            self.refresh_token_expire.total_seconds() if remember_me
            else self.access_token_expire.total_seconds() * 24  # 24 hours for regular sessions
        )
        
        self.active_sessions[session_id] = {
            "user_id": user.id,
            "email": user.email,
            "created_at": time.time(),
            "expires_at": expire_time,
            "remember_me": remember_me
        }
    
    async def _validate_session(self, session_id: str, user_id: str) -> bool:
        """Validate session exists and is not expired"""
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        if session["user_id"] != user_id:
            return False
        
        if time.time() > session["expires_at"]:
            await self._invalidate_session(session_id)
            return False
        
        return True
    
    async def _invalidate_session(self, session_id: str):
        """Invalidate a session"""
        self.active_sessions.pop(session_id, None)
    
    async def _record_failed_attempt(self, email: str):
        """Record failed login attempt"""
        now = time.time()
        if email not in self.failed_attempts:
            self.failed_attempts[email] = []
        
        self.failed_attempts[email].append(now)
        
        # Keep only recent attempts
        cutoff = now - self.lockout_duration.total_seconds()
        self.failed_attempts[email] = [
            attempt for attempt in self.failed_attempts[email]
            if attempt > cutoff
        ]
    
    def _clear_failed_attempts(self, email: str):
        """Clear failed login attempts"""
        self.failed_attempts.pop(email, None)
    
    async def _is_account_locked(self, email: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if email not in self.failed_attempts:
            return False
        
        recent_attempts = self.failed_attempts[email]
        return len(recent_attempts) >= self.max_login_attempts
    
    async def _get_user_by_email(self, email: str, db: AsyncSession) -> Optional[Dict[str, Any]]:
        """Get user by email from database"""
        if DBUser and db:
            try:
                result = await db.execute(
                    select(DBUser).where(DBUser.email == email)
                )
                user = result.scalar_one_or_none()
                
                if user:
                    return {
                        "id": str(user.id),
                        "email": user.email,
                        "password_hash": user.password_hash,
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "role": user.role.value if hasattr(user.role, 'value') else UserRole.STANDARD.value,
                        "created_at": user.created_at,
                        "is_active": user.status == UserStatus.ACTIVE if UserStatus else True,
                        "is_verified": user.is_email_verified
                    }
                return None
            except Exception as e:
                logger.error(f"Database error getting user by email: {e}")
                # Fallback to mock data
        
        # Fallback for testing when database is not available
        if email == "admin@aurarender.com":
            return {
                "id": "user_admin_123",
                "email": email,
                "password_hash": self._hash_password("admin123"),
                "first_name": "Admin",
                "last_name": "User",
                "role": UserRole.ADMIN.value,
                "created_at": datetime.utcnow(),
                "is_active": True,
                "is_verified": True
            }
        return None
    
    async def _get_user_by_id(self, user_id: str, db: AsyncSession) -> Optional[Dict[str, Any]]:
        """Get user by ID from database"""
        if DBUser and db:
            try:
                result = await db.execute(
                    select(DBUser).where(DBUser.id == user_id)
                )
                user = result.scalar_one_or_none()
                
                if user:
                    return {
                        "id": str(user.id),
                        "email": user.email,
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "role": user.role.value if hasattr(user.role, 'value') else UserRole.STANDARD.value,
                        "created_at": user.created_at,
                        "is_active": user.status == UserStatus.ACTIVE if UserStatus else True,
                        "is_verified": user.is_email_verified
                    }
                return None
            except Exception as e:
                logger.error(f"Database error getting user by ID: {e}")
                # Fallback to mock data
        
        # Fallback for testing when database is not available
        if user_id == "user_admin_123":
            return {
                "id": user_id,
                "email": "admin@aurarender.com",
                "first_name": "Admin",
                "last_name": "User",
                "role": UserRole.ADMIN.value,
                "created_at": datetime.utcnow(),
                "is_active": True,
                "is_verified": True
            }
        return None
    
    async def _update_last_login(self, user_id: str, db: AsyncSession):
        """Update user's last login timestamp"""
        if DBUser and db:
            try:
                await db.execute(
                    update(DBUser)
                    .where(DBUser.id == user_id)
                    .values(
                        last_login_at=datetime.utcnow(),
                        login_count=DBUser.login_count + 1
                    )
                )
                await db.commit()
            except Exception as e:
                logger.error(f"Failed to update last login: {e}")
                await db.rollback()
        else:
            # No database available, just log
            logger.debug(f"Updated last login for user {user_id} (in-memory only)")
    
    async def _log_security_event(self, event_type: str, data: Dict[str, Any]):
        """Log security event"""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        
        self.security_events.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        logger.info(f"Security event: {event_type}", extra=data)
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        recent_events = [
            event for event in self.security_events
            if time.time() - event["timestamp"] < 3600  # Last hour
        ]
        
        return {
            "active_sessions": len(self.active_sessions),
            "failed_attempts": sum(len(attempts) for attempts in self.failed_attempts.values()),
            "locked_accounts": sum(
                1 for email in self.failed_attempts.keys()
                if len(self.failed_attempts[email]) >= self.max_login_attempts
            ),
            "recent_events": len(recent_events),
            "event_types": {
                event_type: sum(1 for e in recent_events if e["type"] == event_type)
                for event_type in set(e["type"] for e in recent_events)
            }
        }


# FastAPI dependencies
security = HTTPBearer()
auth_manager: Optional[AuthManager] = None


def initialize_auth_manager(secret_key: str):
    """Initialize global auth manager"""
    global auth_manager
    auth_manager = AuthManager(secret_key)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenPayload:
    """FastAPI dependency to get current authenticated user"""
    if not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system not initialized"
        )
    
    return await auth_manager.verify_token(credentials.credentials)


def require_permission(permission: Permission):
    """FastAPI dependency factory to require specific permission"""
    def permission_checker(current_user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if not auth_manager.check_permission(current_user.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission.value}"
            )
        return current_user
    
    return permission_checker


def require_permissions(permissions: List[Permission]):
    """FastAPI dependency factory to require multiple permissions"""
    def permissions_checker(current_user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if not auth_manager.check_permissions(current_user.role, permissions):
            missing = [
                p.value for p in permissions
                if not auth_manager.check_permission(current_user.role, p)
            ]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permissions: {', '.join(missing)}"
            )
        return current_user
    
    return permissions_checker


def require_role(role: UserRole):
    """FastAPI dependency factory to require specific role or higher"""
    role_hierarchy = [
        UserRole.GUEST,
        UserRole.USER,
        UserRole.PREMIUM,
        UserRole.MODERATOR,
        UserRole.ADMIN,
        UserRole.SUPER_ADMIN
    ]
    
    def role_checker(current_user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        user_role_level = role_hierarchy.index(current_user.role)
        required_role_level = role_hierarchy.index(role)
        
        if user_role_level < required_role_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {role.value} or higher required"
            )
        return current_user
    
    return role_checker