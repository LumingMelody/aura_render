"""
认证服务 - JWT令牌认证和用户授权管理
"""
from typing import Dict, List, Any, Optional, Union
import asyncio
from datetime import datetime, timedelta
import jwt
import hashlib
import secrets
from dataclasses import dataclass, field
from enum import Enum
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

from cache.cache_manager import CacheManager
from database.database_manager import DatabaseManager


class UserRole(Enum):
    """用户角色"""
    GUEST = "guest"
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class Permission(Enum):
    """权限类型"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    CREATE_VIDEO = "create_video"
    UPLOAD_MATERIAL = "upload_material"
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"


@dataclass
class User:
    """用户信息"""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[Permission]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenPayload:
    """JWT载荷"""
    user_id: str
    username: str
    role: str
    permissions: List[str]
    issued_at: datetime
    expires_at: datetime
    token_type: str = "access"


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")
    remember_me: bool = Field(False, description="记住我")


class LoginResponse(BaseModel):
    """登录响应"""
    access_token: str = Field(..., description="访问令牌")
    refresh_token: str = Field(..., description="刷新令牌")
    token_type: str = Field("bearer", description="令牌类型")
    expires_in: int = Field(..., description="过期时间(秒)")
    user: Dict[str, Any] = Field(..., description="用户信息")


class RefreshRequest(BaseModel):
    """刷新令牌请求"""
    refresh_token: str = Field(..., description="刷新令牌")


class AuthService:
    """认证服务"""

    def __init__(self, cache_manager: Optional[CacheManager] = None,
                 database_manager: Optional[DatabaseManager] = None):
        self.cache_manager = cache_manager
        self.database_manager = database_manager

        # JWT配置
        self.secret_key = self._generate_secret_key()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 60
        self.refresh_token_expire_days = 30

        # 用户存储（生产环境应使用数据库）
        self.users: Dict[str, User] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.blacklisted_tokens: set = set()

        # 角色权限映射
        self.role_permissions = {
            UserRole.GUEST: [Permission.READ],
            UserRole.USER: [Permission.READ, Permission.WRITE, Permission.CREATE_VIDEO],
            UserRole.PREMIUM: [Permission.READ, Permission.WRITE, Permission.CREATE_VIDEO, Permission.UPLOAD_MATERIAL],
            UserRole.ADMIN: [Permission.READ, Permission.WRITE, Permission.DELETE, Permission.CREATE_VIDEO,
                           Permission.UPLOAD_MATERIAL, Permission.MANAGE_USERS, Permission.VIEW_ANALYTICS],
            UserRole.SUPER_ADMIN: list(Permission)
        }

        # 初始化默认用户
        self._create_default_users()

    def _generate_secret_key(self) -> str:
        """生成密钥"""
        return secrets.token_urlsafe(32)

    def _create_default_users(self):
        """创建默认用户"""
        # 管理员用户
        admin_user = User(
            user_id="admin_001",
            username="admin",
            email="admin@example.com",
            role=UserRole.ADMIN,
            permissions=self.role_permissions[UserRole.ADMIN],
            created_at=datetime.now()
        )
        self.users["admin"] = admin_user

        # 测试用户
        test_user = User(
            user_id="user_001",
            username="testuser",
            email="test@example.com",
            role=UserRole.USER,
            permissions=self.role_permissions[UserRole.USER],
            created_at=datetime.now()
        )
        self.users["testuser"] = test_user

        # 访客用户
        guest_user = User(
            user_id="guest_001",
            username="guest",
            email="guest@example.com",
            role=UserRole.GUEST,
            permissions=self.role_permissions[UserRole.GUEST],
            created_at=datetime.now()
        )
        self.users["guest"] = guest_user

    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()

    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """验证密码"""
        return self._hash_password(password) == hashed_password

    def create_access_token(self, user: User) -> str:
        """创建访问令牌"""
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=self.access_token_expire_minutes)

        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "iat": now,
            "exp": expires_at,
            "type": "access"
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, user: User) -> str:
        """创建刷新令牌"""
        now = datetime.utcnow()
        expires_at = now + timedelta(days=self.refresh_token_expire_days)

        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "iat": now,
            "exp": expires_at,
            "type": "refresh"
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """验证令牌"""
        try:
            # 检查黑名单
            if token in self.blacklisted_tokens:
                return None

            # 解码令牌
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            return TokenPayload(
                user_id=payload["user_id"],
                username=payload["username"],
                role=payload["role"],
                permissions=payload["permissions"],
                issued_at=datetime.fromtimestamp(payload["iat"]),
                expires_at=datetime.fromtimestamp(payload["exp"]),
                token_type=payload.get("type", "access")
            )

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """用户认证"""
        # 简化的认证逻辑（生产环境应从数据库获取）
        if username not in self.users:
            return None

        user = self.users[username]

        # 简单的密码验证（实际应用中应使用安全的密码哈希）
        if username == "admin" and password == "admin123":
            return user
        elif username == "testuser" and password == "test123":
            return user
        elif username == "guest" and password == "guest":
            return user

        return None

    async def login(self, request: LoginRequest) -> LoginResponse:
        """用户登录"""
        user = await self.authenticate_user(request.username, request.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled"
            )

        # 创建令牌
        access_token = self.create_access_token(user)
        refresh_token = self.create_refresh_token(user)

        # 更新用户最后登录时间
        user.last_login = datetime.now()

        # 创建会话
        session_id = secrets.token_urlsafe(16)
        self.user_sessions[session_id] = {
            "user_id": user.user_id,
            "username": user.username,
            "login_time": datetime.now(),
            "refresh_token": refresh_token
        }

        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_token_expire_minutes * 60,
            user={
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "permissions": [p.value for p in user.permissions]
            }
        )

    async def refresh_token(self, request: RefreshRequest) -> LoginResponse:
        """刷新令牌"""
        token_payload = self.verify_token(request.refresh_token)
        if not token_payload or token_payload.token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        # 获取用户信息
        user = self.users.get(token_payload.username)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )

        # 创建新的访问令牌
        access_token = self.create_access_token(user)

        return LoginResponse(
            access_token=access_token,
            refresh_token=request.refresh_token,  # 刷新令牌保持不变
            expires_in=self.access_token_expire_minutes * 60,
            user={
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "permissions": [p.value for p in user.permissions]
            }
        )

    async def logout(self, token: str) -> bool:
        """用户登出"""
        # 将令牌加入黑名单
        self.blacklisted_tokens.add(token)

        # 移除相关会话
        token_payload = self.verify_token(token)
        if token_payload:
            sessions_to_remove = []
            for session_id, session in self.user_sessions.items():
                if session["user_id"] == token_payload.user_id:
                    sessions_to_remove.append(session_id)

            for session_id in sessions_to_remove:
                del self.user_sessions[session_id]

        return True

    def get_user_from_token(self, token: str) -> Optional[User]:
        """从令牌获取用户信息"""
        token_payload = self.verify_token(token)
        if not token_payload:
            return None

        return self.users.get(token_payload.username)

    def check_permission(self, user: User, required_permission: Permission) -> bool:
        """检查用户权限"""
        return required_permission in user.permissions

    def check_role(self, user: User, required_role: UserRole) -> bool:
        """检查用户角色"""
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.USER: 1,
            UserRole.PREMIUM: 2,
            UserRole.ADMIN: 3,
            UserRole.SUPER_ADMIN: 4
        }

        user_level = role_hierarchy.get(user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)

        return user_level >= required_level

    async def create_user(self, username: str, email: str, password: str,
                         role: UserRole = UserRole.USER) -> User:
        """创建新用户"""
        if username in self.users:
            raise ValueError("Username already exists")

        user_id = f"user_{len(self.users) + 1:03d}"
        permissions = self.role_permissions.get(role, [])

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            permissions=permissions,
            created_at=datetime.now()
        )

        self.users[username] = user

        # 存储到数据库（如果可用）
        if self.database_manager:
            await self._store_user_to_db(user, password)

        return user

    async def _store_user_to_db(self, user: User, password: str):
        """存储用户到数据库"""
        try:
            hashed_password = self._hash_password(password)
            query = """
            INSERT INTO users (user_id, username, email, password_hash, role, permissions, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            params = [
                user.user_id, user.username, user.email, hashed_password,
                user.role.value, [p.value for p in user.permissions], user.created_at
            ]
            await self.database_manager.execute_query(query, params)
        except Exception as e:
            print(f"❌ Failed to store user to database: {e}")

    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户会话"""
        sessions = []
        for session_id, session in self.user_sessions.items():
            if session["user_id"] == user_id:
                sessions.append({
                    "session_id": session_id,
                    "login_time": session["login_time"].isoformat(),
                    "username": session["username"]
                })
        return sessions

    def revoke_user_sessions(self, user_id: str) -> int:
        """撤销用户所有会话"""
        sessions_to_remove = []
        for session_id, session in self.user_sessions.items():
            if session["user_id"] == user_id:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.user_sessions[session_id]

        return len(sessions_to_remove)

    def get_auth_stats(self) -> Dict[str, Any]:
        """获取认证统计"""
        return {
            "total_users": len(self.users),
            "active_sessions": len(self.user_sessions),
            "blacklisted_tokens": len(self.blacklisted_tokens),
            "user_roles": {
                role.value: len([u for u in self.users.values() if u.role == role])
                for role in UserRole
            }
        }


class AuthMiddleware(BaseHTTPMiddleware):
    """认证中间件"""

    def __init__(self, app, auth_service: AuthService):
        super().__init__(app)
        self.auth_service = auth_service
        self.security = HTTPBearer(auto_error=False)

        # 不需要认证的路径
        self.public_paths = {
            "/", "/health", "/info", "/docs", "/openapi.json",
            "/api/v1/auth/login", "/api/v1/auth/refresh"
        }

    async def dispatch(self, request: Request, call_next):
        # 检查是否为公开路径
        if request.url.path in self.public_paths or request.url.path.startswith("/docs"):
            return await call_next(request)

        # 获取认证头
        authorization = request.headers.get("Authorization")
        if not authorization:
            # 对于未认证的请求，使用访客用户
            request.state.user = self.auth_service.users.get("guest")
            return await call_next(request)

        try:
            # 提取令牌
            token = authorization.replace("Bearer ", "")
            user = self.auth_service.get_user_from_token(token)

            if user:
                request.state.user = user
            else:
                request.state.user = self.auth_service.users.get("guest")

        except Exception:
            request.state.user = self.auth_service.users.get("guest")

        return await call_next(request)


# 依赖注入函数
def get_current_user(request: Request) -> User:
    """获取当前用户"""
    user = getattr(request.state, 'user', None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user


def require_permission(permission: Permission):
    """权限装饰器"""
    def decorator(user: User = Depends(get_current_user)):
        if not user or permission not in user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission.value}' required"
            )
        return user
    return decorator


def require_role(role: UserRole):
    """角色装饰器"""
    def decorator(user: User = Depends(get_current_user)):
        auth_service = AuthService()  # 这里应该通过依赖注入获取
        if not user or not auth_service.check_role(user, role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role.value}' required"
            )
        return user
    return decorator