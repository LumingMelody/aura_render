"""
API Rate Limiting and Security

Advanced rate limiting, request throttling, and API security measures
to protect against abuse and ensure fair usage.
"""

import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

import redis
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from analytics import get_metrics_collector


class RateLimitType(str, Enum):
    """Types of rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class SecurityLevel(str, Enum):
    """Security levels for different endpoints"""
    PUBLIC = "public"           # No authentication required
    AUTHENTICATED = "authenticated"  # Requires valid token
    PREMIUM = "premium"         # Requires premium subscription
    ADMIN = "admin"            # Requires admin role


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests: int              # Number of requests allowed
    window_seconds: int        # Time window in seconds
    burst_multiplier: float = 1.5  # Allow burst up to this multiplier
    block_duration: int = 300  # Block duration when limit exceeded (seconds)
    
    def __post_init__(self):
        self.burst_limit = int(self.requests * self.burst_multiplier)


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    current_usage: int = 0
    
    def to_headers(self) -> Dict[str, str]:
        headers = {
            "X-RateLimit-Limit": str(self.current_usage + self.remaining),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_time.timestamp()))
        }
        
        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)
            
        return headers


class RateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.metrics = get_metrics_collector()
        self.logger = logging.getLogger(__name__)
        
        # Rate limit configurations for different endpoints
        self.configs: Dict[str, RateLimitConfig] = {
            # Public endpoints
            "public": RateLimitConfig(requests=100, window_seconds=3600),  # 100/hour
            "health": RateLimitConfig(requests=60, window_seconds=60),     # 1/second
            
            # Authentication endpoints
            "auth_login": RateLimitConfig(requests=5, window_seconds=300, block_duration=900),  # 5/5min
            "auth_register": RateLimitConfig(requests=3, window_seconds=3600),  # 3/hour
            
            # API endpoints
            "api_basic": RateLimitConfig(requests=1000, window_seconds=3600),   # 1000/hour
            "api_premium": RateLimitConfig(requests=5000, window_seconds=3600),  # 5000/hour
            "api_admin": RateLimitConfig(requests=10000, window_seconds=3600),   # 10000/hour
            
            # Video generation (expensive operations)
            "video_generation": RateLimitConfig(requests=10, window_seconds=3600, block_duration=1800),  # 10/hour
            "batch_processing": RateLimitConfig(requests=5, window_seconds=3600, block_duration=3600),   # 5/hour
            
            # Image generation
            "image_generation": RateLimitConfig(requests=100, window_seconds=3600),  # 100/hour
            
            # Export operations
            "export": RateLimitConfig(requests=50, window_seconds=3600),  # 50/hour
            
            # WebSocket connections
            "websocket": RateLimitConfig(requests=10, window_seconds=60),  # 10/minute
        }
        
        # Blocked IPs and users
        self.blocked_ips: Dict[str, datetime] = {}
        self.blocked_users: Dict[str, datetime] = {}
        
        # Security patterns
        self.suspicious_patterns = [
            r"(?i)(union|select|insert|delete|drop|create|alter|exec)",  # SQL injection
            r"<script|javascript:|vbscript:",  # XSS
            r"\.\.\/|\.\.\\",  # Path traversal
            r"(cmd|powershell|bash|sh)\s",  # Command injection
        ]
        
        # API key management
        self.api_keys: Dict[str, Dict[str, Any]] = {}
    
    async def check_rate_limit(
        self,
        key: str,
        config_name: str,
        request: Optional[Request] = None
    ) -> RateLimitResult:
        """Check rate limit for a given key and configuration"""
        config = self.configs.get(config_name)
        if not config:
            # Default to basic rate limit if config not found
            config = self.configs["api_basic"]
        
        try:
            # Check if key is currently blocked
            if await self._is_blocked(key, config_name):
                remaining_block_time = await self._get_remaining_block_time(key, config_name)
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=datetime.now() + timedelta(seconds=remaining_block_time),
                    retry_after=remaining_block_time,
                    current_usage=config.requests
                )
            
            # Use sliding window rate limiting
            result = await self._sliding_window_check(key, config_name, config)
            
            # Record metrics
            await self.metrics.increment_counter(
                f"rate_limit.{config_name}.{'allowed' if result.allowed else 'blocked'}"
            )
            
            if not result.allowed:
                await self.metrics.increment_counter(f"rate_limit.{config_name}.limit_exceeded")
                
                # Block key if limit exceeded
                await self._block_key(key, config_name, config.block_duration)
                
                # Log rate limit violation
                self.logger.warning(
                    f"Rate limit exceeded for key {key[:10]}... in {config_name}"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Rate limit check failed for {key}: {e}")
            # Fail open - allow request if rate limiter fails
            return RateLimitResult(
                allowed=True,
                remaining=config.requests,
                reset_time=datetime.now() + timedelta(seconds=config.window_seconds)
            )
    
    async def _sliding_window_check(
        self,
        key: str,
        config_name: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Implement sliding window rate limiting using Redis"""
        now = time.time()
        window_start = now - config.window_seconds
        
        # Redis key for this rate limit
        redis_key = f"rate_limit:{config_name}:{key}"
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        # Remove expired entries
        pipe.zremrangebyscore(redis_key, 0, window_start)
        
        # Count current requests
        pipe.zcard(redis_key)
        
        # Add current request timestamp
        pipe.zadd(redis_key, {str(now): now})
        
        # Set expiry for the key
        pipe.expire(redis_key, config.window_seconds + 1)
        
        # Execute pipeline
        results = pipe.execute()
        current_count = results[1]  # Number of requests in current window
        
        # Calculate remaining requests
        remaining = max(0, config.requests - current_count)
        allowed = current_count < config.requests
        
        # Calculate reset time (start of next window)
        reset_time = datetime.fromtimestamp(now + config.window_seconds)
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            current_usage=current_count
        )
    
    async def _is_blocked(self, key: str, config_name: str) -> bool:
        """Check if a key is currently blocked"""
        block_key = f"blocked:{config_name}:{key}"
        return await asyncio.to_thread(self.redis.exists, block_key)
    
    async def _get_remaining_block_time(self, key: str, config_name: str) -> int:
        """Get remaining block time for a key"""
        block_key = f"blocked:{config_name}:{key}"
        ttl = await asyncio.to_thread(self.redis.ttl, block_key)
        return max(0, ttl)
    
    async def _block_key(self, key: str, config_name: str, duration: int):
        """Block a key for a specified duration"""
        block_key = f"blocked:{config_name}:{key}"
        await asyncio.to_thread(self.redis.setex, block_key, duration, "1")
        
        self.logger.warning(f"Blocked key {key[:10]}... for {duration} seconds")
    
    async def check_security_patterns(self, request: Request) -> bool:
        """Check request for suspicious patterns"""
        import re
        
        # Check URL path
        path = str(request.url.path)
        query = str(request.url.query) if request.url.query else ""
        
        # Check headers
        user_agent = request.headers.get("user-agent", "")
        
        # Combine all text to check
        text_to_check = f"{path} {query} {user_agent}".lower()
        
        # Check against suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text_to_check):
                await self.metrics.increment_counter("security.suspicious_pattern_detected")
                self.logger.warning(
                    f"Suspicious pattern detected: {pattern} in request from "
                    f"{request.client.host if request.client else 'unknown'}"
                )
                return False
        
        return True
    
    def get_client_identifier(self, request: Request) -> str:
        """Get unique identifier for the client"""
        # Try to get user ID from request context (if authenticated)
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            return f"user:{user_id}"
        
        # Get API key if present
        api_key = request.headers.get("x-api-key")
        if api_key:
            # Hash the API key for privacy
            return f"api:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        
        # Fall back to IP address
        client_ip = "unknown"
        if request.client:
            client_ip = request.client.host
        
        # Check for forwarded IP headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            client_ip = real_ip.strip()
        
        return f"ip:{client_ip}"
    
    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return associated metadata"""
        # Hash the API key for storage/lookup
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Check Redis for API key info
        api_key_info = await asyncio.to_thread(
            self.redis.hgetall, f"api_key:{key_hash}"
        )
        
        if not api_key_info:
            return None
        
        # Check if API key is active
        if api_key_info.get("status") != "active":
            return None
        
        # Check expiration
        expires_at = api_key_info.get("expires_at")
        if expires_at and datetime.fromisoformat(expires_at) < datetime.now():
            return None
        
        return {
            "user_id": api_key_info.get("user_id"),
            "name": api_key_info.get("name"),
            "permissions": api_key_info.get("permissions", "").split(","),
            "rate_limit_tier": api_key_info.get("rate_limit_tier", "api_basic"),
            "created_at": api_key_info.get("created_at"),
            "last_used": api_key_info.get("last_used")
        }
    
    async def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: List[str],
        rate_limit_tier: str = "api_basic",
        expires_days: Optional[int] = None
    ) -> str:
        """Create a new API key"""
        # Generate secure API key
        api_key = f"ak_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Set expiration if specified
        expires_at = None
        if expires_days:
            expires_at = (datetime.now() + timedelta(days=expires_days)).isoformat()
        
        # Store API key info in Redis
        key_info = {
            "user_id": user_id,
            "name": name,
            "permissions": ",".join(permissions),
            "rate_limit_tier": rate_limit_tier,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "last_used": "",
        }
        
        if expires_at:
            key_info["expires_at"] = expires_at
        
        await asyncio.to_thread(
            self.redis.hset, f"api_key:{key_hash}", mapping=key_info
        )
        
        # Add to user's API keys list
        await asyncio.to_thread(
            self.redis.sadd, f"user_api_keys:{user_id}", key_hash
        )
        
        self.logger.info(f"Created API key for user {user_id}: {name}")
        
        return api_key
    
    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Get key info first
        key_info = await asyncio.to_thread(
            self.redis.hgetall, f"api_key:{key_hash}"
        )
        
        if not key_info:
            return False
        
        # Mark as revoked
        await asyncio.to_thread(
            self.redis.hset, f"api_key:{key_hash}", "status", "revoked"
        )
        
        user_id = key_info.get("user_id")
        if user_id:
            await asyncio.to_thread(
                self.redis.srem, f"user_api_keys:{user_id}", key_hash
            )
        
        self.logger.info(f"Revoked API key: {key_info.get('name', 'unknown')}")
        
        return True
    
    async def update_api_key_usage(self, api_key: str):
        """Update API key last used timestamp"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        await asyncio.to_thread(
            self.redis.hset,
            f"api_key:{key_hash}",
            "last_used",
            datetime.now().isoformat()
        )
    
    async def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        stats = {}
        
        for config_name in self.configs.keys():
            allowed = await self.metrics.get_counter(f"rate_limit.{config_name}.allowed")
            blocked = await self.metrics.get_counter(f"rate_limit.{config_name}.blocked")
            
            total = allowed + blocked
            block_rate = (blocked / total * 100) if total > 0 else 0
            
            stats[config_name] = {
                "total_requests": total,
                "allowed_requests": allowed,
                "blocked_requests": blocked,
                "block_rate_percent": round(block_rate, 2)
            }
        
        return stats


class SecurityMiddleware:
    """Security middleware for FastAPI applications"""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger(__name__)
        
        # Define rate limit tiers for different endpoint patterns
        self.endpoint_configs = {
            r"^/health": "health",
            r"^/api/auth/login": "auth_login",
            r"^/api/auth/register": "auth_register",
            r"^/api/tasks/video": "video_generation",
            r"^/api/batch": "batch_processing",
            r"^/api/image": "image_generation",
            r"^/api/export": "export",
            r"^/ws": "websocket",
            r"^/api": "api_basic",  # Default for API endpoints
            r".*": "public"  # Default for all other endpoints
        }
    
    async def __call__(self, request: Request, call_next):
        """Security middleware implementation"""
        start_time = time.time()
        
        try:
            # Check security patterns first
            if not await self.rate_limiter.check_security_patterns(request):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Request contains suspicious patterns"
                )
            
            # Get client identifier
            client_id = self.rate_limiter.get_client_identifier(request)
            
            # Determine rate limit configuration
            config_name = self._get_config_for_path(request.url.path)
            
            # Check API key if present
            api_key = request.headers.get("x-api-key")
            if api_key:
                api_key_info = await self.rate_limiter.validate_api_key(api_key)
                if not api_key_info:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid API key"
                    )
                
                # Use API key rate limit tier
                config_name = api_key_info.get("rate_limit_tier", config_name)
                client_id = f"api_user:{api_key_info['user_id']}"
                
                # Store user info in request state
                request.state.user_id = api_key_info["user_id"]
                request.state.api_key_info = api_key_info
                
                # Update API key usage
                await self.rate_limiter.update_api_key_usage(api_key)
            
            # Check rate limit
            rate_limit_result = await self.rate_limiter.check_rate_limit(
                client_id, config_name, request
            )
            
            if not rate_limit_result.allowed:
                # Add rate limit headers
                headers = rate_limit_result.to_headers()
                
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers=headers
                )
            
            # Process the request
            response = await call_next(request)
            
            # Add rate limit headers to response
            headers = rate_limit_result.to_headers()
            for key, value in headers.items():
                response.headers[key] = value
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            # Record response time metric
            response_time = (time.time() - start_time) * 1000
            await self.rate_limiter.metrics.record_timer(
                f"request.{config_name}.response_time",
                response_time
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Security middleware error: {e}")
            # Continue with request processing if middleware fails
            return await call_next(request)
    
    def _get_config_for_path(self, path: str) -> str:
        """Get rate limit configuration name for a given path"""
        import re
        
        for pattern, config_name in self.endpoint_configs.items():
            if re.match(pattern, path):
                return config_name
        
        return "public"


# Global rate limiter instance
_rate_limiter = None

def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter