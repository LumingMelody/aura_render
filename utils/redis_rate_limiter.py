"""
Redis-based Distributed Rate Limiter

Implements distributed rate limiting using Redis with:
- Sliding window algorithm
- Token bucket algorithm
- Fixed window algorithm
- Multi-tier rate limiting
- Real-time monitoring and metrics
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_second: int = 10
    requests_per_minute: int = 600
    requests_per_hour: int = 36000
    requests_per_day: int = 864000
    burst_size: int = 50
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    
    # Redis settings
    redis_url: str = "redis://localhost:6379/0"
    key_prefix: str = "rate_limit"
    ttl_seconds: int = 86400  # 24 hours
    
    # Distributed settings
    enable_distributed: bool = True
    node_id: Optional[str] = None


@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[int] = None
    current_usage: int = 0
    limit: int = 0
    algorithm: str = ""
    metadata: Dict[str, Any] = None


class RedisRateLimiter:
    """Redis-based distributed rate limiter"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.redis_client = None
        self.connection_pool = None
        self.lua_scripts = {}
        self._initialize_lua_scripts()
        
    async def initialize(self):
        """Initialize Redis connection"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, rate limiting will be disabled")
            return False
            
        try:
            # Create connection pool
            self.connection_pool = aioredis.ConnectionPool.from_url(
                self.config.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                decode_responses=True
            )
            
            self.redis_client = aioredis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis rate limiter initialized successfully")
            
            # Load Lua scripts
            await self._load_lua_scripts()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis rate limiter: {e}")
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.aclose()
        if self.connection_pool:
            await self.connection_pool.disconnect()
    
    def _initialize_lua_scripts(self):
        """Initialize Lua scripts for atomic operations"""
        
        # Sliding window rate limiter script
        self.lua_scripts['sliding_window'] = """
            local key = KEYS[1]
            local window = tonumber(ARGV[1])
            local limit = tonumber(ARGV[2])
            local current_time = tonumber(ARGV[3])
            local ttl = tonumber(ARGV[4])
            
            -- Remove expired entries
            redis.call('ZREMRANGEBYSCORE', key, 0, current_time - window)
            
            -- Count current requests
            local current_count = redis.call('ZCARD', key)
            
            if current_count < limit then
                -- Add current request
                redis.call('ZADD', key, current_time, current_time)
                redis.call('EXPIRE', key, ttl)
                return {1, limit - current_count - 1, current_time + window}
            else
                -- Get oldest entry for reset time calculation
                local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
                local reset_time = current_time + window
                if #oldest > 0 then
                    reset_time = tonumber(oldest[2]) + window
                end
                return {0, 0, reset_time}
            end
        """
        
        # Token bucket script
        self.lua_scripts['token_bucket'] = """
            local key = KEYS[1]
            local capacity = tonumber(ARGV[1])
            local refill_rate = tonumber(ARGV[2])
            local current_time = tonumber(ARGV[3])
            local requested_tokens = tonumber(ARGV[4])
            local ttl = tonumber(ARGV[5])
            
            -- Get current bucket state
            local bucket_data = redis.call('HMGET', key, 'tokens', 'last_refill')
            local current_tokens = tonumber(bucket_data[1]) or capacity
            local last_refill = tonumber(bucket_data[2]) or current_time
            
            -- Calculate tokens to add based on time passed
            local time_passed = current_time - last_refill
            local tokens_to_add = math.floor(time_passed * refill_rate)
            current_tokens = math.min(capacity, current_tokens + tokens_to_add)
            
            if current_tokens >= requested_tokens then
                -- Grant request
                current_tokens = current_tokens - requested_tokens
                redis.call('HMSET', key, 'tokens', current_tokens, 'last_refill', current_time)
                redis.call('EXPIRE', key, ttl)
                return {1, current_tokens, current_time + (capacity - current_tokens) / refill_rate}
            else
                -- Deny request
                redis.call('HMSET', key, 'tokens', current_tokens, 'last_refill', current_time)
                redis.call('EXPIRE', key, ttl)
                local wait_time = (requested_tokens - current_tokens) / refill_rate
                return {0, current_tokens, current_time + wait_time}
            end
        """
        
        # Fixed window script
        self.lua_scripts['fixed_window'] = """
            local key = KEYS[1]
            local limit = tonumber(ARGV[1])
            local window = tonumber(ARGV[2])
            local current_time = tonumber(ARGV[3])
            local ttl = tonumber(ARGV[4])
            
            -- Calculate window start
            local window_start = math.floor(current_time / window) * window
            local window_key = key .. ':' .. window_start
            
            -- Get current count
            local current_count = tonumber(redis.call('GET', window_key)) or 0
            
            if current_count < limit then
                -- Increment counter
                local new_count = redis.call('INCR', window_key)
                redis.call('EXPIRE', window_key, ttl)
                return {1, limit - new_count, window_start + window}
            else
                return {0, 0, window_start + window}
            end
        """
    
    async def _load_lua_scripts(self):
        """Load Lua scripts into Redis"""
        if not self.redis_client:
            return
            
        for script_name, script_code in self.lua_scripts.items():
            try:
                script_sha = await self.redis_client.script_load(script_code)
                self.lua_scripts[script_name] = {
                    'code': script_code,
                    'sha': script_sha
                }
                logger.debug(f"Loaded Lua script {script_name}: {script_sha}")
            except Exception as e:
                logger.error(f"Failed to load Lua script {script_name}: {e}")
    
    async def check_rate_limit(
        self,
        identifier: str,
        resource: str = "default",
        custom_limits: Optional[Dict[str, int]] = None
    ) -> RateLimitResult:
        """
        Check if request is allowed under rate limits
        
        Args:
            identifier: Client identifier (IP, user ID, etc.)
            resource: Resource being accessed
            custom_limits: Optional custom limits override
            
        Returns:
            RateLimitResult with decision and metadata
        """
        if not self.redis_client:
            # Fallback: allow all requests if Redis is not available
            return RateLimitResult(
                allowed=True,
                remaining=999,
                reset_time=time.time() + 60,
                algorithm="fallback"
            )
        
        # Generate rate limit key
        key = self._generate_key(identifier, resource)
        current_time = time.time()
        
        try:
            # Check multiple time windows
            results = []
            
            # Per-second limit
            if self.config.requests_per_second > 0:
                result = await self._check_window(
                    key + ":second",
                    1,  # 1 second window
                    custom_limits.get("per_second", self.config.requests_per_second) if custom_limits else self.config.requests_per_second,
                    current_time
                )
                results.append(("second", result))
            
            # Per-minute limit
            if self.config.requests_per_minute > 0:
                result = await self._check_window(
                    key + ":minute",
                    60,  # 1 minute window
                    custom_limits.get("per_minute", self.config.requests_per_minute) if custom_limits else self.config.requests_per_minute,
                    current_time
                )
                results.append(("minute", result))
            
            # Per-hour limit
            if self.config.requests_per_hour > 0:
                result = await self._check_window(
                    key + ":hour",
                    3600,  # 1 hour window
                    custom_limits.get("per_hour", self.config.requests_per_hour) if custom_limits else self.config.requests_per_hour,
                    current_time
                )
                results.append(("hour", result))
            
            # Find the most restrictive result
            allowed = True
            min_remaining = float('inf')
            earliest_reset = float('inf')
            limiting_window = None
            
            for window_name, (is_allowed, remaining, reset_time) in results:
                if not is_allowed:
                    allowed = False
                if remaining < min_remaining:
                    min_remaining = remaining
                    limiting_window = window_name
                if reset_time < earliest_reset:
                    earliest_reset = reset_time
            
            # Calculate retry_after if blocked
            retry_after = None
            if not allowed:
                retry_after = max(0, int(earliest_reset - current_time))
            
            return RateLimitResult(
                allowed=allowed,
                remaining=int(min_remaining) if min_remaining != float('inf') else 0,
                reset_time=earliest_reset if earliest_reset != float('inf') else current_time + 60,
                retry_after=retry_after,
                current_usage=0,  # Will be populated by monitoring
                limit=self.config.requests_per_minute,  # Primary limit for display
                algorithm=self.config.algorithm.value,
                metadata={
                    "windows_checked": len(results),
                    "limiting_window": limiting_window,
                    "node_id": self.config.node_id
                }
            )
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request but log error
            return RateLimitResult(
                allowed=True,
                remaining=0,
                reset_time=current_time + 60,
                algorithm="error_fallback",
                metadata={"error": str(e)}
            )
    
    async def _check_window(
        self,
        key: str,
        window_seconds: int,
        limit: int,
        current_time: float
    ) -> Tuple[bool, int, float]:
        """Check rate limit for a specific time window"""
        
        if self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return await self._check_sliding_window(key, window_seconds, limit, current_time)
        elif self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return await self._check_token_bucket(key, window_seconds, limit, current_time)
        elif self.config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return await self._check_fixed_window(key, window_seconds, limit, current_time)
        else:
            # Default to sliding window
            return await self._check_sliding_window(key, window_seconds, limit, current_time)
    
    async def _check_sliding_window(
        self,
        key: str,
        window_seconds: int,
        limit: int,
        current_time: float
    ) -> Tuple[bool, int, float]:
        """Check sliding window rate limit"""
        try:
            script_data = self.lua_scripts['sliding_window']
            result = await self.redis_client.evalsha(
                script_data['sha'],
                1,
                key,
                window_seconds,
                limit,
                current_time,
                self.config.ttl_seconds
            )
            
            return bool(result[0]), int(result[1]), float(result[2])
            
        except Exception as e:
            logger.error(f"Sliding window check failed: {e}")
            return True, limit, current_time + window_seconds
    
    async def _check_token_bucket(
        self,
        key: str,
        window_seconds: int,
        limit: int,
        current_time: float
    ) -> Tuple[bool, int, float]:
        """Check token bucket rate limit"""
        try:
            # Token bucket parameters
            capacity = limit
            refill_rate = limit / window_seconds  # Tokens per second
            
            script_data = self.lua_scripts['token_bucket']
            result = await self.redis_client.evalsha(
                script_data['sha'],
                1,
                key,
                capacity,
                refill_rate,
                current_time,
                1,  # Request 1 token
                self.config.ttl_seconds
            )
            
            return bool(result[0]), int(result[1]), float(result[2])
            
        except Exception as e:
            logger.error(f"Token bucket check failed: {e}")
            return True, limit, current_time + window_seconds
    
    async def _check_fixed_window(
        self,
        key: str,
        window_seconds: int,
        limit: int,
        current_time: float
    ) -> Tuple[bool, int, float]:
        """Check fixed window rate limit"""
        try:
            script_data = self.lua_scripts['fixed_window']
            result = await self.redis_client.evalsha(
                script_data['sha'],
                1,
                key,
                limit,
                window_seconds,
                current_time,
                self.config.ttl_seconds
            )
            
            return bool(result[0]), int(result[1]), float(result[2])
            
        except Exception as e:
            logger.error(f"Fixed window check failed: {e}")
            return True, limit, current_time + window_seconds
    
    def _generate_key(self, identifier: str, resource: str) -> str:
        """Generate Redis key for rate limiting"""
        # Create a hash for long identifiers to avoid key length issues
        id_hash = hashlib.md5(f"{identifier}:{resource}".encode()).hexdigest()[:16]
        return f"{self.config.key_prefix}:{id_hash}"
    
    async def get_usage_stats(self, identifier: str, resource: str = "default") -> Dict[str, Any]:
        """Get current usage statistics for an identifier"""
        if not self.redis_client:
            return {}
        
        key = self._generate_key(identifier, resource)
        current_time = time.time()
        stats = {}
        
        try:
            # Get stats for each time window
            for window_name, window_seconds in [("second", 1), ("minute", 60), ("hour", 3600)]:
                window_key = f"{key}:{window_name}"
                
                if self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                    # Count entries in sliding window
                    count = await self.redis_client.zcount(
                        window_key,
                        current_time - window_seconds,
                        current_time
                    )
                    stats[window_name] = {
                        "current_usage": count,
                        "limit": getattr(self.config, f"requests_per_{window_name}"),
                        "remaining": max(0, getattr(self.config, f"requests_per_{window_name}") - count)
                    }
                
                elif self.config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                    # Get fixed window count
                    window_start = int(current_time // window_seconds) * window_seconds
                    count = await self.redis_client.get(f"{window_key}:{window_start}") or 0
                    stats[window_name] = {
                        "current_usage": int(count),
                        "limit": getattr(self.config, f"requests_per_{window_name}"),
                        "remaining": max(0, getattr(self.config, f"requests_per_{window_name}") - int(count))
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return {}
    
    async def reset_limits(self, identifier: str, resource: str = "default"):
        """Reset rate limits for an identifier"""
        if not self.redis_client:
            return
        
        key = self._generate_key(identifier, resource)
        
        try:
            # Delete all related keys
            pattern = f"{key}:*"
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
            
            logger.info(f"Reset rate limits for {identifier}:{resource}")
            
        except Exception as e:
            logger.error(f"Failed to reset limits: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the rate limiter"""
        health = {
            "status": "healthy",
            "redis_connected": False,
            "lua_scripts_loaded": 0,
            "error": None
        }
        
        try:
            if self.redis_client:
                # Test Redis connection
                await self.redis_client.ping()
                health["redis_connected"] = True
                
                # Check loaded scripts
                health["lua_scripts_loaded"] = len([
                    s for s in self.lua_scripts.values() 
                    if isinstance(s, dict) and 'sha' in s
                ])
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health


# Factory function
def create_redis_rate_limiter(config: Optional[Dict[str, Any]] = None) -> RedisRateLimiter:
    """Create a Redis rate limiter with configuration"""
    rate_config = RateLimitConfig()
    
    if config:
        for key, value in config.items():
            if hasattr(rate_config, key):
                setattr(rate_config, key, value)
    
    return RedisRateLimiter(rate_config)


# Usage example and testing
async def test_rate_limiter():
    """Test the Redis rate limiter"""
    config = RateLimitConfig(
        requests_per_second=5,
        requests_per_minute=100,
        requests_per_hour=1000,
        algorithm=RateLimitAlgorithm.SLIDING_WINDOW
    )
    
    limiter = RedisRateLimiter(config)
    
    if await limiter.initialize():
        # Test rate limiting
        for i in range(10):
            result = await limiter.check_rate_limit("test_user", "api_call")
            print(f"Request {i+1}: {'ALLOWED' if result.allowed else 'BLOCKED'}, "
                  f"Remaining: {result.remaining}, "
                  f"Reset: {result.reset_time}")
            
            if not result.allowed:
                print(f"Retry after: {result.retry_after} seconds")
            
            await asyncio.sleep(0.1)
        
        # Get usage stats
        stats = await limiter.get_usage_stats("test_user", "api_call")
        print(f"Usage stats: {stats}")
        
        await limiter.close()
    
    else:
        print("Failed to initialize Redis rate limiter")


if __name__ == "__main__":
    asyncio.run(test_rate_limiter())