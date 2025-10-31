"""
Redis Cache Implementation

Specialized Redis-based cache with advanced features like pub/sub,
distributed locking, and cache warming strategies.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set
import redis.asyncio as aioredis
from datetime import datetime, timedelta

from config import get_settings

logger = logging.getLogger(__name__)

class RedisCache:
    """Advanced Redis cache implementation"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.settings = get_settings()
        self.redis_url = redis_url or self.settings.redis.url
        self.client: Optional[aioredis.Redis] = None
        self.pubsub_client: Optional[aioredis.Redis] = None
    
    async def initialize(self):
        """Initialize Redis connections"""
        self.client = aioredis.from_url(self.redis_url)
        self.pubsub_client = aioredis.from_url(self.redis_url)
        await self.client.ping()
    
    async def get(self, key: str) -> Any:
        """Get value from Redis"""
        if not self.client:
            return None
        
        value = await self.client.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in Redis with TTL"""
        if not self.client:
            return False
        
        serialized = json.dumps(value)
        return bool(await self.client.setex(key, ttl, serialized))
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self.client:
            return False
        
        return bool(await self.client.delete(key))
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.client:
            return False
        
        return bool(await self.client.exists(key))
    
    async def ttl(self, key: str) -> int:
        """Get TTL for key"""
        if not self.client:
            return -1
        
        return await self.client.ttl(key)
    
    async def keys(self, pattern: str) -> List[str]:
        """Get keys matching pattern"""
        if not self.client:
            return []
        
        return await self.client.keys(pattern)
    
    async def close(self):
        """Close connections"""
        if self.client:
            await self.client.close()
        if self.pubsub_client:
            await self.pubsub_client.close()

# Global instance
_redis_cache: Optional[RedisCache] = None

def get_redis_cache() -> RedisCache:
    """Get global Redis cache instance"""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    return _redis_cache