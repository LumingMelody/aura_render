"""
Cache Management System

Redis-based caching layer for the Aura Render system.
"""

import json
import asyncio
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
import logging
from config import Settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis-based cache manager"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.redis_client = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize Redis connection"""
        if self._initialized:
            return
            
        try:
            # Parse Redis URL
            redis_config = self.settings.get_redis_config()
            
            self.redis_client = redis.from_url(
                redis_config["url"],
                max_connections=redis_config["max_connections"],
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            self._initialized = True
            logger.info("Cache manager initialized with Redis")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            # Fall back to in-memory cache
            self.redis_client = None
            self._memory_cache = {}
            self._initialized = True
            logger.warning("Using in-memory cache fallback")
            
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            
    # Basic cache operations
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._initialized:
            await self.initialize()
            
        try:
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                # Memory fallback
                return self._memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            
        return None
        
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        if not self._initialized:
            await self.initialize()
            
        ttl = ttl or self.settings.redis_ttl
        
        try:
            if self.redis_client:
                serialized_value = json.dumps(value, default=str)
                await self.redis_client.set(key, serialized_value, ex=ttl)
                return True
            else:
                # Memory fallback with expiration
                self._memory_cache[key] = {
                    'value': value,
                    'expires_at': datetime.now() + timedelta(seconds=ttl)
                }
                return True
                
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self._initialized:
            await self.initialize()
            
        try:
            if self.redis_client:
                result = await self.redis_client.delete(key)
                return bool(result)
            else:
                # Memory fallback
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    return True
                    
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            
        return False
        
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self._initialized:
            await self.initialize()
            
        try:
            if self.redis_client:
                return bool(await self.redis_client.exists(key))
            else:
                # Memory fallback with expiration check
                if key in self._memory_cache:
                    entry = self._memory_cache[key]
                    if datetime.now() < entry['expires_at']:
                        return True
                    else:
                        del self._memory_cache[key]
                        
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            
        return False
        
    # Pattern-based operations
    async def get_keys(self, pattern: str) -> List[str]:
        """Get keys matching pattern"""
        if not self._initialized:
            await self.initialize()
            
        try:
            if self.redis_client:
                keys = await self.redis_client.keys(pattern)
                return keys or []
            else:
                # Memory fallback - simple pattern matching
                import fnmatch
                return [
                    key for key in self._memory_cache.keys()
                    if fnmatch.fnmatch(key, pattern)
                ]
                
        except Exception as e:
            logger.error(f"Cache get_keys error for pattern {pattern}: {e}")
            return []
            
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        keys = await self.get_keys(pattern)
        if not keys:
            return 0
            
        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1
                
        return count
        
    # Hash operations
    async def hset(self, name: str, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set hash fields"""
        if not self._initialized:
            await self.initialize()
            
        ttl = ttl or self.settings.redis_ttl
        
        try:
            if self.redis_client:
                # Convert values to strings for Redis
                str_mapping = {k: json.dumps(v, default=str) for k, v in mapping.items()}
                await self.redis_client.hset(name, mapping=str_mapping)
                if ttl:
                    await self.redis_client.expire(name, ttl)
                return True
            else:
                # Memory fallback
                self._memory_cache[name] = {
                    'value': mapping,
                    'expires_at': datetime.now() + timedelta(seconds=ttl),
                    'type': 'hash'
                }
                return True
                
        except Exception as e:
            logger.error(f"Cache hset error for hash {name}: {e}")
            return False
            
    async def hget(self, name: str, key: str) -> Optional[Any]:
        """Get hash field"""
        if not self._initialized:
            await self.initialize()
            
        try:
            if self.redis_client:
                value = await self.redis_client.hget(name, key)
                if value:
                    return json.loads(value)
            else:
                # Memory fallback
                entry = self._memory_cache.get(name)
                if (entry and entry.get('type') == 'hash' and 
                    datetime.now() < entry['expires_at']):
                    return entry['value'].get(key)
                    
        except Exception as e:
            logger.error(f"Cache hget error for hash {name}, key {key}: {e}")
            
        return None
        
    async def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all hash fields"""
        if not self._initialized:
            await self.initialize()
            
        try:
            if self.redis_client:
                data = await self.redis_client.hgetall(name)
                if data:
                    return {k: json.loads(v) for k, v in data.items()}
            else:
                # Memory fallback
                entry = self._memory_cache.get(name)
                if (entry and entry.get('type') == 'hash' and 
                    datetime.now() < entry['expires_at']):
                    return entry['value']
                    
        except Exception as e:
            logger.error(f"Cache hgetall error for hash {name}: {e}")
            
        return {}
        
    # Application-specific cache methods
    async def cache_task_result(self, task_id: str, result: Dict[str, Any], ttl: int = 86400):
        """Cache task processing result"""
        key = f"task_result:{task_id}"
        return await self.set(key, result, ttl)
        
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get cached task result"""
        key = f"task_result:{task_id}"
        return await self.get(key)
        
    async def cache_material_search(
        self, 
        query: str, 
        material_type: str, 
        results: List[Dict[str, Any]], 
        ttl: int = 3600
    ):
        """Cache material search results"""
        key = f"material_search:{material_type}:{hash(query)}"
        return await self.set(key, results, ttl)
        
    async def get_material_search(self, query: str, material_type: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached material search results"""
        key = f"material_search:{material_type}:{hash(query)}"
        return await self.get(key)
        
    async def cache_ai_response(self, prompt_hash: str, response: Dict[str, Any], ttl: int = 7200):
        """Cache AI service response"""
        key = f"ai_response:{prompt_hash}"
        return await self.set(key, response, ttl)
        
    async def get_ai_response(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached AI response"""
        key = f"ai_response:{prompt_hash}"
        return await self.get(key)
        
    async def cache_render_progress(self, task_id: str, progress: float, message: str = ""):
        """Cache render progress"""
        key = f"render_progress:{task_id}"
        data = {
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        return await self.set(key, data, ttl=300)  # 5 minute TTL for progress
        
    async def get_render_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get cached render progress"""
        key = f"render_progress:{task_id}"
        return await self.get(key)
        
    # Cache statistics and maintenance
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._initialized:
            await self.initialize()
            
        stats = {
            "cache_type": "redis" if self.redis_client else "memory",
            "connected": bool(self.redis_client),
        }
        
        try:
            if self.redis_client:
                info = await self.redis_client.info()
                stats.update({
                    "used_memory": info.get("used_memory", 0),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                })
            else:
                # Memory cache stats
                valid_entries = 0
                expired_entries = 0
                now = datetime.now()
                
                for entry in self._memory_cache.values():
                    if isinstance(entry, dict) and 'expires_at' in entry:
                        if now < entry['expires_at']:
                            valid_entries += 1
                        else:
                            expired_entries += 1
                            
                stats.update({
                    "total_entries": len(self._memory_cache),
                    "valid_entries": valid_entries,
                    "expired_entries": expired_entries
                })
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            
        return stats
        
    async def clear_expired_memory_cache(self):
        """Clear expired entries from memory cache"""
        if self.redis_client:
            return  # Redis handles expiration automatically
            
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self._memory_cache.items():
            if isinstance(entry, dict) and 'expires_at' in entry:
                if now >= entry['expires_at']:
                    expired_keys.append(key)
                    
        for key in expired_keys:
            del self._memory_cache[key]
            
        logger.info(f"Cleared {len(expired_keys)} expired cache entries")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(settings: Optional[Settings] = None) -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(settings)
    return _cache_manager