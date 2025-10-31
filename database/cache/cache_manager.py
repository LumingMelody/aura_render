"""
Advanced Cache Manager

Intelligent multi-layer cache management with Redis backend,
automatic key generation, cache warming, and performance optimization.
"""

import asyncio
import json
import pickle
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import redis.asyncio as aioredis
import zlib

from config import get_settings
from monitoring import get_error_handler, get_metrics_collector
from monitoring.error_handler import ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache levels for multi-layer caching"""
    MEMORY = "memory"
    REDIS = "redis"
    DISTRIBUTED = "distributed"

class CacheStrategy(Enum):
    """Cache invalidation strategies"""
    TTL = "ttl"              # Time-to-live
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    REFRESH_AHEAD = "refresh_ahead"

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    ttl: Optional[int] = None
    hit_count: int = 0
    size_bytes: int = 0
    compressed: bool = False
    level: CacheLevel = CacheLevel.REDIS
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if not self.ttl:
            return False
        return datetime.now(timezone.utc) > self.created_at + timedelta(seconds=self.ttl)
    
    @property
    def age_seconds(self) -> int:
        """Get age of cache entry in seconds"""
        return int((datetime.now(timezone.utc) - self.created_at).total_seconds())

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    avg_response_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return 100.0 - self.hit_rate

class CacheKeyBuilder:
    """Intelligent cache key generation"""
    
    def __init__(self, prefix: str = "aura_cache", separator: str = ":"):
        self.prefix = prefix
        self.separator = separator
    
    def build_key(self, *parts: Any, namespace: Optional[str] = None) -> str:
        """Build cache key from parts"""
        key_parts = [self.prefix]
        
        if namespace:
            key_parts.append(namespace)
        
        for part in parts:
            if isinstance(part, dict):
                # Sort dict items for consistent key generation
                sorted_items = sorted(part.items())
                part_str = hashlib.md5(json.dumps(sorted_items, sort_keys=True).encode()).hexdigest()[:8]
            elif hasattr(part, '__dict__'):
                # For objects, use their dict representation
                part_str = hashlib.md5(str(part.__dict__).encode()).hexdigest()[:8]
            else:
                part_str = str(part)
            
            key_parts.append(part_str)
        
        return self.separator.join(key_parts)
    
    def build_pattern(self, *parts: Any, namespace: Optional[str] = None) -> str:
        """Build cache key pattern for bulk operations"""
        pattern_parts = [self.prefix]
        
        if namespace:
            pattern_parts.append(namespace)
        
        pattern_parts.extend([str(part) if part != "*" else "*" for part in parts])
        return self.separator.join(pattern_parts)

class AdvancedCacheManager:
    """
    Advanced multi-layer cache manager with intelligent features:
    - Multi-level caching (memory + Redis)
    - Compression for large values
    - Cache warming and pre-loading
    - Smart invalidation patterns
    - Performance monitoring
    """
    
    def __init__(self, redis_url: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.settings = get_settings()
        self.error_handler = get_error_handler()
        self.metrics = get_metrics_collector()
        
        # Configuration
        self.config = config or {}
        self.redis_url = redis_url or self.settings.redis.url
        
        # Cache components
        self.redis_client: Optional[aioredis.Redis] = None
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.key_builder = CacheKeyBuilder()
        
        # Statistics
        self.stats = CacheStats()
        
        # Cache settings
        self.default_ttl = self.config.get('default_ttl', 3600)  # 1 hour
        self.max_memory_entries = self.config.get('max_memory_entries', 1000)
        self.compression_threshold = self.config.get('compression_threshold', 1024)  # bytes
        self.enable_compression = self.config.get('enable_compression', True)
        
        # Performance tracking
        self.response_times: List[float] = []
        
        logger.info("Advanced Cache Manager initialized")
    
    async def initialize(self):
        """Initialize cache connections"""
        try:
            # Initialize Redis connection
            self.redis_client = aioredis.from_url(
                self.redis_url,
                encoding='utf-8',
                decode_responses=False,  # We handle serialization
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis cache connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            await self.error_handler.handle_error(
                exception=e,
                category=ErrorCategory.CACHE,
                severity=ErrorSeverity.HIGH,
                context={"redis_url": self.redis_url}
            )
            raise
    
    async def get(self, key: str, default: Any = None, 
                  deserializer: Optional[Callable] = None) -> Any:
        """Get value from cache with multi-level lookup"""
        start_time = time.time()
        
        try:
            # Try memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.is_expired:
                    entry.accessed_at = datetime.now(timezone.utc)
                    entry.hit_count += 1
                    self.stats.hits += 1
                    
                    response_time = (time.time() - start_time) * 1000
                    self._update_response_time(response_time)
                    
                    return entry.value
                else:
                    # Remove expired entry
                    del self.memory_cache[key]
            
            # Try Redis cache
            if self.redis_client:
                cached_data = await self.redis_client.get(key)
                if cached_data:
                    value = self._deserialize_value(cached_data, deserializer)
                    
                    # Cache in memory for faster access
                    await self._store_in_memory(key, value, self.default_ttl)
                    
                    self.stats.hits += 1
                    
                    response_time = (time.time() - start_time) * 1000
                    self._update_response_time(response_time)
                    
                    return value
            
            # Cache miss
            self.stats.misses += 1
            
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            
            return default
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                  serializer: Optional[Callable] = None, compress: bool = True) -> bool:
        """Set value in cache with intelligent storage"""
        start_time = time.time()
        
        try:
            ttl = ttl or self.default_ttl
            
            # Serialize value
            serialized_value = self._serialize_value(value, serializer)
            
            # Determine if compression is needed
            should_compress = (compress and self.enable_compression and 
                             len(serialized_value) > self.compression_threshold)
            
            if should_compress:
                serialized_value = zlib.compress(serialized_value)
            
            # Store in Redis
            if self.redis_client:
                await self.redis_client.setex(key, ttl, serialized_value)
            
            # Store in memory cache
            await self._store_in_memory(key, value, ttl, compressed=should_compress)
            
            self.stats.sets += 1
            self.stats.total_size_bytes += len(serialized_value)
            
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            await self.error_handler.handle_error(
                exception=e,
                category=ErrorCategory.CACHE,
                severity=ErrorSeverity.MEDIUM,
                context={"key": key, "operation": "set"}
            )
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels"""
        try:
            deleted = False
            
            # Delete from memory
            if key in self.memory_cache:
                del self.memory_cache[key]
                deleted = True
            
            # Delete from Redis
            if self.redis_client:
                result = await self.redis_client.delete(key)
                deleted = deleted or bool(result)
            
            if deleted:
                self.stats.deletes += 1
            
            return deleted
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        try:
            cleared = 0
            
            if self.redis_client:
                # Get matching keys
                keys = await self.redis_client.keys(pattern)
                if keys:
                    cleared = await self.redis_client.delete(*keys)
                
                # Clear from memory cache
                memory_keys = [k for k in self.memory_cache.keys() 
                             if self._match_pattern(k, pattern)]
                for key in memory_keys:
                    del self.memory_cache[key]
                
                cleared += len(memory_keys)
            
            return cleared
            
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            # Check memory first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.is_expired:
                    return True
                else:
                    del self.memory_cache[key]
            
            # Check Redis
            if self.redis_client:
                return bool(await self.redis_client.exists(key))
            
            return False
            
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key"""
        try:
            if self.redis_client:
                ttl = await self.redis_client.ttl(key)
                return ttl if ttl > 0 else None
            return None
            
        except Exception as e:
            logger.error(f"Cache TTL error for key {key}: {e}")
            return None
    
    async def extend_ttl(self, key: str, ttl: int) -> bool:
        """Extend TTL for existing key"""
        try:
            if self.redis_client:
                return bool(await self.redis_client.expire(key, ttl))
            return False
            
        except Exception as e:
            logger.error(f"Cache extend TTL error for key {key}: {e}")
            return False
    
    async def get_multi(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple keys efficiently"""
        try:
            results = {}
            redis_keys = []
            
            # Check memory cache first
            for key in keys:
                if key in self.memory_cache:
                    entry = self.memory_cache[key]
                    if not entry.is_expired:
                        results[key] = entry.value
                        entry.accessed_at = datetime.now(timezone.utc)
                        entry.hit_count += 1
                    else:
                        del self.memory_cache[key]
                        redis_keys.append(key)
                else:
                    redis_keys.append(key)
            
            # Get remaining keys from Redis
            if redis_keys and self.redis_client:
                redis_values = await self.redis_client.mget(redis_keys)
                for key, value in zip(redis_keys, redis_values):
                    if value:
                        deserialized = self._deserialize_value(value)
                        results[key] = deserialized
                        # Cache in memory
                        await self._store_in_memory(key, deserialized, self.default_ttl)
            
            # Update stats
            self.stats.hits += len(results)
            self.stats.misses += len(keys) - len(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Cache get_multi error: {e}")
            return {}
    
    async def set_multi(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple key-value pairs efficiently"""
        try:
            ttl = ttl or self.default_ttl
            
            if self.redis_client:
                # Prepare pipeline
                pipeline = self.redis_client.pipeline()
                
                for key, value in mapping.items():
                    serialized_value = self._serialize_value(value)
                    pipeline.setex(key, ttl, serialized_value)
                    
                    # Store in memory
                    await self._store_in_memory(key, value, ttl)
                
                # Execute pipeline
                await pipeline.execute()
                
                self.stats.sets += len(mapping)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache set_multi error: {e}")
            return False
    
    async def warm_cache(self, warm_functions: List[Callable]) -> int:
        """Warm cache with pre-computed values"""
        warmed = 0
        
        try:
            for func in warm_functions:
                try:
                    if asyncio.iscoroutinefunction(func):
                        await func()
                    else:
                        func()
                    warmed += 1
                except Exception as e:
                    logger.warning(f"Cache warming function failed: {e}")
            
            logger.info(f"Cache warmed with {warmed} functions")
            return warmed
            
        except Exception as e:
            logger.error(f"Cache warming error: {e}")
            return warmed
    
    async def get_stats(self) -> CacheStats:
        """Get cache performance statistics"""
        # Update average response time
        if self.response_times:
            self.stats.avg_response_time_ms = sum(self.response_times) / len(self.response_times)
        
        # Update memory cache size
        memory_size = sum(entry.size_bytes for entry in self.memory_cache.values())
        self.stats.total_size_bytes = memory_size
        
        return self.stats
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries from memory cache"""
        expired_keys = []
        
        for key, entry in self.memory_cache.items():
            if entry.is_expired:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        if expired_keys:
            self.stats.evictions += len(expired_keys)
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    async def _store_in_memory(self, key: str, value: Any, ttl: int, compressed: bool = False):
        """Store entry in memory cache with LRU eviction"""
        # Check if we need to evict entries
        if len(self.memory_cache) >= self.max_memory_entries:
            # Remove least recently used entries
            lru_keys = sorted(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].accessed_at
            )[:10]  # Remove oldest 10 entries
            
            for lru_key in lru_keys:
                del self.memory_cache[lru_key]
                self.stats.evictions += 1
        
        # Calculate size
        size_bytes = len(pickle.dumps(value)) if not compressed else len(str(value).encode())
        
        # Store entry
        self.memory_cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(timezone.utc),
            accessed_at=datetime.now(timezone.utc),
            ttl=ttl,
            size_bytes=size_bytes,
            compressed=compressed,
            level=CacheLevel.MEMORY
        )
    
    def _serialize_value(self, value: Any, serializer: Optional[Callable] = None) -> bytes:
        """Serialize value for cache storage"""
        if serializer:
            return serializer(value)
        
        try:
            # Try JSON first (more readable, smaller)
            return json.dumps(value).encode('utf-8')
        except (TypeError, ValueError):
            # Fall back to pickle
            return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes, deserializer: Optional[Callable] = None) -> Any:
        """Deserialize value from cache"""
        if deserializer:
            return deserializer(data)
        
        # Check if compressed
        try:
            if data.startswith(b'x\x9c') or data.startswith(b'\x1f\x8b'):
                data = zlib.decompress(data)
        except:
            pass
        
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Simple pattern matching for cache keys"""
        if '*' not in pattern:
            return key == pattern
        
        # Convert pattern to regex-like matching
        import re
        regex_pattern = pattern.replace('*', '.*')
        return bool(re.match(regex_pattern, key))
    
    def _update_response_time(self, response_time_ms: float):
        """Update response time tracking"""
        self.response_times.append(response_time_ms)
        
        # Keep only last 1000 measurements
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
    
    async def close(self):
        """Close cache connections"""
        if self.redis_client:
            await self.redis_client.close()
        
        self.memory_cache.clear()
        logger.info("Cache manager closed")

# Global cache manager instance
_cache_manager: Optional[AdvancedCacheManager] = None

def get_cache_manager() -> AdvancedCacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        settings = get_settings()
        _cache_manager = AdvancedCacheManager(
            redis_url=settings.redis.url,
            config={
                'default_ttl': settings.cache.ttl,
                'enable_compression': settings.cache.enable_compression,
                'compression_threshold': settings.cache.compression_threshold
            }
        )
    return _cache_manager