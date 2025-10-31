"""
Memory Cache Implementation

High-performance in-memory cache with LRU eviction and size limits.
"""

import time
import threading
from typing import Any, Dict, Optional
from collections import OrderedDict
from datetime import datetime, timedelta

class MemoryCache:
    """Thread-safe in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict = OrderedDict()
        self._expiry: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Any:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                return None
            
            # Check expiry
            if key in self._expiry and time.time() > self._expiry[key]:
                del self._cache[key]
                del self._expiry[key]
                return None
            
            # Move to end (most recently used)
            value = self._cache[key]
            del self._cache[key]
            self._cache[key] = value
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        with self._lock:
            # Remove if exists
            if key in self._cache:
                del self._cache[key]
            
            # Evict if necessary
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._expiry.pop(oldest_key, None)
            
            # Add new entry
            self._cache[key] = value
            
            # Set expiry
            if ttl is None:
                ttl = self.default_ttl
            
            if ttl > 0:
                self._expiry[key] = time.time() + ttl
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._expiry.pop(key, None)
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return self.get(key) is not None
    
    def clear(self):
        """Clear all entries"""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, expiry in self._expiry.items()
                if current_time > expiry
            ]
            
            for key in expired_keys:
                self._cache.pop(key, None)
                self._expiry.pop(key, None)
            
            return len(expired_keys)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "fill_ratio": len(self._cache) / self.max_size,
                "expired_count": sum(
                    1 for expiry in self._expiry.values()
                    if time.time() > expiry
                )
            }

# Global instance
_memory_cache: Optional[MemoryCache] = None

def get_memory_cache() -> MemoryCache:
    """Get global memory cache instance"""
    global _memory_cache
    if _memory_cache is None:
        _memory_cache = MemoryCache()
    return _memory_cache