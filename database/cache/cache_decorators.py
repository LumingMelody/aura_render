"""
Cache Decorators

Powerful caching decorators for functions and methods with intelligent
key generation, TTL management, and cache invalidation patterns.
"""

import asyncio
import functools
import hashlib
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

from .cache_manager import get_cache_manager, CacheKeyBuilder

logger = logging.getLogger(__name__)

def cache_key(*args, **kwargs):
    """Generate cache key from function arguments"""
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if hasattr(arg, '__dict__'):
            key_parts.append(str(hash(frozenset(arg.__dict__.items()))))
        else:
            key_parts.append(str(arg))
    
    # Add keyword arguments
    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        key_parts.append(str(hash(frozenset(sorted_kwargs))))
    
    return hashlib.md5('|'.join(key_parts).encode()).hexdigest()

def cached(
    ttl: int = 3600,
    namespace: Optional[str] = None,
    key_func: Optional[Callable] = None,
    condition: Optional[Callable] = None,
    unless: Optional[Callable] = None,
    compress: bool = True,
    invalidate_on_error: bool = False
):
    """
    Decorator for caching function results
    
    Args:
        ttl: Time-to-live in seconds
        namespace: Cache namespace for key isolation
        key_func: Custom key generation function
        condition: Only cache if condition returns True
        unless: Don't cache if unless condition returns True
        compress: Enable compression for large values
        invalidate_on_error: Invalidate cache on function error
    """
    def decorator(func: Callable) -> Callable:
        cache_manager = get_cache_manager()
        key_builder = CacheKeyBuilder()
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key_str = key_func(*args, **kwargs)
            else:
                func_name = f"{func.__module__}.{func.__qualname__}"
                arg_key = cache_key(*args, **kwargs)
                cache_key_str = key_builder.build_key(func_name, arg_key, namespace=namespace)
            
            # Check conditions
            if condition and not condition(*args, **kwargs):
                return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            if unless and unless(*args, **kwargs):
                return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key_str)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key_str}")
                return cached_result
            
            # Execute function
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache the result
                await cache_manager.set(cache_key_str, result, ttl=ttl, compress=compress)
                logger.debug(f"Cached result for key: {cache_key_str}")
                
                return result
                
            except Exception as e:
                if invalidate_on_error:
                    await cache_manager.delete(cache_key_str)
                raise e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to run the async cache operations
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def cached_method(
    ttl: int = 3600,
    namespace: Optional[str] = None,
    include_self: bool = False,
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching class method results
    
    Args:
        ttl: Time-to-live in seconds
        namespace: Cache namespace
        include_self: Include self object in key generation
        key_func: Custom key generation function
    """
    def decorator(func: Callable) -> Callable:
        cache_manager = get_cache_manager()
        key_builder = CacheKeyBuilder()
        
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key_str = key_func(self, *args, **kwargs)
            else:
                func_name = f"{func.__qualname__}"
                
                key_parts = []
                if include_self:
                    key_parts.append(str(id(self)))  # Use object ID
                
                if args or kwargs:
                    arg_key = cache_key(*args, **kwargs)
                    key_parts.append(arg_key)
                
                cache_key_str = key_builder.build_key(func_name, *key_parts, namespace=namespace)
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key_str)
            if cached_result is not None:
                return cached_result
            
            # Execute method
            if asyncio.iscoroutinefunction(func):
                result = await func(self, *args, **kwargs)
            else:
                result = func(self, *args, **kwargs)
            
            # Cache the result
            await cache_manager.set(cache_key_str, result, ttl=ttl)
            return result
        
        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(self, *args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def timed_cache(
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    namespace: Optional[str] = None
):
    """
    Decorator for time-based caching with human-readable duration
    
    Args:
        hours: Cache duration in hours
        minutes: Cache duration in minutes  
        seconds: Cache duration in seconds
        namespace: Cache namespace
    """
    total_seconds = hours * 3600 + minutes * 60 + seconds
    if total_seconds <= 0:
        total_seconds = 3600  # Default 1 hour
    
    return cached(ttl=total_seconds, namespace=namespace)

def smart_cache(
    base_ttl: int = 3600,
    max_ttl: int = 86400,
    hit_multiplier: float = 1.5,
    namespace: Optional[str] = None
):
    """
    Smart caching with adaptive TTL based on cache hit patterns
    
    Args:
        base_ttl: Base TTL in seconds
        max_ttl: Maximum TTL in seconds
        hit_multiplier: Multiplier for TTL extension on hits
        namespace: Cache namespace
    """
    def decorator(func: Callable) -> Callable:
        cache_manager = get_cache_manager()
        key_builder = CacheKeyBuilder()
        hit_counts: Dict[str, int] = {}
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            arg_key = cache_key(*args, **kwargs)
            cache_key_str = key_builder.build_key(func_name, arg_key, namespace=namespace)
            
            # Check cache
            cached_result = await cache_manager.get(cache_key_str)
            if cached_result is not None:
                # Update hit count and extend TTL
                hit_counts[cache_key_str] = hit_counts.get(cache_key_str, 0) + 1
                
                # Calculate adaptive TTL
                adaptive_ttl = min(
                    max_ttl,
                    int(base_ttl * (hit_multiplier ** hit_counts[cache_key_str]))
                )
                
                # Extend TTL
                await cache_manager.extend_ttl(cache_key_str, adaptive_ttl)
                
                return cached_result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache with base TTL
            await cache_manager.set(cache_key_str, result, ttl=base_ttl)
            hit_counts[cache_key_str] = 0
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def cache_invalidate(
    pattern: Optional[str] = None,
    namespace: Optional[str] = None,
    keys: Optional[List[str]] = None,
    condition: Optional[Callable] = None
):
    """
    Decorator for cache invalidation after function execution
    
    Args:
        pattern: Key pattern to invalidate
        namespace: Namespace to invalidate
        keys: Specific keys to invalidate
        condition: Only invalidate if condition returns True
    """
    def decorator(func: Callable) -> Callable:
        cache_manager = get_cache_manager()
        key_builder = CacheKeyBuilder()
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Execute function first
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Check condition
            if condition and not condition(result, *args, **kwargs):
                return result
            
            # Invalidate cache
            try:
                if keys:
                    for key in keys:
                        await cache_manager.delete(key)
                
                if pattern:
                    if namespace:
                        full_pattern = key_builder.build_pattern(pattern, namespace=namespace)
                    else:
                        full_pattern = pattern
                    await cache_manager.clear_pattern(full_pattern)
                
                if namespace and not pattern:
                    # Clear entire namespace
                    namespace_pattern = key_builder.build_pattern("*", namespace=namespace)
                    await cache_manager.clear_pattern(namespace_pattern)
                
                logger.debug(f"Cache invalidated after {func.__name__}")
                
            except Exception as e:
                logger.warning(f"Cache invalidation failed: {e}")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def cache_warm(warm_keys: List[Tuple[str, Callable]], ttl: int = 3600):
    """
    Decorator to warm cache with pre-computed values
    
    Args:
        warm_keys: List of (cache_key, value_function) tuples
        ttl: TTL for warmed values
    """
    def decorator(func: Callable) -> Callable:
        cache_manager = get_cache_manager()
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Warm cache
            for cache_key_str, value_func in warm_keys:
                try:
                    if not await cache_manager.exists(cache_key_str):
                        if asyncio.iscoroutinefunction(value_func):
                            warm_value = await value_func(*args, **kwargs)
                        else:
                            warm_value = value_func(*args, **kwargs)
                        
                        await cache_manager.set(cache_key_str, warm_value, ttl=ttl)
                        logger.debug(f"Cache warmed for key: {cache_key_str}")
                        
                except Exception as e:
                    logger.warning(f"Cache warming failed for {cache_key_str}: {e}")
            
            # Execute original function
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Utility functions

async def clear_cache_namespace(namespace: str):
    """Clear all cache entries in a namespace"""
    cache_manager = get_cache_manager()
    key_builder = CacheKeyBuilder()
    
    pattern = key_builder.build_pattern("*", namespace=namespace)
    cleared = await cache_manager.clear_pattern(pattern)
    
    logger.info(f"Cleared {cleared} entries from namespace: {namespace}")
    return cleared

async def get_cache_info(namespace: Optional[str] = None) -> Dict[str, Any]:
    """Get cache information and statistics"""
    cache_manager = get_cache_manager()
    stats = await cache_manager.get_stats()
    
    return {
        "stats": {
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": stats.hit_rate,
            "sets": stats.sets,
            "deletes": stats.deletes,
            "evictions": stats.evictions,
            "total_size_bytes": stats.total_size_bytes,
            "avg_response_time_ms": stats.avg_response_time_ms
        },
        "namespace": namespace,
        "timestamp": datetime.now().isoformat()
    }