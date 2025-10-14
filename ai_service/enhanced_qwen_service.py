#!/usr/bin/env python3
"""
Enhanced Qwen Service for Aura Render

Provides advanced Qwen integration with features:
- Multi-model support (Qwen-Max, Qwen-Plus, Qwen-Turbo)
- Intelligent model selection
- Advanced retry logic with exponential backoff
- Response caching and optimization
- Context-aware generation
- Performance monitoring
- JSON parsing and validation
"""

import asyncio
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import httpx
from pydantic import BaseModel, Field, validator
import logging
from pathlib import Path
import redis.asyncio as redis

try:
    import dashscope
    from dashscope import Generation
    from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

from config import settings
from utils.logger import get_logger


class ResponseStatus(Enum):
    """Response status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    CACHED = "cached"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"


class ModelType(Enum):
    """Qwen model types"""
    QWEN_MAX = "qwen-max"
    QWEN_PLUS = "qwen-plus"  
    QWEN_TURBO = "qwen-turbo"
    QWEN_LONG = "qwen-long"


@dataclass
class GenerationConfig:
    """Generation configuration"""
    model: ModelType = ModelType.QWEN_MAX
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2000
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    result_format: str = "message"
    enable_search: bool = False
    incremental_output: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API call"""
        config = {
            "model": self.model.value,
            "parameters": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "result_format": self.result_format,
                "enable_search": self.enable_search,
                "incremental_output": self.incremental_output
            }
        }
        
        if self.seed is not None:
            config["parameters"]["seed"] = self.seed
        if self.stop:
            config["parameters"]["stop"] = self.stop
            
        return config


@dataclass
class GenerationResponse:
    """Enhanced generation response"""
    status: ResponseStatus
    content: Union[str, Dict[str, Any], List[Any]]
    model_used: str
    response_time: float
    retry_count: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    tokens_used: Optional[int] = None
    confidence: Optional[float] = None
    
    @property
    def is_success(self) -> bool:
        """Check if response is successful"""
        return self.status == ResponseStatus.SUCCESS
    
    @property
    def text_content(self) -> str:
        """Get text content from response"""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, dict):
            return json.dumps(self.content, ensure_ascii=False, indent=2)
        elif isinstance(self.content, list):
            return json.dumps(self.content, ensure_ascii=False, indent=2)
        else:
            return str(self.content)


class QwenServiceMetrics:
    """Service metrics tracking"""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        self.total_response_time = 0.0
        self.total_tokens_used = 0
        self.model_usage = {}
        self.error_counts = {}
        
    def record_request(self, response: GenerationResponse):
        """Record a request in metrics"""
        self.total_requests += 1
        self.total_response_time += response.response_time
        
        if response.cache_hit:
            self.cache_hits += 1
        
        if response.is_success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            error_type = response.error_message or "unknown_error"
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Track model usage
        model = response.model_used
        self.model_usage[model] = self.model_usage.get(model, 0) + 1
        
        # Track token usage
        if response.tokens_used:
            self.total_tokens_used += response.tokens_used
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time"""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "cache_hits": self.cache_hits,
            "success_rate": self.success_rate,
            "average_response_time": self.average_response_time,
            "cache_hit_rate": self.cache_hit_rate,
            "total_tokens_used": self.total_tokens_used,
            "model_usage": self.model_usage,
            "error_counts": self.error_counts
        }


class EnhancedQwenService:
    """Enhanced Qwen service with advanced features"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 redis_client: Optional[redis.Redis] = None,
                 cache_ttl: int = 3600,
                 enable_cache: bool = True,
                 enable_metrics: bool = True):
        """Initialize enhanced Qwen service"""
        self.logger = get_logger(__name__)
        
        # API configuration
        self.api_key = api_key or settings.ai.dashscope_api_key
        if not self.api_key:
            raise ValueError("Qwen API key not provided")
        
        if DASHSCOPE_AVAILABLE:
            dashscope.api_key = self.api_key
        else:
            self.logger.warning("DashScope SDK not available, using HTTP API")
        
        # Cache configuration
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
        self.enable_cache = enable_cache and redis_client is not None
        
        # Metrics
        self.enable_metrics = enable_metrics
        self.metrics = QwenServiceMetrics()
        
        # Rate limiting
        self.request_semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests
        self.last_request_time = 0.0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Model intelligence
        self.model_performance = {}
        self.model_selection_strategy = "adaptive"  # adaptive, fixed, round_robin
        
        self.logger.info("Enhanced Qwen service initialized")
    
    async def _get_cache_key(self, prompt: str, config: GenerationConfig, 
                           context: Optional[Dict] = None) -> str:
        """Generate cache key for request"""
        cache_data = {
            "prompt": prompt,
            "model": config.model.value,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "context": context or {}
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        return f"qwen_cache:{hashlib.md5(cache_str.encode()).hexdigest()}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[GenerationResponse]:
        """Get cached response"""
        if not self.enable_cache or not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                response = GenerationResponse(**data)
                response.cache_hit = True
                self.logger.debug(f"Cache hit for key: {cache_key}")
                return response
        except Exception as e:
            self.logger.warning(f"Cache read error: {e}")
        
        return None
    
    async def _cache_response(self, cache_key: str, response: GenerationResponse):
        """Cache response"""
        if not self.enable_cache or not self.redis_client or response.cache_hit:
            return
        
        try:
            # Don't cache error responses
            if not response.is_success:
                return
            
            cache_data = {
                "status": response.status.value,
                "content": response.content,
                "model_used": response.model_used,
                "response_time": response.response_time,
                "metadata": response.metadata,
                "tokens_used": response.tokens_used,
                "confidence": response.confidence
            }
            
            await self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(cache_data, ensure_ascii=False)
            )
            
            self.logger.debug(f"Response cached with key: {cache_key}")
            
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")
    
    def _select_model(self, prompt: str, context: Optional[Dict] = None) -> ModelType:
        """Intelligently select model based on request characteristics"""
        if self.model_selection_strategy == "fixed":
            return ModelType.QWEN_MAX
        
        # Analyze prompt characteristics
        prompt_length = len(prompt)
        
        # Simple heuristics for model selection
        if prompt_length > 8000:  # Very long context
            return ModelType.QWEN_LONG
        elif prompt_length > 2000:  # Medium context, need high quality
            return ModelType.QWEN_MAX
        elif context and context.get("task_complexity") == "simple":
            return ModelType.QWEN_TURBO  # Fast response for simple tasks
        else:
            return ModelType.QWEN_PLUS  # Balanced choice
    
    async def _rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _make_api_call(self, 
                           prompt: str, 
                           config: GenerationConfig,
                           context: Optional[Dict] = None) -> GenerationResponse:
        """Make actual API call to Qwen"""
        start_time = time.time()
        
        try:
            # Apply rate limiting
            await self._rate_limit()
            
            if DASHSCOPE_AVAILABLE:
                response = await self._call_dashscope_api(prompt, config, context)
            else:
                response = await self._call_http_api(prompt, config, context)
            
            response.response_time = time.time() - start_time
            return response
            
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            return GenerationResponse(
                status=ResponseStatus.ERROR,
                content="",
                model_used=config.model.value,
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _call_dashscope_api(self, 
                                prompt: str, 
                                config: GenerationConfig,
                                context: Optional[Dict] = None) -> GenerationResponse:
        """Call DashScope API"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Add conversation history if available
            if context and "conversation_history" in context:
                history = context["conversation_history"]
                messages = history + messages
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: Generation.call(
                    model=config.model.value,
                    messages=messages,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=config.max_tokens,
                    result_format=config.result_format,
                    enable_search=config.enable_search,
                    incremental_output=config.incremental_output
                )
            )
            
            if response.status_code == 200:
                content = response.output.choices[0]["message"]["content"]
                
                return GenerationResponse(
                    status=ResponseStatus.SUCCESS,
                    content=content,
                    model_used=config.model.value,
                    response_time=0.0,  # Will be set by caller
                    tokens_used=response.usage.total_tokens if response.usage else None,
                    metadata={
                        "request_id": response.request_id,
                        "usage": response.usage.to_dict() if response.usage else {}
                    }
                )
            else:
                error_msg = f"API error {response.status_code}: {response.message}"
                return GenerationResponse(
                    status=ResponseStatus.ERROR,
                    content="",
                    model_used=config.model.value,
                    response_time=0.0,
                    error_message=error_msg
                )
                
        except Exception as e:
            return GenerationResponse(
                status=ResponseStatus.ERROR,
                content="",
                model_used=config.model.value,
                response_time=0.0,
                error_message=str(e)
            )
    
    async def _call_http_api(self, 
                           prompt: str, 
                           config: GenerationConfig,
                           context: Optional[Dict] = None) -> GenerationResponse:
        """Call HTTP API directly"""
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": config.model.value,
            "input": {"prompt": prompt},
            "parameters": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": config.max_tokens
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["output"]["text"]
                    
                    return GenerationResponse(
                        status=ResponseStatus.SUCCESS,
                        content=content,
                        model_used=config.model.value,
                        response_time=0.0,
                        tokens_used=result.get("usage", {}).get("total_tokens"),
                        metadata={"request_id": result.get("request_id")}
                    )
                else:
                    error_msg = f"HTTP error {response.status_code}: {response.text}"
                    return GenerationResponse(
                        status=ResponseStatus.ERROR,
                        content="",
                        model_used=config.model.value,
                        response_time=0.0,
                        error_message=error_msg
                    )
        except httpx.TimeoutException:
            return GenerationResponse(
                status=ResponseStatus.TIMEOUT,
                content="",
                model_used=config.model.value,
                response_time=0.0,
                error_message="Request timeout"
            )
        except Exception as e:
            return GenerationResponse(
                status=ResponseStatus.ERROR,
                content="",
                model_used=config.model.value,
                response_time=0.0,
                error_message=str(e)
            )
    
    async def generate_async(self, 
                           prompt: str,
                           model: Optional[ModelType] = None,
                           config: Optional[GenerationConfig] = None,
                           context: Optional[Dict] = None,
                           parse_json: bool = False,
                           json_schema: Optional[Dict] = None,
                           max_retries: int = 3) -> GenerationResponse:
        """Generate text asynchronously with advanced features"""
        
        # Prepare configuration
        if config is None:
            selected_model = model or self._select_model(prompt, context)
            config = GenerationConfig(model=selected_model)
        
        # Check cache first
        cache_key = await self._get_cache_key(prompt, config, context)
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            if self.enable_metrics:
                self.metrics.record_request(cached_response)
            return cached_response
        
        # Acquire semaphore for rate limiting
        async with self.request_semaphore:
            # Retry logic with exponential backoff
            retry_count = 0
            last_error = None
            
            while retry_count <= max_retries:
                try:
                    response = await self._make_api_call(prompt, config, context)
                    response.retry_count = retry_count
                    
                    # Post-process response
                    if response.is_success and parse_json:
                        response = await self._parse_json_response(response, json_schema)
                    
                    # Cache successful responses
                    if response.is_success:
                        await self._cache_response(cache_key, response)
                    
                    # Record metrics
                    if self.enable_metrics:
                        self.metrics.record_request(response)
                    
                    return response
                    
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    
                    if retry_count <= max_retries:
                        # Exponential backoff
                        sleep_time = (2 ** retry_count) * 0.1
                        await asyncio.sleep(sleep_time)
                        self.logger.warning(f"Retry {retry_count}/{max_retries} after error: {e}")
            
            # All retries failed
            error_response = GenerationResponse(
                status=ResponseStatus.ERROR,
                content="",
                model_used=config.model.value,
                response_time=0.0,
                retry_count=retry_count,
                error_message=f"Max retries exceeded. Last error: {last_error}"
            )
            
            if self.enable_metrics:
                self.metrics.record_request(error_response)
            
            return error_response
    
    async def _parse_json_response(self, 
                                 response: GenerationResponse,
                                 json_schema: Optional[Dict] = None) -> GenerationResponse:
        """Parse JSON response and validate against schema"""
        try:
            content = response.text_content.strip()
            
            # Try to extract JSON from markdown code blocks
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            
            # Parse JSON
            parsed_content = json.loads(content)
            
            # Validate against schema if provided
            if json_schema:
                # Simple schema validation (could be enhanced with jsonschema library)
                if not self._validate_json_schema(parsed_content, json_schema):
                    return GenerationResponse(
                        status=ResponseStatus.ERROR,
                        content=response.content,
                        model_used=response.model_used,
                        response_time=response.response_time,
                        error_message="JSON does not match required schema"
                    )
            
            # Update response with parsed content
            response.content = parsed_content
            response.metadata["json_parsed"] = True
            
            return response
            
        except json.JSONDecodeError as e:
            return GenerationResponse(
                status=ResponseStatus.ERROR,
                content=response.content,
                model_used=response.model_used,
                response_time=response.response_time,
                error_message=f"JSON parsing failed: {e}"
            )
    
    def _validate_json_schema(self, data: Any, schema: Dict) -> bool:
        """Simple JSON schema validation"""
        # This is a basic implementation - could use jsonschema library for full validation
        if "type" in schema:
            expected_type = schema["type"]
            if expected_type == "object" and not isinstance(data, dict):
                return False
            elif expected_type == "array" and not isinstance(data, list):
                return False
            elif expected_type == "string" and not isinstance(data, str):
                return False
            elif expected_type == "number" and not isinstance(data, (int, float)):
                return False
        
        if "required" in schema and isinstance(data, dict):
            for required_field in schema["required"]:
                if required_field not in data:
                    return False
        
        return True
    
    def generate(self, *args, **kwargs) -> GenerationResponse:
        """Synchronous wrapper for generate_async"""
        return asyncio.run(self.generate_async(*args, **kwargs))
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get service health information"""
        return {
            "status": "healthy" if self.api_key else "unhealthy",
            "metrics": self.metrics.to_dict(),
            "cache_enabled": self.enable_cache,
            "dashscope_sdk_available": DASHSCOPE_AVAILABLE,
            "model_selection_strategy": self.model_selection_strategy,
            "current_time": datetime.now().isoformat()
        }
    
    def reset_service_metrics(self):
        """Reset service metrics"""
        self.metrics = QwenServiceMetrics()
        self.logger.info("Service metrics reset")


# Global instance
_enhanced_qwen_service: Optional[EnhancedQwenService] = None


async def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client for caching"""
    try:
        if not hasattr(settings, 'redis') or not settings.redis.url:
            return None
        
        client = redis.from_url(settings.redis.url)
        await client.ping()
        return client
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Redis not available: {e}")
        return None


def get_enhanced_qwen_service() -> EnhancedQwenService:
    """Get or create enhanced Qwen service instance"""
    global _enhanced_qwen_service
    
    if _enhanced_qwen_service is None:
        # Get Redis client if available
        try:
            redis_client = asyncio.run(get_redis_client())
        except:
            redis_client = None
        
        _enhanced_qwen_service = EnhancedQwenService(
            redis_client=redis_client,
            enable_cache=redis_client is not None
        )
    
    return _enhanced_qwen_service