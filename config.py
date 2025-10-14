#!/usr/bin/env python3
"""
Aura Render Configuration Management

Centralized configuration management using pydantic-settings for
type safety and automatic environment variable loading.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root directory
PROJECT_ROOT = Path(__file__).parent


class AIServiceConfig(BaseModel):
    """AI Service Configuration"""
    dashscope_api_key: str = Field(default="", description="DashScope API Key")
    qwen_model_name: str = Field(default="qwen-max", description="Qwen model name")
    qwen_vl_model_name: str = Field(default="qwen-vl-max", description="Qwen VL model name")
    ai_request_timeout: int = Field(default=120, description="AI request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    
    # Image Generation Services
    openai_api_key: str = Field(default="", description="OpenAI API Key for DALL-E")
    stability_api_key: str = Field(default="", description="Stability AI API Key")
    midjourney_api_key: str = Field(default="", description="Midjourney API Key")
    replicate_api_token: str = Field(default="", description="Replicate API Token")
    
    # Image Generation Settings
    default_image_provider: str = Field(default="openai_dalle", description="Default image generation provider")
    default_image_style: str = Field(default="photorealistic", description="Default image style")
    default_image_size: str = Field(default="1024x1024", description="Default image size")
    max_concurrent_image_generations: int = Field(default=3, description="Max concurrent image generations")
    image_generation_timeout: int = Field(default=60, description="Image generation timeout in seconds")


class ServerConfig(BaseModel):
    """Server Configuration"""
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()


class APIConfig(BaseModel):
    """API Configuration"""
    title: str = Field(default="Aura Render Video Generation API", description="API title")
    version: str = Field(default="1.0.0", description="API version")
    description: str = Field(
        default="AI-Powered Video Generation and Rendering Pipeline",
        description="API description"
    )
    docs_url: Optional[str] = Field(default="/docs", description="Swagger UI URL")
    redoc_url: Optional[str] = Field(default="/redoc", description="ReDoc URL")


class RedisConfig(BaseModel):
    """Redis Configuration"""
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_url: Optional[str] = Field(default=None, description="Redis URL (overrides individual settings)")
    
    def get_redis_url(self) -> str:
        """Get Redis URL for connections"""
        # Check if explicit redis_url is provided (use __dict__ to avoid recursion)
        if hasattr(self, 'redis_url') and self.__dict__.get('redis_url'):
            return self.__dict__['redis_url']

        auth_part = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth_part}{self.redis_host}:{self.redis_port}/{self.redis_db}"


class ExternalServicesConfig(BaseModel):
    """External Services Configuration"""
    callback_url: str = Field(
        default="http://192.168.10.16:1689/chat/notify",
        description="Callback notification URL"
    )
    save_url: str = Field(
        default="https://agent.cstlanbaai.com/gateway/admin-api/agent/global-key/create-or-update",
        description="Global key save URL"
    )
    redis_save_url: str = Field(
        default="http://192.168.10.16:1689/redis/add",
        description="Redis save URL"
    )
    material_api_base_url: str = Field(
        default="https://api.yourmateriallibrary.com/v1",
        description="Material library API base URL"
    )
    music_search_api: str = Field(
        default="https://api.yourbgmservice.com/v1/tracks/search",
        description="Music search API URL"
    )


class FileStorageConfig(BaseModel):
    """File Storage Configuration"""
    max_file_size: int = Field(default=15, description="Max file size in MB")
    max_image_file_size: int = Field(default=10, description="Max image file size in MB")
    max_audio_file_size: int = Field(default=50, description="Max audio file size in MB") 
    max_video_file_size: int = Field(default=100, description="Max video file size in MB")
    
    temp_dir: Path = Field(default=Path("/tmp/aura_render"), description="Temporary directory")
    output_dir: Path = Field(default=Path("./outputs"), description="Output directory")
    cache_dir: Path = Field(default=Path("./cache"), description="Cache directory")
    
    def ensure_directories(self):
        """Ensure all directories exist"""
        for dir_path in [self.temp_dir, self.output_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class PerformanceConfig(BaseModel):
    """Performance Configuration"""
    http_timeout: int = Field(default=30, description="HTTP timeout in seconds")
    max_concurrent_tasks: int = Field(default=5, description="Max concurrent tasks")
    max_workers: int = Field(default=4, description="Max worker threads")


class MaterialProvidersConfig(BaseModel):
    """Material Providers Configuration"""
    # External Material Service (主要素材服务)
    external_base_url: str = Field(default="https://api.materials-provider.com/v1", description="External material service base URL")
    external_api_key: str = Field(default="", description="External material service API key")
    
    # Pexels API
    pexels_api_key: str = Field(default="", description="Pexels API Key")
    
    # Pixabay API
    pixabay_api_key: str = Field(default="", description="Pixabay API Key")
    
    # Unsplash API
    unsplash_access_key: str = Field(default="", description="Unsplash Access Key")
    
    # Freesound API
    freesound_api_key: str = Field(default="", description="Freesound API Key")
    
    # Provider preferences
    preferred_video_providers: List[str] = Field(
        default=["external", "pexels_video", "pixabay_video", "mock_video"],
        description="Preferred video providers in order"
    )
    preferred_audio_providers: List[str] = Field(
        default=["external", "freesound", "mock_audio"], 
        description="Preferred audio providers in order"
    )
    preferred_image_providers: List[str] = Field(
        default=["external", "unsplash", "pexels_images", "mock_image"],
        description="Preferred image providers in order"
    )
    
    # Rate limiting
    max_concurrent_requests: int = Field(default=3, description="Max concurrent provider requests")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    
    # Caching
    enable_material_cache: bool = Field(default=True, description="Enable material search caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")


class DevelopmentConfig(BaseModel):
    """Development Configuration"""
    enable_mock_services: bool = Field(default=True, description="Enable mock services")
    enable_detailed_logging: bool = Field(default=True, description="Enable detailed logging")
    save_intermediate_results: bool = Field(default=True, description="Save intermediate results")
    enable_reload: bool = Field(default=True, description="Enable auto-reload in development mode")


class Settings(BaseSettings):
    """Main Settings Class"""
    
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Configuration sections
    ai: AIServiceConfig = Field(default_factory=AIServiceConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    external: ExternalServicesConfig = Field(default_factory=ExternalServicesConfig)
    storage: FileStorageConfig = Field(default_factory=FileStorageConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    materials: MaterialProvidersConfig = Field(default_factory=MaterialProvidersConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist on initialization
        self.storage.ensure_directories()
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.server.debug or self.development.enable_mock_services
    
    @property
    def redis_host(self) -> str:
        """Get Redis host for backward compatibility"""
        return self.redis.redis_host
    
    @property
    def redis_port(self) -> int:
        """Get Redis port for backward compatibility"""
        return self.redis.redis_port
    
    @property
    def redis_db(self) -> int:
        """Get Redis database for backward compatibility"""
        return self.redis.redis_db
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL"""
        return self.redis.get_redis_url()
    
    def get_ai_headers(self) -> Dict[str, str]:
        """Get headers for AI service requests"""
        return {
            "Authorization": f"Bearer {self.ai.dashscope_api_key}",
            "Content-Type": "application/json"
        }
    
    def model_dump_safe(self) -> Dict[str, Any]:
        """Dump configuration without sensitive information"""
        config = self.model_dump()
        # Remove sensitive keys
        if 'ai' in config:
            ai_sensitive_keys = [
                'dashscope_api_key', 
                'openai_api_key', 
                'stability_api_key', 
                'midjourney_api_key', 
                'replicate_api_token'
            ]
            for key in ai_sensitive_keys:
                if key in config['ai'] and config['ai'][key]:
                    config['ai'][key] = '***HIDDEN***'
        
        # Hide material provider API keys
        if 'materials' in config:
            materials = config['materials']
            sensitive_keys = ['external_api_key', 'pexels_api_key', 'pixabay_api_key', 'unsplash_access_key', 'freesound_api_key']
            for key in sensitive_keys:
                if key in materials and materials[key]:
                    materials[key] = '***HIDDEN***'
        
        return config


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance (for dependency injection)"""
    return settings


def reload_settings() -> Settings:
    """Reload settings (useful for testing)"""
    global settings
    settings = Settings()
    return settings


if __name__ == "__main__":
    # Configuration validation and display
    print("🔧 Aura Render Configuration")
    print("=" * 50)
    
    config = settings.model_dump_safe()
    for section, values in config.items():
        print(f"\n[{section.upper()}]")
        for key, value in values.items():
            print(f"  {key}: {value}")
    
    print(f"\n✅ Configuration loaded successfully")
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"🔍 Development mode: {settings.is_development}")