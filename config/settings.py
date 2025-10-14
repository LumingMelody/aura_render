"""
应用设置模型 - 定义各种配置设置的数据模型
"""
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from enum import Enum


class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """数据库类型"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class CacheType(str, Enum):
    """缓存类型"""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"


# 基础设置模型
class BaseSettings(BaseModel):
    """基础设置"""

    class Config:
        extra = "allow"
        use_enum_values = True


class DatabaseSettings(BaseSettings):
    """数据库设置"""
    type: DatabaseType = DatabaseType.SQLITE
    host: Optional[str] = "localhost"
    port: Optional[int] = None
    database: str = "aura_render"
    username: Optional[str] = None
    password: Optional[str] = None
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    ssl_mode: Optional[str] = None
    charset: str = "utf8mb4"

    @validator('port', always=True)
    def set_default_port(cls, v, values):
        if v is None:
            db_type = values.get('type')
            if db_type == DatabaseType.POSTGRESQL:
                return 5432
            elif db_type == DatabaseType.MYSQL:
                return 3306
        return v


class CacheSettings(BaseSettings):
    """缓存设置"""
    type: CacheType = CacheType.MEMORY
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_size: int = 1000
    default_timeout: int = 3600
    key_prefix: str = "aura:"
    serializer: str = "json"


class APISettings(BaseSettings):
    """API设置"""
    title: str = "Aura Render API"
    description: str = "智能视频生成系统API"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1

    # CORS设置
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]

    # 文档设置
    enable_docs: bool = True
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"

    # 安全设置
    secret_key: str = "your-secret-key-change-this"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30

    # 限制设置
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    request_timeout: float = 300.0  # 5分钟

    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v


class MonitoringSettings(BaseSettings):
    """监控设置"""
    enabled: bool = True
    metrics_enabled: bool = True
    alerts_enabled: bool = True

    # 指标收集
    metrics_interval: int = 60  # 秒
    metrics_retention_days: int = 30

    # 告警设置
    alert_check_interval: int = 60  # 秒
    alert_retention_days: int = 7

    # 性能阈值
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    error_rate_threshold: float = 5.0
    response_time_threshold: float = 2000.0  # 毫秒

    # 外部监控集成
    prometheus_enabled: bool = False
    prometheus_port: int = 9090
    grafana_enabled: bool = False
    grafana_url: Optional[str] = None


class SecuritySettings(BaseSettings):
    """安全设置"""
    # JWT设置
    jwt_secret_key: str = "your-jwt-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # 密码设置
    password_min_length: int = 8
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    password_require_uppercase: bool = True

    # 会话设置
    session_timeout_minutes: int = 30
    max_failed_login_attempts: int = 5
    account_lockout_minutes: int = 15

    # HTTPS设置
    force_https: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

    # CSRF保护
    csrf_protection: bool = True
    csrf_secret_key: str = "your-csrf-secret-key"


class LoggingSettings(BaseSettings):
    """日志设置"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # 文件日志
    file_enabled: bool = True
    file_path: str = "logs/aura_render.log"
    file_max_size: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 5

    # 控制台日志
    console_enabled: bool = True
    console_colors: bool = True

    # 结构化日志
    json_format: bool = False
    include_traceback: bool = True

    # 特定模块日志级别
    module_levels: Dict[str, str] = field(default_factory=dict)


class VideoGenerationSettings(BaseSettings):
    """视频生成设置"""
    # 默认设置
    default_resolution: str = "1080p"
    default_fps: int = 30
    default_duration: int = 60  # 秒

    # 限制设置
    max_duration: int = 600  # 10分钟
    max_resolution: str = "4K"
    max_file_size: int = 2 * 1024 * 1024 * 1024  # 2GB

    # 质量设置
    quality_levels: List[str] = ["low", "medium", "high", "ultra"]
    default_quality: str = "high"

    # 并发设置
    max_concurrent_generations: int = 5
    queue_timeout: int = 3600  # 1小时

    # 模板设置
    template_dir: str = "templates"
    custom_templates_enabled: bool = True

    # 存储设置
    output_dir: str = "output"
    temp_dir: str = "temp"
    cleanup_temp_files: bool = True


class MaterialSettings(BaseSettings):
    """素材设置"""
    # 存储设置
    upload_dir: str = "uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = [".jpg", ".jpeg", ".png", ".gif", ".mp4", ".avi", ".mov", ".mp3", ".wav"]

    # 处理设置
    auto_resize: bool = True
    max_image_size: tuple = (1920, 1080)
    image_quality: int = 85

    # 缓存设置
    cache_processed_materials: bool = True
    cache_duration_days: int = 30

    # API设置
    external_apis_enabled: bool = True
    api_rate_limits: Dict[str, int] = field(default_factory=lambda: {
        "pexels": 200,
        "pixabay": 100,
        "freesound": 50
    })


class Settings(BaseSettings):
    """主设置类"""
    # 应用信息
    app_name: str = "Aura Render"
    app_version: str = "1.0.0"
    app_description: str = "智能视频生成系统"
    environment: str = "development"
    debug: bool = False

    # 各模块设置
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    api: APISettings = Field(default_factory=APISettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    video_generation: VideoGenerationSettings = Field(default_factory=VideoGenerationSettings)
    materials: MaterialSettings = Field(default_factory=MaterialSettings)

    @property
    def redis_url(self) -> str:
        """Get Redis URL for backward compatibility with Celery"""
        if self.cache.type == CacheType.REDIS:
            auth_part = f":{self.cache.password}@" if self.cache.password else ""
            return f"redis://{auth_part}{self.cache.host}:{self.cache.port}/{self.cache.db}"
        else:
            # Default Redis URL for testing/development
            return "redis://localhost:6379/0"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.debug or self.environment == "development"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Settings':
        """从字典创建设置实例"""
        # 处理嵌套配置
        settings_data = {}

        for key, value in config_dict.items():
            if key == 'database' and isinstance(value, dict):
                settings_data[key] = DatabaseSettings(**value)
            elif key == 'cache' and isinstance(value, dict):
                settings_data[key] = CacheSettings(**value)
            elif key == 'api' and isinstance(value, dict):
                settings_data[key] = APISettings(**value)
            elif key == 'monitoring' and isinstance(value, dict):
                settings_data[key] = MonitoringSettings(**value)
            elif key == 'security' and isinstance(value, dict):
                settings_data[key] = SecuritySettings(**value)
            elif key == 'logging' and isinstance(value, dict):
                settings_data[key] = LoggingSettings(**value)
            elif key == 'video_generation' and isinstance(value, dict):
                settings_data[key] = VideoGenerationSettings(**value)
            elif key == 'materials' and isinstance(value, dict):
                settings_data[key] = MaterialSettings(**value)
            else:
                settings_data[key] = value

        return cls(**settings_data)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict()

    def validate_all(self) -> List[str]:
        """验证所有设置"""
        errors = []

        try:
            # 验证数据库设置
            if self.database.type == DatabaseType.POSTGRESQL:
                if not self.database.host or not self.database.username:
                    errors.append("PostgreSQL requires host and username")

            # 验证缓存设置
            if self.cache.type == CacheType.REDIS:
                if not self.cache.host:
                    errors.append("Redis cache requires host")

            # 验证API设置
            if not 1 <= self.api.port <= 65535:
                errors.append("API port must be between 1 and 65535")

            # 验证安全设置
            if len(self.security.jwt_secret_key) < 32:
                errors.append("JWT secret key should be at least 32 characters")

            # 验证视频生成设置
            if self.video_generation.max_duration <= 0:
                errors.append("Max video duration must be positive")

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return errors