"""
配置验证器 - 验证配置的正确性和完整性
"""
from typing import Dict, List, Any, Optional, Union
import os
import re
import socket
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging

from .settings import Settings, DatabaseType, CacheType, LogLevel


class ValidationLevel(Enum):
    """验证级别"""
    ERROR = "error"      # 错误，阻止启动
    WARNING = "warning"  # 警告，可能影响功能
    INFO = "info"       # 信息，建议优化


@dataclass
class ValidationResult:
    """验证结果"""
    level: ValidationLevel
    category: str
    message: str
    suggestion: Optional[str] = None
    config_path: Optional[str] = None


class ConfigValidator:
    """配置验证器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results: List[ValidationResult] = []

    def validate_settings(self, settings: Settings) -> List[ValidationResult]:
        """验证设置对象"""
        self.results = []

        # 验证应用配置
        self._validate_app_config(settings)

        # 验证数据库配置
        self._validate_database_config(settings.database)

        # 验证缓存配置
        self._validate_cache_config(settings.cache)

        # 验证API配置
        self._validate_api_config(settings.api)

        # 验证安全配置
        self._validate_security_config(settings.security)

        # 验证日志配置
        self._validate_logging_config(settings.logging)

        # 验证视频生成配置
        self._validate_video_generation_config(settings.video_generation)

        # 验证素材配置
        self._validate_materials_config(settings.materials)

        # 验证监控配置
        self._validate_monitoring_config(settings.monitoring)

        return self.results

    def _add_result(self, level: ValidationLevel, category: str, message: str,
                   suggestion: Optional[str] = None, config_path: Optional[str] = None):
        """添加验证结果"""
        result = ValidationResult(
            level=level,
            category=category,
            message=message,
            suggestion=suggestion,
            config_path=config_path
        )
        self.results.append(result)

    def _validate_app_config(self, settings: Settings):
        """验证应用配置"""
        if not settings.app_name:
            self._add_result(
                ValidationLevel.WARNING,
                "app",
                "Application name is not set",
                "Set app_name for better identification"
            )

        if not settings.app_version:
            self._add_result(
                ValidationLevel.WARNING,
                "app",
                "Application version is not set",
                "Set app_version for version tracking"
            )

        if settings.environment not in ["development", "testing", "staging", "production"]:
            self._add_result(
                ValidationLevel.WARNING,
                "app",
                f"Unknown environment: {settings.environment}",
                "Use standard environment names: development, testing, staging, production"
            )

    def _validate_database_config(self, db_config):
        """验证数据库配置"""
        if db_config.type == DatabaseType.POSTGRESQL:
            if not db_config.host:
                self._add_result(
                    ValidationLevel.ERROR,
                    "database",
                    "PostgreSQL host is required",
                    "Set database.host for PostgreSQL connection"
                )

            if not db_config.username:
                self._add_result(
                    ValidationLevel.ERROR,
                    "database",
                    "PostgreSQL username is required",
                    "Set database.username for PostgreSQL connection"
                )

            if not db_config.password:
                self._add_result(
                    ValidationLevel.WARNING,
                    "database",
                    "PostgreSQL password is not set",
                    "Set database.password for secure connection"
                )

            if not db_config.database:
                self._add_result(
                    ValidationLevel.ERROR,
                    "database",
                    "PostgreSQL database name is required",
                    "Set database.database name"
                )

            # 验证端口
            if not self._is_port_valid(db_config.port):
                self._add_result(
                    ValidationLevel.ERROR,
                    "database",
                    f"Invalid database port: {db_config.port}",
                    "Set a valid port number (1-65535)"
                )

        elif db_config.type == DatabaseType.MYSQL:
            if not db_config.host:
                self._add_result(
                    ValidationLevel.ERROR,
                    "database",
                    "MySQL host is required",
                    "Set database.host for MySQL connection"
                )

        elif db_config.type == DatabaseType.SQLITE:
            if db_config.database and db_config.database != ":memory:":
                db_path = Path(db_config.database)
                db_dir = db_path.parent
                if not db_dir.exists():
                    self._add_result(
                        ValidationLevel.WARNING,
                        "database",
                        f"SQLite database directory does not exist: {db_dir}",
                        f"Create directory: {db_dir}"
                    )

        # 验证连接池配置
        if db_config.pool_size <= 0:
            self._add_result(
                ValidationLevel.ERROR,
                "database",
                "Database pool_size must be positive",
                "Set pool_size to a positive integer"
            )

        if db_config.max_overflow < 0:
            self._add_result(
                ValidationLevel.ERROR,
                "database",
                "Database max_overflow cannot be negative",
                "Set max_overflow to a non-negative integer"
            )

    def _validate_cache_config(self, cache_config):
        """验证缓存配置"""
        if cache_config.type == CacheType.REDIS:
            if not cache_config.host:
                self._add_result(
                    ValidationLevel.ERROR,
                    "cache",
                    "Redis host is required",
                    "Set cache.host for Redis connection"
                )

            if not self._is_port_valid(cache_config.port):
                self._add_result(
                    ValidationLevel.ERROR,
                    "cache",
                    f"Invalid Redis port: {cache_config.port}",
                    "Set a valid port number (1-65535)"
                )

            if cache_config.db < 0 or cache_config.db > 15:
                self._add_result(
                    ValidationLevel.WARNING,
                    "cache",
                    f"Redis database index out of typical range: {cache_config.db}",
                    "Use Redis database index 0-15"
                )

        if cache_config.max_size <= 0:
            self._add_result(
                ValidationLevel.ERROR,
                "cache",
                "Cache max_size must be positive",
                "Set max_size to a positive integer"
            )

        if cache_config.default_timeout <= 0:
            self._add_result(
                ValidationLevel.WARNING,
                "cache",
                "Cache default_timeout should be positive",
                "Set default_timeout to a positive value"
            )

    def _validate_api_config(self, api_config):
        """验证API配置"""
        if not self._is_port_valid(api_config.port):
            self._add_result(
                ValidationLevel.ERROR,
                "api",
                f"Invalid API port: {api_config.port}",
                "Set a valid port number (1-65535)"
            )

        if api_config.workers <= 0:
            self._add_result(
                ValidationLevel.ERROR,
                "api",
                "API workers must be positive",
                "Set workers to a positive integer"
            )

        if api_config.workers > 10:
            self._add_result(
                ValidationLevel.WARNING,
                "api",
                f"High number of API workers: {api_config.workers}",
                "Consider system resources when setting worker count"
            )

        # 验证CORS配置
        if "*" in api_config.cors_origins and len(api_config.cors_origins) > 1:
            self._add_result(
                ValidationLevel.WARNING,
                "api",
                "CORS origins contains '*' with other origins",
                "Use either '*' or specific origins, not both"
            )

        # 验证安全密钥
        if api_config.secret_key == "your-secret-key-change-this":
            self._add_result(
                ValidationLevel.ERROR,
                "api",
                "Default secret key is being used",
                "Change api.secret_key to a secure random string"
            )

        if len(api_config.secret_key) < 32:
            self._add_result(
                ValidationLevel.WARNING,
                "api",
                "API secret key is too short",
                "Use a secret key of at least 32 characters"
            )

        # 验证超时配置
        if api_config.request_timeout <= 0:
            self._add_result(
                ValidationLevel.ERROR,
                "api",
                "Request timeout must be positive",
                "Set request_timeout to a positive value"
            )

        if api_config.request_timeout > 600:
            self._add_result(
                ValidationLevel.WARNING,
                "api",
                f"Very long request timeout: {api_config.request_timeout}s",
                "Consider shorter timeout for better user experience"
            )

    def _validate_security_config(self, security_config):
        """验证安全配置"""
        # 验证JWT配置
        if security_config.jwt_secret_key in ["your-jwt-secret-key", "test-jwt-secret"]:
            self._add_result(
                ValidationLevel.ERROR,
                "security",
                "Default JWT secret key is being used",
                "Change security.jwt_secret_key to a secure random string"
            )

        if len(security_config.jwt_secret_key) < 32:
            self._add_result(
                ValidationLevel.WARNING,
                "security",
                "JWT secret key is too short",
                "Use a JWT secret key of at least 32 characters"
            )

        # 验证密码策略
        if security_config.password_min_length < 8:
            self._add_result(
                ValidationLevel.WARNING,
                "security",
                f"Weak password minimum length: {security_config.password_min_length}",
                "Set password_min_length to at least 8 characters"
            )

        # 验证会话配置
        if security_config.session_timeout_minutes <= 0:
            self._add_result(
                ValidationLevel.ERROR,
                "security",
                "Session timeout must be positive",
                "Set session_timeout_minutes to a positive value"
            )

        if security_config.max_failed_login_attempts <= 0:
            self._add_result(
                ValidationLevel.WARNING,
                "security",
                "Max failed login attempts should be positive",
                "Set max_failed_login_attempts to a positive value"
            )

        # 验证SSL配置
        if security_config.force_https:
            if security_config.ssl_cert_path and not Path(security_config.ssl_cert_path).exists():
                self._add_result(
                    ValidationLevel.ERROR,
                    "security",
                    f"SSL certificate file not found: {security_config.ssl_cert_path}",
                    "Provide a valid SSL certificate path"
                )

            if security_config.ssl_key_path and not Path(security_config.ssl_key_path).exists():
                self._add_result(
                    ValidationLevel.ERROR,
                    "security",
                    f"SSL key file not found: {security_config.ssl_key_path}",
                    "Provide a valid SSL key path"
                )

    def _validate_logging_config(self, logging_config):
        """验证日志配置"""
        # 验证日志级别
        valid_levels = [level.value for level in LogLevel]
        if logging_config.level not in valid_levels:
            self._add_result(
                ValidationLevel.ERROR,
                "logging",
                f"Invalid log level: {logging_config.level}",
                f"Use one of: {', '.join(valid_levels)}"
            )

        # 验证文件日志配置
        if logging_config.file_enabled:
            log_path = Path(logging_config.file_path)
            log_dir = log_path.parent

            if not log_dir.exists():
                self._add_result(
                    ValidationLevel.WARNING,
                    "logging",
                    f"Log directory does not exist: {log_dir}",
                    f"Create log directory: {log_dir}"
                )

            if logging_config.file_max_size <= 0:
                self._add_result(
                    ValidationLevel.ERROR,
                    "logging",
                    "Log file max size must be positive",
                    "Set file_max_size to a positive value"
                )

            if logging_config.file_backup_count < 0:
                self._add_result(
                    ValidationLevel.ERROR,
                    "logging",
                    "Log file backup count cannot be negative",
                    "Set file_backup_count to a non-negative value"
                )

    def _validate_video_generation_config(self, video_config):
        """验证视频生成配置"""
        # 验证分辨率
        valid_resolutions = ["480p", "720p", "1080p", "1440p", "4K"]
        if video_config.default_resolution not in valid_resolutions:
            self._add_result(
                ValidationLevel.WARNING,
                "video_generation",
                f"Unknown default resolution: {video_config.default_resolution}",
                f"Use one of: {', '.join(valid_resolutions)}"
            )

        if video_config.max_resolution not in valid_resolutions:
            self._add_result(
                ValidationLevel.WARNING,
                "video_generation",
                f"Unknown max resolution: {video_config.max_resolution}",
                f"Use one of: {', '.join(valid_resolutions)}"
            )

        # 验证FPS
        if video_config.default_fps <= 0 or video_config.default_fps > 120:
            self._add_result(
                ValidationLevel.WARNING,
                "video_generation",
                f"Unusual default FPS: {video_config.default_fps}",
                "Use a reasonable FPS value (1-120)"
            )

        # 验证时长
        if video_config.default_duration <= 0:
            self._add_result(
                ValidationLevel.ERROR,
                "video_generation",
                "Default duration must be positive",
                "Set default_duration to a positive value"
            )

        if video_config.max_duration <= 0:
            self._add_result(
                ValidationLevel.ERROR,
                "video_generation",
                "Max duration must be positive",
                "Set max_duration to a positive value"
            )

        if video_config.default_duration > video_config.max_duration:
            self._add_result(
                ValidationLevel.ERROR,
                "video_generation",
                "Default duration exceeds max duration",
                "Set default_duration <= max_duration"
            )

        # 验证并发配置
        if video_config.max_concurrent_generations <= 0:
            self._add_result(
                ValidationLevel.ERROR,
                "video_generation",
                "Max concurrent generations must be positive",
                "Set max_concurrent_generations to a positive value"
            )

        # 验证目录
        for dir_name, dir_path in [
            ("template_dir", video_config.template_dir),
            ("output_dir", video_config.output_dir),
            ("temp_dir", video_config.temp_dir)
        ]:
            if not Path(dir_path).exists():
                self._add_result(
                    ValidationLevel.WARNING,
                    "video_generation",
                    f"{dir_name} does not exist: {dir_path}",
                    f"Create directory: {dir_path}"
                )

    def _validate_materials_config(self, materials_config):
        """验证素材配置"""
        # 验证上传目录
        upload_dir = Path(materials_config.upload_dir)
        if not upload_dir.exists():
            self._add_result(
                ValidationLevel.WARNING,
                "materials",
                f"Upload directory does not exist: {upload_dir}",
                f"Create upload directory: {upload_dir}"
            )

        # 验证文件大小
        if materials_config.max_file_size <= 0:
            self._add_result(
                ValidationLevel.ERROR,
                "materials",
                "Max file size must be positive",
                "Set max_file_size to a positive value"
            )

        # 验证扩展名
        for ext in materials_config.allowed_extensions:
            if not ext.startswith('.'):
                self._add_result(
                    ValidationLevel.WARNING,
                    "materials",
                    f"File extension should start with dot: {ext}",
                    "Use format like '.jpg', '.mp4'"
                )

        # 验证图片处理配置
        if materials_config.auto_resize:
            max_size = materials_config.max_image_size
            if len(max_size) != 2 or any(s <= 0 for s in max_size):
                self._add_result(
                    ValidationLevel.ERROR,
                    "materials",
                    f"Invalid max image size: {max_size}",
                    "Set max_image_size as [width, height] with positive values"
                )

        if not 0 <= materials_config.image_quality <= 100:
            self._add_result(
                ValidationLevel.ERROR,
                "materials",
                f"Invalid image quality: {materials_config.image_quality}",
                "Set image_quality between 0 and 100"
            )

    def _validate_monitoring_config(self, monitoring_config):
        """验证监控配置"""
        if monitoring_config.enabled:
            # 验证时间间隔
            if monitoring_config.metrics_interval <= 0:
                self._add_result(
                    ValidationLevel.ERROR,
                    "monitoring",
                    "Metrics interval must be positive",
                    "Set metrics_interval to a positive value"
                )

            if monitoring_config.alert_check_interval <= 0:
                self._add_result(
                    ValidationLevel.ERROR,
                    "monitoring",
                    "Alert check interval must be positive",
                    "Set alert_check_interval to a positive value"
                )

            # 验证阈值
            thresholds = [
                ("cpu_threshold", monitoring_config.cpu_threshold),
                ("memory_threshold", monitoring_config.memory_threshold),
                ("disk_threshold", monitoring_config.disk_threshold)
            ]

            for name, value in thresholds:
                if not 0 <= value <= 100:
                    self._add_result(
                        ValidationLevel.WARNING,
                        "monitoring",
                        f"Invalid {name}: {value}",
                        f"Set {name} between 0 and 100"
                    )

            if monitoring_config.error_rate_threshold < 0:
                self._add_result(
                    ValidationLevel.WARNING,
                    "monitoring",
                    f"Invalid error rate threshold: {monitoring_config.error_rate_threshold}",
                    "Set error_rate_threshold to a non-negative value"
                )

            if monitoring_config.response_time_threshold <= 0:
                self._add_result(
                    ValidationLevel.WARNING,
                    "monitoring",
                    "Response time threshold should be positive",
                    "Set response_time_threshold to a positive value"
                )

        # 验证Prometheus配置
        if monitoring_config.prometheus_enabled:
            if not self._is_port_valid(monitoring_config.prometheus_port):
                self._add_result(
                    ValidationLevel.ERROR,
                    "monitoring",
                    f"Invalid Prometheus port: {monitoring_config.prometheus_port}",
                    "Set a valid port number (1-65535)"
                )

    def _is_port_valid(self, port: int) -> bool:
        """检查端口是否有效"""
        return isinstance(port, int) and 1 <= port <= 65535

    def _is_host_reachable(self, host: str, port: int, timeout: int = 5) -> bool:
        """检查主机是否可达"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def has_errors(self) -> bool:
        """检查是否有错误"""
        return any(result.level == ValidationLevel.ERROR for result in self.results)

    def has_warnings(self) -> bool:
        """检查是否有警告"""
        return any(result.level == ValidationLevel.WARNING for result in self.results)

    def get_summary(self) -> Dict[str, int]:
        """获取验证结果摘要"""
        summary = {
            "total": len(self.results),
            "errors": 0,
            "warnings": 0,
            "info": 0
        }

        for result in self.results:
            if result.level == ValidationLevel.ERROR:
                summary["errors"] += 1
            elif result.level == ValidationLevel.WARNING:
                summary["warnings"] += 1
            else:
                summary["info"] += 1

        return summary

    def format_results(self, include_suggestions: bool = True) -> str:
        """格式化验证结果"""
        if not self.results:
            return "✅ Configuration validation passed successfully!"

        lines = []
        summary = self.get_summary()

        # 添加摘要
        lines.append(f"Configuration Validation Results:")
        lines.append(f"  Total: {summary['total']}")
        lines.append(f"  Errors: {summary['errors']}")
        lines.append(f"  Warnings: {summary['warnings']}")
        lines.append(f"  Info: {summary['info']}")
        lines.append("")

        # 按级别分组显示
        for level in [ValidationLevel.ERROR, ValidationLevel.WARNING, ValidationLevel.INFO]:
            level_results = [r for r in self.results if r.level == level]
            if not level_results:
                continue

            icon = "❌" if level == ValidationLevel.ERROR else "⚠️" if level == ValidationLevel.WARNING else "ℹ️"
            lines.append(f"{icon} {level.value.upper()}S:")

            for result in level_results:
                lines.append(f"  [{result.category}] {result.message}")
                if include_suggestions and result.suggestion:
                    lines.append(f"    💡 {result.suggestion}")
                lines.append("")

        return "\n".join(lines)


def validate_config_file(config_path: str) -> List[ValidationResult]:
    """验证配置文件"""
    validator = ConfigValidator()

    try:
        from .config_manager import ConfigManager

        config_manager = ConfigManager()
        settings = Settings.from_dict(config_manager.config)

        return validator.validate_settings(settings)

    except Exception as e:
        return [ValidationResult(
            level=ValidationLevel.ERROR,
            category="config_file",
            message=f"Failed to load configuration: {str(e)}",
            suggestion="Check configuration file syntax and structure"
        )]


def main():
    """命令行入口"""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Validate Aura Render configuration")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--environment", "-e", default="development",
                       help="Environment name (development, testing, staging, production)")
    parser.add_argument("--no-suggestions", action="store_true",
                       help="Don't show suggestions in output")

    args = parser.parse_args()

    # 设置环境变量
    if args.environment:
        os.environ["AURA_ENV"] = args.environment

    # 验证配置
    if args.config:
        results = validate_config_file(args.config)
    else:
        from .config_manager import get_config_manager
        config_manager = get_config_manager()
        settings = Settings.from_dict(config_manager.config)

        validator = ConfigValidator()
        results = validator.validate_settings(settings)

    # 显示结果
    validator = ConfigValidator()
    validator.results = results

    print(validator.format_results(include_suggestions=not args.no_suggestions))

    # 返回适当的退出码
    if validator.has_errors():
        sys.exit(1)
    elif validator.has_warnings():
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()