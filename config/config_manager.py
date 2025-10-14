"""
配置管理器 - 统一管理应用配置和环境变量
"""
from typing import Dict, Any, Optional, Union, List
import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime


class Environment(Enum):
    """环境类型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ConfigSource:
    """配置源"""
    name: str
    type: str  # file, env, remote
    path: Optional[str] = None
    priority: int = 0
    enabled: bool = True
    last_loaded: Optional[datetime] = None


class ConfigManager:
    """配置管理器"""

    def __init__(self, environment: Environment = Environment.DEVELOPMENT,
                 config_dir: str = "config", auto_reload: bool = False):
        self.environment = environment
        self.config_dir = Path(config_dir)
        self.auto_reload = auto_reload

        # 配置存储
        self.config: Dict[str, Any] = {}
        self.sources: List[ConfigSource] = []
        self.watchers: Dict[str, Any] = {}

        # 日志
        self.logger = logging.getLogger(__name__)

        # 初始化
        self._setup_default_sources()
        self.load_all_configs()

    def _setup_default_sources(self):
        """设置默认配置源"""
        # 环境变量源
        self.add_source(ConfigSource(
            name="environment_variables",
            type="env",
            priority=100,
            enabled=True
        ))

        # 默认配置文件
        self.add_source(ConfigSource(
            name="default_config",
            type="file",
            path=str(self.config_dir / "default.yaml"),
            priority=10,
            enabled=True
        ))

        # 环境特定配置
        env_config_file = self.config_dir / f"{self.environment.value}.yaml"
        if env_config_file.exists():
            self.add_source(ConfigSource(
                name=f"{self.environment.value}_config",
                type="file",
                path=str(env_config_file),
                priority=50,
                enabled=True
            ))

        # 本地覆盖配置
        local_config_file = self.config_dir / "local.yaml"
        if local_config_file.exists():
            self.add_source(ConfigSource(
                name="local_config",
                type="file",
                path=str(local_config_file),
                priority=80,
                enabled=True
            ))

    def add_source(self, source: ConfigSource):
        """添加配置源"""
        self.sources.append(source)
        # 按优先级排序
        self.sources.sort(key=lambda s: s.priority)
        self.logger.info(f"Added config source: {source.name}")

    def remove_source(self, source_name: str) -> bool:
        """移除配置源"""
        for i, source in enumerate(self.sources):
            if source.name == source_name:
                del self.sources[i]
                self.logger.info(f"Removed config source: {source_name}")
                return True
        return False

    def load_all_configs(self):
        """加载所有配置源"""
        self.config = {}

        for source in self.sources:
            if not source.enabled:
                continue

            try:
                config_data = self._load_source(source)
                if config_data:
                    self._merge_config(config_data)
                    source.last_loaded = datetime.now()
                    self.logger.debug(f"Loaded config from source: {source.name}")

            except Exception as e:
                self.logger.error(f"Failed to load config from {source.name}: {e}")

        self.logger.info(f"Loaded configuration for environment: {self.environment.value}")

    def _load_source(self, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """加载单个配置源"""
        if source.type == "file":
            return self._load_file_config(source.path)
        elif source.type == "env":
            return self._load_env_config()
        elif source.type == "remote":
            return self._load_remote_config(source.path)
        else:
            self.logger.warning(f"Unknown source type: {source.type}")
            return None

    def _load_file_config(self, file_path: str) -> Optional[Dict[str, Any]]:
        """加载文件配置"""
        if not file_path or not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    return yaml.safe_load(f)
                elif file_path.endswith('.json'):
                    return json.load(f)
                else:
                    self.logger.warning(f"Unsupported file format: {file_path}")
                    return None

        except Exception as e:
            self.logger.error(f"Failed to load config file {file_path}: {e}")
            return None

    def _load_env_config(self) -> Dict[str, Any]:
        """加载环境变量配置"""
        config = {}

        # 加载以AURA_开头的环境变量
        prefix = "AURA_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace('_', '.')
                config = self._set_nested_value(config, config_key, self._parse_env_value(value))

        return config

    def _load_remote_config(self, url: str) -> Optional[Dict[str, Any]]:
        """加载远程配置"""
        # 这里可以实现从远程服务器加载配置的逻辑
        # 例如：从配置中心、Consul、etcd等获取配置
        self.logger.info(f"Remote config loading not implemented for: {url}")
        return None

    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值"""
        # 尝试解析为JSON
        try:
            return json.loads(value)
        except:
            pass

        # 解析布尔值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # 解析数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except:
            pass

        # 返回原始字符串
        return value

    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
        """设置嵌套值"""
        keys = key.split('.')
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value
        return config

    def _merge_config(self, new_config: Dict[str, Any]):
        """合并配置"""
        self._deep_merge(self.config, new_config)

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]):
        """深度合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        current = self.config

        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """设置配置值"""
        self.config = self._set_nested_value(self.config, key, value)

    def has(self, key: str) -> bool:
        """检查配置是否存在"""
        return self.get(key) is not None

    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置段"""
        return self.get(section, {})

    def get_database_config(self) -> Dict[str, Any]:
        """获取数据库配置"""
        return self.get_section('database')

    def get_cache_config(self) -> Dict[str, Any]:
        """获取缓存配置"""
        return self.get_section('cache')

    def get_api_config(self) -> Dict[str, Any]:
        """获取API配置"""
        return self.get_section('api')

    def get_monitoring_config(self) -> Dict[str, Any]:
        """获取监控配置"""
        return self.get_section('monitoring')

    def get_security_config(self) -> Dict[str, Any]:
        """获取安全配置"""
        return self.get_section('security')

    def reload_config(self):
        """重新加载配置"""
        self.logger.info("Reloading configuration...")
        self.load_all_configs()

    def validate_config(self) -> List[str]:
        """验证配置"""
        errors = []

        # 验证必需的配置项
        required_keys = [
            'database.default.type',
            'cache.default.type',
            'api.host',
            'api.port'
        ]

        for key in required_keys:
            if not self.has(key):
                errors.append(f"Missing required config: {key}")

        # 验证数据库配置
        db_config = self.get_database_config()
        if db_config:
            default_db = db_config.get('default', {})
            if default_db.get('type') == 'postgresql':
                required_db_keys = ['host', 'port', 'database', 'username']
                for key in required_db_keys:
                    if key not in default_db:
                        errors.append(f"Missing database config: {key}")

        # 验证API配置
        api_config = self.get_api_config()
        if api_config:
            port = api_config.get('port')
            if port and (not isinstance(port, int) or port < 1 or port > 65535):
                errors.append("Invalid API port number")

        return errors

    def export_config(self, format: str = 'yaml') -> str:
        """导出配置"""
        if format == 'yaml':
            return yaml.dump(self.config, default_flow_style=False)
        elif format == 'json':
            return json.dumps(self.config, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            'environment': self.environment.value,
            'config_dir': str(self.config_dir),
            'sources': [
                {
                    'name': source.name,
                    'type': source.type,
                    'path': source.path,
                    'priority': source.priority,
                    'enabled': source.enabled,
                    'last_loaded': source.last_loaded.isoformat() if source.last_loaded else None
                }
                for source in self.sources
            ],
            'total_keys': len(self._flatten_dict(self.config)),
            'sections': list(self.config.keys())
        }

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """扁平化字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def create_default_config_files(self):
        """创建默认配置文件"""
        # 确保配置目录存在
        self.config_dir.mkdir(exist_ok=True)

        # 创建默认配置文件
        default_config = {
            'app': {
                'name': 'Aura Render',
                'version': '1.0.0',
                'debug': False
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'cors_origins': ['*'],
                'enable_docs': True
            },
            'database': {
                'default': {
                    'type': 'sqlite',
                    'database': 'aura_render.db'
                }
            },
            'cache': {
                'default': {
                    'type': 'memory',
                    'max_size': 1000
                }
            },
            'monitoring': {
                'enabled': True,
                'metrics_interval': 60,
                'alerts_enabled': True
            },
            'security': {
                'jwt_secret_key': 'your-secret-key-here',
                'token_expire_minutes': 60
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }

        default_file = self.config_dir / 'default.yaml'
        if not default_file.exists():
            with open(default_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            self.logger.info(f"Created default config file: {default_file}")

        # 创建开发环境配置
        dev_config = {
            'app': {'debug': True},
            'api': {'enable_docs': True},
            'database': {
                'default': {
                    'type': 'sqlite',
                    'database': 'aura_render_dev.db'
                }
            },
            'logging': {'level': 'DEBUG'}
        }

        dev_file = self.config_dir / 'development.yaml'
        if not dev_file.exists():
            with open(dev_file, 'w', encoding='utf-8') as f:
                yaml.dump(dev_config, f, default_flow_style=False)
            self.logger.info(f"Created development config file: {dev_file}")

        # 创建生产环境配置
        prod_config = {
            'app': {'debug': False},
            'api': {
                'enable_docs': False,
                'cors_origins': ['https://yourdomain.com']
            },
            'database': {
                'default': {
                    'type': 'postgresql',
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'aura_render',
                    'username': 'aura_user',
                    'password': '${AURA_DB_PASSWORD}'
                }
            },
            'cache': {
                'default': {
                    'type': 'redis',
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0
                }
            },
            'logging': {'level': 'WARNING'}
        }

        prod_file = self.config_dir / 'production.yaml'
        if not prod_file.exists():
            with open(prod_file, 'w', encoding='utf-8') as f:
                yaml.dump(prod_config, f, default_flow_style=False)
            self.logger.info(f"Created production config file: {prod_file}")


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    global _config_manager
    if _config_manager is None:
        env = Environment(os.getenv('AURA_ENV', 'development'))
        _config_manager = ConfigManager(environment=env)
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """获取配置值的便捷函数"""
    return get_config_manager().get(key, default)


def reload_config():
    """重新加载配置的便捷函数"""
    get_config_manager().reload_config()