"""
配置管理模块 - 环境配置和应用设置管理
"""
from .config_manager import ConfigManager, Environment
from .settings import Settings, DatabaseSettings, CacheSettings, APISettings
from .deployment_config import DeploymentConfig, DockerConfig, KubernetesConfig

# 为了兼容 app.py 中的导入，也从根目录的 config.py 导入
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 创建默认设置实例
settings = Settings()

def get_settings():
    """获取设置实例"""
    global settings
    if settings is None:
        settings = Settings()
    return settings

__all__ = [
    'ConfigManager',
    'Environment',
    'Settings',
    'DatabaseSettings',
    'CacheSettings',
    'APISettings',
    'DeploymentConfig',
    'DockerConfig',
    'KubernetesConfig',
    'settings',
    'get_settings'
]