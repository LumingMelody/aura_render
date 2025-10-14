"""
密钥管理器 - 安全管理API密钥和敏感配置
"""

import os
import json
import base64
import hashlib
from typing import Any, Dict, Optional
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import logging

logger = logging.getLogger(__name__)


class SecretManager:
    """
    安全的密钥管理器，支持加密存储和环境变量管理
    """
    
    def __init__(self, secret_file: str = ".secrets.enc", master_key: Optional[str] = None):
        """
        初始化密钥管理器
        
        Args:
            secret_file: 加密密钥文件路径
            master_key: 主密钥（如果未提供，将从环境变量读取）
        """
        self.secret_file = Path(secret_file)
        self.master_key = master_key or os.getenv("MASTER_KEY")
        self._cipher = None
        self._secrets_cache = {}
        
        if self.master_key:
            self._initialize_cipher()
    
    def _initialize_cipher(self):
        """初始化加密器"""
        # 使用PBKDF2从主密钥派生加密密钥
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'aura_render_salt',  # 在生产环境应使用随机盐
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        self._cipher = Fernet(key)
    
    def _generate_master_key(self) -> str:
        """生成新的主密钥"""
        return Fernet.generate_key().decode()
    
    def save_secrets(self, secrets: Dict[str, Any]):
        """
        加密并保存密钥
        
        Args:
            secrets: 要保存的密钥字典
        """
        if not self._cipher:
            raise ValueError("未初始化加密器，请提供主密钥")
        
        # 加密密钥数据
        encrypted_data = self._cipher.encrypt(json.dumps(secrets).encode())
        
        # 保存到文件
        self.secret_file.write_bytes(encrypted_data)
        logger.info(f"密钥已加密保存到 {self.secret_file}")
        
        # 更新缓存
        self._secrets_cache = secrets
    
    def load_secrets(self) -> Dict[str, Any]:
        """
        加载并解密密钥
        
        Returns:
            解密后的密钥字典
        """
        if not self._cipher:
            raise ValueError("未初始化加密器，请提供主密钥")
        
        if not self.secret_file.exists():
            logger.warning(f"密钥文件 {self.secret_file} 不存在")
            return {}
        
        # 从文件读取加密数据
        encrypted_data = self.secret_file.read_bytes()
        
        # 解密
        try:
            decrypted_data = self._cipher.decrypt(encrypted_data)
            secrets = json.loads(decrypted_data.decode())
            self._secrets_cache = secrets
            return secrets
        except Exception as e:
            logger.error(f"解密密钥失败: {e}")
            return {}
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        获取单个密钥值
        
        Args:
            key: 密钥名称
            default: 默认值
        
        Returns:
            密钥值或默认值
        """
        # 优先从环境变量读取
        env_value = os.getenv(key)
        if env_value:
            return env_value
        
        # 从缓存或文件读取
        if not self._secrets_cache:
            self._secrets_cache = self.load_secrets()
        
        return self._secrets_cache.get(key, default)
    
    def set_secret(self, key: str, value: str):
        """
        设置单个密钥值
        
        Args:
            key: 密钥名称
            value: 密钥值
        """
        if not self._secrets_cache:
            self._secrets_cache = self.load_secrets()
        
        self._secrets_cache[key] = value
        self.save_secrets(self._secrets_cache)
    
    def validate_required_secrets(self, required_keys: list) -> tuple[bool, list]:
        """
        验证必需的密钥是否存在
        
        Args:
            required_keys: 必需的密钥列表
        
        Returns:
            (是否全部存在, 缺失的密钥列表)
        """
        missing_keys = []
        
        for key in required_keys:
            if not self.get_secret(key):
                missing_keys.append(key)
        
        return len(missing_keys) == 0, missing_keys
    
    def mask_secret(self, secret: str, visible_chars: int = 4) -> str:
        """
        掩码密钥用于日志输出
        
        Args:
            secret: 原始密钥
            visible_chars: 显示的字符数
        
        Returns:
            掩码后的密钥
        """
        if not secret or len(secret) <= visible_chars * 2:
            return "*" * len(secret) if secret else ""
        
        return f"{secret[:visible_chars]}{'*' * (len(secret) - visible_chars * 2)}{secret[-visible_chars:]}"
    
    def rotate_secret(self, key: str, new_value: str, keep_old: bool = True):
        """
        轮换密钥
        
        Args:
            key: 密钥名称
            new_value: 新密钥值
            keep_old: 是否保留旧密钥
        """
        if keep_old:
            old_value = self.get_secret(key)
            if old_value:
                self.set_secret(f"{key}_OLD", old_value)
        
        self.set_secret(key, new_value)
        logger.info(f"密钥 {key} 已轮换")
    
    @staticmethod
    def generate_api_key(prefix: str = "sk", length: int = 32) -> str:
        """
        生成新的API密钥
        
        Args:
            prefix: 密钥前缀
            length: 密钥长度
        
        Returns:
            生成的API密钥
        """
        random_bytes = os.urandom(length)
        key_hash = hashlib.sha256(random_bytes).hexdigest()[:length]
        return f"{prefix}-{key_hash}"
    
    def export_env_file(self, output_path: str = ".env.secure"):
        """
        导出密钥为环境变量文件
        
        Args:
            output_path: 输出文件路径
        """
        secrets = self.load_secrets()
        
        with open(output_path, 'w') as f:
            f.write("# Aura Render Secure Environment Variables\n")
            f.write("# Generated by SecretManager\n\n")
            
            for key, value in secrets.items():
                # 对敏感值进行部分掩码
                if any(sensitive in key.upper() for sensitive in ['KEY', 'SECRET', 'PASSWORD', 'TOKEN']):
                    display_value = self.mask_secret(value)
                    f.write(f"# {key}={display_value} (masked)\n")
                    f.write(f"{key}={value}\n")
                else:
                    f.write(f"{key}={value}\n")
        
        logger.info(f"环境变量已导出到 {output_path}")


class EnvValidator:
    """
    环境配置验证器
    """
    
    @staticmethod
    def validate_env_file(env_path: str = ".env") -> Dict[str, list]:
        """
        验证环境配置文件
        
        Args:
            env_path: 环境文件路径
        
        Returns:
            验证结果字典
        """
        results = {
            "missing": [],
            "invalid": [],
            "warnings": [],
            "info": []
        }
        
        # 必需的配置项
        required_configs = {
            "DASHSCOPE_API_KEY": "阿里云API密钥",
            "SECRET_KEY": "应用密钥",
            "DATABASE_URL": "数据库连接",
        }
        
        # 推荐的配置项
        recommended_configs = {
            "REDIS_URL": "Redis连接",
            "SENTRY_DSN": "错误监控",
            "BACKUP_ENABLED": "备份设置",
        }
        
        # 读取环境文件
        env_vars = {}
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        
        # 检查必需配置
        for key, desc in required_configs.items():
            if key not in env_vars:
                results["missing"].append(f"{key} ({desc})")
            elif not env_vars[key] or env_vars[key] == f"your-{key.lower().replace('_', '-')}":
                results["invalid"].append(f"{key} ({desc}) - 使用默认值")
        
        # 检查推荐配置
        for key, desc in recommended_configs.items():
            if key not in env_vars:
                results["warnings"].append(f"{key} ({desc}) - 建议配置")
        
        # 安全检查
        if "SECRET_KEY" in env_vars:
            if len(env_vars["SECRET_KEY"]) < 32:
                results["warnings"].append("SECRET_KEY 长度建议至少32个字符")
            if env_vars["SECRET_KEY"] == "your-super-secret-key-change-in-production":
                results["invalid"].append("SECRET_KEY 使用了默认值，存在安全风险")
        
        # 数据库检查
        if "DATABASE_URL" in env_vars:
            if "sqlite" in env_vars["DATABASE_URL"].lower():
                results["info"].append("使用SQLite数据库，仅适合开发环境")
        
        # 环境检查
        if "ENVIRONMENT" in env_vars:
            if env_vars["ENVIRONMENT"] == "production" and env_vars.get("DEBUG") == "true":
                results["warnings"].append("生产环境不应启用DEBUG模式")
        
        return results
    
    @staticmethod
    def print_validation_results(results: Dict[str, list]):
        """打印验证结果"""
        print("\n=== 环境配置验证结果 ===\n")
        
        if results["missing"]:
            print("❌ 缺失的必需配置:")
            for item in results["missing"]:
                print(f"  - {item}")
        
        if results["invalid"]:
            print("\n⚠️  无效的配置:")
            for item in results["invalid"]:
                print(f"  - {item}")
        
        if results["warnings"]:
            print("\n⚠️  警告:")
            for item in results["warnings"]:
                print(f"  - {item}")
        
        if results["info"]:
            print("\nℹ️  信息:")
            for item in results["info"]:
                print(f"  - {item}")
        
        if not any(results.values()):
            print("✅ 所有配置验证通过!")


# 使用示例
if __name__ == "__main__":
    # 1. 初始化密钥管理器
    manager = SecretManager()
    
    # 2. 生成新的API密钥
    new_api_key = SecretManager.generate_api_key(prefix="aura", length=40)
    print(f"生成的API密钥: {manager.mask_secret(new_api_key)}")
    
    # 3. 验证环境配置
    validator = EnvValidator()
    results = validator.validate_env_file(".env")
    validator.print_validation_results(results)
    
    # 4. 检查必需的密钥
    required = ["DASHSCOPE_API_KEY", "SECRET_KEY", "DATABASE_URL"]
    is_valid, missing = manager.validate_required_secrets(required)
    if not is_valid:
        print(f"\n缺失的密钥: {missing}")