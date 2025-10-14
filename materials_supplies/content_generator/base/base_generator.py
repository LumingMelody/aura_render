# ai_content_pipeline/core/base_generator.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseGenerator(ABC):
    """
    所有智能生成器的抽象基类
    """

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        返回生成器元信息
        """
        pass

    @abstractmethod
    async def generate(self, **kwargs) -> Dict[str, Any]:
        """
        核心生成接口
        """
        pass

    async def validate(self) -> bool:
        """
        验证配置有效性
        """
        return True

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        """
        return {"status": "healthy", "generator": self.metadata["name"]}