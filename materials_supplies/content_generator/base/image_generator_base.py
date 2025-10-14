# ai_content_generator/base/image_generator_base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class ImageGeneratorBase(BaseGenerator):
    """
    图片生成器抽象基类
    所有图片生成模型需继承此类
    """

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        返回该生成器的元信息（必须实现）
        """
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style: Optional[str] = None,
        size: str = "1024x1024",
        quality: str = "standard",  # standard, high
        n: int = 1,  # 生成数量
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成图片的核心接口

        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词
            style: 风格（如: realistic, anime, oil_painting）
            size: 图片尺寸（如: 1024x1024, 1280x720）
            quality: 质量等级
            n: 生成数量
            **kwargs: 扩展参数（如 seed, steps, sampler 等）

        Returns:
            {
                "images": [{"url": "...", "base64": "...", "size": "..."}],
                "prompt": "实际使用的提示词",
                "metadata": {生成参数},
                "duration": 2.3,
                "status": "success"
            }
        """
        pass

    async def enhance_prompt(self, prompt: str, style: str = None) -> str:
        """
        可选：默认提示词增强（可被子类重写）
        """
        if style:
            return f"[风格: {style}] {prompt}"
        return prompt

    async def validate_config(self) -> bool:
        """
        可选：检查 API 密钥、服务连通性
        """
        return True