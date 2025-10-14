# ai_content_generator/factory.py
from .registry import GENERATOR_REGISTRY
from typing import Dict, Any, List

class AIContentGeneratorFactory:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._instances = {}

    async def generate_chunk(self, **kwargs) -> Dict:
        is_talking_head = kwargs.get("is_talking_head", False)
        preferred_provider = kwargs.get("preferred_provider")  # 可选：指定供应商

        # 查找匹配的生成器
        candidates = [
            item for item in GENERATOR_REGISTRY.values()
            if item["is_talking_head"] == is_talking_head
        ]

        if preferred_provider:
            candidates = [c for c in candidates if c["metadata"].get("provider") == preferred_provider]

        if not candidates:
            raise RuntimeError(f"No generator found for is_talking_head={is_talking_head}")

        # 选择第一个（可扩展为评分机制）
        selected = candidates[0]
        cls = selected["class"]

        # 缓存实例
        if cls not in self._instances:
            self._instances[cls] = cls(**self._extract_config(cls))

        generator = self._instances[cls]
        return await generator.generate(**{k: v for k, v in kwargs.items() if k != 'preferred_provider'})

    def list_generators(self, is_talking_head: bool = None) -> List[Dict]:
        """
        列出所有可用生成器
        """
        items = GENERATOR_REGISTRY.values()
        if is_talking_head is not None:
            items = [i for i in items if i["is_talking_head"] == is_talking_head]
        return [
            {
                "key": key,
                "name": item["metadata"].get("name", key),
                "description": item["metadata"].get("description"),
                "features": item["metadata"].get("features", []),
                "tags": item["tags"]
            }
            for key, item in GENERATOR_REGISTRY.items()
        ]
    
    async def generate_image(self, prompt: str, style: str = None, generator_key: str = None, **kwargs):
        candidates = [
            item for item in GENERATOR_REGISTRY.values()
            if ImageGeneratorBase in item["class"].__bases__  # 判断是否为图片生成器
        ]
        if generator_key:
            candidates = [c for c in candidates if c["key"] == generator_key]
        
        if not candidates:
            raise RuntimeError("No image generator available")

        selected = candidates[0]
        cls = selected["class"]
        instance = self._instances.get(cls) or cls(**self._extract_config(cls))
        self._instances[cls] = instance

        return await instance.generate(prompt=prompt, style=style, **kwargs)
    
    # ai_content_generator/factory.py

async def generate_tts(self, text: str, voice: str = "default", generator_key: str = None, **kwargs) -> Dict:
    """
    生成语音
    """
    candidates = [
        item for item in GENERATOR_REGISTRY.values()
        if TTSGeneratorBase in item["class"].__bases__
    ]
    if generator_key:
        candidates = [c for c in candidates if c["key"] == generator_key]
    
    if not candidates:
        raise RuntimeError("No TTS generator available")

    selected = candidates[0]
    cls = selected["class"]
    instance = self._instances.get(cls) or cls(**self._extract_config(cls))
    self._instances[cls] = instance

    return await instance.generate(text=text, voice=voice, **kwargs)