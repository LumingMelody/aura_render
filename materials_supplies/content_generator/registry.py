# ai_content_generator/registry.py
from typing import Dict, List, Type
from .base.talking_head_base import TalkingHeadBase
from .base.pure_ai_video_base import PureAIVideoBase
from .base.image_generator_base import ImageGeneratorBase  # 新增导入

# 全局注册表
GENERATOR_REGISTRY: Dict[str, dict] = {}

def register_generator(
    key: str,
    cls: Type,
    description: str,
    tags: List[str],
    is_talking_head: bool
):
    """
    注册一个生成器实现
    """
    if key in GENERATOR_REGISTRY:
        raise ValueError(f"Generator {key} already registered")

    # 实例化获取 metadata
    try:
        instance = cls()  # 注意：这里需要无参构造或默认参数
        metadata = instance.metadata
    except Exception as e:
        metadata = {
            "name": key,
            "description": description,
            "error": str(e)
        }

    GENERATOR_REGISTRY[key] = {
        "class": cls,
        "metadata": metadata,
        "description": description,
        "tags": tags,
        "is_talking_head": is_talking_head
    }

# 使用装饰器自动注册（推荐）
def generator_plugin(key: str, description: str, tags: List[str], is_talking_head: bool):
    def wrapper(cls):
        register_generator(key, cls, description, tags, is_talking_head)
        return cls
    return wrapper