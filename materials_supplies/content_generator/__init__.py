# ai_content_generator/__init__.py
from .factory import AIContentGeneratorFactory

# 也可暴露具体模块
from .talking_head import TalkingHeadGenerator
from .pure_ai_video import PureAIVideoGenerator