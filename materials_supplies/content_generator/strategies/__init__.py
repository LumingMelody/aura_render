# strategies/__init__.py
from .strategy_registry import register_strategy
from .storyboard_to_video_strategy import StoryboardToVideoStrategy

# strategies/__init__.py
from .strategy_registry import register_strategy
from .talking_avatar_strategy import TalkingAvatarStoryStrategy

register_strategy("talking_avatar")(TalkingAvatarStoryStrategy)

register_strategy("storyboard_to_video")(StoryboardToVideoStrategy)