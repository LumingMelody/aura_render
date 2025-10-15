"""
Materials Matcher Module

Provides different types of material matchers for video generation:
- Main video matcher for primary video content
- BGM matcher for background music
- Intelligent video matcher with AI capabilities
- Supplement matcher for additional materials
- And more...
"""

# Import only what actually exists
try:
    from .main_video_matcher import MainVideoMatcher
except ImportError:
    MainVideoMatcher = None

try:
    from .intelligent_video_matcher import Intelligent_video_matcher
except ImportError:
    Intelligent_video_matcher = None

try:
    from .bgm_matcher import match_bgm
except ImportError:
    match_bgm = None

__all__ = [
    'MainVideoMatcher',
    'Intelligent_video_matcher',
    'match_bgm',
]
