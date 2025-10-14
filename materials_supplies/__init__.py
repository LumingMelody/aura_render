
from .matcher.bgm_matcher import match_bgm
from .matcher.sfx_matcher import match_sfx
from .matcher.font_matcher import match_font
from .matcher.supplement_matcher import match_supplement
from .matcher.intelligent_video_matcher import Intelligent_video_matcher
from .matcher.introoutro_matcher import match_introoutro
from .matcher.tts_matcher import match_tts


from .models import BGMRequest, BGMResponse, SFXRequest, SFXResponse, FontRequest, FontResponse, SupplementRequest, SupplementResponse,  IntroOutroRequest, IntroOutroResponse, TTSRequest, TTSResponse

async def match_intelligent_video(context: dict) -> dict:
    ivm = Intelligent_video_matcher()
    return await ivm.match_intelligent_video(context)

