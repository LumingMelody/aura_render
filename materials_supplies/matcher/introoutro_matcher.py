from materials_supplies.models import IntroOutroRequest, IntroOutroResponse
import random
import asyncio
from typing import List

# 模拟可用的片头片尾视频资源（含画面+音频）
INTRO_VIDEO_CANDIDATES = [
    {"url": "https://media.com/intro-tech-animation.mp4", "duration": 6.0},
    {"url": "https://media.com/intro-logo-reveal.mp4", "duration": 4.0},
    {"url": "https://media.com/intro-cinematic-open.mp4", "duration": 8.0},
]

OUTRO_VIDEO_CANDIDATES = [
    {"url": "https://media.com/outro-simple-fade.mp4", "duration": 5.0},
    {"url": "https://media.com/outro-epic-end.mp4", "duration": 10.0},
    {"url": "https://media.com/outro-call-to-action-chinese.mp4", "duration": 7.0},
]

async def match_introoutro(request: IntroOutroRequest) -> List[IntroOutroResponse]:
    """
    模拟匹配合适的片头/片尾 **视频素材**（含画面与音频）
    返回裁剪区间，供后续视频合成使用
    """
    await asyncio.sleep(0.1)  # 模拟服务调用延迟

    results: List[IntroOutroResponse] = []

    # === 匹配片头视频 ===
    if request.intro_required:
        candidate = random.choice(INTRO_VIDEO_CANDIDATES)
        total_duration = candidate["duration"]
        video_url = candidate["url"]

        if request.duration >= total_duration:
            # 如果所需时长更长，就用完整视频
            cut_start = 0.0
            cut_end = total_duration
        else:
            # 否则随机裁剪一段 request.duration 长度
            cut_start = random.uniform(0, total_duration - request.duration)
            cut_end = cut_start + request.duration

        results.append(IntroOutroResponse(
            type="intro",
            video_url=video_url,
            audio_embedded=True,        # 视频自带音频
            total_duration=total_duration,
            cut_start=cut_start,
            cut_end=cut_end
        ))

    # === 匹配片尾视频 ===
    if request.outro_required:
        candidate = random.choice(OUTRO_VIDEO_CANDIDATES)
        total_duration = candidate["duration"]
        video_url = candidate["url"]

        if request.duration >= total_duration:
            cut_start = 0.0
            cut_end = total_duration
        else:
            cut_start = random.uniform(0, total_duration - request.duration)
            cut_end = cut_start + request.duration

        results.append(IntroOutroResponse(
            type="outro",
            video_url=video_url,
            audio_embedded=True,
            total_duration=total_duration,
            cut_start=cut_start,
            cut_end=cut_end
        ))

    return results