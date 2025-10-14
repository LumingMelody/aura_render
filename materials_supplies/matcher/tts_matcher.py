from materials_supplies.models import TTSRequest, TTSResponse  # 替换为你的实际路径
import random
import asyncio
from typing import List

# 模拟可用的语音角色（音色）
VOICES = ["zh-CN-XiaoxiaoNeural", "zh-CN-YunyangNeural", "en-US-JennyNeural", "ja-JP-AoiNeural"]

# 模拟音频基础时长（根据文本长度估算，单位：秒/字，含随机波动）
def estimate_duration(text: str, speed: float) -> float:
    base_duration_per_char = 0.06  # 平均每字0.06秒
    fluctuation = random.uniform(-0.01, 0.01)
    total_chars = len(text)
    estimated = total_chars * (base_duration_per_char + fluctuation)
    return max(1.0, estimated / speed)  # 考虑语速调整

async def match_tts(request: TTSRequest) -> List[TTSResponse]:
    """
    模拟生成 TTS 语音文件，返回一个或多个候选语音结果
    """
    # 模拟网络延迟或服务处理时间
    await asyncio.sleep(0.1)

    results = []

    # 可生成多个候选（例如不同音色）
    candidate_voices = [request.voice] if request.voice else random.sample(VOICES, 2)

    for voice in candidate_voices:
        # 模拟生成音频 URL（可加入哈希或随机标识）
        audio_id = random.randint(10000, 99999)
        url = f"https://tts-audio.com/generated/{audio_id}.mp3"

        # 估算语音时长
        duration = estimate_duration(request.text, request.speed)

        # 如果请求中指定了期望时长，可做简单对齐（如拉伸或裁剪提示）
        if request.duration > 0:
            # 模拟通过变速等方式尽量匹配目标时长
            speed_adjust = duration / request.duration
            adjusted_duration = duration / speed_adjust
            speed_used = request.speed * speed_adjust
        else:
            adjusted_duration = duration
            speed_used = request.speed

        result = TTSResponse(
            url=url,
            text=request.text,
            voice=voice,
            duration=round(adjusted_duration, 2),
            speed=round(speed_used, 2)
        )
        results.append(result)

    return results