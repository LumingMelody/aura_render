"""
TTS服务基础客户端 - 统一接口定义
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
import time


class VoiceGender(Enum):
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceStyle(Enum):
    NORMAL = "normal"
    CHEERFUL = "cheerful"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    FRIENDLY = "friendly"
    CALM = "calm"
    PROFESSIONAL = "professional"


@dataclass
class VoiceProfile:
    """语音配置文件"""
    voice_id: str
    name: str
    language: str
    gender: VoiceGender
    age_range: str  # e.g., "adult", "child", "elderly"
    description: str
    styles: List[VoiceStyle]
    sample_rate: int = 24000
    quality_score: float = 0.8
    is_premium: bool = False


@dataclass
class TTSRequest:
    """TTS请求参数"""
    text: str
    voice_id: str
    language: str = "zh-CN"
    style: VoiceStyle = VoiceStyle.NORMAL
    speed: float = 1.0  # 语速 0.5-2.0
    pitch: float = 1.0  # 音调 0.5-2.0
    volume: float = 1.0  # 音量 0.0-1.0
    sample_rate: int = 24000
    format: str = "mp3"  # mp3, wav, pcm
    emotions: Optional[Dict[str, float]] = None  # 情感参数

    def __post_init__(self):
        # 限制参数范围
        self.speed = max(0.5, min(2.0, self.speed))
        self.pitch = max(0.5, min(2.0, self.pitch))
        self.volume = max(0.0, min(1.0, self.volume))


@dataclass
class TTSResponse:
    """TTS响应结果"""
    audio_url: str
    audio_data: Optional[bytes] = None
    duration: float = 0.0
    sample_rate: int = 24000
    format: str = "mp3"
    size_bytes: int = 0
    processing_time_ms: int = 0
    voice_id: str = ""
    text: str = ""
    cost: float = 0.0  # API调用成本


@dataclass
class EmotionMapping:
    """情感到语音参数的映射"""
    joy: Dict[str, float]
    sadness: Dict[str, float]
    anger: Dict[str, float]
    fear: Dict[str, float]
    surprise: Dict[str, float]
    neutral: Dict[str, float]


class BaseTTSClient(ABC):
    """TTS服务基础客户端"""

    def __init__(self, api_key: str, base_url: str, timeout: int = 30):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.last_request_time = 0
        self.rate_limit_delay = 1.0
        self.session: Optional[aiohttp.ClientSession] = None

        # 默认情感映射
        self.emotion_mapping = EmotionMapping(
            joy={"speed": 1.1, "pitch": 1.1, "volume": 1.0},
            sadness={"speed": 0.9, "pitch": 0.9, "volume": 0.8},
            anger={"speed": 1.2, "pitch": 1.1, "volume": 1.0},
            fear={"speed": 1.1, "pitch": 1.2, "volume": 0.9},
            surprise={"speed": 1.1, "pitch": 1.2, "volume": 1.0},
            neutral={"speed": 1.0, "pitch": 1.0, "volume": 1.0}
        )

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @abstractmethod
    async def synthesize_speech(self, request: TTSRequest) -> TTSResponse:
        """合成语音"""
        pass

    @abstractmethod
    async def get_available_voices(self, language: str = None) -> List[VoiceProfile]:
        """获取可用语音列表"""
        pass

    @abstractmethod
    async def get_voice_details(self, voice_id: str) -> Optional[VoiceProfile]:
        """获取语音详情"""
        pass

    async def batch_synthesize(self, requests: List[TTSRequest]) -> List[TTSResponse]:
        """批量合成语音"""
        tasks = []
        for request in requests:
            task = self.synthesize_speech(request)
            tasks.append(task)

        # 控制并发数量
        semaphore = asyncio.Semaphore(5)  # 最多5个并发请求

        async def limited_synthesize(request):
            async with semaphore:
                return await self.synthesize_speech(request)

        return await asyncio.gather(*[limited_synthesize(req) for req in requests])

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """发送HTTP请求"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))

        # 速率限制
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)

        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = self._get_auth_headers()

        if 'headers' in kwargs:
            headers.update(kwargs.pop('headers'))

        try:
            async with self.session.request(method, url, headers=headers, **kwargs) as response:
                self.last_request_time = time.time()

                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"TTS API error {response.status}: {error_text}")

                # 根据content-type决定如何处理响应
                content_type = response.headers.get('content-type', '')

                if 'application/json' in content_type:
                    return await response.json()
                elif 'audio/' in content_type or 'application/octet-stream' in content_type:
                    return {"audio_data": await response.read()}
                else:
                    return {"text": await response.text()}

        except Exception as e:
            print(f"TTS request failed: {e}")
            raise

    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """获取认证头"""
        pass

    def apply_emotion_parameters(self, request: TTSRequest, dominant_emotion: str, intensity: float = 1.0) -> TTSRequest:
        """根据情感调整语音参数"""
        emotion_params = getattr(self.emotion_mapping, dominant_emotion.lower(), self.emotion_mapping.neutral)

        # 应用情感强度
        request.speed = request.speed * (1 + (emotion_params["speed"] - 1) * intensity)
        request.pitch = request.pitch * (1 + (emotion_params["pitch"] - 1) * intensity)
        request.volume = request.volume * (1 + (emotion_params["volume"] - 1) * intensity)

        # 确保参数在有效范围内
        request.speed = max(0.5, min(2.0, request.speed))
        request.pitch = max(0.5, min(2.0, request.pitch))
        request.volume = max(0.0, min(1.0, request.volume))

        return request

    def select_voice_by_criteria(self, voices: List[VoiceProfile],
                               gender: VoiceGender = None,
                               style: VoiceStyle = None,
                               language: str = None,
                               prefer_premium: bool = False) -> Optional[VoiceProfile]:
        """根据条件选择语音"""
        filtered_voices = voices

        if gender:
            filtered_voices = [v for v in filtered_voices if v.gender == gender]

        if style:
            filtered_voices = [v for v in filtered_voices if style in v.styles]

        if language:
            filtered_voices = [v for v in filtered_voices if v.language == language]

        if not filtered_voices:
            return None

        # 排序逻辑：优先级 > 质量分数
        filtered_voices.sort(key=lambda v: (
            v.is_premium if prefer_premium else not v.is_premium,
            v.quality_score
        ), reverse=True)

        return filtered_voices[0]

    async def get_voice_preview(self, voice_id: str, sample_text: str = "这是一个语音预览示例") -> Optional[TTSResponse]:
        """获取语音预览"""
        try:
            request = TTSRequest(
                text=sample_text,
                voice_id=voice_id,
                format="mp3"
            )
            return await self.synthesize_speech(request)
        except Exception as e:
            print(f"Failed to get voice preview: {e}")
            return None

    def estimate_cost(self, text: str, voice_id: str) -> float:
        """估算合成成本（字符数基础）"""
        # 基础实现，子类可以重写
        char_count = len(text)
        base_cost_per_char = 0.001  # 每字符基础成本
        return char_count * base_cost_per_char

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            voices = await self.get_available_voices()
            return len(voices) > 0
        except Exception:
            return False