"""
OpenAI Text-to-Speech 客户端
"""
from typing import Dict, List, Any, Optional
import asyncio
import json
from .base_tts_client import (
    BaseTTSClient,
    TTSRequest,
    TTSResponse,
    VoiceProfile,
    VoiceGender,
    VoiceStyle
)


class OpenAITTSClient(BaseTTSClient):
    """OpenAI TTS 客户端"""

    def __init__(self, api_key: str, model: str = "tts-1-hd"):
        super().__init__(api_key, "https://api.openai.com/v1", timeout=30)
        self.model = model  # tts-1 或 tts-1-hd

        # OpenAI预设语音
        self.openai_voices = {
            "alloy": VoiceProfile(
                voice_id="alloy",
                name="Alloy",
                language="en-US",
                gender=VoiceGender.NEUTRAL,
                age_range="adult",
                description="Balanced and natural voice",
                styles=[VoiceStyle.NORMAL, VoiceStyle.PROFESSIONAL],
                quality_score=0.85,
                is_premium=False
            ),
            "echo": VoiceProfile(
                voice_id="echo",
                name="Echo",
                language="en-US",
                gender=VoiceGender.MALE,
                age_range="adult",
                description="Warm and engaging male voice",
                styles=[VoiceStyle.NORMAL, VoiceStyle.FRIENDLY],
                quality_score=0.8,
                is_premium=False
            ),
            "fable": VoiceProfile(
                voice_id="fable",
                name="Fable",
                language="en-US",
                gender=VoiceGender.MALE,
                age_range="adult",
                description="Expressive storytelling voice",
                styles=[VoiceStyle.NORMAL, VoiceStyle.EXCITED],
                quality_score=0.85,
                is_premium=False
            ),
            "onyx": VoiceProfile(
                voice_id="onyx",
                name="Onyx",
                language="en-US",
                gender=VoiceGender.MALE,
                age_range="adult",
                description="Deep and authoritative voice",
                styles=[VoiceStyle.NORMAL, VoiceStyle.PROFESSIONAL],
                quality_score=0.85,
                is_premium=False
            ),
            "nova": VoiceProfile(
                voice_id="nova",
                name="Nova",
                language="en-US",
                gender=VoiceGender.FEMALE,
                age_range="adult",
                description="Bright and energetic female voice",
                styles=[VoiceStyle.NORMAL, VoiceStyle.CHEERFUL, VoiceStyle.EXCITED],
                quality_score=0.9,
                is_premium=False
            ),
            "shimmer": VoiceProfile(
                voice_id="shimmer",
                name="Shimmer",
                language="en-US",
                gender=VoiceGender.FEMALE,
                age_range="adult",
                description="Gentle and soothing female voice",
                styles=[VoiceStyle.NORMAL, VoiceStyle.CALM, VoiceStyle.FRIENDLY],
                quality_score=0.9,
                is_premium=False
            )
        }

    def _get_auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def synthesize_speech(self, request: TTSRequest) -> TTSResponse:
        """合成语音"""
        start_time = asyncio.get_event_loop().time()

        # OpenAI TTS请求参数
        payload = {
            "model": self.model,
            "input": request.text,
            "voice": request.voice_id,
            "response_format": self._map_format(request.format),
            "speed": max(0.25, min(4.0, request.speed))  # OpenAI支持0.25-4.0
        }

        try:
            # OpenAI TTS API返回音频流
            headers = self._get_auth_headers()
            headers["Accept"] = "audio/*"

            response = await self._make_request(
                "POST",
                "/audio/speech",
                json=payload,
                headers=headers
            )

            processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            audio_data = response.get("audio_data")

            if not audio_data:
                raise Exception("No audio data received from OpenAI TTS")

            return TTSResponse(
                audio_url="",  # OpenAI返回音频数据，不是URL
                audio_data=audio_data,
                duration=self._estimate_duration(request.text, request.speed),
                sample_rate=24000,
                format=request.format,
                size_bytes=len(audio_data),
                processing_time_ms=processing_time,
                voice_id=request.voice_id,
                text=request.text,
                cost=self.estimate_cost(request.text, request.voice_id)
            )

        except Exception as e:
            print(f"OpenAI TTS synthesis failed: {e}")
            raise

    async def get_available_voices(self, language: str = None) -> List[VoiceProfile]:
        """获取可用语音列表"""
        voices = list(self.openai_voices.values())

        # OpenAI的语音主要是英文，但支持多语言输入
        if language and not language.startswith("en"):
            # 为非英文语言创建多语言变体
            multilingual_voices = []
            for voice in voices:
                multilingual_voice = VoiceProfile(
                    voice_id=voice.voice_id,
                    name=f"{voice.name} ({language})",
                    language=language,
                    gender=voice.gender,
                    age_range=voice.age_range,
                    description=f"{voice.description} - Multilingual support",
                    styles=voice.styles,
                    quality_score=voice.quality_score * 0.9,  # 略降质量分数
                    is_premium=voice.is_premium
                )
                multilingual_voices.append(multilingual_voice)
            return multilingual_voices

        return voices

    async def get_voice_details(self, voice_id: str) -> Optional[VoiceProfile]:
        """获取语音详情"""
        return self.openai_voices.get(voice_id)

    def _map_format(self, format: str) -> str:
        """映射音频格式"""
        format_map = {
            "mp3": "mp3",
            "wav": "wav",
            "pcm": "pcm",
            "opus": "opus",
            "aac": "aac",
            "flac": "flac"
        }
        return format_map.get(format.lower(), "mp3")

    def _estimate_duration(self, text: str, speed: float = 1.0) -> float:
        """估算音频时长"""
        # 英文：平均每分钟150-200词
        # 中文：平均每分钟300-400字
        char_count = len(text)

        # 检测主要语言
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            # 包含中文
            base_chars_per_second = 5.5  # 约330字/分钟
        else:
            # 英文等其他语言
            word_count = len(text.split())
            base_words_per_second = 2.8  # 约170词/分钟
            duration = word_count / (base_words_per_second * speed)
            return max(0.1, duration)

        duration = char_count / (base_chars_per_second * speed)
        return max(0.1, duration)

    def estimate_cost(self, text: str, voice_id: str) -> float:
        """估算合成成本"""
        char_count = len(text)

        # OpenAI按字符计费
        if self.model == "tts-1-hd":
            cost_per_1k_chars = 0.030  # $0.030 per 1K characters for HD model
        else:
            cost_per_1k_chars = 0.015  # $0.015 per 1K characters for standard model

        return (char_count / 1000) * cost_per_1k_chars

    async def create_speech_with_emotions(self, text: str, voice_id: str, emotions: Dict[str, float]) -> TTSResponse:
        """根据情感参数合成语音"""
        # OpenAI TTS本身不支持情感参数，但我们可以通过调整语速来模拟
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        intensity = max(emotions.values())

        request = TTSRequest(
            text=text,
            voice_id=voice_id,
            speed=1.0,
            pitch=1.0,
            volume=1.0
        )

        # 应用情感参数
        request = self.apply_emotion_parameters(request, dominant_emotion, intensity)

        return await self.synthesize_speech(request)

    async def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model": self.model,
            "quality": "HD" if "hd" in self.model else "Standard",
            "supported_formats": ["mp3", "wav", "pcm", "opus", "aac", "flac"],
            "speed_range": [0.25, 4.0],
            "max_input_length": 4096,  # 字符数限制
            "languages_supported": [
                "en", "zh", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "ar", "hi"
            ]
        }

    async def validate_input(self, text: str) -> Dict[str, Any]:
        """验证输入文本"""
        result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }

        if len(text) > 4096:
            result["valid"] = False
            result["errors"].append("Text exceeds maximum length of 4096 characters")

        if not text.strip():
            result["valid"] = False
            result["errors"].append("Text cannot be empty")

        # 检查可能的问题字符
        if any(ord(char) > 0x10FFFF for char in text):
            result["warnings"].append("Text contains unsupported Unicode characters")

        return result