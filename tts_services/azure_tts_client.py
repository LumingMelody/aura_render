"""
Azure Text-to-Speech 客户端
"""
from typing import Dict, List, Any, Optional
import asyncio
import xml.etree.ElementTree as ET
from .base_tts_client import (
    BaseTTSClient,
    TTSRequest,
    TTSResponse,
    VoiceProfile,
    VoiceGender,
    VoiceStyle
)


class AzureTTSClient(BaseTTSClient):
    """Azure TTS 客户端"""

    def __init__(self, subscription_key: str, region: str = "eastus"):
        self.subscription_key = subscription_key
        self.region = region
        base_url = f"https://{region}.tts.speech.microsoft.com"
        super().__init__(subscription_key, base_url, timeout=30)

        # Azure语音样式映射
        self.azure_styles = {
            VoiceStyle.NORMAL: "general",
            VoiceStyle.CHEERFUL: "cheerful",
            VoiceStyle.SAD: "sad",
            VoiceStyle.ANGRY: "angry",
            VoiceStyle.EXCITED: "excited",
            VoiceStyle.FRIENDLY: "friendly",
            VoiceStyle.CALM: "calm",
            VoiceStyle.PROFESSIONAL: "newscast"
        }

        # 中文语音映射
        self.chinese_voices = {
            "xiaoxiao": VoiceProfile(
                voice_id="zh-CN-XiaoxiaoNeural",
                name="晓晓",
                language="zh-CN",
                gender=VoiceGender.FEMALE,
                age_range="adult",
                description="温柔甜美的女声",
                styles=[VoiceStyle.NORMAL, VoiceStyle.CHEERFUL, VoiceStyle.CALM],
                quality_score=0.9,
                is_premium=True
            ),
            "yunxi": VoiceProfile(
                voice_id="zh-CN-YunxiNeural",
                name="云希",
                language="zh-CN",
                gender=VoiceGender.MALE,
                age_range="adult",
                description="成熟稳重的男声",
                styles=[VoiceStyle.NORMAL, VoiceStyle.PROFESSIONAL, VoiceStyle.CALM],
                quality_score=0.9,
                is_premium=True
            ),
            "xiaoyi": VoiceProfile(
                voice_id="zh-CN-XiaoyiNeural",
                name="晓伊",
                language="zh-CN",
                gender=VoiceGender.FEMALE,
                age_range="adult",
                description="活泼可爱的女声",
                styles=[VoiceStyle.CHEERFUL, VoiceStyle.EXCITED, VoiceStyle.FRIENDLY],
                quality_score=0.85,
                is_premium=True
            )
        }

    def _get_auth_headers(self) -> Dict[str, str]:
        return {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-160kbitrate-mono-mp3"
        }

    async def synthesize_speech(self, request: TTSRequest) -> TTSResponse:
        """合成语音"""
        start_time = asyncio.get_event_loop().time()

        # 构建SSML
        ssml = self._build_ssml(request)

        # 设置输出格式
        format_map = {
            "mp3": "audio-24khz-160kbitrate-mono-mp3",
            "wav": "riff-24khz-16bit-mono-pcm",
            "pcm": "raw-24khz-16bit-mono-pcm"
        }

        headers = self._get_auth_headers()
        headers["X-Microsoft-OutputFormat"] = format_map.get(request.format, format_map["mp3"])

        try:
            response = await self._make_request(
                "POST",
                "/cognitiveservices/v1",
                data=ssml.encode('utf-8'),
                headers=headers
            )

            processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            audio_data = response.get("audio_data")

            if not audio_data:
                raise Exception("No audio data received from Azure TTS")

            return TTSResponse(
                audio_url="",  # Azure返回音频数据，不是URL
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
            print(f"Azure TTS synthesis failed: {e}")
            raise

    async def get_available_voices(self, language: str = None) -> List[VoiceProfile]:
        """获取可用语音列表"""
        try:
            # Azure提供语音列表API
            response = await self._make_request(
                "GET",
                "/cognitiveservices/voices/list",
                headers={"Ocp-Apim-Subscription-Key": self.subscription_key}
            )

            voices = []
            voice_list = response if isinstance(response, list) else response.get("voices", [])

            for voice_data in voice_list:
                if language and not voice_data.get("Locale", "").startswith(language):
                    continue

                # 解析语音信息
                gender = VoiceGender.FEMALE if voice_data.get("Gender") == "Female" else VoiceGender.MALE

                # 获取支持的样式
                styles = [VoiceStyle.NORMAL]
                style_list = voice_data.get("StyleList", [])
                for style in style_list:
                    for voice_style, azure_style in self.azure_styles.items():
                        if style.lower() == azure_style.lower():
                            styles.append(voice_style)

                voice_profile = VoiceProfile(
                    voice_id=voice_data.get("ShortName", ""),
                    name=voice_data.get("DisplayName", ""),
                    language=voice_data.get("Locale", ""),
                    gender=gender,
                    age_range="adult",
                    description=voice_data.get("Description", ""),
                    styles=list(set(styles)),
                    quality_score=0.9 if "Neural" in voice_data.get("VoiceType", "") else 0.7,
                    is_premium="Neural" in voice_data.get("VoiceType", "")
                )
                voices.append(voice_profile)

            # 如果API调用失败，返回预设的中文语音
            if not voices and (not language or language.startswith("zh")):
                voices = list(self.chinese_voices.values())

            return voices

        except Exception as e:
            print(f"Failed to get Azure voices: {e}")
            # 返回预设语音作为后备
            if not language or language.startswith("zh"):
                return list(self.chinese_voices.values())
            return []

    async def get_voice_details(self, voice_id: str) -> Optional[VoiceProfile]:
        """获取语音详情"""
        voices = await self.get_available_voices()
        for voice in voices:
            if voice.voice_id == voice_id:
                return voice
        return None

    def _build_ssml(self, request: TTSRequest) -> str:
        """构建SSML文档"""
        # 确定语音样式
        style = self.azure_styles.get(request.style, "general")

        # 构建SSML
        ssml_parts = [
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"',
            ' xmlns:mstts="https://www.w3.org/2001/mstts"',
            f' xml:lang="{request.language}">',
            f'<voice name="{request.voice_id}">'
        ]

        # 添加样式和参数
        if request.style != VoiceStyle.NORMAL:
            ssml_parts.append(f'<mstts:express-as style="{style}">')

        # 添加韵律参数
        prosody_attrs = []
        if request.speed != 1.0:
            speed_percent = f"{int((request.speed - 1) * 100):+d}%"
            prosody_attrs.append(f'rate="{speed_percent}"')

        if request.pitch != 1.0:
            pitch_percent = f"{int((request.pitch - 1) * 50):+d}%"
            prosody_attrs.append(f'pitch="{pitch_percent}"')

        if request.volume != 1.0:
            volume_percent = f"{int(request.volume * 100)}%"
            prosody_attrs.append(f'volume="{volume_percent}"')

        if prosody_attrs:
            ssml_parts.append(f'<prosody {" ".join(prosody_attrs)}>')

        # 添加文本内容
        ssml_parts.append(self._escape_ssml(request.text))

        # 关闭标签
        if prosody_attrs:
            ssml_parts.append('</prosody>')

        if request.style != VoiceStyle.NORMAL:
            ssml_parts.append('</mstts:express-as>')

        ssml_parts.extend(['</voice>', '</speak>'])

        return ''.join(ssml_parts)

    def _escape_ssml(self, text: str) -> str:
        """转义SSML特殊字符"""
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        return text

    def _estimate_duration(self, text: str, speed: float = 1.0) -> float:
        """估算音频时长"""
        # 中文：平均每分钟400字
        # 英文：平均每分钟200词
        char_count = len(text)

        # 基础语速（字符/秒）
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            # 包含中文
            base_chars_per_second = 6.7  # 400字/分钟
        else:
            # 英文
            base_chars_per_second = 3.3  # 200词/分钟，假设平均每词5字符

        duration = char_count / (base_chars_per_second * speed)
        return max(0.1, duration)  # 最少0.1秒

    def estimate_cost(self, text: str, voice_id: str) -> float:
        """估算合成成本"""
        char_count = len(text)

        # Azure按字符计费，神经语音更贵
        if "Neural" in voice_id:
            cost_per_char = 0.000016  # $0.000016 per character for Neural voices
        else:
            cost_per_char = 0.000004  # $0.000004 per character for Standard voices

        return char_count * cost_per_char

    async def get_voice_styles(self, voice_id: str) -> List[str]:
        """获取语音支持的样式"""
        voice = await self.get_voice_details(voice_id)
        if voice:
            return [style.value for style in voice.styles]
        return ["general"]