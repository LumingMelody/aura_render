"""
Microsoft Edge TTS 客户端 - 免费语音合成服务
"""
from typing import Dict, List, Any, Optional
import asyncio
import uuid
import json
import websockets
import ssl
from .base_tts_client import (
    BaseTTSClient,
    TTSRequest,
    TTSResponse,
    VoiceProfile,
    VoiceGender,
    VoiceStyle
)


class EdgeTTSClient(BaseTTSClient):
    """Microsoft Edge TTS 客户端（免费）"""

    def __init__(self):
        # Edge TTS是免费服务，不需要API key
        super().__init__("", "wss://speech.platform.bing.com", timeout=30)
        self.rate_limit_delay = 0.1  # 较短的延迟

        # Edge TTS 预设语音
        self.edge_voices = {
            # 中文语音
            "xiaoxiao": VoiceProfile(
                voice_id="zh-CN-XiaoxiaoNeural",
                name="晓晓",
                language="zh-CN",
                gender=VoiceGender.FEMALE,
                age_range="adult",
                description="温柔自然的女声",
                styles=[VoiceStyle.NORMAL, VoiceStyle.CHEERFUL, VoiceStyle.CALM],
                quality_score=0.85
            ),
            "xiaoyi": VoiceProfile(
                voice_id="zh-CN-XiaoyiNeural",
                name="晓伊",
                language="zh-CN",
                gender=VoiceGender.FEMALE,
                age_range="adult",
                description="活泼可爱的女声",
                styles=[VoiceStyle.CHEERFUL, VoiceStyle.EXCITED],
                quality_score=0.8
            ),
            "yunjian": VoiceProfile(
                voice_id="zh-CN-YunjianNeural",
                name="云健",
                language="zh-CN",
                gender=VoiceGender.MALE,
                age_range="adult",
                description="成熟稳重的男声",
                styles=[VoiceStyle.NORMAL, VoiceStyle.PROFESSIONAL],
                quality_score=0.85
            ),
            "yunxi": VoiceProfile(
                voice_id="zh-CN-YunxiNeural",
                name="云希",
                language="zh-CN",
                gender=VoiceGender.MALE,
                age_range="adult",
                description="年轻活力的男声",
                styles=[VoiceStyle.NORMAL, VoiceStyle.FRIENDLY],
                quality_score=0.8
            ),
            # 英文语音
            "aria": VoiceProfile(
                voice_id="en-US-AriaNeural",
                name="Aria",
                language="en-US",
                gender=VoiceGender.FEMALE,
                age_range="adult",
                description="Clear and professional female voice",
                styles=[VoiceStyle.NORMAL, VoiceStyle.PROFESSIONAL, VoiceStyle.FRIENDLY],
                quality_score=0.9
            ),
            "davis": VoiceProfile(
                voice_id="en-US-DavisNeural",
                name="Davis",
                language="en-US",
                gender=VoiceGender.MALE,
                age_range="adult",
                description="Warm and engaging male voice",
                styles=[VoiceStyle.NORMAL, VoiceStyle.FRIENDLY],
                quality_score=0.85
            )
        }

    def _get_auth_headers(self) -> Dict[str, str]:
        # Edge TTS不需要认证
        return {}

    async def synthesize_speech(self, request: TTSRequest) -> TTSResponse:
        """合成语音"""
        start_time = asyncio.get_event_loop().time()

        try:
            # 构建SSML
            ssml = self._build_edge_ssml(request)

            # 通过WebSocket连接Edge TTS
            audio_data = await self._synthesize_via_websocket(ssml)

            processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

            return TTSResponse(
                audio_url="",
                audio_data=audio_data,
                duration=self._estimate_duration(request.text, request.speed),
                sample_rate=24000,
                format="mp3",
                size_bytes=len(audio_data) if audio_data else 0,
                processing_time_ms=processing_time,
                voice_id=request.voice_id,
                text=request.text,
                cost=0.0  # Edge TTS是免费的
            )

        except Exception as e:
            print(f"Edge TTS synthesis failed: {e}")
            raise

    async def _synthesize_via_websocket(self, ssml: str) -> bytes:
        """通过WebSocket连接进行语音合成"""
        url = "wss://speech.platform.bing.com/consumer/speech/synthesize/realtimestreaming/edge/v1"

        # 生成请求ID
        request_id = str(uuid.uuid4()).replace("-", "")

        # 创建SSL上下文
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
            "Origin": "chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold"
        }

        try:
            async with websockets.connect(url, ssl=ssl_context, extra_headers=headers) as websocket:
                # 发送配置消息
                config_message = (
                    f"X-Timestamp:{self._get_timestamp()}\r\n"
                    f"Content-Type:application/json; charset=utf-8\r\n"
                    f"Path:speech.config\r\n\r\n"
                    f'{{"context":{{"synthesis":{{"audio":{{"metadataoptions":{{"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"true"}},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}}}}}'
                )
                await websocket.send(config_message)

                # 发送SSML消息
                ssml_message = (
                    f"X-RequestId:{request_id}\r\n"
                    f"X-Timestamp:{self._get_timestamp()}\r\n"
                    f"Content-Type:application/ssml+xml\r\n"
                    f"Path:ssml\r\n\r\n"
                    f"{ssml}"
                )
                await websocket.send(ssml_message)

                # 接收音频数据
                audio_data = b""
                async for message in websocket:
                    if isinstance(message, bytes):
                        # 解析二进制消息
                        separator = b'\r\n\r\n'
                        if separator in message:
                            header, body = message.split(separator, 1)
                            header_str = header.decode('utf-8')

                            if 'Path:audio' in header_str:
                                audio_data += body
                    else:
                        # 文本消息，检查是否结束
                        if 'Path:turn.end' in message:
                            break

                return audio_data

        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            raise

    def _build_edge_ssml(self, request: TTSRequest) -> str:
        """构建Edge TTS的SSML"""
        # 映射语音样式
        edge_style = "general"
        if request.style == VoiceStyle.CHEERFUL:
            edge_style = "cheerful"
        elif request.style == VoiceStyle.SAD:
            edge_style = "sad"
        elif request.style == VoiceStyle.CALM:
            edge_style = "calm"
        elif request.style == VoiceStyle.FRIENDLY:
            edge_style = "friendly"

        # 构建SSML
        ssml = f'''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{request.language}">
    <voice name="{request.voice_id}">
        <mstts:express-as style="{edge_style}">
            <prosody rate="{self._format_rate(request.speed)}" pitch="{self._format_pitch(request.pitch)}" volume="{self._format_volume(request.volume)}">
                {self._escape_ssml(request.text)}
            </prosody>
        </mstts:express-as>
    </voice>
</speak>'''
        return ssml

    def _format_rate(self, speed: float) -> str:
        """格式化语速"""
        if speed == 1.0:
            return "medium"
        elif speed < 0.8:
            return "slow"
        elif speed < 1.2:
            return "medium"
        else:
            return "fast"

    def _format_pitch(self, pitch: float) -> str:
        """格式化音调"""
        if pitch == 1.0:
            return "medium"
        elif pitch < 0.8:
            return "low"
        elif pitch < 1.2:
            return "medium"
        else:
            return "high"

    def _format_volume(self, volume: float) -> str:
        """格式化音量"""
        if volume < 0.3:
            return "silent"
        elif volume < 0.6:
            return "soft"
        elif volume < 0.9:
            return "medium"
        else:
            return "loud"

    def _escape_ssml(self, text: str) -> str:
        """转义SSML特殊字符"""
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        return text

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        import datetime
        return datetime.datetime.utcnow().strftime('%a %b %d %Y %H:%M:%S GMT+0000 (UTC)')

    async def get_available_voices(self, language: str = None) -> List[VoiceProfile]:
        """获取可用语音列表"""
        voices = list(self.edge_voices.values())

        if language:
            voices = [v for v in voices if v.language.startswith(language)]

        return voices

    async def get_voice_details(self, voice_id: str) -> Optional[VoiceProfile]:
        """获取语音详情"""
        for voice in self.edge_voices.values():
            if voice.voice_id == voice_id:
                return voice
        return None

    def _estimate_duration(self, text: str, speed: float = 1.0) -> float:
        """估算音频时长"""
        char_count = len(text)

        # 基础语速（字符/秒）
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            # 中文
            base_chars_per_second = 6.0
        else:
            # 英文
            base_chars_per_second = 3.0

        duration = char_count / (base_chars_per_second * speed)
        return max(0.1, duration)

    def estimate_cost(self, text: str, voice_id: str) -> float:
        """估算合成成本"""
        return 0.0  # Edge TTS是免费的

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 尝试简单的语音合成测试
            test_request = TTSRequest(
                text="测试",
                voice_id="zh-CN-XiaoxiaoNeural"
            )
            result = await self.synthesize_speech(test_request)
            return result.audio_data is not None and len(result.audio_data) > 0
        except Exception:
            return False

    async def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表"""
        languages = set()
        for voice in self.edge_voices.values():
            languages.add(voice.language)
        return sorted(list(languages))

    async def test_voice(self, voice_id: str, test_text: str = None) -> Optional[TTSResponse]:
        """测试语音"""
        if not test_text:
            # 根据语音语言选择测试文本
            voice = await self.get_voice_details(voice_id)
            if voice and voice.language.startswith("zh"):
                test_text = "这是一个语音测试。"
            else:
                test_text = "This is a voice test."

        try:
            request = TTSRequest(
                text=test_text,
                voice_id=voice_id
            )
            return await self.synthesize_speech(request)
        except Exception as e:
            print(f"Voice test failed: {e}")
            return None