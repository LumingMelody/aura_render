# ai_content_generator/base/tts_generator_base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class TTSGeneratorBase(BaseGenerator):
    """
    TTS（文本转语音）生成器抽象基类
    所有语音合成模型需继承此类
    """

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        返回该生成器的元信息（必须实现）
        """
        pass

    @abstractmethod
    async def generate(
        self,
        text: str,
        voice: str = "default",
        language: str = "zh",
        speed: float = 1.0,
        pitch: float = 1.0,
        style: Optional[str] = None,
        style_intensity: float = 1.0,
        output_format: str = "mp3",
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成语音的核心接口

        Args:
            text: 输入文本
            voice: 音色名称（如 "xiaoyi", "zh-CN-Xiaoxiao"）
            language: 语言代码（如 "zh", "en"）
            speed: 语速（0.5 ~ 2.0）
            pitch: 音调
            style: 语音风格（如 "calm", "excited", "narration"）
            style_intensity: 风格强度
            output_format: 输出格式（"mp3", "wav", "pcm"）
            **kwargs: 扩展参数（如 seed, sample_rate, sentence_break 等）

        Returns:
            {
                "audio_url": "https://.../speech.mp3",
                "local_path": "/tmp/speech.mp3",
                "duration": 12.3,
                "sample_rate": 24000,
                "format": "mp3",
                "visemes": [ {"time": 0.1, "phoneme": "AH", "duration": 0.08}, ... ],  # 口型数据
                "words": [ {"word": "你好", "start": 0.0, "end": 0.5}, ... ],  # 分词时间戳
                "status": "success"
            }
        """
        pass

    async def validate_config(self) -> bool:
        """
        可选：检查 API 密钥、服务连通性
        """
        return True

    async def list_voices(self, language: str = None) -> List[Dict[str, Any]]:
        """
        可选：列出支持的音色
        """
        return []