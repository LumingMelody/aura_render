# ai_content_generator/base/talking_head_base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class TalkingHeadBase(BaseGenerator):
    """
    数字人生成器基类（驱动已有形象视频 + 文本/语音 合成新口型视频）

    核心能力：
    - 接受一个数字人形象视频（含人脸）
    - 接受朗读文本（用于TTS或唇形驱动）
    - 接受语音（可选，若未提供则使用文本生成TTS或提取原视频语音）
    - 输出口型同步的新视频
    """

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        返回该实现类的元信息，例如：
        {
            "name": "XXX数字人",
            "provider": "xxx",
            "supports_video_input": True,
            "requires_audio": False,
            "can_extract_audio": True,
            "output_formats": ["mp4", "webm"],
            "max_duration": 120
        }
        """
        pass

    @abstractmethod
    async def generate(
        self,
        video_url: str,
        text: str,
        audio_url: Optional[str] = None,
        voice_style: Optional[str] = None,
        output_format: str = "mp4",
        callback_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成口型同步的数字人视频

        Args:
            video_url (str): 数字人形象视频路径（本地路径或云存储URL）
            text (str): 要朗读的文本内容（用于生成语音或唇形驱动）
            audio_url (Optional[str]): 外部语音文件路径（可选）
                                      若为空，则根据文本生成TTS语音
            voice_style (Optional[str]): 语音风格（如 "professional", "friendly"）
            output_format (str): 输出视频格式，默认 mp4
            callback_url (Optional[str]): 异步任务完成后回调地址（可选）
            **kwargs: 扩展参数（如分辨率、唇形精度、GPU加速等）

        Returns:
            Dict[str, Any]: 结果字典，包含：
                {
                    "status": "success" | "failed" | "processing",
                    "video_url": "https://.../output.mp4",      # 输出视频地址
                    "audio_url": "https://.../tts.mp3",         # 实际使用的语音（TTS生成或原输入）
                    "duration": 15.3,                           # 视频时长
                    "text_used": "实际朗读文本",
                    "visemes": [ {"time": 0.1, "phoneme": "M"}, ... ],  # 口型数据（可选）
                    "task_id": "xxx",                           # 异步任务ID
                    "error": "错误信息（失败时）"
                }
        """
        pass

    async def validate_config(self) -> bool:
        """
        可选：检查 API Key、服务连通性、依赖环境（如FFmpeg）
        """
        return True

    async def extract_audio_from_video(self, video_url: str) -> str:
        """
        可选：从视频中提取音频（可用于语音驱动或TTS对齐）
        返回音频文件路径或URL
        """
        raise NotImplementedError("子类可实现音频提取逻辑")