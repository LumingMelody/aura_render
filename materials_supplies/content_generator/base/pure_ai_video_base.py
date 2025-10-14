# ai_content_generator/base/pure_ai_video_base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class PureAIVideoBase(BaseGenerator):
    """
    纯 AI 视频生成器基类（文生视频 / 图生视频 / 首尾帧控制视频）

    核心能力：
    - 根据文本提示词生成视频
    - 可选：使用首帧图像作为起始画面
    - 可选：使用尾帧图像作为结束画面（需首帧存在）
    - 支持多种生成模式：text-to-video, image-to-video, image+text-to-video, image-pair-to-video
    """

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        返回该实现类的元信息，例如：
        {
            "name": "Pika Labs",
            "provider": "pika",
            "supports_image_start": True,
            "supports_image_end": True,
            "supported_formats": ["mp4", "webm"],
            "max_duration": 5.0,
            "description": "支持首帧控制的AI视频生成"
        }
        """
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        image_start: Optional[str] = None,
        image_end: Optional[str] = None,
        duration: float = 2.0,
        output_format: str = "mp4",
        motion_intensity: str = "medium",  # "low", "medium", "high"
        callback_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成 AI 视频

        Args:
            prompt (str): 视频内容提示词（必填）
            image_start (Optional[str]): 首帧图像路径或URL（可选）
            image_end (Optional[str]): 尾帧图像路径或URL（可选，但若提供则 image_start 必须存在）
            duration (float): 视频时长（秒），需在模型支持范围内
            output_format (str): 输出格式，如 "mp4", "webm"
            motion_intensity (str): 动作强度（用于控制镜头运动）
            callback_url (Optional[str]): 异步任务完成后的回调地址
            **kwargs: 扩展参数（如 seed, fps, resolution, camera_motion 等）

        Returns:
            Dict[str, Any]: 生成结果，包含：
                {
                    "status": "success" | "failed" | "processing",
                    "video_url": "https://.../output.mp4",
                    "duration": 3.0,
                    "prompt_used": "实际使用的提示词",
                    "image_start_used": "https://.../start.png",  # 实际使用的首帧
                    "image_end_used": "https://.../end.png",      # 实际使用的尾帧
                    "task_id": "xxx",
                    "metadata": {
                        "model": "pika-1.0",
                        "motion": "medium",
                        "format": "mp4"
                    },
                    "error": "错误信息（失败时）"
                }
        """
        pass

    async def enhance_prompt(self, prompt: str) -> str:
        """
        可选：提示词增强逻辑（可被子类重写）
        """
        # 示例：添加通用质量词
        quality_booster = "high quality, 4K, detailed, cinematic"
        return f"{prompt}, {quality_booster}"

    async def validate_config(self) -> bool:
        """
        可选：检查 API Key、服务状态等
        """
        return True

    async def validate_frames(self, image_start: Optional[str], image_end: Optional[str]) -> bool:
        """
        验证首尾帧逻辑合法性

        Raises:
            ValueError: 当 image_end 存在但 image_start 为空时抛出
        """
        if image_end is not None and image_start is None:
            raise ValueError("Error: 'image_end' is provided but 'image_start' is missing. "
                           "A start frame is required when an end frame is specified.")
        return True