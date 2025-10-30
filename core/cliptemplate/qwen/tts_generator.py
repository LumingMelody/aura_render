"""
千问 TTS 语音合成模块

根据字幕文本生成语音音频并上传到OSS
"""

import os
import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# 使用项目的日志系统
try:
    from utils.logger import get_logger, LogCategory
    logger = get_logger("qwen.tts_generator").with_context(category=LogCategory.SYSTEM)
except ImportError:
    logger = logging.getLogger(__name__)


class QwenTTSGenerator:
    """千问TTS语音生成器"""

    def __init__(self, api_key: str = None):
        """
        初始化千问TTS生成器

        Args:
            api_key: DashScope API密钥，如果未提供则从环境变量获取
        """
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY') or os.getenv('AI__DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("未找到DashScope API密钥，请设置DASHSCOPE_API_KEY环境变量")

        # ✅ 修复：使用正确的千问TTS endpoint
        self.endpoint = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
            # 千问TTS使用同步模式，不需要X-DashScope-Async
        }

        # ✅ 不再需要OSS上传器，直接使用千问临时URL（3小时有效）
        logger.info("✅ TTS生成器初始化完成，将使用千问临时URL")

    async def generate_speech(
        self,
        text: str,
        voice: str = "Cherry",  # ✅ 使用阿里云Qwen3-TTS支持的音色（芊悦-女声）
        speed: float = 1.0
    ) -> Optional[str]:
        """
        生成语音并返回音频URL

        Args:
            text: 待合成的文本
            voice: 音色选择（Qwen3-TTS支持17种音色），推荐值：
                - "Cherry": 芊悦（温柔女声，支持多语言）
                - "Ethan": 晨煦（沉稳男声，支持多语言）
                - "Nofish": 不吃鱼（活力女声）
                - "Jennifer": 詹妮弗（知性女声）
                - "Ryan": 甜茶（清新男声）
                - "Jada": 上海-阿珍（上海话女声）
                - "Dylan": 北京-晓东（北京话男声）
                - "Sunny": 四川-晴儿（四川话女声）
            speed: 语速，范围 0.5-2.0，默认1.0

        Returns:
            音频URL（千问临时URL，3小时有效），失败返回None

        Example:
            >>> generator = QwenTTSGenerator()
            >>> audio_url = await generator.generate_speech("欢迎来到机器学习的世界")
            >>> print(audio_url)
        """
        try:
            logger.info(f"🎤 开始生成语音，文本长度: {len(text)} 字符")
            logger.info(f"   音色: {voice}, 语速: {speed}")

            # ✅ 修复：构建正确的千问TTS请求体
            request_body = {
                "model": "qwen3-tts-flash",
                "input": {
                    "text": text,
                    "voice": voice,  # 音色放在input中
                    "language_type": "Chinese"  # 语言类型
                },
                "parameters": {
                    "format": "mp3",  # 输出格式
                    "sample_rate": 24000,  # 采样率
                    "speech_rate": speed  # 语速（千问TTS用speech_rate，不是rate）
                }
            }

            # 发送请求（千问TTS是同步返回，直接返回音频URL）
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    headers=self.headers,
                    json=request_body
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"❌ TTS API请求失败: {response.status} - {error_text}")
                        return None

                    result = await response.json()

                    # ✅ 修复：正确解析阿里云TTS响应格式
                    # 响应格式: {"output": {"audio": {"url": "...", "id": "...", "expires_at": ...}}}
                    output = result.get("output", {})
                    audio_info = output.get("audio", {})
                    audio_url = audio_info.get("url")  # ✅ 正确的字段路径

                    if not audio_url:
                        logger.error(f"❌ TTS响应中缺少audio.url: {result}")
                        return None

                    logger.info(f"✅ 千问TTS生成成功: {audio_url[:80]}...")

            # ✅ 直接返回千问临时URL（3小时有效，足够使用）
            # 千问TTS返回的URL格式：http://dashscope-result-*.oss-*.aliyuncs.com/...
            # 有效期：3小时，对于视频生成流程完全够用
            logger.info(f"✅ 使用千问临时URL（3小时有效）")
            return audio_url

        except Exception as e:
            logger.error(f"❌ TTS生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _wait_for_completion(self, task_id: str, timeout: int = 60) -> Optional[str]:
        """
        等待TTS任务完成并返回音频URL

        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）

        Returns:
            音频URL，失败返回None
        """
        status_endpoint = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        status_endpoint,
                        headers=self.headers
                    ) as response:
                        if response.status != 200:
                            logger.warning(f"⚠️ 查询任务状态失败: {response.status}")
                            await asyncio.sleep(2)
                            continue

                        result = await response.json()
                        output = result.get("output", {})
                        task_status = output.get("task_status")

                        if task_status == "SUCCEEDED":
                            # 提取音频URL
                            audio_url = output.get("audio_url")
                            if audio_url:
                                return audio_url
                            else:
                                logger.error(f"❌ 任务成功但未找到audio_url: {output}")
                                return None

                        elif task_status in ["FAILED", "UNKNOWN"]:
                            error_msg = output.get("message", "Unknown error")
                            logger.error(f"❌ TTS任务失败: {error_msg}")
                            return None

                        elif task_status in ["PENDING", "RUNNING"]:
                            logger.info(f"⏳ TTS任务进行中... ({task_status})")
                            await asyncio.sleep(2)
                        else:
                            logger.warning(f"⚠️ 未知任务状态: {task_status}")
                            await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"❌ 查询任务状态异常: {e}")
                await asyncio.sleep(2)

        logger.error("❌ TTS任务超时")
        return None

    async def _download_audio(self, url: str) -> str:
        """
        下载音频到临时文件

        Args:
            url: 音频URL

        Returns:
            临时文件路径
        """
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.mp3',
            prefix='tts_audio_'
        )
        temp_path = temp_file.name
        temp_file.close()

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(temp_path, 'wb') as f:
                        f.write(content)
                    return temp_path
                else:
                    raise Exception(f"下载音频失败: {response.status}")


# 全局实例（单例模式）
_qwen_tts_generator = None


def get_qwen_tts_generator() -> QwenTTSGenerator:
    """获取全局千问TTS生成器实例"""
    global _qwen_tts_generator
    if _qwen_tts_generator is None:
        _qwen_tts_generator = QwenTTSGenerator()
    return _qwen_tts_generator


async def generate_speech_from_text(
    text: str,
    voice: str = "Cherry",  # ✅ 使用阿里云Qwen3-TTS支持的音色（芊悦-女声）
    speed: float = 1.0,
    upload_to_oss: bool = True
) -> Optional[str]:
    """
    便捷函数：生成语音

    Args:
        text: 待合成的文本
        voice: 音色
        speed: 语速
        upload_to_oss: 是否上传到OSS

    Returns:
        音频URL

    Examples:
        >>> await generate_speech_from_text("欢迎来到机器学习的世界")
    """
    generator = get_qwen_tts_generator()
    return await generator.generate_speech(text, voice, speed, upload_to_oss)
