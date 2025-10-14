#!/usr/bin/env python3
"""
阿里云视频生成器 - 使用DashScope视频生成API
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import requests
import uuid

logger = logging.getLogger(__name__)

# 确保输出目录存在
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class AliyunVideoGenerator:
    """阿里云DashScope视频生成器"""

    def __init__(self, api_key: Optional[str] = None):
        """初始化阿里云视频生成器"""
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("需要设置 DASHSCOPE_API_KEY 环境变量或传入 API Key")

        # DashScope视频生成API端点 - 使用最新的文生视频API
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        self.query_url = "https://dashscope.aliyuncs.com/api/v1/tasks"

        # 请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable"  # 启用异步模式
        }

    def generate_video(self,
                       task_id: str,
                       description: str,
                       keywords: List[str],
                       duration: int = 30,
                       theme: str = "科技创新") -> Dict[str, Any]:
        """
        使用阿里云DashScope生成视频
        """
        try:
            logger.info(f"🎬 开始使用阿里云DashScope生成视频 - Task ID: {task_id}")
            logger.info(f"📋 主题: {theme}, 时长: {duration}秒")
            logger.info(f"🔑 关键词: {keywords}")
            logger.info(f"📝 描述: {description}")

            # 构建视频生成提示词
            prompt = self._build_video_prompt(description, keywords, theme, duration)
            logger.info(f"🎯 生成提示词: {prompt}")

            # 调用阿里云视频生成API
            task_response = self._submit_video_generation_task(prompt, duration)

            if not task_response.get("success"):
                return {
                    "success": False,
                    "error": f"提交视频生成任务失败: {task_response.get('error')}",
                    "task_id": task_id
                }

            # 获取任务ID
            aliyun_task_id = task_response["task_id"]
            logger.info(f"✅ 视频生成任务已提交，阿里云任务ID: {aliyun_task_id}")

            # 轮询查询任务状态
            result = self._poll_task_status(aliyun_task_id, task_id, max_wait_time=300)

            return result

        except Exception as e:
            logger.error(f"❌ 阿里云视频生成失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }

    def _build_video_prompt(self, description: str, keywords: List[str], theme: str, duration: int) -> str:
        """构建视频生成提示词"""

        # 主题相关的视觉风格
        theme_styles = {
            "科技创新": "现代科技风格，蓝色调，数字化元素，未来感强",
            "现代都市": "都市夜景，霓虹灯光，摩天大楼，现代化氛围",
            "自然风光": "自然清新，绿色主调，阳光明媚，生机勃勃",
            "商务专业": "简洁专业，深色调，商务场景，稳重大气",
            "创意艺术": "色彩丰富，创意元素，艺术感强，视觉冲击力"
        }

        style_desc = theme_styles.get(theme, "现代简洁风格")
        keywords_text = "、".join(keywords)

        # 构建详细的提示词
        prompt = f"""
创建一个{duration}秒的高质量宣传视频，主题：{theme}。

内容描述：{description}

关键元素：{keywords_text}

视觉风格：{style_desc}

技术要求：
- 视频分辨率：1920x1080 (Full HD)
- 帧率：30fps
- 时长：{duration}秒
- 画面清晰流畅，色彩饱和度适中
- 包含平滑的转场效果
- 画面构图美观，符合视觉美学

场景要求：
1. 开场：引人注目的开场画面，突出主题
2. 中段：展示关键词相关的核心内容场景
3. 结尾：有力的结尾画面，给人深刻印象

镜头要求：
- 使用多种镜头角度（特写、中景、远景）
- 适当的摄像机运动（推拉摇移）
- 画面节奏感强，符合主题气氛

请生成专业级别的视频内容。
        """.strip()

        return prompt

    def _submit_video_generation_task(self, prompt: str, duration: int) -> Dict[str, Any]:
        """提交视频生成任务到阿里云"""
        try:
            # 构建请求数据 - 使用最新的wan2.2-t2v-plus模型
            data = {
                "model": "wan2.2-t2v-plus",  # 最新的文生视频模型
                "input": {
                    "prompt": prompt  # 文本提示词
                },
                "parameters": {
                    "size": "1920*1080"  # 视频尺寸，使用1080P
                }
            }

            logger.info(f"🚀 向阿里云提交视频生成请求...")
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("output") and result["output"].get("task_id"):
                    return {
                        "success": True,
                        "task_id": result["output"]["task_id"],
                        "message": "视频生成任务提交成功"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"API响应格式异常: {result}"
                    }
            else:
                error_detail = response.text
                logger.error(f"❌ API请求失败: {response.status_code} - {error_detail}")
                return {
                    "success": False,
                    "error": f"API请求失败: {response.status_code} - {error_detail}"
                }

        except Exception as e:
            logger.error(f"❌ 提交视频生成任务异常: {str(e)}")
            return {
                "success": False,
                "error": f"提交任务异常: {str(e)}"
            }

    def _poll_task_status(self, aliyun_task_id: str, local_task_id: str, max_wait_time: int = 300) -> Dict[str, Any]:
        """轮询查询任务状态"""
        start_time = time.time()
        poll_interval = 10  # 10秒查询一次

        logger.info(f"🔄 开始轮询任务状态，最大等待时间: {max_wait_time}秒")

        while time.time() - start_time < max_wait_time:
            try:
                # 查询任务状态
                query_url = f"{self.query_url}/{aliyun_task_id}"
                response = requests.get(
                    query_url,
                    headers=self.headers,
                    timeout=30
                )

                if response.status_code != 200:
                    logger.warning(f"⚠️ 查询任务状态失败: {response.status_code}")
                    time.sleep(poll_interval)
                    continue

                result = response.json()
                status = result.get("output", {}).get("task_status", "UNKNOWN")

                logger.info(f"📊 任务状态: {status}")

                if status == "SUCCEEDED":
                    # 任务成功完成
                    output = result.get("output", {})
                    # 获取视频URL（已确认在output.video_url中）
                    video_url = output.get("video_url")

                    logger.info(f"📋 完整响应: {json.dumps(result, indent=2, ensure_ascii=False)}")

                    if video_url:
                        # 下载视频文件
                        download_result = self._download_video(video_url, local_task_id)
                        if download_result["success"]:
                            file_size = Path(download_result["local_path"]).stat().st_size / (1024 * 1024)
                            return {
                                "success": True,
                                "output_path": download_result["local_path"],
                                "duration": 30,  # 实际时长需要从视频文件获取
                                "resolution": "1280x720",
                                "file_size_mb": round(file_size, 2),
                                "source": "aliyun_dashscope",
                                "aliyun_task_id": aliyun_task_id,
                                "video_url": video_url,
                                "timestamp": datetime.now().isoformat()
                            }
                        else:
                            return {
                                "success": False,
                                "error": f"下载视频失败: {download_result['error']}",
                                "video_url": video_url
                            }
                    else:
                        return {
                            "success": False,
                            "error": "视频生成成功但未获取到下载URL"
                        }

                elif status == "FAILED":
                    # 任务失败
                    error_message = result.get("output", {}).get("message", "未知错误")
                    return {
                        "success": False,
                        "error": f"阿里云视频生成失败: {error_message}",
                        "aliyun_task_id": aliyun_task_id
                    }

                elif status in ["PENDING", "RUNNING"]:
                    # 任务进行中，继续等待
                    progress = result.get("output", {}).get("progress", 0)
                    logger.info(f"⏳ 视频生成中，进度: {progress}%")
                    time.sleep(poll_interval)
                    continue

                else:
                    # 未知状态
                    logger.warning(f"⚠️ 未知任务状态: {status}")
                    time.sleep(poll_interval)
                    continue

            except Exception as e:
                logger.error(f"❌ 查询任务状态异常: {str(e)}")
                time.sleep(poll_interval)
                continue

        # 超时
        return {
            "success": False,
            "error": f"视频生成超时（超过{max_wait_time}秒）",
            "aliyun_task_id": aliyun_task_id
        }

    def _download_video(self, video_url: str, task_id: str) -> Dict[str, Any]:
        """下载视频文件到本地"""
        try:
            logger.info(f"📥 开始下载视频: {video_url}")

            # 本地文件路径
            local_filename = f"video_{task_id}_aliyun.mp4"
            local_path = OUTPUT_DIR / local_filename

            # 下载文件
            response = requests.get(video_url, stream=True, timeout=120)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"✅ 视频下载完成: {local_path}")

            return {
                "success": True,
                "local_path": str(local_path),
                "filename": local_filename
            }

        except Exception as e:
            logger.error(f"❌ 下载视频失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表"""
        return ["wan2.2-t2v-plus", "wanx-v1"]

    def get_supported_resolutions(self) -> List[str]:
        """获取支持的分辨率列表"""
        return ["1080*1920", "1920*1080", "1440*1440", "1632*1248", "1248*1632", "480*832", "832*480", "624*624"]

    def estimate_generation_time(self, duration: int) -> int:
        """估算视频生成时间（秒）"""
        # 根据经验，阿里云视频生成大约需要视频时长的3-5倍时间
        return duration * 4


# 单例模式
_aliyun_generator_instance = None

def get_aliyun_video_generator(api_key: Optional[str] = None) -> AliyunVideoGenerator:
    """获取阿里云视频生成器实例"""
    global _aliyun_generator_instance
    if _aliyun_generator_instance is None:
        _aliyun_generator_instance = AliyunVideoGenerator(api_key)
    return _aliyun_generator_instance


if __name__ == "__main__":
    # 测试阿里云视频生成
    import sys

    # 检查API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 请设置 DASHSCOPE_API_KEY 环境变量")
        sys.exit(1)

    generator = get_aliyun_video_generator()
    result = generator.generate_video(
        task_id="aliyun_test_001",
        description="制作一个关于科技创新的30秒宣传视频，包含现代城市场景和科技元素",
        keywords=["现代城市", "科技元素", "创新", "未来"],
        duration=5,  # 测试用较短时长
        theme="科技创新"
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))