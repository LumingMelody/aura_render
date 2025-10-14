#!/usr/bin/env python3
"""
阿里云视频生成器增强版 - 支持生成更长时间的视频
通过生成多个片段并拼接实现
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
import subprocess
import tempfile

logger = logging.getLogger(__name__)

# 确保输出目录存在
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class AliyunVideoGeneratorExtended:
    """阿里云DashScope视频生成器 - 支持长视频"""

    def __init__(self, api_key: Optional[str] = None):
        """初始化阿里云视频生成器"""
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("需要设置 DASHSCOPE_API_KEY 环境变量或传入 API Key")

        # DashScope视频生成API端点
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        self.query_url = "https://dashscope.aliyuncs.com/api/v1/tasks"

        # 请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable"
        }

    def generate_video(self,
                       task_id: str,
                       description: str,
                       keywords: List[str],
                       duration: int = 30,
                       theme: str = "科技创新") -> Dict[str, Any]:
        """
        生成指定时长的视频（通过片段拼接）
        """
        try:
            logger.info(f"🎬 开始生成{duration}秒视频 - Task ID: {task_id}")

            # 计算需要生成的片段数（每片段5秒）
            segment_duration = 5
            num_segments = max(1, duration // segment_duration)

            logger.info(f"📋 将生成{num_segments}个{segment_duration}秒片段")

            # 为不同片段生成不同的场景描述
            segment_prompts = self._generate_segment_prompts(
                description, keywords, theme, num_segments
            )

            # 生成所有视频片段
            segment_files = []
            for i, prompt in enumerate(segment_prompts):
                logger.info(f"🎯 生成片段 {i+1}/{num_segments}")

                segment_result = self._generate_single_segment(
                    f"{task_id}_segment_{i+1}",
                    prompt,
                    theme
                )

                if segment_result["success"]:
                    segment_files.append(segment_result["output_path"])
                    logger.info(f"✅ 片段 {i+1} 生成成功: {segment_result['output_path']}")
                else:
                    logger.error(f"❌ 片段 {i+1} 生成失败: {segment_result.get('error')}")
                    # 如果某个片段失败，使用已生成的片段
                    if len(segment_files) == 0:
                        return segment_result
                    break

            # 如果只有一个片段，直接返回
            if len(segment_files) == 1:
                final_path = OUTPUT_DIR / f"video_{task_id}_aliyun.mp4"
                os.rename(segment_files[0], final_path)

                return {
                    "success": True,
                    "output_path": str(final_path),
                    "duration": segment_duration,
                    "resolution": "1920x1080",
                    "segments_generated": 1,
                    "source": "aliyun_dashscope"
                }

            # 拼接所有片段
            logger.info(f"🎬 开始拼接{len(segment_files)}个片段...")
            final_video = self._concat_videos(segment_files, task_id)

            if final_video["success"]:
                # 删除临时片段文件
                for seg_file in segment_files:
                    try:
                        os.remove(seg_file)
                    except:
                        pass

                return {
                    "success": True,
                    "output_path": final_video["output_path"],
                    "duration": len(segment_files) * segment_duration,
                    "resolution": "1920x1080",
                    "segments_generated": len(segment_files),
                    "source": "aliyun_dashscope"
                }
            else:
                return final_video

        except Exception as e:
            logger.error(f"❌ 视频生成失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }

    def _generate_segment_prompts(self, description: str, keywords: List[str],
                                  theme: str, num_segments: int) -> List[str]:
        """为不同片段生成不同的提示词"""

        # 场景进展模板
        scene_templates = {
            1: "开场画面，引入主题",
            2: "展开内容，深入展示",
            3: "核心展示，重点呈现",
            4: "转折变化，新的视角",
            5: "高潮部分，震撼画面",
            6: "收尾总结，回归主题"
        }

        prompts = []
        for i in range(num_segments):
            segment_num = min(i + 1, 6)  # 使用1-6的模板循环
            scene_desc = scene_templates.get(segment_num, "继续展示")

            # 选择本片段重点展示的关键词
            if keywords:
                focus_keyword = keywords[i % len(keywords)]
            else:
                focus_keyword = theme

            prompt = f"""
{description}

【片段{i+1}/{num_segments}】{scene_desc}
重点展示：{focus_keyword}
主题风格：{theme}

要求：
- 高质量1920x1080视频
- 画面流畅，色彩鲜明
- 与主题{theme}风格一致
- 突出{focus_keyword}元素
"""
            prompts.append(prompt.strip())

        return prompts

    def _generate_single_segment(self, segment_id: str, prompt: str, theme: str) -> Dict[str, Any]:
        """生成单个视频片段"""
        try:
            # 提交任务
            data = {
                "model": "wan2.2-t2v-plus",
                "input": {
                    "prompt": prompt
                },
                "parameters": {
                    "size": "1920*1080"
                }
            }

            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
                timeout=30
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"API请求失败: {response.status_code}"
                }

            result = response.json()
            if not result.get("output", {}).get("task_id"):
                return {
                    "success": False,
                    "error": "未获取到任务ID"
                }

            aliyun_task_id = result["output"]["task_id"]

            # 轮询等待结果
            max_wait = 180  # 3分钟
            start_time = time.time()

            while time.time() - start_time < max_wait:
                time.sleep(10)

                # 查询任务状态
                query_response = requests.get(
                    f"{self.query_url}/{aliyun_task_id}",
                    headers=self.headers,
                    timeout=30
                )

                if query_response.status_code != 200:
                    continue

                task_result = query_response.json()
                status = task_result.get("output", {}).get("task_status", "UNKNOWN")

                if status == "SUCCEEDED":
                    video_url = task_result.get("output", {}).get("video_url")
                    if video_url:
                        # 下载视频
                        local_path = TEMP_DIR / f"{segment_id}.mp4"
                        download_response = requests.get(video_url, stream=True)

                        with open(local_path, 'wb') as f:
                            for chunk in download_response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)

                        return {
                            "success": True,
                            "output_path": str(local_path)
                        }
                    else:
                        return {
                            "success": False,
                            "error": "未获取到视频URL"
                        }
                elif status == "FAILED":
                    return {
                        "success": False,
                        "error": task_result.get("output", {}).get("message", "生成失败")
                    }

            return {
                "success": False,
                "error": "生成超时"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _concat_videos(self, video_files: List[str], task_id: str) -> Dict[str, Any]:
        """使用ffmpeg拼接视频片段"""
        try:
            # 创建文件列表
            list_file = TEMP_DIR / f"{task_id}_list.txt"
            with open(list_file, 'w') as f:
                for video_file in video_files:
                    # 确保路径格式正确
                    abs_path = Path(video_file).absolute()
                    f.write(f"file '{abs_path}'\n")

            # 输出文件路径
            output_path = OUTPUT_DIR / f"video_{task_id}_aliyun.mp4"

            # 使用ffmpeg拼接
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(list_file),
                '-c', 'copy',  # 不重新编码，直接拼接
                '-y',  # 覆盖已存在的文件
                str(output_path)
            ]

            logger.info(f"📹 执行拼接命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            # 删除临时列表文件
            try:
                os.remove(list_file)
            except:
                pass

            if result.returncode == 0 and output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ 视频拼接成功: {output_path} ({file_size:.2f} MB)")

                return {
                    "success": True,
                    "output_path": str(output_path),
                    "file_size_mb": round(file_size, 2)
                }
            else:
                error_msg = result.stderr if result.stderr else "拼接失败"
                logger.error(f"❌ 视频拼接失败: {error_msg}")
                return {
                    "success": False,
                    "error": f"视频拼接失败: {error_msg}"
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "视频拼接超时"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"视频拼接异常: {str(e)}"
            }


# 单例模式
_extended_generator_instance = None

def get_aliyun_video_generator_extended(api_key: Optional[str] = None) -> AliyunVideoGeneratorExtended:
    """获取阿里云视频生成器增强版实例"""
    global _extended_generator_instance
    if _extended_generator_instance is None:
        _extended_generator_instance = AliyunVideoGeneratorExtended(api_key)
    return _extended_generator_instance


if __name__ == "__main__":
    # 测试生成30秒视频
    from dotenv import load_dotenv
    load_dotenv()

    generator = get_aliyun_video_generator_extended()
    result = generator.generate_video(
        task_id="extended_test_001",
        description="制作一个关于科技创新的宣传视频，包含现代城市场景和科技元素",
        keywords=["现代城市", "科技元素", "创新", "未来", "人工智能", "数字化"],
        duration=30,  # 30秒
        theme="科技创新"
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))