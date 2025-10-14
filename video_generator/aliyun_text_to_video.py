#!/usr/bin/env python3
"""
阿里云万相文生视频API客户端
根据阿里云Model Studio文生视频API实现
"""

import os
import json
import time
import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

class AliyunTextToVideoClient:
    """阿里云万相文生视频API客户端"""

    def __init__(self, api_key: str = None):
        # 从环境变量获取API密钥
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            logger.warning("⚠️ 未找到DashScope API密钥，文生视频功能不可用")

        # API配置 - 使用官方正确的URL和头部
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable"
        }

        self.temp_dir = Path(tempfile.gettempdir()) / "aura_render_videos"
        self.temp_dir.mkdir(exist_ok=True)

    def generate_video_segment(self,
                             text_prompt: str,
                             duration: int = 5,
                             style: str = "realistic") -> Dict[str, Any]:
        """
        生成单个视频片段（5秒）

        Args:
            text_prompt: 文本描述
            duration: 视频时长(秒)，默认5秒
            style: 视频风格，可选值: realistic, cartoon, anime等

        Returns:
            包含视频信息的字典
        """
        if not self.api_key:
            return self._create_mock_video_segment(text_prompt, duration)

        try:
            logger.info(f"🎬 开始生成视频片段: {text_prompt[:50]}...")

            # 构建请求数据 - 使用正确的DashScope格式
            request_data = {
                "model": "wanx-v1",  # 使用正确的模型名称
                "input": {
                    "prompt": text_prompt
                },
                "parameters": {
                    "size": "1280*720",  # 支持的分辨率
                    "length": "5s"  # 视频时长
                }
            }

            # 发送异步生成请求 - 添加详细日志
            logger.info(f"🔍 发送DashScope API请求:")
            logger.info(f"🔍 URL: {self.base_url}")
            logger.info(f"🔍 Headers: {json.dumps(self.headers, indent=2, ensure_ascii=False)}")
            logger.info(f"🔍 Request Data: {json.dumps(request_data, indent=2, ensure_ascii=False)}")

            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=request_data,
                timeout=30
            )

            logger.info(f"🔍 Response Status: {response.status_code}")
            logger.info(f"🔍 Response Headers: {dict(response.headers)}")

            if response.status_code != 200:
                error_detail = response.text if response.text else "无错误详情"
                logger.error(f"❌ 文生视频API请求失败: {response.status_code} - {error_detail}")
                logger.error(f"❌ 请求URL: {self.base_url}")
                logger.error(f"❌ 请求数据: {json.dumps(request_data, indent=2, ensure_ascii=False)}")
                return self._create_mock_video_segment(text_prompt, duration)

            result = response.json()

            # 异步模式：立即返回task_id，需要轮询查询结果
            if "output" in result and "task_id" in result["output"]:
                task_id = result["output"]["task_id"]
                logger.info(f"🔄 获得异步任务ID: {task_id}")
                return self._wait_for_video_generation(task_id, text_prompt, duration)
            else:
                logger.error(f"❌ 文生视频API返回格式错误: {result}")
                return self._create_mock_video_segment(text_prompt, duration)

        except Exception as e:
            logger.error(f"❌ 文生视频API调用异常: {str(e)}")
            return self._create_mock_video_segment(text_prompt, duration)

    def _wait_for_video_generation(self, task_id: str, text_prompt: str, duration: int) -> Dict[str, Any]:
        """等待异步视频生成完成"""
        max_wait_time = 300  # 最大等待5分钟
        start_time = time.time()

        query_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"

        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(query_url, headers=self.headers, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    # 可能的状态字段：task_status 或 status
                    status = result.get("output", {}).get("task_status") or result.get("output", {}).get("status", "")

                    if status == "SUCCEEDED":
                        video_url = result["output"]["video_url"]
                        logger.info(f"✅ 视频生成成功: {text_prompt[:30]}...")
                        return self._download_video_segment(video_url, text_prompt, duration)

                    elif status == "FAILED":
                        logger.error(f"❌ 视频生成失败: {result}")
                        break

                    else:
                        logger.info(f"⏳ 视频生成中... 状态: {status}")
                        time.sleep(10)  # 等待10秒后重试

                else:
                    logger.warning(f"⚠️ 查询任务状态失败: {response.status_code}")
                    time.sleep(5)

            except Exception as e:
                logger.error(f"❌ 查询视频生成状态异常: {str(e)}")
                time.sleep(5)

        logger.error(f"❌ 视频生成超时: {text_prompt}")
        return self._create_mock_video_segment(text_prompt, duration)

    def _download_video_segment(self, video_url: str, text_prompt: str, duration: int) -> Dict[str, Any]:
        """下载生成的视频片段"""
        try:
            # 生成本地文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in text_prompt[:20] if c.isalnum() or c in (' ', '-', '_')).strip()
            filename = f"video_segment_{timestamp}_{safe_prompt}.mp4"
            local_path = self.temp_dir / filename

            # 下载视频文件
            logger.info(f"📥 下载视频片段: {video_url}")
            response = requests.get(video_url, timeout=60)

            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)

                file_size = local_path.stat().st_size / (1024 * 1024)  # MB

                return {
                    "success": True,
                    "local_path": str(local_path),
                    "video_url": video_url,
                    "text_prompt": text_prompt,
                    "duration": duration,
                    "file_size_mb": round(file_size, 2),
                    "resolution": "1920x1080",
                    "fps": 24,
                    "timestamp": datetime.now().isoformat()
                }

            else:
                logger.error(f"❌ 下载视频失败: {response.status_code}")
                return self._create_mock_video_segment(text_prompt, duration)

        except Exception as e:
            logger.error(f"❌ 下载视频异常: {str(e)}")
            return self._create_mock_video_segment(text_prompt, duration)

    def _create_mock_video_segment(self, text_prompt: str, duration: int) -> Dict[str, Any]:
        """创建Mock视频片段信息（当API不可用时）"""
        return {
            "success": False,
            "local_path": None,
            "video_url": None,
            "text_prompt": text_prompt,
            "duration": duration,
            "file_size_mb": 0,
            "resolution": "1920x1080",
            "fps": 24,
            "timestamp": datetime.now().isoformat(),
            "error": "阿里云文生视频API不可用，使用Mock数据"
        }

    def generate_multi_segment_video(self,
                                   text_prompts: List[str],
                                   target_duration: int = 30) -> List[Dict[str, Any]]:
        """
        生成多个视频片段

        Args:
            text_prompts: 文本描述列表
            target_duration: 目标总时长

        Returns:
            视频片段信息列表
        """
        segment_duration = 5  # 每个片段5秒
        needed_segments = max(1, target_duration // segment_duration)

        # 如果文本数量不足，重复使用
        if len(text_prompts) < needed_segments:
            extended_prompts = []
            for i in range(needed_segments):
                extended_prompts.append(text_prompts[i % len(text_prompts)])
            text_prompts = extended_prompts
        else:
            text_prompts = text_prompts[:needed_segments]

        logger.info(f"🎬 开始生成 {len(text_prompts)} 个视频片段，总时长 {len(text_prompts) * segment_duration} 秒")

        segments = []
        for i, prompt in enumerate(text_prompts):
            logger.info(f"📽️ 生成片段 {i+1}/{len(text_prompts)}: {prompt[:50]}...")
            segment_result = self.generate_video_segment(prompt, segment_duration)
            segments.append(segment_result)

            # 避免API频率限制 - 增加延迟时间
            if i < len(text_prompts) - 1:
                logger.info("⏳ 等待5秒避免API频率限制...")
                time.sleep(5)

        return segments

# 全局实例
_aliyun_client_instance = None

def get_aliyun_text_to_video_client() -> AliyunTextToVideoClient:
    """获取阿里云文生视频客户端实例"""
    global _aliyun_client_instance
    if _aliyun_client_instance is None:
        _aliyun_client_instance = AliyunTextToVideoClient()
    return _aliyun_client_instance

if __name__ == "__main__":
    # 测试客户端
    client = get_aliyun_text_to_video_client()

    # 测试生成单个片段
    test_prompts = [
        "现代化城市的科技大楼，展示创新与未来",
        "人工智能机器人在实验室中工作",
        "高科技数据中心闪烁的服务器灯光"
    ]

    segments = client.generate_multi_segment_video(test_prompts, target_duration=15)

    for i, segment in enumerate(segments):
        print(f"片段 {i+1}: {json.dumps(segment, indent=2, ensure_ascii=False)}")