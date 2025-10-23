"""
千问API集成 - 首尾帧视频生成
"""
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import base64
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.oss_uploader import get_oss_uploader

# 配置日志
logger = logging.getLogger(__name__)


class QwenVideoGenerator:
    """千问视频生成器"""

    def __init__(self, api_key: str, endpoint: str = None):
        self.api_key = api_key
        # 通义万相图生视频API端点 (正确的URL)
        self.endpoint = endpoint or "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable"
        }
        # 初始化OSS上传器
        try:
            self.oss_uploader = get_oss_uploader()
            self.use_oss = True
            logger.info("✅ OSS上传器初始化成功，将使用OSS URL方式")
        except Exception as e:
            logger.warning(f"⚠️ OSS上传器初始化失败，将使用base64方式: {e}")
            self.oss_uploader = None
            self.use_oss = False

    async def generate_video_from_frames(self,
                                        start_image_path_or_url: str,
                                        end_image_path_or_url: str,
                                        duration_seconds: float = 5.0,
                                        video_prompt: str = None) -> Dict:
        """
        使用首尾帧生成视频

        参数:
            start_image_path_or_url: 首帧图片路径或URL
            end_image_path_or_url: 尾帧图片路径或URL
            duration_seconds: 视频时长（秒），默认5秒
            video_prompt: 视频描述提示词（可选，用于指导视频运动）

        返回:
            生成的视频信息
        """

        # 检查是否已经是URL（万相返回的URL可直接使用）
        from urllib.parse import urlparse
        parsed = urlparse(start_image_path_or_url)
        if parsed.scheme in ('http', 'https'):
            # 已经是URL，直接使用
            img_url = start_image_path_or_url
            logger.info(f"✅ 使用万相图片URL: {img_url}")
        else:
            # 本地路径，需要上传或编码
            if self.use_oss and self.oss_uploader:
                try:
                    # 上传图片到OSS并获取公网URL
                    img_url = self.oss_uploader.upload_image(start_image_path_or_url)
                    logger.info(f"📤 图片已上传到OSS: {img_url}")
                except Exception as e:
                    logger.warning(f"⚠️ OSS上传失败，降级使用base64: {e}")
                    img_url = self._encode_image(start_image_path_or_url)
            else:
                # 使用base64编码
                img_url = self._encode_image(start_image_path_or_url)

        # 构建图生视频描述提示词
        if video_prompt:
            # 使用传入的refined_prompt（包含明确的动态运动描述）
            prompt = f"{video_prompt}，画面运动流畅自然"
            logger.info(f"📋 使用动态提示词: {video_prompt[:50]}...")
        else:
            # 默认通用提示词
            prompt = f"基于输入的图片生成一个自然流畅的视频，保持图片中的主要元素和风格，画面平滑过渡"

        # 构建请求 - 使用正确的DashScope图生视频API格式
        request_body = {
            "model": "wan2.5-i2v-preview",  # 图生视频模型 (支持音频的最新版本)
            "input": {
                "img_url": img_url,  # 图片URL (OSS公网URL或base64 data URI)
                "prompt": prompt  # 文本描述 (可选)
            },
            "parameters": {
                "resolution": "720P",  # 分辨率 (720P或1080P)
                "duration": int(duration_seconds),  # 视频时长 (5或10秒)
                "audio": False  # 暂不生成音频，减少处理时间
            }
        }

        # 发送请求
        logger.info(f"🚀 发送图生视频请求到: {self.endpoint}")
        logger.info(f"📦 请求体: model={request_body['model']}, img_url={img_url[:100]}...")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint,
                headers=self.headers,
                json=request_body
            ) as response:
                response_text = await response.text()

                if response.status == 200:
                    result = json.loads(response_text)
                    # 通义万相异步API返回格式：{"output": {"task_id": "..."}, "request_id": "..."}
                    task_id = result.get("output", {}).get("task_id")
                    if task_id:
                        logger.info(f"✅ 任务创建成功, task_id: {task_id}")
                        return {
                            "success": True,
                            "task_id": task_id,
                            "video_url": None,  # 需要后续查询获取
                            "status": "processing"
                        }
                    else:
                        logger.info(f"❌ API响应中缺少task_id: {result}")
                        return {
                            "success": False,
                            "error": f"API响应中缺少task_id: {result}"
                        }
                else:
                    logger.info(f"❌ API请求失败 (status {response.status}): {response_text}")
                    return {
                        "success": False,
                        "error": f"API error {response.status}: {response_text}"
                    }

    async def get_task_status(self, task_id: str) -> Dict:
        """查询任务状态"""

        # 通义万相2.1任务状态查询端点
        status_endpoint = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                status_endpoint,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Failed to get status: {response.status}"}

    def _encode_image(self, image_path: str) -> str:
        """将图片编码为base64 data URI格式，支持本地文件和URL"""
        import requests
        from urllib.parse import urlparse
        import mimetypes

        # 检查是否为URL
        parsed = urlparse(image_path)
        if parsed.scheme in ('http', 'https'):
            # 下载图片内容
            response = requests.get(image_path, timeout=30)
            response.raise_for_status()
            content = response.content
            mime_type = response.headers.get('content-type', 'image/jpeg')
        else:
            # 本地文件路径
            with open(image_path, "rb") as f:
                content = f.read()
            # 根据文件扩展名确定MIME类型
            mime_type = mimetypes.guess_type(image_path)[0] or 'image/jpeg'

        # 返回完整的data URI格式
        base64_data = base64.b64encode(content).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"

    async def wait_for_completion(self, task_id: str, timeout: int = 300) -> Dict:
        """等待任务完成"""

        start_time = asyncio.get_event_loop().time()

        while True:
            # 检查超时
            if asyncio.get_event_loop().time() - start_time > timeout:
                return {"success": False, "error": "Timeout waiting for task completion"}

            # 查询状态
            status_response = await self.get_task_status(task_id)

            # 根据文档,状态在 output.task_status 字段中
            output = status_response.get("output", {})
            task_status = output.get("task_status")

            if task_status == "SUCCEEDED":
                # 尝试获取URL - 优先video_url（图生视频），其次results[0].url（文生图）
                url = output.get("video_url")
                if not url:
                    results = output.get("results", [])
                    if results and len(results) > 0:
                        url = results[0].get("url")

                return {
                    "success": True,
                    "video_url": url  # 统一使用video_url字段名
                }
            elif task_status in ["FAILED", "UNKNOWN"]:
                return {
                    "success": False,
                    "error": status_response.get("message", output.get("message", "Task failed"))
                }
            elif task_status in ["PENDING", "RUNNING"]:
                # 任务仍在进行中，继续等待
                logger.info(f"⏳ 任务 {task_id[:8]}... 状态: {task_status}")
                pass
            else:
                # 未知状态,打印调试信息
                logger.info(f"⚠️ 未知任务状态: {task_status}, 完整响应: {status_response}")

            # 等待后重试
            await asyncio.sleep(5)

    async def submit_image_edit_task(self, base_image_url: str, prompt: str, function: str = "stylization_all") -> str:
        """
        提交图片编辑任务（图生图）

        参数:
            base_image_url: 基础图片URL
            prompt: 编辑描述提示词
            function: 编辑功能类型，可选值：
                - "stylization_all": 全局风格迁移
                - "description_edit": 内容编辑

        返回:
            任务ID
        """
        # 万相图编辑API端点
        endpoint = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis"

        request_body = {
            "model": "wanx2.1-imageedit",  # 图编辑模型
            "input": {
                "base_image_url": base_image_url,
                "function": function,
                "prompt": prompt
            },
            "parameters": {
                "size": "1280*720"  # 16:9比例
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    headers=self.headers,
                    json=request_body
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        task_id = result.get("output", {}).get("task_id")

                        if task_id:
                            return task_id
                        else:
                            logger.info(f"❌ 图编辑API未返回task_id: {result}")
                            return None
                    else:
                        error_text = await response.text()
                        logger.info(f"❌ 图编辑API错误 {response.status}: {error_text[:200]}")
                        return None
        except Exception as e:
            logger.info(f"❌ 图编辑API异常: {e}")
            import traceback
            traceback.print_exc()
            return None


class StoryboardToVideoProcessor:
    """分镜到视频处理器"""

    def __init__(self, qwen_api_key: str):
        self.qwen = QwenVideoGenerator(qwen_api_key)
        self.temp_videos = []

    async def process_storyboard_frames(self,
                                       keyframes: List[Dict],
                                       output_dir: str) -> List[str]:
        """
        处理分镜关键帧，生成视频片段

        参数:
            keyframes: 关键帧列表
            output_dir: 输出目录

        返回:
            生成的视频片段路径列表
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_clips = []

        # 将关键帧配对（首帧-尾帧）
        frame_pairs = self._pair_frames(keyframes)

        # 串行生成视频片段以避免API限流 (每次1个请求，间隔2秒)
        results = []
        for i, (start_frame, end_frame) in enumerate(frame_pairs):
            try:
                logger.info(f"🎬 正在生成视频片段 {i+1}/{len(frame_pairs)}...")
                result = await self._generate_clip(
                    start_frame,
                    end_frame,
                    output_dir / f"clip_{i:03d}.mp4"
                )
                results.append(result)
                logger.info(f"✅ 视频片段 {i+1} 生成成功")
                logger.info(f"   📹 视频URL: {result.get('url', 'N/A')}")
                logger.info(f"   ⏱️  时长: {result.get('duration', 0)}秒")

                # 添加延迟避免限流
                if i < len(frame_pairs) - 1:
                    await asyncio.sleep(2)
            except Exception as e:
                logger.info(f"❌ 视频片段 {i} 生成失败: {e}")
                results.append(e)

        # 处理结果
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.info(f"Clip {i} generation failed: {result}")
            else:
                video_clips.append(result)

        return video_clips

    def _pair_frames(self, keyframes: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """
        配对关键帧，处理帧复用逻辑

        返回:
            [(首帧, 尾帧), ...]
        """

        pairs = []

        i = 0
        while i < len(keyframes) - 1:
            start_frame = keyframes[i]

            # 查找对应的尾帧
            if i + 1 < len(keyframes):
                end_frame = keyframes[i + 1]

                # 检查是否需要复用
                if end_frame.get("is_reused") and i + 2 < len(keyframes):
                    # 如果尾帧被复用，跳过它，使用下一个作为尾帧
                    end_frame = keyframes[i + 2]
                    pairs.append((start_frame, end_frame))
                    i += 3  # 跳过已处理的帧
                else:
                    pairs.append((start_frame, end_frame))
                    i += 2
            else:
                break

        return pairs

    async def _generate_clip(self,
                            start_frame: Dict,
                            end_frame: Dict,
                            output_path: Path) -> Dict:
        """生成单个视频片段,返回视频URL"""

        # 获取图片路径或URL（优先使用URL）
        start_img = start_frame.get("image_url") or start_frame.get("image_path")
        end_img = end_frame.get("image_url") or end_frame.get("image_path")

        if not start_img:
            raise ValueError(f"Start frame missing both image_url and image_path: {start_frame}")
        if not end_img:
            raise ValueError(f"End frame missing both image_url and image_path: {end_frame}")

        # 如果是本地路径，检查文件是否存在
        from urllib.parse import urlparse
        if urlparse(start_img).scheme not in ('http', 'https'):
            if not Path(start_img).exists():
                raise FileNotFoundError(f"Start frame not found: {start_img}")
        if urlparse(end_img).scheme not in ('http', 'https'):
            if not Path(end_img).exists():
                raise FileNotFoundError(f"End frame not found: {end_img}")

        # 调用千问API（支持URL和本地路径）
        result = await self.qwen.generate_video_from_frames(
            start_img,
            end_img,
            duration_seconds=5.0
        )

        if result["success"]:
            # 等待生成完成
            task_id = result["task_id"]
            completion_result = await self.qwen.wait_for_completion(task_id)

            if completion_result["success"]:
                # 获取视频URL(万相返回的阿里云URL,可直接用于IMS)
                video_url = completion_result["video_url"]

                # 返回URL,不需要下载
                return {
                    "url": video_url,
                    "duration": 5.0
                }
            else:
                raise Exception(f"Video generation failed: {completion_result['error']}")
        else:
            raise Exception(f"API call failed: {result['error']}")

    async def process_keyframes_with_consistency(self,
                                                keyframes_with_strategy: List[Dict],
                                                output_dir: str,
                                                product_image_url: str = None) -> List[Dict]:
        """
        处理带一致性策略的关键帧，生成视频片段

        参数:
            keyframes_with_strategy: 带生成策略的关键帧列表，包含：
                - refined_prompt: 细化后的提示词
                - generation_strategy: "text_to_image" 或 "image_to_image"
                - reference_source: "none" 或 "previous_frame" 或 "product_image"
            output_dir: 输出目录
            product_image_url: 产品参考图片URL（可选）

        返回:
            生成的视频片段列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_clips = []
        generated_images = []  # 存储已生成的图片，供后续参考

        logger.info(f"\n{'='*80}")
        logger.info(f"🎬 开始生成 {len(keyframes_with_strategy)} 个视频片段（含一致性保障）")
        if product_image_url:
            logger.info(f"📦 使用产品参考图: {product_image_url[:80]}...")
        logger.info(f"{'='*80}\n")

        # 逐个处理关键帧
        for i, keyframe in enumerate(keyframes_with_strategy):
            try:
                logger.info(f"📸 正在处理关键帧 {i+1}/{len(keyframes_with_strategy)}...")

                strategy = keyframe.get("generation_strategy", "text_to_image")
                reference_source = keyframe.get("reference_source", "none")
                refined_prompt = keyframe.get("refined_prompt", "")

                logger.info(f"   策略: {strategy}")
                logger.info(f"   参考源: {reference_source}")
                logger.info(f"   提示词: {refined_prompt[:60]}...")

                # === 步骤1: 生成或获取关键帧图片 ===
                current_image_url = None

                if strategy == "image_to_image" and reference_source == "product_image" and product_image_url:
                    # ✅ 第一个镜头：直接使用产品原图，不进行图编辑（避免变形）
                    logger.info(f"   📦 使用产品原图（跳过图编辑，避免变形）...")
                    current_image_url = product_image_url  # 直接使用产品图
                elif strategy == "image_to_image" and reference_source == "previous_frame" and generated_images:
                    # 使用前一帧作为参考
                    reference_image_url = generated_images[-1]
                    if reference_image_url:
                        logger.info(f"   🔗 使用前一帧作为参考: {reference_image_url[:60] if reference_image_url else 'None'}...")
                        current_image_url = await self._generate_image_from_image(
                            reference_image_url,
                            refined_prompt
                        )
                    else:
                        logger.info(f"   ⚠️ 前一帧为空，降级为文生图...")
                        current_image_url = await self._generate_image_from_text(refined_prompt)
                else:
                    # 文生图生成当前关键帧
                    logger.info(f"   🎨 使用文生图生成关键帧...")
                    current_image_url = await self._generate_image_from_text(refined_prompt)

                # 检查是否成功生成
                if not current_image_url:
                    logger.info(f"   ❌ 关键帧生成失败，跳过此帧")
                    continue

                # 保存生成的图片URL
                generated_images.append(current_image_url)
                logger.info(f"   ✅ 关键帧生成成功: {current_image_url[:60]}...")

                # === 步骤2: 使用关键帧生成视频 ===
                logger.info(f"   🎥 正在生成视频片段...")
                # 提取动态运动描述（前40个字），用于指导视频生成
                motion_prompt = refined_prompt[:80] if refined_prompt else None
                video_result = await self._generate_video_from_single_image(
                    current_image_url,
                    duration_seconds=5.0,
                    video_prompt=motion_prompt  # 使用refined_prompt指导视频生成
                )

                video_clips.append(video_result)
                logger.info(f"   ✅ 视频片段 {i+1} 生成成功")
                logger.info(f"      URL: {video_result.get('url', 'N/A')[:60]}...")

                # 添加延迟避免限流
                if i < len(keyframes_with_strategy) - 1:
                    await asyncio.sleep(2)

            except Exception as e:
                logger.info(f"   ❌ 关键帧 {i+1} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                # 继续处理下一个

        logger.info(f"\n{'='*80}")
        logger.info(f"✅ 视频生成完成，共 {len(video_clips)} 个片段")
        logger.info(f"{'='*80}\n")

        return video_clips

    async def _generate_image_from_text(self, prompt: str) -> str:
        """
        文生图：使用万相文生图API生成图片

        参数:
            prompt: 图片描述提示词

        返回:
            图片URL
        """
        # 万相文生图API端点
        endpoint = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"

        request_body = {
            "model": "wanx-v1",  # 文生图模型
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "style": "<auto>",  # 自动风格
                "size": "1280*720",  # 16:9比例
                "n": 1
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    headers=self.qwen.headers,
                    json=request_body
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        task_id = result.get("output", {}).get("task_id")

                        if task_id:
                            logger.info(f"      📋 文生图任务已提交, task_id: {task_id[:8]}...")
                            # 等待图片生成完成
                            completion_result = await self.qwen.wait_for_completion(task_id)

                            if completion_result["success"]:
                                # 万相文生图返回的URL在 video_url 字段（复用）
                                image_url = completion_result.get("video_url")

                                if not image_url:
                                    # 如果video_url为空，尝试重新查询任务状态获取URL
                                    status_response = await self.qwen.get_task_status(task_id)
                                    output = status_response.get("output", {})
                                    # 文生图的URL在 output.results[0].url
                                    results = output.get("results", [])
                                    if results and len(results) > 0:
                                        image_url = results[0].get("url")

                                if not image_url:
                                    logger.info(f"      ⚠️ 文生图任务完成，但未找到图片URL")
                                    logger.info(f"      响应: {completion_result}")
                                    return None

                                return image_url
                            else:
                                logger.info(f"      ❌ 文生图失败: {completion_result.get('error')}")
                                return None
                        else:
                            logger.info(f"      ❌ 文生图API未返回task_id: {result}")
                            return None
                    else:
                        error_text = await response.text()
                        logger.info(f"      ❌ 文生图API错误 {response.status}: {error_text[:200]}")
                        return None
        except Exception as e:
            logger.info(f"      ❌ 文生图异常: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _generate_image_from_image(self, reference_image_url: str, prompt: str) -> str:
        """
        图生图：使用参考图生成新图片（保持一致性）

        使用万相的 wanx2.1-imageedit 模型实现图生图功能

        参数:
            reference_image_url: 参考图片URL
            prompt: 图片描述提示词

        返回:
            图片URL
        """
        logger.info(f"      🎨 使用图编辑API (wanx2.1-imageedit) 进行图生图...")
        logger.info(f"      📸 参考图: {reference_image_url[:80]}...")

        try:
            # ✅ 添加保持产品外观的约束到prompt
            constrained_prompt = f"保持产品外观和形态不变，仅调整{prompt}"

            # 使用万相图编辑API
            task_id = await self.qwen.submit_image_edit_task(
                base_image_url=reference_image_url,
                prompt=constrained_prompt,  # 使用增强的prompt
                function="stylization_all"  # 全局风格迁移，保持主体一致性
            )

            if not task_id:
                logger.info(f"      ❌ 图编辑任务提交失败")
                # 降级为文生图
                logger.info(f"      ⚠️ 降级为文生图")
                return await self._generate_image_from_text(prompt)

            logger.info(f"      📋 图编辑任务已提交, task_id: {task_id[:12]}...")

            # 等待任务完成
            result = await self.qwen.wait_for_completion(task_id, timeout=180)

            if not result.get("success"):
                logger.info(f"      ❌ 图编辑任务失败")
                # 降级为文生图
                logger.info(f"      ⚠️ 降级为文生图")
                return await self._generate_image_from_text(prompt)

            image_url = result.get("video_url")  # 图编辑返回的也是这个字段

            if not image_url:
                logger.info(f"      ⚠️ 图编辑任务完成，但未找到图片URL")
                # 降级为文生图
                logger.info(f"      ⚠️ 降级为文生图")
                return await self._generate_image_from_text(prompt)

            logger.info(f"      ✅ 图编辑成功: {image_url[:80]}...")
            return image_url

        except Exception as e:
            logger.info(f"      ❌ 图生图异常: {e}")
            import traceback
            traceback.print_exc()
            # 降级为文生图
            logger.info(f"      ⚠️ 降级为文生图")
            return await self._generate_image_from_text(prompt)

    async def _generate_video_from_single_image(self, image_url: str, duration_seconds: float = 5.0, video_prompt: str = None) -> Dict:
        """
        使用单张图片生成视频

        参数:
            image_url: 图片URL
            duration_seconds: 视频时长
            video_prompt: 视频描述提示词（可选，用于指导视频生成）

        返回:
            视频信息 {"url": ..., "duration": ...}
        """
        result = await self.qwen.generate_video_from_frames(
            image_url,
            image_url,  # 首尾帧相同
            duration_seconds=duration_seconds,
            video_prompt=video_prompt  # 传递视频描述
        )

        if result["success"]:
            task_id = result["task_id"]
            completion_result = await self.qwen.wait_for_completion(task_id)

            if completion_result["success"]:
                return {
                    "url": completion_result["video_url"],
                    "duration": duration_seconds
                }
            else:
                raise Exception(f"Video generation failed: {completion_result['error']}")
        else:
            raise Exception(f"API call failed: {result['error']}")

    async def _download_video(self, url: str, output_path: Path):
        """下载视频文件"""

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(output_path, "wb") as f:
                        f.write(content)
                else:
                    raise Exception(f"Failed to download video: {response.status}")

    def _convert_subtitle_to_ims_format(
        self,
        subtitle_sequence: Dict,
        video_start_time: float = 0.0
    ) -> List[Dict]:
        """
        将subtitle_sequence转换为阿里云IMS的SubtitleTrackClips格式

        参数:
            subtitle_sequence: Node 14生成的字幕数据
            video_start_time: 视频开始时间（用于对齐片头）

        返回:
            IMS SubtitleTrackClips数组
        """
        if not subtitle_sequence or "clips" not in subtitle_sequence:
            return []

        clips = subtitle_sequence.get("clips", [])
        style_config = subtitle_sequence.get("style_config", {})

        # 提取样式配置
        font_color = style_config.get("color", "#FFFFFF")
        stroke_color = style_config.get("stroke", "#000000")
        font_size = style_config.get("font_size", 40)

        ims_subtitles = []

        for clip in clips:
            # 字幕文本
            text = clip.get("text", "")
            if not text:
                continue

            # 时间对齐（加上视频开始时间）
            timeline_in = video_start_time + clip.get("start", 0.0)
            timeline_out = video_start_time + clip.get("end", clip.get("start", 0.0) + clip.get("duration", 0.0))

            # 位置信息 - 针对720p视频优化
            # Y=580 距离底部约140px，适合大多数场景
            y_pos = 580

            # 构建IMS字幕格式
            ims_clip = {
                "Type": "Text",
                "Content": text.replace("\n", "\\N"),  # IMS使用\\N作为换行符
                "X": 0,
                "Y": y_pos,  # 固定为580，适配720p视频
                "Font": "AlibabaPuHuiTi",  # 阿里云内置字体
                "FontSize": font_size,
                "FontColor": font_color,
                "Outline": 2,  # 描边宽度
                "OutlineColour": stroke_color,
                "Alignment": "TopCenter",
                "TimelineIn": round(timeline_in, 2),
                "TimelineOut": round(timeline_out, 2),
                "FontFace": {
                    "Bold": True
                }
            }

            ims_subtitles.append(ims_clip)

        return ims_subtitles

    async def merge_clips(self, clip_data: List[Dict], output_path: str, subtitle_sequence: Dict = None, vgp_context: Dict = None) -> Dict:
        """
        合并视频片段 - 使用阿里云IMS API

        参数:
            clip_data: 视频片段数据列表,每项包含 {"url": ..., "duration": ...}
            output_path: 输出路径(仅用于命名)
            subtitle_sequence: 字幕序列（可选），从Node 14生成
            vgp_context: VGP上下文（包含滤镜、转场、特效等信息）

        返回:
            包含合并后视频URL的字典
        """
        try:
            from alibabacloud_ice20201109 import client as ice_client, models as ice_models
            from alibabacloud_tea_openapi import models as open_api_models
            import json

            # 提取所有视频URL(万相返回的URL可直接用于IMS)
            video_urls = [clip["url"] for clip in clip_data]
            logger.info(f"🎬 使用阿里云IMS合并 {len(video_urls)} 个视频片段...")
            logger.info(f"   视频URL示例: {video_urls[0][:80]}...")

            # 初始化IMS客户端配置
            config = open_api_models.Config(
                access_key_id=os.getenv("OSS_ACCESS_KEY_ID"),
                access_key_secret=os.getenv("OSS_ACCESS_KEY_SECRET"),
                region_id='cn-shanghai',
                endpoint='ice.cn-shanghai.aliyuncs.com'
            )
            client = ice_client.Client(config)

            # 构建基础Timeline
            timeline = {
                "VideoTracks": [{
                    "VideoTrackClips": [
                        {
                            "MediaURL": url,
                            "Effects": []  # ✅ 添加Effects字段用于转场
                        }
                        for url in video_urls
                    ]
                }]
            }

            # ✅ 集成IMS转换器 - 处理转场、滤镜、特效
            if vgp_context:
                try:
                    from ims_converter import IMSConverter

                    logger.info(f"🎨 开始应用VGP特效到IMS Timeline...")
                    converter = IMSConverter(use_filter_preset=True)

                    # 准备VGP输出数据
                    vgp_result = {
                        "filter_sequence_id": vgp_context.get("filter_sequence_id", []),
                        "transition_sequence_id": vgp_context.get("transition_sequence_id", []),
                        "effects_sequence_id": vgp_context.get("effects_sequence_id", [])
                    }

                    # 转换为IMS格式
                    converted = converter.convert(vgp_result)

                    # 合并转换后的轨道
                    if converted.get("VideoTracks"):
                        # 添加转场效果到VideoTrackClips
                        converted_clips = converted["VideoTracks"][0].get("VideoTrackClips", [])
                        for i, clip in enumerate(timeline["VideoTracks"][0]["VideoTrackClips"]):
                            if i < len(converted_clips) and converted_clips[i].get("Effects"):
                                clip["Effects"] = converted_clips[i]["Effects"]
                                logger.info(f"   ✅ Clip {i+1}: 添加 {len(clip['Effects'])} 个转场效果")

                    # 添加滤镜和特效轨道
                    if converted.get("EffectTracks"):
                        if "EffectTracks" not in timeline:
                            timeline["EffectTracks"] = []
                        timeline["EffectTracks"].extend(converted["EffectTracks"])

                        total_effects = sum(len(track.get("EffectTrackItems", [])) for track in converted["EffectTracks"])
                        logger.info(f"   ✅ 添加 {total_effects} 个滤镜/特效")

                    logger.info(f"✨ VGP特效应用完成")

                except ImportError:
                    logger.warning(f"   ⚠️ IMS转换器未安装，跳过转场/滤镜/特效")
                except Exception as e:
                    logger.warning(f"   ⚠️ IMS转换失败: {e}")
                    import traceback
                    traceback.print_exc()

            # 添加字幕轨道
            if subtitle_sequence:
                logger.info(f"📝 添加字幕轨道...")
                subtitle_clips = self._convert_subtitle_to_ims_format(
                    subtitle_sequence,
                    video_start_time=0.0  # 如果有片头，需要传入片头时长
                )

                if subtitle_clips:
                    timeline["SubtitleTracks"] = [{
                        "SubtitleTrackClips": subtitle_clips
                    }]
                    logger.info(f"   ✅ 已添加 {len(subtitle_clips)} 个字幕片段")

                    # ✨ 新增：生成TTS音频并添加到AudioTracks
                    try:
                        from video_generate_protocol.nodes.audio_tts_integration import integrate_tts_to_timeline

                        logger.info(f"🎤 开始生成TTS语音...")
                        timeline = await integrate_tts_to_timeline(
                            timeline,
                            subtitle_sequence,
                            voice="Cherry",        # ✅ 使用阿里云Qwen3-TTS支持的音色（芊悦-温柔女声）
                            speed=1.0,
                            upload_to_oss=True,    # 上传到OSS获取永久URL
                            use_segmented=True     # ✨ 使用分段生成，实现音画精确同步
                        )
                        logger.info(f"   ✅ TTS音频已集成到timeline")
                    except Exception as tts_error:
                        logger.warning(f"   ⚠️ TTS音频生成失败，视频将无声音: {tts_error}")
                        import traceback
                        traceback.print_exc()
                        # TTS失败不影响主流程，继续生成无声视频
                else:
                    logger.info(f"   ⚠️ 字幕序列为空，跳过字幕轨道")
            else:
                logger.info(f"   ℹ️ 未提供字幕序列，跳过字幕轨道和TTS音频")

            # 输出配置
            output_config = {
                "MediaURL": f"https://ai-movie-cloud-v2.oss-cn-shanghai.aliyuncs.com/merged_videos/{Path(output_path).name}",
                "Width": 1280,
                "Height": 720
            }

            # 提交合成任务
            request = ice_models.SubmitMediaProducingJobRequest(
                timeline=json.dumps(timeline, ensure_ascii=False),
                output_media_config=json.dumps(output_config, ensure_ascii=False)
            )

            logger.info(f"📋 Timeline配置: {json.dumps(timeline, indent=2, ensure_ascii=False)}")
            response = client.submit_media_producing_job(request)

            if response.status_code == 200:
                job_id = response.body.job_id
                logger.info(f"✅ IMS合并任务已提交, JobId: {job_id}")
                # 等待任务完成并获取最终视频URL
                final_url = await self._wait_for_ims_job(client, job_id)
                return {
                    "success": True,
                    "video_url": final_url,
                    "job_id": job_id
                }
            else:
                raise Exception(f"IMS合并失败: status={response.status_code}")

        except Exception as e:
            logger.info(f"⚠️ IMS合并失败,降级使用本地ffmpeg: {e}")
            import traceback
            traceback.print_exc()
            # 降级方案: 使用ffmpeg本地合并
            return await self._merge_clips_ffmpeg(clip_data, output_path)

    async def _merge_clips_ffmpeg(self, clip_data: List[Dict], output_path: str) -> Dict:
        """降级方案: 使用ffmpeg本地合并"""
        import subprocess
        import tempfile

        logger.info(f"📥 开始下载视频片段到本地...")
        local_clips = []
        temp_dir = Path(tempfile.mkdtemp(prefix="video_clips_"))

        try:
            # 下载所有视频片段
            for i, clip in enumerate(clip_data):
                video_url = clip.get("url")
                local_path = temp_dir / f"clip_{i:03d}.mp4"

                logger.info(f"   下载片段 {i+1}/{len(clip_data)}: {video_url[:80]}...")
                await self._download_video(video_url, local_path)
                local_clips.append(str(local_path))

            # 创建文件列表
            list_file = temp_dir / "clips.txt"
            with open(list_file, "w") as f:
                for local_path in local_clips:
                    f.write(f"file '{local_path}'\n")

            # 使用ffmpeg合并
            logger.info(f"🎬 使用ffmpeg合并 {len(local_clips)} 个视频片段...")
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c", "copy", "-y", output_path
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise Exception(f"FFmpeg merge failed: {stderr.decode()}")

            logger.info(f"✅ ffmpeg合并完成: {output_path}")
            return {"success": True, "local_path": output_path}

        finally:
            # 清理临时文件
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    async def _wait_for_ims_job(self, client, job_id: str, timeout: int = 300) -> str:
        """等待IMS合并任务完成"""
        from alibabacloud_ice20201109 import models as ice_models
        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            request = ice_models.GetMediaProducingJobRequest(job_id=job_id)
            response = client.get_media_producing_job(request)

            if response.status_code == 200:
                job = response.body.media_producing_job
                status = job.status

                if status == "Success":
                    media_url = job.media_url
                    logger.info(f"✅ IMS合并完成")
                    logger.info(f"   🎬 最终视频URL: {media_url}")
                    return media_url
                elif status == "Failed":
                    raise Exception(f"IMS任务失败: {getattr(job, 'message', 'Unknown error')}")
                else:
                    logger.info(f"⏳ IMS合并中... ({status})")
                    await asyncio.sleep(5)
            else:
                raise Exception(f"查询IMS任务状态失败: {response.status_code}")

        raise Exception("IMS合并超时")


# 使用示例
async def demo():
    """演示完整流程"""

    # 初始化处理器
    processor = StoryboardToVideoProcessor(
        qwen_api_key="your_api_key_here"
    )

    # 模拟关键帧数据
    keyframes = [
        {
            "frame_id": "frame_001",
            "image_path": "/tmp/frame_001.png",
            "is_reused": False
        },
        {
            "frame_id": "frame_002",
            "image_path": "/tmp/frame_002.png",
            "is_reused": False
        },
        {
            "frame_id": "frame_002",  # 复用frame_002作为下一段的首帧
            "image_path": "/tmp/frame_002.png",
            "is_reused": True
        },
        {
            "frame_id": "frame_003",
            "image_path": "/tmp/frame_003.png",
            "is_reused": False
        },
        # ... 更多帧
    ]

    # 生成视频片段
    clips = await processor.process_storyboard_frames(
        keyframes,
        "/tmp/video_output"
    )

    logger.info(f"Generated {len(clips)} video clips")

    # 合并成最终视频
    final_video = await processor.merge_clips(
        clips,
        "/tmp/video_output/final_video.mp4"
    )

    logger.info(f"Final video: {final_video}")

    return final_video


if __name__ == "__main__":
    # 注意：需要设置实际的API密钥
    asyncio.run(demo())