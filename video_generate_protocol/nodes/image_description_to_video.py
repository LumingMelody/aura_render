"""
图片+描述到视频生成器 - 基于图片和描述生成视频
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import json

from image_generation_node import ImageGenerationNode, ImageGenerationTask, ImageGenerationNodeRequest
from vgp_optimization_node import VGPOptimizationNode, VGPOptimizationConfig
from qwen_integration import QwenVideoGenerator
from video_generate_protocol.prompt_manager import get_prompt_manager


@dataclass
class ImageDescriptionToVideoRequest:
    """
    图片+描述到视频请求体

    支持两种输入方式：
    1. 单张图片+整体描述 → 生成完整视频
    2. 多张图片+对应描述 → 基于分镜生成视频
    """

    # 方式1：单张图片+描述
    image_path: Optional[str] = None  # 单张图片路径
    description: Optional[str] = None  # 整体视频描述

    # 方式2：多张图片+多个描述（分镜）
    storyboard_items: Optional[List[Dict[str, str]]] = None  # [{"image": "path", "description": "..."}, ...]

    # 共同参数
    total_duration_seconds: int = 30  # 总时长

    # 生成参数
    product_info: Optional[Dict] = None  # 产品信息（保持一致性）
    style_config: Dict[str, Any] = field(default_factory=lambda: {
        "visual_style": "realistic",
        "motion_intensity": "medium",
        "transition_type": "smooth"
    })

    # 高级选项
    auto_generate_intermediate_frames: bool = True  # 自动生成中间帧
    use_vl_validation: bool = True  # 使用VL验证

    # 输出配置
    output_path: Optional[str] = None
    save_intermediate_frames: bool = False

    # VGP系统必需的ID参数
    theme_id: Optional[str] = None
    keywords_id: Optional[str] = None
    target_duration_id: Optional[str] = None
    user_description_id: Optional[str] = None


@dataclass
class VideoSegmentPlan:
    """视频段落规划"""
    segment_id: int
    start_time_ms: int
    end_time_ms: int
    duration_ms: int

    # 关键帧
    start_frame: Dict[str, Any]  # {"image_path": str, "description": str, "is_generated": bool}
    end_frame: Dict[str, Any]

    # 生成策略
    generation_mode: str  # "first_last_frame" or "single_frame_extend"
    needs_generation: bool  # 是否需要生成新帧

    # 描述
    segment_description: str
    transition_from_previous: Optional[str] = None


class ImageDescriptionToVideoOrchestrator:
    """
    图片+描述到视频编排器

    核心功能：
    1. 解析图片和描述，生成分镜计划
    2. 根据描述生成缺失的关键帧
    3. 使用千问API生成5秒视频片段
    4. 合并成最终视频
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 初始化各个节点
        self.image_generation_node = ImageGenerationNode(config)

        # VGP优化节点
        vgp_config = VGPOptimizationConfig(
            product_protection_level="maximum" if config.get("product_info") else "medium",
            prefer_wide_shots=True,
            enhance_lighting=True
        )
        self.vgp_node = VGPOptimizationNode(vgp_config)

        # 千问视频生成器
        qwen_key = config.get("qwen_api_key")
        if not qwen_key:
            raise ValueError("qwen_api_key is required")
        self.qwen_generator = QwenVideoGenerator(qwen_key)

        # 工作目录
        self.work_dir = Path(config.get("work_dir", "/tmp/image_desc_video"))
        self.work_dir.mkdir(parents=True, exist_ok=True)

    async def process_request(self, request: ImageDescriptionToVideoRequest) -> Dict[str, Any]:
        """
        处理图片+描述到视频的请求
        """

        print("\n" + "="*60)
        print("🎬 开始处理图片+描述到视频请求")
        print(f"⏱️ 目标时长: {request.total_duration_seconds}秒")
        print("="*60)

        try:
            # 第1步：解析输入，生成分镜计划
            segment_plans = await self._create_segment_plans(request)

            # 第2步：生成缺失的关键帧
            if request.auto_generate_intermediate_frames:
                segment_plans = await self._generate_missing_frames(segment_plans, request)

            # 第3步：VL验证（如果启用）
            if request.use_vl_validation:
                validation_result = await self._validate_frames(segment_plans)
                if not validation_result["passed"]:
                    print(f"⚠️ VL验证未完全通过，但继续生成")

            # 第4步：生成视频片段
            video_clips = await self._generate_video_clips(segment_plans)

            # 第5步：合并最终视频
            final_video = await self._merge_final_video(
                video_clips,
                request.output_path or str(self.work_dir / "final_output.mp4")
            )

            return {
                "success": True,
                "video_path": final_video,
                "duration_seconds": request.total_duration_seconds,
                "segments_count": len(segment_plans),
                "clips_generated": len(video_clips),
                "segment_plans": [self._segment_plan_to_dict(plan) for plan in segment_plans]
            }

        except Exception as e:
            print(f"❌ 处理失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _create_segment_plans(self, request: ImageDescriptionToVideoRequest) -> List[VideoSegmentPlan]:
        """
        第1步：创建视频段落规划
        """

        print("\n[第1步] 📋 创建分镜规划...")

        # 计算需要的5秒片段数
        num_segments = (request.total_duration_seconds + 4) // 5
        segment_duration_ms = 5000

        segment_plans = []

        if request.image_path and request.description:
            # 方式1：单张图片+描述
            print("  模式：单张图片+描述")
            segment_plans = await self._create_plans_from_single_image(
                request.image_path,
                request.description,
                num_segments,
                segment_duration_ms,
                request
            )

        elif request.storyboard_items:
            # 方式2：多张图片+描述（分镜）
            print("  模式：多张分镜图片")
            segment_plans = await self._create_plans_from_storyboard(
                request.storyboard_items,
                num_segments,
                segment_duration_ms,
                request
            )
        else:
            raise ValueError("必须提供 (image_path + description) 或 storyboard_items")

        print(f"  ✅ 创建了 {len(segment_plans)} 个段落规划")

        return segment_plans

    async def _create_plans_from_single_image(self,
                                             image_path: str,
                                             description: str,
                                             num_segments: int,
                                             segment_duration_ms: int,
                                             request: ImageDescriptionToVideoRequest) -> List[VideoSegmentPlan]:
        """
        从单张图片+描述创建分镜规划
        """

        # 解析描述，分解成多个阶段
        stage_descriptions = self._parse_description_to_stages(description, num_segments)

        plans = []

        for i in range(num_segments):
            start_time = i * segment_duration_ms
            end_time = min((i + 1) * segment_duration_ms, request.total_duration_seconds * 1000)

            # 第一个段落使用原始图片作为起点
            if i == 0:
                start_frame = {
                    "image_path": image_path,
                    "description": stage_descriptions[i]["start"],
                    "is_generated": False,
                    "is_original": True
                }
            else:
                # 复用前一段的尾帧（或生成新的）
                start_frame = {
                    "image_path": None,  # 待生成
                    "description": stage_descriptions[i]["start"],
                    "is_generated": True,
                    "is_original": False,
                    "needs_generation": True
                }

            # 尾帧都需要生成
            end_frame = {
                "image_path": None,  # 待生成
                "description": stage_descriptions[i]["end"],
                "is_generated": True,
                "is_original": False,
                "needs_generation": True
            }

            plan = VideoSegmentPlan(
                segment_id=i,
                start_time_ms=start_time,
                end_time_ms=end_time,
                duration_ms=end_time - start_time,
                start_frame=start_frame,
                end_frame=end_frame,
                generation_mode="first_last_frame",
                needs_generation=True,
                segment_description=stage_descriptions[i]["description"],
                transition_from_previous=stage_descriptions[i].get("transition")
            )

            plans.append(plan)

        return plans

    async def _create_plans_from_storyboard(self,
                                           storyboard_items: List[Dict],
                                           num_segments: int,
                                           segment_duration_ms: int,
                                           request: ImageDescriptionToVideoRequest) -> List[VideoSegmentPlan]:
        """
        从多张分镜图创建规划
        """

        plans = []

        # 分配分镜到段落
        items_per_segment = max(1, len(storyboard_items) // num_segments)

        for i in range(num_segments):
            start_time = i * segment_duration_ms
            end_time = min((i + 1) * segment_duration_ms, request.total_duration_seconds * 1000)

            # 获取对应的分镜项
            start_idx = min(i * items_per_segment, len(storyboard_items) - 1)
            end_idx = min((i + 1) * items_per_segment, len(storyboard_items) - 1)

            start_item = storyboard_items[start_idx]
            end_item = storyboard_items[end_idx] if end_idx != start_idx else storyboard_items[min(start_idx + 1, len(storyboard_items) - 1)]

            # 判断是否需要复用帧
            if i > 0:
                prev_plan = plans[-1]
                # 检查是否连续场景
                if self._is_continuous_scene(prev_plan.end_frame["description"], start_item["description"]):
                    # 复用前一段的尾帧
                    start_frame = {
                        "image_path": prev_plan.end_frame["image_path"],
                        "description": start_item["description"],
                        "is_generated": False,
                        "is_reused": True
                    }
                else:
                    # 使用新的分镜图
                    start_frame = {
                        "image_path": start_item["image"],
                        "description": start_item["description"],
                        "is_generated": False,
                        "is_original": True
                    }
            else:
                # 第一段
                start_frame = {
                    "image_path": start_item["image"],
                    "description": start_item["description"],
                    "is_generated": False,
                    "is_original": True
                }

            # 尾帧
            end_frame = {
                "image_path": end_item["image"] if end_item != start_item else None,
                "description": end_item["description"],
                "is_generated": end_item == start_item,  # 如果是同一个项，需要生成
                "needs_generation": end_item == start_item
            }

            plan = VideoSegmentPlan(
                segment_id=i,
                start_time_ms=start_time,
                end_time_ms=end_time,
                duration_ms=end_time - start_time,
                start_frame=start_frame,
                end_frame=end_frame,
                generation_mode="first_last_frame",
                needs_generation=end_frame.get("needs_generation", False),
                segment_description=f"{start_item['description']} → {end_item['description']}"
            )

            plans.append(plan)

        return plans

    def _parse_description_to_stages(self, description: str, num_segments: int) -> List[Dict]:
        """
        将描述解析成多个阶段
        """

        # 尝试按句号、分号等分割
        sentences = description.replace('。', '.').replace('；', ';').replace('，', ',').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        stages = []

        if len(sentences) >= num_segments:
            # 句子够多，直接分配
            for i in range(num_segments):
                stage_desc = sentences[i] if i < len(sentences) else sentences[-1]
                stages.append({
                    "description": stage_desc,
                    "start": f"Beginning of: {stage_desc}",
                    "end": f"Completion of: {stage_desc}"
                })
        else:
            # 句子不够，需要插值
            for i in range(num_segments):
                progress = i / max(1, num_segments - 1)

                if progress < 0.3:
                    stage = "opening"
                    desc = f"Opening scene: {description[:50]}"
                elif progress < 0.7:
                    stage = "development"
                    desc = f"Developing: {description}"
                else:
                    stage = "conclusion"
                    desc = f"Concluding: {description}"

                stages.append({
                    "description": desc,
                    "start": f"{stage} - beginning",
                    "end": f"{stage} - end",
                    "transition": "smooth" if i > 0 else None
                })

        return stages

    async def _generate_missing_frames(self,
                                      segment_plans: List[VideoSegmentPlan],
                                      request: ImageDescriptionToVideoRequest) -> List[VideoSegmentPlan]:
        """
        第2步：生成缺失的关键帧
        """

        print("\n[第2步] 🎨 生成缺失的关键帧...")

        frames_to_generate = []

        # 收集需要生成的帧
        for plan in segment_plans:
            if plan.start_frame.get("needs_generation"):
                frames_to_generate.append(("start", plan.segment_id, plan.start_frame))
            if plan.end_frame.get("needs_generation"):
                frames_to_generate.append(("end", plan.segment_id, plan.end_frame))

        if not frames_to_generate:
            print("  ✅ 所有帧都已提供，无需生成")
            return segment_plans

        print(f"  需要生成 {len(frames_to_generate)} 个关键帧")

        # 批量生成
        image_tasks = []
        for position, seg_id, frame_info in frames_to_generate:
            # 构建提示词 - 使用PromptManager
            prompt = self._build_frame_prompt_with_manager(frame_info["description"], request)

            # 确定参考图像
            reference_image = None
            if request.image_path:
                # 优先使用提供的产品图片作为参考
                reference_image = request.image_path
            elif frame_info.get("image_path") and Path(frame_info["image_path"]).exists():
                # 使用已有的帧图片
                reference_image = frame_info["image_path"]

            task = ImageGenerationTask(
                prompt=prompt,
                reference_image=reference_image,  # 添加参考图像
                style=request.style_config.get("visual_style", "realistic"),
                quality="high",
                aspect_ratio="16:9",
                width=1920,
                height=1080
            )

            image_tasks.append({
                "task": task,
                "position": position,
                "segment_id": seg_id
            })

        # 调用图像生成节点
        tasks_only = [item["task"] for item in image_tasks]

        generation_request = ImageGenerationNodeRequest(
            tasks=tasks_only,
            batch_mode=True,
            generation_config={
                "enhance_prompts": True,
                "fallback_enabled": True
            }
        )

        print("  🔄 批量生成图像...")
        response = await self.image_generation_node.process(generation_request)

        if response.generated_images:
            print(f"  ✅ 成功生成 {len(response.generated_images)} 个关键帧")

            # 更新segment_plans中的图片路径
            for i, generated_image in enumerate(response.generated_images):
                task_info = image_tasks[i]
                seg_id = task_info["segment_id"]
                position = task_info["position"]

                # 找到对应的plan并更新
                for plan in segment_plans:
                    if plan.segment_id == seg_id:
                        if position == "start":
                            plan.start_frame["image_path"] = generated_image.image_path
                            plan.start_frame["is_generated"] = True
                        else:
                            plan.end_frame["image_path"] = generated_image.image_path
                            plan.end_frame["is_generated"] = True
                        break
        else:
            print("  ❌ 图像生成失败")

        # 处理帧复用逻辑
        segment_plans = self._apply_frame_reuse(segment_plans)

        return segment_plans

    def _apply_frame_reuse(self, segment_plans: List[VideoSegmentPlan]) -> List[VideoSegmentPlan]:
        """
        应用帧复用逻辑
        """

        for i in range(1, len(segment_plans)):
            curr_plan = segment_plans[i]
            prev_plan = segment_plans[i-1]

            # 判断是否应该复用
            if self._should_reuse_frame(prev_plan, curr_plan):
                # 复用前一段的尾帧作为当前段的首帧
                curr_plan.start_frame["image_path"] = prev_plan.end_frame["image_path"]
                curr_plan.start_frame["is_reused"] = True
                print(f"    🔄 段{curr_plan.segment_id}复用段{prev_plan.segment_id}的尾帧")

        return segment_plans

    def _should_reuse_frame(self, prev_plan: VideoSegmentPlan, curr_plan: VideoSegmentPlan) -> bool:
        """
        判断是否应该复用帧
        """

        # 检查描述的连续性
        prev_desc = prev_plan.segment_description.lower()
        curr_desc = curr_plan.segment_description.lower()

        # 连续性关键词
        continuity_keywords = ["continue", "then", "next", "follow", "progress", "develop"]

        for keyword in continuity_keywords:
            if keyword in curr_desc:
                return True

        # 场景相似度判断
        return self._is_continuous_scene(prev_desc, curr_desc)

    def _is_continuous_scene(self, desc1: str, desc2: str) -> bool:
        """
        判断是否连续场景
        """

        # 简单的词汇重叠判断
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))

        similarity = overlap / total if total > 0 else 0

        return similarity > 0.5

    async def _validate_frames(self, segment_plans: List[VideoSegmentPlan]) -> Dict:
        """
        第3步：VL验证
        """

        print("\n[第3步] 👁️ VL视觉验证...")

        # 这里应该调用实际的VL模型
        # 暂时返回模拟结果

        return {
            "passed": True,
            "confidence": 0.9,
            "issues": []
        }

    async def _generate_video_clips(self, segment_plans: List[VideoSegmentPlan]) -> List[str]:
        """
        第4步：生成视频片段
        """

        print("\n[第4步] 🎥 生成5秒视频片段...")

        video_clips = []

        for plan in segment_plans:
            print(f"  生成片段 {plan.segment_id + 1}/{len(segment_plans)}")

            # 确保帧路径存在
            if not plan.start_frame.get("image_path") or not plan.end_frame.get("image_path"):
                print(f"    ⚠️ 片段{plan.segment_id}缺少关键帧，跳过")
                continue

            # 调用千问API生成视频
            video_result = await self.qwen_generator.generate_video_from_frames(
                start_image_path=plan.start_frame["image_path"],
                end_image_path=plan.end_frame["image_path"],
                duration_seconds=plan.duration_ms / 1000
            )

            if video_result["success"]:
                # 等待生成完成
                completion = await self.qwen_generator.wait_for_completion(video_result["task_id"])

                if completion["success"]:
                    # 下载视频
                    clip_path = str(self.work_dir / f"clip_{plan.segment_id:03d}.mp4")
                    await self._download_video(completion["video_url"], clip_path)
                    video_clips.append(clip_path)
                    print(f"    ✅ 片段{plan.segment_id}生成成功")
                else:
                    print(f"    ❌ 片段{plan.segment_id}生成失败: {completion.get('error')}")
            else:
                print(f"    ❌ 片段{plan.segment_id} API调用失败")

        print(f"  ✅ 成功生成 {len(video_clips)} 个视频片段")

        return video_clips

    async def _merge_final_video(self, video_clips: List[str], output_path: str) -> str:
        """
        第5步：合并视频
        """

        print("\n[第5步] 🔄 合并最终视频...")

        # 创建ffmpeg文件列表
        list_file = self.work_dir / "clips.txt"
        with open(list_file, "w") as f:
            for clip_path in video_clips:
                f.write(f"file '{clip_path}'\n")

        # 使用ffmpeg合并
        import subprocess

        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-y",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"视频合并失败: {result.stderr}")

        print(f"  ✅ 最终视频: {output_path}")

        return output_path

    async def _download_video(self, url: str, output_path: str):
        """下载视频"""

        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(output_path, "wb") as f:
                        f.write(content)
                else:
                    raise Exception(f"下载失败: {response.status}")

    def _build_frame_prompt(self, description: str, request: ImageDescriptionToVideoRequest) -> str:
        """构建帧提示词（保留向后兼容）"""
        return self._build_frame_prompt_with_manager(description, request)

    def _build_frame_prompt_with_manager(self, description: str, request: ImageDescriptionToVideoRequest) -> str:
        """使用PromptManager构建帧提示词"""

        # 获取PromptManager
        prompt_manager = get_prompt_manager()

        # 基础提示词
        base_prompt = description

        # 添加产品信息（如果有）
        if request.product_info:
            product_name = request.product_info.get("name", "product")
            base_prompt = f"{description}, featuring {product_name}"

        # 使用PromptManager增强提示词
        enhanced_prompt = prompt_manager.enhance_prompt(
            base_prompt,
            "frame_refinement",  # 首帧细化阶段
            context={
                "product": request.product_info,
                "input": description,
                "style": request.style_config
            }
        )

        return enhanced_prompt

    def _segment_plan_to_dict(self, plan: VideoSegmentPlan) -> Dict:
        """转换为字典格式"""

        return {
            "segment_id": plan.segment_id,
            "duration_ms": plan.duration_ms,
            "description": plan.segment_description,
            "start_frame": plan.start_frame,
            "end_frame": plan.end_frame,
            "generation_mode": plan.generation_mode
        }


# 便捷API
async def generate_video_from_image_and_description(
    image_path: str,
    description: str,
    duration_seconds: int = 30,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    便捷API：从图片+描述生成视频

    参数:
        image_path: 图片路径
        description: 视频描述
        duration_seconds: 视频时长
        config: 配置

    返回:
        生成结果
    """

    if not config:
        config = {
            "qwen_api_key": "your_qwen_api_key",
            "openai_api_key": "your_openai_api_key",
            "work_dir": "/tmp/image_desc_video"
        }

    orchestrator = ImageDescriptionToVideoOrchestrator(config)

    request = ImageDescriptionToVideoRequest(
        image_path=image_path,
        description=description,
        total_duration_seconds=duration_seconds
    )

    return await orchestrator.process_request(request)


# 使用示例
async def demo():
    """演示用法"""

    config = {
        "qwen_api_key": "your_actual_key",
        "openai_api_key": "your_actual_key",
        "work_dir": "/tmp/demo_video"
    }

    # 示例1：单张图片+描述
    print("\n🎬 示例1: 单张图片+描述")
    result1 = await generate_video_from_image_and_description(
        image_path="/path/to/product.jpg",
        description="""
        展示智能手表的完整功能。
        首先展示外观设计，360度旋转。
        然后展示屏幕界面和操作。
        接着演示运动追踪功能。
        最后展示充电和配件。
        """,
        duration_seconds=20,
        config=config
    )

    if result1["success"]:
        print(f"✅ 视频生成成功: {result1['video_path']}")

    # 示例2：多张分镜图
    print("\n🎬 示例2: 多张分镜图")

    orchestrator = ImageDescriptionToVideoOrchestrator(config)

    request2 = ImageDescriptionToVideoRequest(
        storyboard_items=[
            {"image": "/path/to/scene1.jpg", "description": "产品整体展示"},
            {"image": "/path/to/scene2.jpg", "description": "细节特写"},
            {"image": "/path/to/scene3.jpg", "description": "使用场景"},
            {"image": "/path/to/scene4.jpg", "description": "最终效果"}
        ],
        total_duration_seconds=20,
        product_info={
            "name": "SmartWatch Pro",
            "constraints": ["保持颜色一致"]
        }
    )

    result2 = await orchestrator.process_request(request2)

    if result2["success"]:
        print(f"✅ 分镜视频生成成功: {result2['video_path']}")
        print(f"📊 生成了 {result2['segments_count']} 个片段")


if __name__ == "__main__":
    asyncio.run(demo())