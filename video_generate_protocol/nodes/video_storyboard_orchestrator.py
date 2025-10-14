"""
视频分镜编排器 - 整合现有节点架构实现分镜到视频的完整流程
"""
from typing import Dict, List, Any, Optional
import asyncio
from dataclasses import dataclass
from pathlib import Path

# 导入现有节点
from video_generate_protocol.nodes.image_generation_node import (
    ImageGenerationNode,
    ImageGenerationTask,
    ImageGenerationNodeRequest
)
from vgp_optimization_node import VGPOptimizationNode, VGPOptimizationConfig
from storyboard_sequence_node import StoryboardSequenceNode
from qwen_integration import StoryboardToVideoProcessor


@dataclass
class VideoStoryboardRequest:
    """视频分镜请求"""
    text_description: str
    duration_seconds: int
    product_info: Optional[Dict] = None
    style_config: Optional[Dict] = None
    output_path: Optional[str] = None

    # VGP系统必需的ID参数
    theme_id: Optional[str] = None
    keywords_id: Optional[str] = None
    target_duration_id: Optional[str] = None
    user_description_id: Optional[str] = None


class VideoStoryboardOrchestrator:
    """
    视频分镜编排器

    整合现有的 video_generate_protocol/nodes 中的节点：
    - ImageGenerationNode: 图像生成
    - VGPOptimizationNode: 分镜优化
    - StoryboardSequenceNode: 分镜序列
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 初始化现有节点
        self.image_generation_node = ImageGenerationNode(config)

        # VGP优化配置
        vgp_config = VGPOptimizationConfig(
            product_protection_level="maximum",
            era_preference=config.get("era_preference", "modern"),
            forbidden_elements=config.get("forbidden_elements", []),
            prefer_wide_shots=True,
            enhance_lighting=True
        )
        self.vgp_optimization_node = VGPOptimizationNode(vgp_config)

        # 分镜序列节点
        self.storyboard_sequence_node = StoryboardSequenceNode(config)

        # 千问视频处理器
        qwen_key = config.get("qwen_api_key")
        if not qwen_key:
            raise ValueError("qwen_api_key is required")
        self.video_processor = StoryboardToVideoProcessor(qwen_key)

    async def process_video_request(self, request: VideoStoryboardRequest) -> Dict[str, Any]:
        """
        处理视频生成请求 - 使用现有节点架构
        """

        print(f"\n🎬 开始视频生成流程 ({request.duration_seconds}秒)")
        print("="*60)

        try:
            # 第1步：VGP优化分镜规划
            storyboard_plan = await self._optimize_storyboard(request)

            # 第2步：使用现有ImageGenerationNode生成关键帧
            keyframes = await self._generate_keyframes_with_existing_node(storyboard_plan)

            # 第3步：使用千问API生成5秒视频片段
            video_clips = await self._generate_video_clips(keyframes)

            # 第4步：合并最终视频
            final_video = await self._merge_final_video(video_clips, request.output_path)

            return {
                "success": True,
                "video_path": final_video,
                "duration_seconds": request.duration_seconds,
                "segments_count": len(video_clips),
                "keyframes_count": len(keyframes),
                "storyboard_plan": storyboard_plan
            }

        except Exception as e:
            print(f"❌ 视频生成失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _optimize_storyboard(self, request: VideoStoryboardRequest) -> Dict:
        """第1步：使用VGP节点优化分镜"""

        print("\n[第1步] 📋 VGP优化分镜规划...")

        # 解析文本为原始段落
        raw_segments = self._parse_text_to_segments(request.text_description)

        # 调用现有的VGP优化节点
        optimization_result = await self.vgp_optimization_node.optimize_storyboard_sequence(
            raw_segments=raw_segments,
            product_info=request.product_info,
            total_duration_ms=request.duration_seconds * 1000
        )

        # 显示优化结果
        frames = optimization_result['optimized_frames']
        segments_count = (request.duration_seconds + 4) // 5  # 5秒片段数

        print(f"  ✅ 生成了 {len(frames)} 个关键帧")
        print(f"  ✅ 规划了 {segments_count} 个5秒片段")

        # 分析生成模式分布
        mode_stats = {}
        for frame in frames:
            mode = frame.generation_mode.value
            mode_stats[mode] = mode_stats.get(mode, 0) + 1

        print("  📊 生成模式分布:")
        for mode, count in mode_stats.items():
            print(f"    {mode}: {count}")

        return optimization_result

    async def _generate_keyframes_with_existing_node(self, storyboard_plan: Dict) -> List[Dict]:
        """第2步：使用现有ImageGenerationNode生成关键帧"""

        print("\n[第2步] 🎨 生成关键帧图像...")

        optimized_frames = storyboard_plan['optimized_frames']
        generated_keyframes = []

        # 转换为ImageGenerationTask格式
        image_tasks = []
        for frame in optimized_frames:
            # 获取优化后的提示词
            prompt = frame.prompt_optimization.get('base_description', frame.description)

            # 添加风格标签
            style_tags = frame.prompt_optimization.get('style_tags', [])
            if style_tags:
                prompt += f", {', '.join(style_tags)}"

            # 添加质量标签
            quality_tags = frame.prompt_optimization.get('quality_tags', [])
            if quality_tags:
                prompt += f", {', '.join(quality_tags)}"

            # 创建图像生成任务
            task = ImageGenerationTask(
                prompt=prompt,
                negative_prompt="low quality, blurry, distorted",
                style="realistic",
                quality="high",
                aspect_ratio="16:9",  # 视频比例
                width=1920,
                height=1080,
                reference_image=frame.reference_product_image if hasattr(frame, 'reference_product_image') else None
            )
            image_tasks.append((frame, task))

        # 批量生成图像
        tasks_only = [task for _, task in image_tasks]
        request = ImageGenerationNodeRequest(
            tasks=tasks_only,
            batch_mode=True,
            generation_config={
                "enhance_prompts": True,
                "fallback_enabled": True
            }
        )

        print(f"  🔄 批量生成 {len(tasks_only)} 张图像...")
        response = await self.image_generation_node.process(request)

        if response.generated_images:
            print(f"  ✅ 成功生成 {len(response.generated_images)} 张图像")
            print(f"  💰 总成本: ${response.total_cost:.4f}")
            print(f"  ⏱️ 总时间: {response.total_time_ms/1000:.2f}秒")

            # 转换为keyframe格式
            for i, (frame, generated_image) in enumerate(zip([f for f, _ in image_tasks], response.generated_images)):
                keyframe = {
                    "frame_id": frame.frame_id,
                    "segment_id": frame.segment_id,
                    "image_path": generated_image.image_path,
                    "generation_mode": frame.generation_mode.value,
                    "is_reused": False,
                    "prompt": generated_image.prompt,
                    "revised_prompt": generated_image.revised_prompt
                }
                generated_keyframes.append(keyframe)
        else:
            raise Exception("图像生成失败")

        # 处理帧复用逻辑
        processed_keyframes = self._apply_frame_reuse_logic(generated_keyframes)

        return processed_keyframes

    def _apply_frame_reuse_logic(self, keyframes: List[Dict]) -> List[Dict]:
        """
        应用灵活的帧复用逻辑
        根据段落间的连续性判断是否复用首尾帧
        """

        print("  🔄 应用灵活帧复用逻辑...")

        # 按segment_id排序
        keyframes.sort(key=lambda x: (x['segment_id'], x['frame_id']))

        processed = []
        segments = {}

        # 按段落分组
        for kf in keyframes:
            seg_id = kf['segment_id']
            if seg_id not in segments:
                segments[seg_id] = []
            segments[seg_id].append(kf)

        # 处理每个段落
        for seg_id in sorted(segments.keys()):
            seg_frames = segments[seg_id]

            if len(seg_frames) >= 2:
                start_frame = seg_frames[0]
                end_frame = seg_frames[1]

                # 首帧：判断是否需要复用前一段的尾帧
                if seg_id > 0 and processed:
                    prev_end_frame = processed[-1]

                    # 判断连续性：是否应该复用
                    should_reuse = self._should_reuse_frame(
                        prev_segment_id=seg_id-1,
                        curr_segment_id=seg_id,
                        prev_end_frame=prev_end_frame,
                        curr_start_frame=start_frame
                    )

                    if should_reuse:
                        # 复用前一段的尾帧作为首帧
                        reused_start_frame = {
                            **prev_end_frame,
                            "frame_id": f"frame_{seg_id:03d}_start_reused",
                            "segment_id": seg_id,
                            "is_reused": True,
                            "source_frame_id": prev_end_frame["frame_id"],
                            "reuse_reason": "scene_continuity"
                        }
                        processed.append(reused_start_frame)
                        print(f"    ✅ 段{seg_id}复用前段尾帧（连续场景）")
                    else:
                        # 不复用，使用独立首帧
                        processed.append(start_frame)
                        print(f"    🎬 段{seg_id}使用独立首帧（场景切换）")
                else:
                    # 第一段使用原始首帧
                    processed.append(start_frame)

                # 尾帧
                processed.append(end_frame)

        reuse_count = sum(1 for kf in processed if kf.get('is_reused', False))
        independent_count = len(processed) - reuse_count

        print(f"  📊 帧统计: {independent_count}个独立帧 + {reuse_count}个复用帧 = {len(processed)}总帧")

        return processed

    def _should_reuse_frame(self,
                           prev_segment_id: int,
                           curr_segment_id: int,
                           prev_end_frame: Dict,
                           curr_start_frame: Dict) -> bool:
        """
        判断是否应该复用帧 - 核心连续性逻辑

        决策依据：
        1. 场景连续性（同场景、同物体）
        2. 镜头切换检测（场景变化、视角变化）
        3. 产品一致性要求
        """

        # 提取帧的描述信息
        prev_prompt = prev_end_frame.get('prompt', '').lower()
        curr_prompt = curr_start_frame.get('prompt', '').lower()

        # 判断因素1：场景元素相似性
        scene_similarity = self._calculate_scene_similarity(prev_prompt, curr_prompt)

        # 判断因素2：是否有明确的场景切换词汇
        has_scene_change = self._detect_scene_change(prev_prompt, curr_prompt)

        # 判断因素3：产品连续性
        has_product_continuity = self._check_product_continuity(prev_prompt, curr_prompt)

        # 决策逻辑
        if has_scene_change:
            # 明确场景切换 → 不复用
            return False
        elif has_product_continuity and scene_similarity > 0.6:
            # 产品连续 + 场景相似 → 复用
            return True
        elif scene_similarity > 0.8:
            # 高度相似 → 复用
            return True
        else:
            # 默认不复用（保守策略）
            return False

    def _calculate_scene_similarity(self, prompt1: str, prompt2: str) -> float:
        """计算场景相似度"""

        # 关键场景元素
        scene_keywords = [
            'office', 'studio', 'outdoor', 'indoor', 'kitchen', 'bedroom',
            'meeting room', 'gym', 'park', 'street', 'home', 'workplace'
        ]

        object_keywords = [
            'watch', 'phone', 'computer', 'desk', 'chair', 'product',
            'table', 'background', 'wall', 'window'
        ]

        person_keywords = [
            'person', 'man', 'woman', 'professional', 'athlete', 'user'
        ]

        all_keywords = scene_keywords + object_keywords + person_keywords

        # 统计共同关键词
        common_keywords = 0
        total_keywords = 0

        for keyword in all_keywords:
            in_prompt1 = keyword in prompt1
            in_prompt2 = keyword in prompt2

            if in_prompt1 or in_prompt2:
                total_keywords += 1
                if in_prompt1 and in_prompt2:
                    common_keywords += 1

        if total_keywords == 0:
            return 0.5  # 无法判断时返回中性值

        return common_keywords / total_keywords

    def _detect_scene_change(self, prompt1: str, prompt2: str) -> bool:
        """检测明确的场景切换"""

        # 场景切换关键词
        scene_change_indicators = [
            'cut to', 'switch to', 'transition to', 'move to',
            'location change', 'scene change', 'new scene',
            'different location', 'another place', 'elsewhere',
            '切换到', '转场到', '场景切换', '位置变化', '换到'
        ]

        # 场景类型词汇
        location_words = [
            'office', 'home', 'outdoor', 'indoor', 'studio', 'gym',
            'meeting room', 'kitchen', 'bedroom', 'street', 'park',
            '办公室', '家里', '户外', '室内', '工作室', '健身房'
        ]

        # 检查明确的切换指示
        for indicator in scene_change_indicators:
            if indicator in prompt1 or indicator in prompt2:
                return True

        # 检查完全不同的场景类型
        prompt1_locations = [word for word in location_words if word in prompt1]
        prompt2_locations = [word for word in location_words if word in prompt2]

        if prompt1_locations and prompt2_locations:
            # 如果两个提示词都有场景词且完全不同
            if not set(prompt1_locations).intersection(set(prompt2_locations)):
                return True

        return False

    def _check_product_continuity(self, prompt1: str, prompt2: str) -> bool:
        """检查产品连续性"""

        product_keywords = [
            'watch', 'smartwatch', 'product', 'device', 'gadget',
            '手表', '产品', '设备'
        ]

        # 检查两个提示词是否都包含产品
        has_product_1 = any(keyword in prompt1 for keyword in product_keywords)
        has_product_2 = any(keyword in prompt2 for keyword in product_keywords)

        return has_product_1 and has_product_2

    async def _generate_video_clips(self, keyframes: List[Dict]) -> List[str]:
        """第3步：生成5秒视频片段"""

        print("\n[第3步] 🎥 生成5秒视频片段...")

        # 使用千问处理器生成视频
        clips = await self.video_processor.process_storyboard_frames(
            keyframes,
            str(Path(self.config.get("work_dir", "/tmp")) / "video_clips")
        )

        print(f"  ✅ 生成了 {len(clips)} 个5秒视频片段")

        return clips

    async def _merge_final_video(self, clips: List[str], output_path: Optional[str]) -> str:
        """第4步：合并最终视频"""

        print("\n[第4步] 🔄 合并最终视频...")

        if not output_path:
            output_path = str(Path(self.config.get("work_dir", "/tmp")) / "final_video.mp4")

        final_video = await self.video_processor.merge_clips(clips, output_path)

        print(f"  ✅ 最终视频: {final_video}")

        return final_video

    def _parse_text_to_segments(self, text: str) -> List[Dict]:
        """解析文本为段落"""
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        return [{"description": line, "index": i} for i, line in enumerate(lines)]


# 使用示例函数
async def create_video_from_text(
    text_description: str,
    duration_seconds: int,
    product_info: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    便捷函数：从文本生成视频

    参数:
        text_description: 视频描述文本
        duration_seconds: 视频时长（秒）
        product_info: 产品信息（可选）
        config: 配置信息

    返回:
        生成结果
    """

    if not config:
        config = {
            "qwen_api_key": "your_qwen_api_key",
            "openai_api_key": "your_openai_api_key",
            "work_dir": "/tmp/video_generation"
        }

    # 创建编排器
    orchestrator = VideoStoryboardOrchestrator(config)

    # 创建请求
    request = VideoStoryboardRequest(
        text_description=text_description,
        duration_seconds=duration_seconds,
        product_info=product_info
    )

    # 处理请求
    result = await orchestrator.process_video_request(request)

    return result


# 演示用法
async def demo():
    """演示40秒产品视频生成"""

    text = """
    展示我们的新款智能手表SmartWatch Pro。
    开场：产品360度旋转展示，突出设计美感。
    聚焦表盘：展示AMOLED屏幕和精美界面。
    功能演示：运动追踪，心率监测，GPS定位。
    生活场景：商务会议中查看消息和日程。
    运动场景：跑步时监测运动数据。
    防水测试：展示IPX8防水能力。
    充电展示：无线充电底座和快速充电。
    包装展示：精美包装盒和配件全览。
    """

    product_info = {
        "name": "SmartWatch Pro",
        "constraints": [
            "产品颜色必须保持一致（黑色表带+银色表壳）",
            "必须清晰显示品牌logo",
            "避免出现竞品手表"
        ],
        "reference_images": ["smartwatch_product.jpg"],
        "attributes": {
            "color": "black/silver",
            "features": ["AMOLED", "heart_rate", "waterproof", "GPS"]
        }
    }

    config = {
        "qwen_api_key": "your_actual_key_here",
        "openai_api_key": "your_actual_key_here",
        "work_dir": "/tmp/smartwatch_video",
        "era_preference": "modern",
        "forbidden_elements": ["competitor", "low quality", "cheap"]
    }

    # 生成40秒视频（8个5秒片段）
    result = await create_video_from_text(
        text_description=text,
        duration_seconds=40,
        product_info=product_info,
        config=config
    )

    if result["success"]:
        print(f"\n🎉 视频生成成功！")
        print(f"📁 文件路径: {result['video_path']}")
        print(f"⏱️ 视频时长: {result['duration_seconds']}秒")
        print(f"🎬 片段数量: {result['segments_count']} (每段5秒)")
        print(f"🖼️ 关键帧数: {result['keyframes_count']}")
    else:
        print(f"❌ 生成失败: {result['error']}")

    return result


if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo())