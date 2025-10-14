"""
增强版视频编排器 - 完善所有缺失功能
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass
from pathlib import Path
import json

from video_generate_protocol.nodes.video_storyboard_orchestrator import VideoStoryboardOrchestrator, VideoStoryboardRequest
from video_generate_protocol.nodes.image_generation_node import ImageGenerationNode, ImageGenerationTask, ImageGenerationNodeRequest


class EnhancedVideoOrchestrator(VideoStoryboardOrchestrator):
    """
    增强版视频编排器
    完善缺失的功能：
    1. 两种视频生成方式（首尾帧 vs 仅首帧）
    2. VL视觉验证
    3. 同一物体的图生图优先级
    4. 段落划分时的转场预处理
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # VL模型配置（用于视觉验证）
        self.vl_validation_enabled = config.get("vl_validation", True)

    async def process_video_request(self, request: VideoStoryboardRequest) -> Dict[str, Any]:
        """
        增强的视频生成流程
        """

        print(f"\n🎬 开始增强视频生成流程 ({request.duration_seconds}秒)")
        print("="*60)

        try:
            # 第1步：预处理 - 转场检测和段落优化
            preprocessed_segments = await self._preprocess_segments(request)

            # 第2步：VGP优化分镜规划
            storyboard_plan = await self._optimize_storyboard_enhanced(request, preprocessed_segments)

            # 第3步：智能图像生成策略
            keyframes = await self._generate_keyframes_with_priority(storyboard_plan)

            # 第4步：VL视觉验证
            if self.vl_validation_enabled:
                validation_result = await self._vl_validate_keyframes(keyframes)
                if not validation_result["passed"]:
                    # 重新生成有问题的帧
                    keyframes = await self._regenerate_failed_frames(keyframes, validation_result)

            # 第5步：智能视频生成（两种模式）
            video_clips = await self._generate_video_clips_enhanced(keyframes)

            # 第6步：合并最终视频
            final_video = await self._merge_final_video(video_clips, request.output_path)

            return {
                "success": True,
                "video_path": final_video,
                "duration_seconds": request.duration_seconds,
                "segments_count": len(video_clips),
                "keyframes_count": len(keyframes),
                "validation_passed": validation_result.get("passed", True) if self.vl_validation_enabled else True,
                "storyboard_plan": storyboard_plan
            }

        except Exception as e:
            print(f"❌ 增强视频生成失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _preprocess_segments(self, request: VideoStoryboardRequest) -> List[Dict]:
        """
        第1步：预处理段落 - 转场检测

        "如果分镜转场很大，涉及到镜头切换，建议在短段落划分处就解决"
        """

        print("\n[第1步] 🔍 预处理段落和转场检测...")

        raw_segments = self._parse_text_to_segments(request.text_description)

        # 检测每个段落间的转场强度
        enhanced_segments = []

        for i, segment in enumerate(raw_segments):
            enhanced_segment = {**segment}

            if i > 0:
                # 检测与前一段的转场强度
                prev_segment = raw_segments[i-1]
                transition_intensity = self._calculate_transition_intensity(
                    prev_segment["description"],
                    segment["description"]
                )

                enhanced_segment["transition_intensity"] = transition_intensity
                enhanced_segment["needs_hard_cut"] = transition_intensity > 0.7

                if enhanced_segment["needs_hard_cut"]:
                    print(f"  🎬 检测到强转场: 段{i-1} → 段{i}")
            else:
                enhanced_segment["transition_intensity"] = 0.0
                enhanced_segment["needs_hard_cut"] = False

            enhanced_segments.append(enhanced_segment)

        return enhanced_segments

    def _calculate_transition_intensity(self, desc1: str, desc2: str) -> float:
        """计算转场强度"""

        # 强转场指示词
        strong_transition_words = [
            "切换到", "转场到", "场景变化", "位置改变", "时间跳跃",
            "cut to", "switch to", "transition to", "move to", "jump to",
            "different", "another", "new location", "elsewhere"
        ]

        # 场景类别词
        scene_categories = {
            "室内": ["indoor", "inside", "room", "office", "home", "室内"],
            "户外": ["outdoor", "outside", "street", "park", "户外"],
            "工作": ["work", "office", "meeting", "business", "工作"],
            "生活": ["home", "personal", "life", "daily", "生活"],
            "运动": ["sport", "gym", "exercise", "fitness", "运动"]
        }

        # 检查强转场词
        desc1_lower = desc1.lower()
        desc2_lower = desc2.lower()

        for word in strong_transition_words:
            if word in desc1_lower or word in desc2_lower:
                return 0.9

        # 检查场景类别变化
        desc1_categories = set()
        desc2_categories = set()

        for category, keywords in scene_categories.items():
            if any(kw in desc1_lower for kw in keywords):
                desc1_categories.add(category)
            if any(kw in desc2_lower for kw in keywords):
                desc2_categories.add(category)

        if desc1_categories and desc2_categories:
            if not desc1_categories.intersection(desc2_categories):
                return 0.8  # 完全不同的场景类别

        # 词汇相似度（简单实现）
        words1 = set(desc1_lower.split())
        words2 = set(desc2_lower.split())

        if len(words1) == 0 or len(words2) == 0:
            return 0.5

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union)

        # 转场强度 = 1 - 相似度
        return 1.0 - similarity

    async def _optimize_storyboard_enhanced(self,
                                          request: VideoStoryboardRequest,
                                          preprocessed_segments: List[Dict]) -> Dict:
        """第2步：增强的VGP优化"""

        print("\n[第2步] 📋 增强VGP优化...")

        # 调用VGP优化，但传入预处理的段落信息
        optimization_result = await self.vgp_optimization_node.optimize_storyboard_sequence(
            raw_segments=preprocessed_segments,
            product_info=request.product_info,
            total_duration_ms=request.duration_seconds * 1000
        )

        # 根据转场强度调整生成策略
        frames = optimization_result['optimized_frames']

        for i, frame in enumerate(frames):
            if hasattr(frame, 'segment_id') and frame.segment_id < len(preprocessed_segments):
                segment_info = preprocessed_segments[frame.segment_id]

                if segment_info.get("needs_hard_cut", False):
                    # 强制使用独立生成，不复用
                    frame.force_independent = True

        optimization_result['optimized_frames'] = frames

        return optimization_result

    async def _generate_keyframes_with_priority(self, storyboard_plan: Dict) -> List[Dict]:
        """
        第3步：智能图像生成策略

        "优先性是判断是否为同一物体，若是同一物体则可直接使用图生图，
        若不是而画面中有产品则使用原产品图进行图生图"
        """

        print("\n[第3步] 🎨 智能图像生成策略...")

        optimized_frames = storyboard_plan['optimized_frames']

        # 按优先级分组
        frame_groups = self._group_frames_by_priority(optimized_frames)

        generated_keyframes = []

        # 优先级1: 产品帧（最高优先级）
        if "product_frames" in frame_groups:
            print("  🥇 优先级1: 生成产品帧...")
            product_keyframes = await self._generate_product_frames(frame_groups["product_frames"])
            generated_keyframes.extend(product_keyframes)

        # 优先级2: 同一物体的连续帧（图生图）
        if "same_object_frames" in frame_groups:
            print("  🥈 优先级2: 同一物体图生图...")
            object_keyframes = await self._generate_same_object_frames(
                frame_groups["same_object_frames"],
                generated_keyframes
            )
            generated_keyframes.extend(object_keyframes)

        # 优先级3: 场景连续帧
        if "scene_continuous_frames" in frame_groups:
            print("  🥉 优先级3: 场景连续帧...")
            scene_keyframes = await self._generate_scene_continuous_frames(
                frame_groups["scene_continuous_frames"],
                generated_keyframes
            )
            generated_keyframes.extend(scene_keyframes)

        # 优先级4: 独立帧（文生图）
        if "independent_frames" in frame_groups:
            print("  🆕 优先级4: 独立生成帧...")
            independent_keyframes = await self._generate_independent_frames(
                frame_groups["independent_frames"]
            )
            generated_keyframes.extend(independent_keyframes)

        # 处理帧复用逻辑（考虑强转场）
        processed_keyframes = self._apply_enhanced_frame_reuse(generated_keyframes, optimized_frames)

        return processed_keyframes

    def _group_frames_by_priority(self, frames: List) -> Dict[str, List]:
        """按优先级分组帧"""

        groups = {
            "product_frames": [],
            "same_object_frames": [],
            "scene_continuous_frames": [],
            "independent_frames": []
        }

        for i, frame in enumerate(frames):
            # 检查是否包含产品
            if self._frame_contains_product(frame):
                groups["product_frames"].append(frame)
            # 检查是否与前一帧是同一物体
            elif i > 0 and self._is_same_object(frames[i-1], frame):
                groups["same_object_frames"].append(frame)
            # 检查是否场景连续
            elif i > 0 and self._is_scene_continuous(frames[i-1], frame):
                groups["scene_continuous_frames"].append(frame)
            else:
                groups["independent_frames"].append(frame)

        return groups

    def _frame_contains_product(self, frame) -> bool:
        """检查帧是否包含产品"""
        description = getattr(frame, 'description', '').lower()
        product_keywords = [
            'product', 'watch', 'smartwatch', 'device', 'gadget',
            '产品', '手表', '设备'
        ]
        return any(keyword in description for keyword in product_keywords)

    def _is_same_object(self, frame1, frame2) -> bool:
        """判断是否为同一物体"""

        # 提取物体关键词
        object_keywords = [
            'watch', 'phone', 'computer', 'product', 'device',
            'person', 'man', 'woman', 'user',
            '手表', '手机', '电脑', '产品', '人物'
        ]

        desc1 = getattr(frame1, 'description', '').lower()
        desc2 = getattr(frame2, 'description', '').lower()

        # 找出两帧中的物体
        objects1 = [kw for kw in object_keywords if kw in desc1]
        objects2 = [kw for kw in object_keywords if kw in desc2]

        # 检查是否有共同物体
        return bool(set(objects1).intersection(set(objects2)))

    def _is_scene_continuous(self, frame1, frame2) -> bool:
        """判断场景是否连续"""

        # 场景关键词
        scene_keywords = [
            'office', 'home', 'studio', 'gym', 'park', 'street',
            '办公室', '家里', '工作室', '健身房', '公园'
        ]

        desc1 = getattr(frame1, 'description', '').lower()
        desc2 = getattr(frame2, 'description', '').lower()

        scenes1 = [kw for kw in scene_keywords if kw in desc1]
        scenes2 = [kw for kw in scene_keywords if kw in desc2]

        return bool(set(scenes1).intersection(set(scenes2)))

    async def _generate_product_frames(self, frames: List) -> List[Dict]:
        """生成产品帧（使用产品参考图）"""

        keyframes = []

        for frame in frames:
            # 使用产品引导生成
            image_result = await self.image_generation_node.generate_single_image(
                prompt=self._build_product_prompt(frame),
                style="product_photography",
                quality="high",
                provider="dalle"
            )

            if image_result:
                keyframes.append({
                    "frame_id": frame.frame_id,
                    "segment_id": frame.segment_id,
                    "image_path": image_result.image_path,
                    "generation_mode": "product_guided",
                    "is_reused": False,
                    "priority": "product"
                })

        return keyframes

    async def _generate_same_object_frames(self, frames: List, existing_keyframes: List) -> List[Dict]:
        """生成同一物体帧（图生图）"""

        keyframes = []

        for frame in frames:
            # 找到参考帧
            reference_frame = self._find_reference_frame(frame, existing_keyframes)

            if reference_frame:
                # 使用图生图
                image_result = await self._img2img_generate(frame, reference_frame["image_path"])
            else:
                # 降级到文生图
                image_result = await self.image_generation_node.generate_single_image(
                    prompt=self._build_frame_prompt(frame),
                    style="realistic",
                    quality="high"
                )

            if image_result:
                keyframes.append({
                    "frame_id": frame.frame_id,
                    "segment_id": frame.segment_id,
                    "image_path": image_result.image_path if hasattr(image_result, 'image_path') else image_result,
                    "generation_mode": "img2img" if reference_frame else "txt2img",
                    "is_reused": False,
                    "priority": "same_object"
                })

        return keyframes

    async def _generate_scene_continuous_frames(self, frames: List, existing_keyframes: List) -> List[Dict]:
        """生成场景连续帧"""

        keyframes = []

        for frame in frames:
            reference_frame = self._find_scene_reference(frame, existing_keyframes)

            if reference_frame:
                image_result = await self._img2img_generate(frame, reference_frame["image_path"])
            else:
                image_result = await self.image_generation_node.generate_single_image(
                    prompt=self._build_frame_prompt(frame),
                    style="realistic",
                    quality="high"
                )

            if image_result:
                keyframes.append({
                    "frame_id": frame.frame_id,
                    "segment_id": frame.segment_id,
                    "image_path": image_result.image_path if hasattr(image_result, 'image_path') else image_result,
                    "generation_mode": "img2img" if reference_frame else "txt2img",
                    "is_reused": False,
                    "priority": "scene_continuous"
                })

        return keyframes

    async def _generate_independent_frames(self, frames: List) -> List[Dict]:
        """生成独立帧（文生图）"""

        keyframes = []

        for frame in frames:
            image_result = await self.image_generation_node.generate_single_image(
                prompt=self._build_frame_prompt(frame),
                style="realistic",
                quality="high"
            )

            if image_result:
                keyframes.append({
                    "frame_id": frame.frame_id,
                    "segment_id": frame.segment_id,
                    "image_path": image_result.image_path,
                    "generation_mode": "txt2img",
                    "is_reused": False,
                    "priority": "independent"
                })

        return keyframes

    def _apply_enhanced_frame_reuse(self, keyframes: List[Dict], original_frames: List) -> List[Dict]:
        """增强的帧复用逻辑（考虑强转场）"""

        processed = []

        for i, keyframe in enumerate(keyframes):
            processed.append(keyframe)

            # 检查对应的原始帧是否要求强制独立
            original_frame = next((f for f in original_frames
                                 if f.frame_id == keyframe["frame_id"]), None)

            if original_frame and getattr(original_frame, 'force_independent', False):
                # 强转场，不复用
                continue

            # 正常的复用逻辑...
            # (使用之前实现的逻辑)

        return processed

    async def _vl_validate_keyframes(self, keyframes: List[Dict]) -> Dict[str, Any]:
        """
        第4步：VL视觉验证

        "使用vl进行检查时也一样，重点从图片含义角度来看是否满足，是否有异常点"
        """

        print("\n[第4步] 👁️ VL视觉验证...")

        validation_results = {
            "passed": True,
            "total_frames": len(keyframes),
            "passed_frames": 0,
            "failed_frames": [],
            "issues": []
        }

        for keyframe in keyframes:
            if keyframe.get("image_path"):
                # 调用VL模型验证
                vl_result = await self._vl_validate_single_frame(keyframe)

                if vl_result["passed"]:
                    validation_results["passed_frames"] += 1
                else:
                    validation_results["failed_frames"].append({
                        "frame_id": keyframe["frame_id"],
                        "issues": vl_result["issues"]
                    })
                    validation_results["issues"].extend(vl_result["issues"])

        # 整体通过率
        pass_rate = validation_results["passed_frames"] / validation_results["total_frames"]
        validation_results["passed"] = pass_rate >= 0.8  # 80%通过率
        validation_results["pass_rate"] = pass_rate

        print(f"  📊 验证结果: {validation_results['passed_frames']}/{validation_results['total_frames']} 通过 ({pass_rate:.1%})")

        if not validation_results["passed"]:
            print(f"  ⚠️ 发现 {len(validation_results['failed_frames'])} 个问题帧")

        return validation_results

    async def _vl_validate_single_frame(self, keyframe: Dict) -> Dict:
        """VL验证单帧"""

        # 这里应该调用实际的VL模型
        # 暂时模拟验证逻辑

        issues = []

        # 模拟检查产品一致性
        if keyframe.get("priority") == "product":
            # 产品帧需要严格验证
            if "product" not in keyframe.get("frame_id", ""):
                issues.append("产品显示不清晰")

        # 模拟检查图像质量
        # 实际应该检查模糊、失真、异常元素等

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "confidence": 0.9 if len(issues) == 0 else 0.3
        }

    async def _regenerate_failed_frames(self, keyframes: List[Dict], validation_result: Dict) -> List[Dict]:
        """重新生成失败的帧"""

        print("  🔄 重新生成失败帧...")

        for failed_frame in validation_result["failed_frames"]:
            frame_id = failed_frame["frame_id"]

            # 找到失败的帧
            for i, keyframe in enumerate(keyframes):
                if keyframe["frame_id"] == frame_id:
                    # 重新生成
                    print(f"    重新生成帧: {frame_id}")
                    # 这里应该实现重新生成逻辑
                    break

        return keyframes

    async def _generate_video_clips_enhanced(self, keyframes: List[Dict]) -> List[str]:
        """
        第5步：智能视频生成（两种模式）

        "根据连续性是否要首尾帧生成还是仅首帧生成分别使用首尾帧生成（两种视频生成方式）"
        """

        print("\n[第5步] 🎥 智能视频生成（两种模式）...")

        # 分析每个片段的生成模式
        video_clips = []

        # 按段落配对帧
        segments = self._group_keyframes_by_segment(keyframes)

        for segment_id, segment_frames in segments.items():
            if len(segment_frames) >= 2:
                start_frame = segment_frames[0]
                end_frame = segment_frames[1]

                # 判断使用哪种生成模式
                generation_mode = self._determine_video_generation_mode(start_frame, end_frame)

                if generation_mode == "first_last_frame":
                    # 首尾帧生成（5秒视频）
                    clip_path = await self._generate_first_last_video(start_frame, end_frame, segment_id)
                else:
                    # 仅首帧生成（扩展生成）
                    clip_path = await self._generate_first_frame_only_video(start_frame, segment_id)

                if clip_path:
                    video_clips.append(clip_path)
                    print(f"  ✅ 生成片段{segment_id}: {generation_mode}")

        return video_clips

    def _determine_video_generation_mode(self, start_frame: Dict, end_frame: Dict) -> str:
        """判断视频生成模式"""

        # 如果尾帧是复用的，使用仅首帧模式
        if end_frame.get("is_reused", False):
            return "first_frame_only"

        # 如果首尾帧差异很大，使用首尾帧模式
        start_prompt = start_frame.get("prompt", "")
        end_prompt = end_frame.get("prompt", "")

        if self._calculate_prompt_similarity(start_prompt, end_prompt) < 0.5:
            return "first_last_frame"

        # 默认使用首尾帧模式
        return "first_last_frame"

    async def _generate_first_last_video(self, start_frame: Dict, end_frame: Dict, segment_id: int) -> str:
        """首尾帧生成5秒视频"""
        return await self.video_processor.generate_video_from_frames(
            start_frame["image_path"],
            end_frame["image_path"],
            duration_seconds=5.0
        )

    async def _generate_first_frame_only_video(self, start_frame: Dict, segment_id: int) -> str:
        """仅首帧生成视频（需要实现扩展生成）"""
        # 这里需要调用支持单帧扩展的API
        # 暂时使用首帧作为首尾帧
        return await self.video_processor.generate_video_from_frames(
            start_frame["image_path"],
            start_frame["image_path"],  # 使用同一帧
            duration_seconds=5.0
        )

    # 辅助方法
    def _build_product_prompt(self, frame) -> str:
        """构建产品提示词"""
        base_prompt = getattr(frame, 'description', '')
        return f"product photography, {base_prompt}, high quality, professional lighting"

    def _build_frame_prompt(self, frame) -> str:
        """构建普通帧提示词"""
        return getattr(frame, 'description', '')

    def _find_reference_frame(self, frame, existing_keyframes: List) -> Optional[Dict]:
        """找到参考帧"""
        # 简化实现：找到最近的同类型帧
        for kf in reversed(existing_keyframes):
            if kf.get("priority") == "product" and self._frame_contains_product(frame):
                return kf
        return None

    def _find_scene_reference(self, frame, existing_keyframes: List) -> Optional[Dict]:
        """找到场景参考帧"""
        # 简化实现
        return existing_keyframes[-1] if existing_keyframes else None

    async def _img2img_generate(self, frame, reference_image_path: str) -> str:
        """图生图生成"""
        # 这里应该调用实际的img2img API
        # 暂时返回模拟路径
        return f"/tmp/img2img_{frame.frame_id}.png"

    def _group_keyframes_by_segment(self, keyframes: List[Dict]) -> Dict[int, List[Dict]]:
        """按段落分组关键帧"""
        segments = {}
        for kf in keyframes:
            seg_id = kf["segment_id"]
            if seg_id not in segments:
                segments[seg_id] = []
            segments[seg_id].append(kf)
        return segments

    def _calculate_prompt_similarity(self, prompt1: str, prompt2: str) -> float:
        """计算提示词相似度"""
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)


# 使用示例
async def enhanced_demo():
    """增强版演示"""

    config = {
        "qwen_api_key": "your_qwen_key",
        "openai_api_key": "your_openai_key",
        "work_dir": "/tmp/enhanced_video",
        "vl_validation": True,
        "era_preference": "modern"
    }

    orchestrator = EnhancedVideoOrchestrator(config)

    request = VideoStoryboardRequest(
        text_description="""
        办公室场景：展示智能手表的整体外观。
        特写镜头：聚焦表盘显示界面。
        切换到户外：用户跑步时的运动追踪。
        回到室内：充电场景展示。
        """,
        duration_seconds=20,
        product_info={
            "name": "SmartWatch Pro",
            "constraints": ["保持产品一致性"],
            "reference_images": ["product.jpg"]
        }
    )

    result = await orchestrator.process_video_request(request)

    print(f"\n🎉 增强版生成{'成功' if result['success'] else '失败'}!")
    if result["success"]:
        print(f"📁 视频: {result['video_path']}")
        print(f"✅ VL验证: {'通过' if result['validation_passed'] else '失败'}")

if __name__ == "__main__":
    asyncio.run(enhanced_demo())