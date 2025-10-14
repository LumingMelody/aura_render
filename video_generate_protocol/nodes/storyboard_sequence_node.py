"""
分镜序列生成节点 - 处理自然语言到视频的完整流程
支持千问首尾帧生成5秒视频的连续拼接
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import hashlib
import tempfile

class TransitionType(Enum):
    """转场类型"""
    CONTINUOUS = "continuous"  # 连续镜头，需要帧复用
    CUT = "cut"  # 硬切，允许场景切换
    FADE = "fade"  # 淡入淡出
    DISSOLVE = "dissolve"  # 溶解

class SceneConsistency(Enum):
    """场景一致性级别"""
    HIGH = "high"  # 同一场景，同一物体/人物，需要图生图
    MEDIUM = "medium"  # 同一场景，不同角度或细节
    LOW = "low"  # 不同场景，可以独立生成

@dataclass
class StoryboardSegment:
    """分镜段落"""
    segment_id: int
    description: str  # 段落描述
    duration_ms: int  # 段落时长（毫秒）
    scene_elements: Dict[str, Any]  # 场景元素
    camera_movement: Optional[str] = None  # 镜头运动
    transition_type: TransitionType = TransitionType.CONTINUOUS
    consistency_level: SceneConsistency = SceneConsistency.HIGH
    product_constraints: List[str] = field(default_factory=list)  # 产品约束
    style_preferences: Dict[str, Any] = field(default_factory=dict)  # 风格偏好

@dataclass
class KeyFrame:
    """关键帧"""
    frame_id: str
    segment_id: int
    position: str  # "start" or "end"
    prompt: str  # 生成提示词
    refined_prompt: str  # 细化后的提示词
    image_path: Optional[str] = None
    is_reused: bool = False  # 是否是复用帧
    source_frame_id: Optional[str] = None  # 复用源帧ID
    generation_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VideoClip:
    """视频片段"""
    clip_id: str
    segment_id: int
    start_frame: KeyFrame
    end_frame: KeyFrame
    duration_ms: int
    video_path: Optional[str] = None
    generation_status: str = "pending"

class StoryboardSequenceNode:
    """分镜序列生成节点"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.clip_duration_ms = 5000  # 千问API限制：5秒
        self.temp_dir = Path(tempfile.mkdtemp(prefix="storyboard_"))

        # 默认配置
        self.default_style = {
            "visual_style": "cinematic",
            "color_palette": "vibrant",
            "aspect_ratio": "16:9",
            "resolution": "1920x1080"
        }

        self.optimization_rules = {
            "prefer_wide_shots": True,  # AI更擅长大场景
            "minimize_close_ups": True,
            "enhance_lighting": True,
            "stabilize_camera": True
        }

    async def process_natural_language(self,
                                      text: str,
                                      total_duration_ms: int,
                                      product_info: Optional[Dict] = None) -> Dict[str, Any]:
        """处理自然语言输入，生成完整的视频序列"""

        # 1. 解析并生成分镜段落
        segments = await self._parse_and_segment(text, total_duration_ms, product_info)

        # 2. 分析场景一致性
        segments = await self._analyze_consistency(segments)

        # 3. 生成关键帧序列
        keyframes = await self._generate_keyframe_sequence(segments)

        # 4. 创建视频片段规划
        video_clips = self._plan_video_clips(segments, keyframes)

        # 5. 执行图像生成
        await self._generate_keyframe_images(keyframes)

        # 6. 生成视频片段
        await self._generate_video_clips(video_clips)

        return {
            "segments": segments,
            "keyframes": keyframes,
            "video_clips": video_clips,
            "total_duration_ms": total_duration_ms,
            "output_path": str(self.temp_dir)
        }

    async def _parse_and_segment(self,
                                 text: str,
                                 total_duration_ms: int,
                                 product_info: Optional[Dict]) -> List[StoryboardSegment]:
        """解析文本并划分成5秒段落"""

        # 计算需要的段落数
        num_segments = (total_duration_ms + self.clip_duration_ms - 1) // self.clip_duration_ms

        segments = []

        # 使用LLM分析整体基调和场景
        overall_analysis = await self._analyze_overall_tone(text, product_info)

        # 智能划分段落
        segment_descriptions = await self._smart_segment_division(
            text, num_segments, overall_analysis
        )

        for i, desc in enumerate(segment_descriptions):
            # 判断转场类型
            if i > 0:
                transition = self._determine_transition(
                    segment_descriptions[i-1], desc
                )
            else:
                transition = TransitionType.CONTINUOUS

            segment = StoryboardSegment(
                segment_id=i,
                description=desc["content"],
                duration_ms=min(self.clip_duration_ms,
                              total_duration_ms - i * self.clip_duration_ms),
                scene_elements=desc.get("elements", {}),
                camera_movement=desc.get("camera", None),
                transition_type=transition,
                product_constraints=product_info.get("constraints", []) if product_info else [],
                style_preferences=overall_analysis.get("style", self.default_style)
            )
            segments.append(segment)

        return segments

    async def _analyze_overall_tone(self, text: str, product_info: Optional[Dict]) -> Dict:
        """分析整体基调"""
        # 这里应该调用LLM来分析
        # 模拟返回
        return {
            "mood": "professional",
            "pace": "moderate",
            "style": {
                "visual_style": "modern",
                "color_palette": "corporate",
                "lighting": "bright"
            },
            "key_elements": ["product", "benefits", "usage"],
            "avoid_elements": ["competitors", "negative_scenarios"]
        }

    async def _smart_segment_division(self,
                                     text: str,
                                     num_segments: int,
                                     overall_analysis: Dict) -> List[Dict]:
        """智能段落划分"""
        # 这里应该调用LLM进行智能划分
        # 模拟返回段落描述
        segments = []
        for i in range(num_segments):
            segments.append({
                "content": f"Segment {i+1} content based on: {text[:50]}...",
                "elements": {
                    "subject": "product",
                    "background": "studio",
                    "action": "showcase"
                },
                "camera": "pan" if i % 2 == 0 else "static"
            })
        return segments

    async def _analyze_consistency(self, segments: List[StoryboardSegment]) -> List[StoryboardSegment]:
        """分析场景一致性"""
        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]

            # 判断场景一致性
            consistency = self._calculate_consistency(current, next_seg)
            next_seg.consistency_level = consistency

        return segments

    def _calculate_consistency(self, seg1: StoryboardSegment, seg2: StoryboardSegment) -> SceneConsistency:
        """计算两个段落的一致性"""
        # 简化的一致性判断逻辑
        if seg1.scene_elements.get("subject") == seg2.scene_elements.get("subject"):
            if seg1.scene_elements.get("background") == seg2.scene_elements.get("background"):
                return SceneConsistency.HIGH
            return SceneConsistency.MEDIUM
        return SceneConsistency.LOW

    def _determine_transition(self, prev_desc: Dict, curr_desc: Dict) -> TransitionType:
        """判断转场类型"""
        # 如果场景变化很大，使用切换
        if prev_desc.get("elements", {}).get("background") != curr_desc.get("elements", {}).get("background"):
            return TransitionType.CUT
        # 否则保持连续
        return TransitionType.CONTINUOUS

    async def _generate_keyframe_sequence(self, segments: List[StoryboardSegment]) -> List[KeyFrame]:
        """生成关键帧序列（处理帧复用）"""
        keyframes = []
        frame_counter = 0

        for i, segment in enumerate(segments):
            # 生成首帧
            if i == 0 or segment.transition_type == TransitionType.CUT:
                # 需要新生成首帧
                start_frame = KeyFrame(
                    frame_id=f"frame_{frame_counter:04d}",
                    segment_id=segment.segment_id,
                    position="start",
                    prompt=self._generate_prompt(segment, "start"),
                    refined_prompt=await self._refine_prompt(segment, "start"),
                    generation_params=self._get_generation_params(segment)
                )
                frame_counter += 1
            else:
                # 复用前一段的尾帧作为首帧
                prev_end_frame = keyframes[-1]
                start_frame = KeyFrame(
                    frame_id=prev_end_frame.frame_id,  # 使用相同ID表示复用
                    segment_id=segment.segment_id,
                    position="start",
                    prompt=prev_end_frame.prompt,
                    refined_prompt=prev_end_frame.refined_prompt,
                    is_reused=True,
                    source_frame_id=prev_end_frame.frame_id
                )

            keyframes.append(start_frame)

            # 生成尾帧
            end_frame = KeyFrame(
                frame_id=f"frame_{frame_counter:04d}",
                segment_id=segment.segment_id,
                position="end",
                prompt=self._generate_prompt(segment, "end"),
                refined_prompt=await self._refine_prompt(segment, "end"),
                generation_params=self._get_generation_params(segment)
            )
            frame_counter += 1
            keyframes.append(end_frame)

        return keyframes

    def _generate_prompt(self, segment: StoryboardSegment, position: str) -> str:
        """生成基础提示词"""
        base_prompt = segment.description

        # 添加位置特定的描述
        if position == "start":
            prompt = f"Beginning of scene: {base_prompt}"
        else:
            prompt = f"End of scene: {base_prompt}"

        # 添加产品约束
        if segment.product_constraints:
            prompt += f", featuring {', '.join(segment.product_constraints)}"

        # 添加风格要求
        style = segment.style_preferences
        if style:
            prompt += f", {style.get('visual_style', '')} style"

        return prompt

    async def _refine_prompt(self, segment: StoryboardSegment, position: str) -> str:
        """细化提示词（考虑AI能力限制）"""
        base_prompt = self._generate_prompt(segment, position)

        # 应用优化规则
        if self.optimization_rules["prefer_wide_shots"]:
            base_prompt = base_prompt.replace("close-up", "medium shot")
            base_prompt = base_prompt.replace("detail shot", "wide angle shot")

        # 增强场景描述
        refined = f"{base_prompt}, professional lighting, high quality, detailed"

        return refined

    def _get_generation_params(self, segment: StoryboardSegment) -> Dict:
        """获取图像生成参数"""
        return {
            "style": segment.style_preferences.get("visual_style", "realistic"),
            "quality": "high",
            "aspect_ratio": segment.style_preferences.get("aspect_ratio", "16:9"),
            "seed": None  # 可以设置种子以保持一致性
        }

    def _plan_video_clips(self,
                         segments: List[StoryboardSegment],
                         keyframes: List[KeyFrame]) -> List[VideoClip]:
        """规划视频片段"""
        video_clips = []

        # 将关键帧配对成视频片段
        frame_pairs = []
        for i in range(0, len(keyframes) - 1, 2):
            if i + 1 < len(keyframes):
                frame_pairs.append((keyframes[i], keyframes[i + 1]))

        for i, (start_frame, end_frame) in enumerate(frame_pairs):
            clip = VideoClip(
                clip_id=f"clip_{i:04d}",
                segment_id=segments[i].segment_id,
                start_frame=start_frame,
                end_frame=end_frame,
                duration_ms=segments[i].duration_ms
            )
            video_clips.append(clip)

        return video_clips

    async def _generate_keyframe_images(self, keyframes: List[KeyFrame]):
        """生成关键帧图像"""
        # 过滤出需要生成的帧（非复用帧）
        frames_to_generate = [kf for kf in keyframes if not kf.is_reused]

        # 检查是否需要图生图
        for i, frame in enumerate(frames_to_generate):
            segment_id = frame.segment_id

            # 查找是否有产品图或参考图
            if self._has_product_reference(segment_id):
                # 使用图生图保持产品一致性
                frame.generation_params["mode"] = "img2img"
                frame.generation_params["reference_image"] = self._get_product_reference(segment_id)
            else:
                # 检查是否需要基于前一帧生成（保持场景连续性）
                if i > 0 and self._should_use_previous_frame(frames_to_generate[i-1], frame):
                    frame.generation_params["mode"] = "img2img"
                    frame.generation_params["reference_image"] = frames_to_generate[i-1].image_path
                else:
                    frame.generation_params["mode"] = "txt2img"

        # 批量生成图像
        for frame in frames_to_generate:
            # 这里应该调用实际的图像生成API
            frame.image_path = str(self.temp_dir / f"{frame.frame_id}.png")
            print(f"Generating frame {frame.frame_id}: {frame.refined_prompt[:50]}...")

        # 更新复用帧的图像路径
        for frame in keyframes:
            if frame.is_reused and frame.source_frame_id:
                source_frame = next(f for f in keyframes if f.frame_id == frame.source_frame_id)
                frame.image_path = source_frame.image_path

    def _has_product_reference(self, segment_id: int) -> bool:
        """检查是否有产品参考图"""
        # 实际实现中应该检查产品库
        return False

    def _get_product_reference(self, segment_id: int) -> str:
        """获取产品参考图路径"""
        # 实际实现中应该返回产品图路径
        return ""

    def _should_use_previous_frame(self, prev_frame: KeyFrame, curr_frame: KeyFrame) -> bool:
        """判断是否应该基于前一帧生成"""
        # 如果是同一段落内的帧，通常需要保持连续性
        return prev_frame.segment_id == curr_frame.segment_id

    async def _generate_video_clips(self, video_clips: List[VideoClip]):
        """生成视频片段（调用千问API）"""
        for clip in video_clips:
            try:
                # 调用千问首尾帧生成视频API
                video_path = await self._call_qwen_video_api(
                    start_image=clip.start_frame.image_path,
                    end_image=clip.end_frame.image_path,
                    duration_seconds=clip.duration_ms / 1000
                )

                clip.video_path = video_path
                clip.generation_status = "completed"
                print(f"Generated video clip {clip.clip_id}")

            except Exception as e:
                print(f"Failed to generate clip {clip.clip_id}: {e}")
                clip.generation_status = "failed"

    async def _call_qwen_video_api(self,
                                  start_image: str,
                                  end_image: str,
                                  duration_seconds: float) -> str:
        """调用千问视频生成API"""
        # 这里应该实现实际的API调用
        # 参考：https://help.aliyun.com/zh/model-studio/image-to-video-by-first-and-last-frame-api-reference

        output_path = str(self.temp_dir / f"video_{hashlib.md5(f'{start_image}{end_image}'.encode()).hexdigest()}.mp4")

        # 模拟API调用
        print(f"Calling Qwen API: {start_image} -> {end_image} ({duration_seconds}s)")

        return output_path

    async def merge_video_clips(self, video_clips: List[VideoClip], output_path: str) -> str:
        """合并视频片段"""
        # 获取所有成功的视频路径
        valid_clips = [c for c in video_clips if c.generation_status == "completed" and c.video_path]

        if not valid_clips:
            raise ValueError("No valid video clips to merge")

        # 使用ffmpeg合并视频
        clip_paths = [c.video_path for c in valid_clips]

        # 这里应该调用ffmpeg进行合并
        print(f"Merging {len(clip_paths)} clips into {output_path}")

        return output_path

    async def validate_with_vl(self, keyframes: List[KeyFrame]) -> Dict[str, Any]:
        """使用视觉语言模型验证分镜"""
        validation_results = {
            "overall_score": 0.0,
            "frame_scores": [],
            "issues": [],
            "suggestions": []
        }

        for frame in keyframes:
            if frame.image_path and not frame.is_reused:
                # 验证单帧
                score, issues = await self._validate_single_frame(frame)
                validation_results["frame_scores"].append({
                    "frame_id": frame.frame_id,
                    "score": score,
                    "issues": issues
                })

                if issues:
                    validation_results["issues"].extend(issues)

        # 计算整体分数
        if validation_results["frame_scores"]:
            total_score = sum(f["score"] for f in validation_results["frame_scores"])
            validation_results["overall_score"] = total_score / len(validation_results["frame_scores"])

        # 生成改进建议
        if validation_results["overall_score"] < 0.7:
            validation_results["suggestions"].append("Consider regenerating low-scoring frames")

        return validation_results

    async def _validate_single_frame(self, frame: KeyFrame) -> Tuple[float, List[str]]:
        """验证单个关键帧"""
        # 这里应该调用VL模型进行验证
        # 检查：
        # 1. 是否符合提示词描述
        # 2. 是否有异常元素
        # 3. 产品是否正确显示
        # 4. 整体质量评分

        # 模拟返回
        score = 0.85
        issues = []

        if "product" in frame.prompt and score < 0.8:
            issues.append(f"Product visibility issue in frame {frame.frame_id}")

        return score, issues


# 使用示例
async def demo():
    """演示用法"""
    node = StoryboardSequenceNode()

    # 用户只提供自然语言描述
    user_input = """
    展示我们的新款智能手表，首先是产品的整体外观，
    然后展示表盘的细节和功能，接着演示运动追踪功能，
    最后展示充电和包装。
    """

    # 产品信息（可选）
    product_info = {
        "name": "SmartWatch Pro",
        "constraints": ["必须显示品牌logo", "保持产品颜色一致"],
        "reference_images": ["product_ref.jpg"]
    }

    # 生成30秒视频（6个5秒片段）
    result = await node.process_natural_language(
        text=user_input,
        total_duration_ms=30000,
        product_info=product_info
    )

    # 合并视频
    final_video = await node.merge_video_clips(
        result["video_clips"],
        "output/final_video.mp4"
    )

    # 验证结果
    validation = await node.validate_with_vl(result["keyframes"])

    print(f"Video generated: {final_video}")
    print(f"Validation score: {validation['overall_score']}")

    return result

if __name__ == "__main__":
    asyncio.run(demo())