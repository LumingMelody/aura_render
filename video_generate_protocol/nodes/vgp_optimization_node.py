"""
VGP (Visual Generation Protocol) 优化节点
处理分镜生成的核心优化逻辑，确保产品一致性和视觉连续性
"""
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

class ElementType(Enum):
    """元素类型"""
    PRODUCT = "product"  # 产品（最高优先级）
    PERSON = "person"  # 人物
    SCENE = "scene"  # 场景
    OBJECT = "object"  # 物体
    BACKGROUND = "background"  # 背景

class GenerationMode(Enum):
    """生成模式"""
    TEXT_TO_IMAGE = "txt2img"  # 文生图
    IMAGE_TO_IMAGE = "img2img"  # 图生图
    PRODUCT_GUIDED = "product_guided"  # 产品引导生成
    REFERENCE_BASED = "reference_based"  # 基于参考图生成

@dataclass
class SceneElement:
    """场景元素"""
    element_id: str
    element_type: ElementType
    description: str
    importance: float  # 0-1, 重要性评分
    consistency_requirement: float  # 0-1, 一致性要求
    reference_images: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StoryboardFrame:
    """优化后的分镜帧"""
    frame_id: str
    segment_id: int
    timestamp_ms: int
    description: str
    elements: List[SceneElement]
    generation_mode: GenerationMode
    reference_frame_id: Optional[str] = None
    reference_product_image: Optional[str] = None
    prompt_optimization: Dict[str, Any] = field(default_factory=dict)
    continuity_score: float = 0.0
    needs_regeneration: bool = False

@dataclass
class VGPOptimizationConfig:
    """VGP优化配置"""
    # 产品保护设置
    product_protection_level: str = "maximum"  # maximum, high, medium
    product_consistency_threshold: float = 0.95

    # 时代与风格偏好
    era_preference: str = "modern"  # modern, classic, futuristic
    brand_style: Dict[str, Any] = field(default_factory=dict)

    # 雷点规避
    forbidden_elements: List[str] = field(default_factory=list)
    competitor_brands: List[str] = field(default_factory=list)

    # AI能力优化
    prefer_wide_shots: bool = True
    minimize_close_ups: bool = True
    enhance_lighting: bool = True
    stabilize_compositions: bool = True

    # 场景连续性
    continuity_threshold: float = 0.7
    max_scene_variations: int = 3

class VGPOptimizationNode:
    """VGP优化节点"""

    def __init__(self, config: VGPOptimizationConfig = None):
        self.config = config or VGPOptimizationConfig()
        self.element_registry: Dict[str, SceneElement] = {}
        self.frame_clusters: List[List[StoryboardFrame]] = []

    async def optimize_storyboard_sequence(self,
                                          raw_segments: List[Dict],
                                          product_info: Optional[Dict] = None,
                                          total_duration_ms: int = 30000) -> Dict[str, Any]:
        """优化分镜序列"""

        # 1. 合并并分析整体基调
        overall_context = await self._merge_and_analyze(raw_segments, product_info)

        # 2. 智能划分5秒段落
        optimized_segments = await self._smart_segmentation(
            overall_context,
            total_duration_ms
        )

        # 3. 识别并注册所有元素
        self._identify_elements(optimized_segments, product_info)

        # 4. 判断场景一致性和生成模式
        frames = self._determine_generation_modes(optimized_segments)

        # 5. 应用VGP优化规则
        frames = self._apply_vgp_optimizations(frames)

        # 6. 生成元素聚类（用于批量生成）
        self.frame_clusters = self._cluster_frames(frames)

        return {
            "overall_context": overall_context,
            "optimized_frames": frames,
            "frame_clusters": self.frame_clusters,
            "element_registry": self.element_registry,
            "optimization_report": self._generate_report(frames)
        }

    async def _merge_and_analyze(self,
                                raw_segments: List[Dict],
                                product_info: Optional[Dict]) -> Dict:
        """合并分镜并分析整体基调"""

        # 合并所有描述
        merged_description = " ".join([seg.get("description", "") for seg in raw_segments])

        # 分析整体基调
        context = {
            "merged_description": merged_description,
            "main_theme": self._extract_theme(merged_description),
            "emotional_tone": self._analyze_emotion(merged_description),
            "visual_style": self._determine_visual_style(merged_description, product_info),
            "key_moments": self._identify_key_moments(raw_segments),
            "product_focus": product_info is not None and len(product_info.get("constraints", [])) > 0
        }

        # 根据配置调整风格
        if self.config.era_preference:
            context["visual_style"]["era"] = self.config.era_preference

        if self.config.brand_style:
            context["visual_style"].update(self.config.brand_style)

        return context

    async def _smart_segmentation(self,
                                 overall_context: Dict,
                                 total_duration_ms: int) -> List[Dict]:
        """智能5秒段落划分"""

        num_segments = (total_duration_ms + 4999) // 5000
        segments = []

        # 提取关键时刻
        key_moments = overall_context.get("key_moments", [])

        # 智能分配内容到每个5秒段
        for i in range(num_segments):
            start_time = i * 5000
            end_time = min((i + 1) * 5000, total_duration_ms)

            # 确定该段的重点内容
            segment_focus = self._determine_segment_focus(
                i, num_segments, key_moments, overall_context
            )

            # 判断与前后段的关系
            continuity_type = self._determine_continuity(i, num_segments, segment_focus)

            segment = {
                "segment_id": i,
                "start_time_ms": start_time,
                "end_time_ms": end_time,
                "duration_ms": end_time - start_time,
                "focus": segment_focus,
                "continuity": continuity_type,
                "description": self._generate_segment_description(segment_focus, overall_context)
            }

            segments.append(segment)

        return segments

    def _determine_segment_focus(self,
                                segment_idx: int,
                                total_segments: int,
                                key_moments: List,
                                overall_context: Dict) -> Dict:
        """确定段落焦点"""

        # 根据位置确定默认焦点
        if segment_idx == 0:
            focus_type = "opening"
        elif segment_idx == total_segments - 1:
            focus_type = "closing"
        else:
            focus_type = "development"

        # 检查是否有特定的关键时刻
        for moment in key_moments:
            moment_segment = moment.get("segment_index", -1)
            if moment_segment == segment_idx:
                focus_type = moment.get("type", focus_type)
                break

        return {
            "type": focus_type,
            "elements": self._get_focus_elements(focus_type, overall_context),
            "camera_movement": self._suggest_camera_movement(focus_type)
        }

    def _determine_continuity(self,
                            segment_idx: int,
                            total_segments: int,
                            segment_focus: Dict) -> str:
        """判断连续性类型"""

        if segment_idx == 0:
            return "start"
        elif segment_idx == total_segments - 1:
            return "end"

        # 基于焦点类型判断
        focus_type = segment_focus.get("type", "")

        if focus_type in ["product_closeup", "detail_shot"]:
            return "continuous"  # 需要连续性
        elif focus_type in ["scene_change", "location_switch"]:
            return "cut"  # 允许切换
        else:
            return "continuous"  # 默认保持连续

    def _identify_elements(self,
                          segments: List[Dict],
                          product_info: Optional[Dict]):
        """识别并注册所有场景元素"""

        # 首先注册产品元素（最高优先级）
        if product_info:
            product_element = SceneElement(
                element_id=f"product_{hashlib.md5(str(product_info).encode()).hexdigest()[:8]}",
                element_type=ElementType.PRODUCT,
                description=product_info.get("name", "main product"),
                importance=1.0,
                consistency_requirement=self.config.product_consistency_threshold,
                reference_images=product_info.get("reference_images", []),
                attributes=product_info.get("attributes", {})
            )
            self.element_registry[product_element.element_id] = product_element

        # 分析每个段落中的元素
        for segment in segments:
            elements = self._extract_elements_from_description(segment.get("description", ""))

            for element_desc in elements:
                element_id = self._generate_element_id(element_desc)

                if element_id not in self.element_registry:
                    element = SceneElement(
                        element_id=element_id,
                        element_type=self._classify_element_type(element_desc),
                        description=element_desc,
                        importance=self._calculate_importance(element_desc),
                        consistency_requirement=self._determine_consistency_requirement(element_desc)
                    )
                    self.element_registry[element_id] = element

    def _determine_generation_modes(self, segments: List[Dict]) -> List[StoryboardFrame]:
        """确定每个帧的生成模式"""

        frames = []
        prev_frame = None

        for i, segment in enumerate(segments):
            # 生成首帧
            start_frame = self._create_frame(
                segment,
                position="start",
                prev_frame=prev_frame
            )

            # 生成尾帧
            end_frame = self._create_frame(
                segment,
                position="end",
                prev_frame=start_frame
            )

            frames.extend([start_frame, end_frame])
            prev_frame = end_frame

        return frames

    def _create_frame(self,
                     segment: Dict,
                     position: str,
                     prev_frame: Optional[StoryboardFrame]) -> StoryboardFrame:
        """创建优化的分镜帧"""

        frame_id = f"frame_{segment['segment_id']:03d}_{position}"

        # 提取该帧的元素
        frame_elements = self._get_frame_elements(segment, position)

        # 判断生成模式
        generation_mode, reference_info = self._decide_generation_mode(
            frame_elements,
            prev_frame,
            segment.get("continuity", "continuous")
        )

        # 优化提示词
        prompt_optimization = self._optimize_prompt(segment, position, frame_elements)

        frame = StoryboardFrame(
            frame_id=frame_id,
            segment_id=segment["segment_id"],
            timestamp_ms=segment["start_time_ms"] if position == "start" else segment["end_time_ms"],
            description=segment.get("description", ""),
            elements=frame_elements,
            generation_mode=generation_mode,
            reference_frame_id=reference_info.get("frame_id"),
            reference_product_image=reference_info.get("product_image"),
            prompt_optimization=prompt_optimization,
            continuity_score=self._calculate_continuity_score(frame_elements, prev_frame)
        )

        return frame

    def _decide_generation_mode(self,
                               frame_elements: List[SceneElement],
                               prev_frame: Optional[StoryboardFrame],
                               continuity_type: str) -> Tuple[GenerationMode, Dict]:
        """决定生成模式（核心逻辑）"""

        reference_info = {}

        # 优先级1：检查是否有产品
        product_elements = [e for e in frame_elements if e.element_type == ElementType.PRODUCT]

        if product_elements:
            # 有产品，必须使用产品引导生成
            product = product_elements[0]
            if product.reference_images:
                reference_info["product_image"] = product.reference_images[0]
                return GenerationMode.PRODUCT_GUIDED, reference_info

        # 优先级2：检查与前一帧的元素重叠
        if prev_frame and continuity_type == "continuous":
            shared_elements = self._find_shared_elements(frame_elements, prev_frame.elements)

            # 如果共享关键元素，使用图生图
            if self._has_key_shared_elements(shared_elements):
                reference_info["frame_id"] = prev_frame.frame_id
                return GenerationMode.IMAGE_TO_IMAGE, reference_info

        # 优先级3：检查是否有其他参考图
        for element in frame_elements:
            if element.reference_images:
                reference_info["reference_image"] = element.reference_images[0]
                return GenerationMode.REFERENCE_BASED, reference_info

        # 默认：文生图
        return GenerationMode.TEXT_TO_IMAGE, reference_info

    def _find_shared_elements(self,
                             elements1: List[SceneElement],
                             elements2: List[SceneElement]) -> List[SceneElement]:
        """查找共享元素"""

        shared = []
        element_ids1 = {e.element_id for e in elements1}

        for element in elements2:
            if element.element_id in element_ids1:
                shared.append(element)

        return shared

    def _has_key_shared_elements(self, shared_elements: List[SceneElement]) -> bool:
        """判断是否有关键共享元素"""

        if not shared_elements:
            return False

        # 检查是否有高重要性或高一致性要求的元素
        for element in shared_elements:
            if element.importance > 0.7 or element.consistency_requirement > 0.8:
                return True

        # 检查是否有人物或产品
        for element in shared_elements:
            if element.element_type in [ElementType.PRODUCT, ElementType.PERSON]:
                return True

        return False

    def _apply_vgp_optimizations(self, frames: List[StoryboardFrame]) -> List[StoryboardFrame]:
        """应用VGP优化规则"""

        for frame in frames:
            # 应用AI能力优化
            if self.config.prefer_wide_shots:
                frame.prompt_optimization["camera_angle"] = "wide shot"

            if self.config.enhance_lighting:
                frame.prompt_optimization["lighting"] = "professional studio lighting"

            if self.config.stabilize_compositions:
                frame.prompt_optimization["composition"] = "balanced, rule of thirds"

            # 检查并移除禁忌元素
            frame = self._remove_forbidden_elements(frame)

            # 检查产品一致性
            frame = self._ensure_product_consistency(frame)

        return frames

    def _remove_forbidden_elements(self, frame: StoryboardFrame) -> StoryboardFrame:
        """移除禁忌元素"""

        if not self.config.forbidden_elements:
            return frame

        # 检查描述中的禁忌词
        for forbidden in self.config.forbidden_elements:
            if forbidden.lower() in frame.description.lower():
                frame.needs_regeneration = True
                frame.prompt_optimization["avoid"] = self.config.forbidden_elements

        # 检查竞品
        for competitor in self.config.competitor_brands:
            if competitor.lower() in frame.description.lower():
                frame.needs_regeneration = True
                frame.prompt_optimization["exclude_brands"] = self.config.competitor_brands

        return frame

    def _ensure_product_consistency(self, frame: StoryboardFrame) -> StoryboardFrame:
        """确保产品一致性"""

        product_elements = [e for e in frame.elements if e.element_type == ElementType.PRODUCT]

        if product_elements:
            # 强制使用产品参考图
            if frame.generation_mode != GenerationMode.PRODUCT_GUIDED:
                frame.generation_mode = GenerationMode.PRODUCT_GUIDED
                frame.needs_regeneration = True

            # 增加产品描述的权重
            frame.prompt_optimization["emphasis"] = "product details and branding"

        return frame

    def _cluster_frames(self, frames: List[StoryboardFrame]) -> List[List[StoryboardFrame]]:
        """聚类帧（用于批量生成）"""

        clusters = []
        current_cluster = []

        for frame in frames:
            if not current_cluster:
                current_cluster.append(frame)
            else:
                # 判断是否应该加入当前聚类
                if self._should_cluster_together(current_cluster[-1], frame):
                    current_cluster.append(frame)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [frame]

        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def _should_cluster_together(self, frame1: StoryboardFrame, frame2: StoryboardFrame) -> bool:
        """判断两帧是否应该聚类在一起"""

        # 同一生成模式
        if frame1.generation_mode != frame2.generation_mode:
            return False

        # 连续性要求
        if abs(frame1.timestamp_ms - frame2.timestamp_ms) <= 5000:
            return True

        # 共享关键元素
        shared = self._find_shared_elements(frame1.elements, frame2.elements)
        if self._has_key_shared_elements(shared):
            return True

        return False

    def _optimize_prompt(self,
                        segment: Dict,
                        position: str,
                        elements: List[SceneElement]) -> Dict[str, Any]:
        """优化提示词"""

        optimization = {
            "base_description": segment.get("description", ""),
            "position_modifier": f"{position} of sequence",
            "style_tags": [],
            "quality_tags": ["high quality", "professional", "cinematic"],
            "technical_specs": {
                "resolution": "1920x1080",
                "aspect_ratio": "16:9"
            }
        }

        # 根据元素类型添加标签
        for element in elements:
            if element.element_type == ElementType.PRODUCT:
                optimization["style_tags"].append("product photography")
            elif element.element_type == ElementType.PERSON:
                optimization["style_tags"].append("portrait")

        # 添加时代风格
        if self.config.era_preference:
            optimization["style_tags"].append(f"{self.config.era_preference} style")

        return optimization

    def _calculate_continuity_score(self,
                                   current_elements: List[SceneElement],
                                   prev_frame: Optional[StoryboardFrame]) -> float:
        """计算连续性分数"""

        if not prev_frame:
            return 1.0

        shared = self._find_shared_elements(current_elements, prev_frame.elements)

        if not current_elements:
            return 0.0

        # 基于共享元素的比例和重要性
        shared_importance = sum(e.importance for e in shared)
        total_importance = sum(e.importance for e in current_elements)

        if total_importance == 0:
            return 0.0

        return shared_importance / total_importance

    def _generate_report(self, frames: List[StoryboardFrame]) -> Dict[str, Any]:
        """生成优化报告"""

        report = {
            "total_frames": len(frames),
            "generation_modes": {},
            "continuity_analysis": {},
            "optimization_suggestions": [],
            "warnings": []
        }

        # 统计生成模式
        for frame in frames:
            mode = frame.generation_mode.value
            report["generation_modes"][mode] = report["generation_modes"].get(mode, 0) + 1

        # 分析连续性
        continuity_scores = [f.continuity_score for f in frames]
        report["continuity_analysis"] = {
            "average_score": sum(continuity_scores) / len(continuity_scores) if continuity_scores else 0,
            "min_score": min(continuity_scores) if continuity_scores else 0,
            "max_score": max(continuity_scores) if continuity_scores else 0
        }

        # 生成建议
        if report["continuity_analysis"]["average_score"] < self.config.continuity_threshold:
            report["optimization_suggestions"].append(
                "Consider increasing scene continuity for smoother transitions"
            )

        # 检查警告
        frames_needing_regen = [f for f in frames if f.needs_regeneration]
        if frames_needing_regen:
            report["warnings"].append(
                f"{len(frames_needing_regen)} frames need regeneration due to policy violations"
            )

        return report

    # 辅助方法
    def _extract_theme(self, description: str) -> str:
        """提取主题"""
        # 简化实现，实际应调用NLP
        if "product" in description.lower():
            return "product_showcase"
        elif "story" in description.lower():
            return "narrative"
        else:
            return "general"

    def _analyze_emotion(self, description: str) -> str:
        """分析情感基调"""
        # 简化实现
        positive_words = ["happy", "exciting", "beautiful", "amazing"]
        negative_words = ["sad", "dark", "serious", "problem"]

        desc_lower = description.lower()
        positive_count = sum(1 for word in positive_words if word in desc_lower)
        negative_count = sum(1 for word in negative_words if word in desc_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _determine_visual_style(self, description: str, product_info: Optional[Dict]) -> Dict:
        """确定视觉风格"""
        style = {
            "primary": "modern",
            "secondary": "clean",
            "color_scheme": "vibrant"
        }

        if product_info:
            brand_style = product_info.get("brand_style", {})
            style.update(brand_style)

        return style

    def _identify_key_moments(self, segments: List[Dict]) -> List[Dict]:
        """识别关键时刻"""
        key_moments = []

        for i, segment in enumerate(segments):
            desc = segment.get("description", "").lower()

            if any(keyword in desc for keyword in ["reveal", "transform", "highlight"]):
                key_moments.append({
                    "segment_index": i,
                    "type": "highlight",
                    "description": segment.get("description", "")
                })

        return key_moments

    def _extract_elements_from_description(self, description: str) -> List[str]:
        """从描述中提取元素"""
        # 简化实现，实际应使用NER
        elements = []

        # 简单的关键词提取
        keywords = ["product", "person", "background", "logo", "text"]
        for keyword in keywords:
            if keyword in description.lower():
                elements.append(keyword)

        return elements

    def _generate_element_id(self, element_desc: str) -> str:
        """生成元素ID"""
        return f"elem_{hashlib.md5(element_desc.encode()).hexdigest()[:8]}"

    def _classify_element_type(self, element_desc: str) -> ElementType:
        """分类元素类型"""
        desc_lower = element_desc.lower()

        if "product" in desc_lower:
            return ElementType.PRODUCT
        elif "person" in desc_lower or "people" in desc_lower:
            return ElementType.PERSON
        elif "scene" in desc_lower or "location" in desc_lower:
            return ElementType.SCENE
        elif "background" in desc_lower:
            return ElementType.BACKGROUND
        else:
            return ElementType.OBJECT

    def _calculate_importance(self, element_desc: str) -> float:
        """计算元素重要性"""
        # 简化实现
        if "product" in element_desc.lower():
            return 1.0
        elif "main" in element_desc.lower() or "key" in element_desc.lower():
            return 0.8
        else:
            return 0.5

    def _determine_consistency_requirement(self, element_desc: str) -> float:
        """确定一致性要求"""
        # 产品需要最高一致性
        if "product" in element_desc.lower():
            return 0.95
        elif "person" in element_desc.lower():
            return 0.85
        elif "logo" in element_desc.lower():
            return 0.9
        else:
            return 0.6

    def _get_focus_elements(self, focus_type: str, overall_context: Dict) -> List[str]:
        """获取焦点元素"""
        if focus_type == "opening":
            return ["establishing shot", "brand introduction"]
        elif focus_type == "closing":
            return ["final message", "call to action"]
        else:
            return ["main content", "product details"]

    def _suggest_camera_movement(self, focus_type: str) -> str:
        """建议镜头运动"""
        movements = {
            "opening": "slow zoom in",
            "closing": "slow zoom out",
            "product_closeup": "steady cam",
            "detail_shot": "macro focus",
            "development": "smooth pan"
        }
        return movements.get(focus_type, "static")

    def _get_frame_elements(self, segment: Dict, position: str) -> List[SceneElement]:
        """获取帧的元素"""
        # 从注册表中获取相关元素
        relevant_elements = []

        description = segment.get("description", "").lower()

        for element_id, element in self.element_registry.items():
            if element.description.lower() in description:
                relevant_elements.append(element)

        return relevant_elements

    def _generate_segment_description(self, focus: Dict, overall_context: Dict) -> str:
        """生成段落描述"""
        focus_type = focus.get("type", "general")
        elements = focus.get("elements", [])

        description = f"{focus_type} segment featuring {', '.join(elements)}"

        # 添加风格描述
        visual_style = overall_context.get("visual_style", {})
        if visual_style:
            description += f" in {visual_style.get('primary', 'modern')} style"

        return description