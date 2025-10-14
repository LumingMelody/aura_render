"""
连续块合并处理器 - 将连续的AI生成任务合并，提高一致性和效率
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .style_anchor_manager import StyleAnchorManager, GlobalElements
from .global_consistency_decision_tree import ConsistencyStrategy


class BlockType(Enum):
    """块类型"""
    TALKING_HEAD = "talking_head"  # 数字人口播
    AI_VIDEO = "ai_video"  # 纯AI视频
    USER_MATERIAL = "user_material"  # 用户素材
    LIBRARY_MATERIAL = "library_material"  # 素材库素材


@dataclass
class GenerationBlock:
    """生成块"""
    block_id: str
    original_indices: List[int]  # 原始分镜索引
    block_type: BlockType
    descriptions: List[str]
    scripts: List[str]  # 对于口播类型
    total_duration: float
    style_requirements: Dict[str, Any]
    context: Dict[str, Any]


@dataclass
class MergedGroup:
    """合并组"""
    group_id: str
    block_type: BlockType
    blocks: List[GenerationBlock]
    total_duration: float
    generation_strategy: str
    priority: int = 0


class ContinuousBlockMerger:
    """连续块合并处理器"""

    def __init__(self, style_manager: StyleAnchorManager):
        self.style_manager = style_manager

        # 合并配置
        self.max_talking_duration = 300  # 数字人最大连续时长（5分钟）
        self.max_ai_video_duration = 60   # AI视频最大连续时长（1分钟）
        self.min_merge_duration = 5       # 最小合并时长
        self.max_segments_per_group = 10  # 每组最大片段数

    def merge_continuous_blocks(
        self,
        shots: List[Dict],
        strategy: ConsistencyStrategy
    ) -> Tuple[List[MergedGroup], Dict[str, Any]]:
        """
        合并连续的生成块
        """
        # Step 1: 识别需要生成的块
        generation_blocks = self._identify_generation_blocks(shots, strategy)

        # Step 2: 按类型和连续性分组
        merged_groups = self._group_continuous_blocks(generation_blocks)

        # Step 3: 优化合并策略
        optimized_groups = self._optimize_merge_strategy(merged_groups)

        # Step 4: 生成合并统计
        merge_stats = self._generate_merge_statistics(shots, optimized_groups)

        return optimized_groups, merge_stats

    def _identify_generation_blocks(
        self,
        shots: List[Dict],
        strategy: ConsistencyStrategy
    ) -> List[GenerationBlock]:
        """识别需要AI生成的块"""
        generation_blocks = []

        for i, shot in enumerate(shots):
            # 根据策略判断是否需要生成
            needs_generation = self._should_generate_block(shot, strategy)

            if needs_generation:
                # 判断块类型
                block_type = self._determine_block_type(shot)

                # 创建生成块
                block = GenerationBlock(
                    block_id=f"gen_block_{i}",
                    original_indices=[i],
                    block_type=block_type,
                    descriptions=[shot.get("description", "")],
                    scripts=[shot.get("script", shot.get("description", ""))],
                    total_duration=shot.get("duration", 3.0),
                    style_requirements=self._extract_style_requirements(shot),
                    context=shot.get("context", {})
                )

                generation_blocks.append(block)

        return generation_blocks

    def _should_generate_block(self, shot: Dict, strategy: ConsistencyStrategy) -> bool:
        """判断是否需要AI生成"""
        if strategy == ConsistencyStrategy.ALL_AI_GENERATION:
            return True

        elif strategy == ConsistencyStrategy.AI_FILL_MISSING:
            # 只生成缺失的块
            return shot.get("asset_status") != "matched"

        elif strategy == ConsistencyStrategy.HYBRID_WITH_STYLE_TRANSFER:
            # 生成缺失的，已有的进行风格迁移
            return shot.get("asset_status") != "matched"

        elif strategy == ConsistencyStrategy.USE_LIBRARY_MATERIALS:
            # 基本不生成，除非是特殊要求
            return shot.get("force_generation", False)

        return False

    def _determine_block_type(self, shot: Dict) -> BlockType:
        """确定块类型"""
        # 检查是否为用户素材
        if shot.get("asset_status") == "matched":
            asset = shot.get("scheduled_asset", {})
            if asset.get("source") == "user_upload":
                return BlockType.USER_MATERIAL
            else:
                return BlockType.LIBRARY_MATERIAL

        # 检查是否为数字人口播
        description = shot.get("description", "").lower()
        script = shot.get("script", "")

        talking_keywords = [
            "讲解", "介绍", "说明", "演讲", "主持", "播报",
            "解说", "阐述", "叙述", "讲述", "表达", "传达",
            "说", "谈", "讲", "述"
        ]

        if any(keyword in description for keyword in talking_keywords) or script:
            return BlockType.TALKING_HEAD
        else:
            return BlockType.AI_VIDEO

    def _extract_style_requirements(self, shot: Dict) -> Dict[str, Any]:
        """提取风格要求"""
        return {
            "style_type": shot.get("style"),
            "mood": shot.get("mood"),
            "lighting": shot.get("lighting"),
            "camera_angle": shot.get("camera_angle"),
            "special_effects": shot.get("special_effects", [])
        }

    def _group_continuous_blocks(self, blocks: List[GenerationBlock]) -> List[MergedGroup]:
        """按连续性分组"""
        if not blocks:
            return []

        merged_groups = []
        current_group_blocks = [blocks[0]]
        group_counter = 0

        for i in range(1, len(blocks)):
            current_block = blocks[i]
            last_block = current_group_blocks[-1]

            # 检查是否可以合并
            if self._can_merge_blocks(last_block, current_block, current_group_blocks):
                current_group_blocks.append(current_block)
            else:
                # 创建当前组
                group = self._create_merged_group(current_group_blocks, group_counter)
                merged_groups.append(group)

                # 开始新组
                current_group_blocks = [current_block]
                group_counter += 1

        # 处理最后一组
        if current_group_blocks:
            group = self._create_merged_group(current_group_blocks, group_counter)
            merged_groups.append(group)

        return merged_groups

    def _can_merge_blocks(
        self,
        last_block: GenerationBlock,
        current_block: GenerationBlock,
        current_group: List[GenerationBlock]
    ) -> bool:
        """判断是否可以合并块"""

        # 类型必须相同
        if last_block.block_type != current_block.block_type:
            return False

        # 检查索引连续性
        last_max_index = max(last_block.original_indices)
        current_min_index = min(current_block.original_indices)
        if current_min_index != last_max_index + 1:
            return False

        # 检查时长限制
        total_duration = sum(block.total_duration for block in current_group) + current_block.total_duration

        if last_block.block_type == BlockType.TALKING_HEAD:
            if total_duration > self.max_talking_duration:
                return False
        elif last_block.block_type == BlockType.AI_VIDEO:
            if total_duration > self.max_ai_video_duration:
                return False

        # 检查组内块数限制
        if len(current_group) >= self.max_segments_per_group:
            return False

        # 检查风格一致性
        if not self._are_styles_compatible(last_block, current_block):
            return False

        return True

    def _are_styles_compatible(self, block1: GenerationBlock, block2: GenerationBlock) -> bool:
        """检查风格兼容性"""
        style1 = block1.style_requirements
        style2 = block2.style_requirements

        # 检查关键风格属性
        key_attributes = ["style_type", "mood", "lighting"]

        for attr in key_attributes:
            val1 = style1.get(attr)
            val2 = style2.get(attr)

            if val1 and val2 and val1 != val2:
                return False

        return True

    def _create_merged_group(self, blocks: List[GenerationBlock], group_id: int) -> MergedGroup:
        """创建合并组"""
        if not blocks:
            return None

        # 合并块信息
        all_indices = []
        all_descriptions = []
        all_scripts = []
        total_duration = 0

        for block in blocks:
            all_indices.extend(block.original_indices)
            all_descriptions.extend(block.descriptions)
            all_scripts.extend(block.scripts)
            total_duration += block.total_duration

        # 确定生成策略
        block_type = blocks[0].block_type
        generation_strategy = self._determine_generation_strategy(blocks)

        # 计算优先级
        priority = self._calculate_group_priority(blocks)

        return MergedGroup(
            group_id=f"merged_group_{group_id}",
            block_type=block_type,
            blocks=blocks,
            total_duration=total_duration,
            generation_strategy=generation_strategy,
            priority=priority
        )

    def _determine_generation_strategy(self, blocks: List[GenerationBlock]) -> str:
        """确定生成策略"""
        block_type = blocks[0].block_type

        if block_type == BlockType.TALKING_HEAD:
            if len(blocks) == 1:
                return "single_talking_head"
            else:
                return "continuous_talking_head"

        elif block_type == BlockType.AI_VIDEO:
            if len(blocks) == 1:
                return "single_ai_video"
            elif len(blocks) <= 3:
                return "short_sequence_ai_video"
            else:
                return "long_sequence_ai_video"

        return "default"

    def _calculate_group_priority(self, blocks: List[GenerationBlock]) -> int:
        """计算组优先级"""
        priority = 0

        # 数字人口播优先级更高
        if blocks[0].block_type == BlockType.TALKING_HEAD:
            priority += 100

        # 连续块数量影响优先级
        priority += len(blocks) * 10

        # 总时长影响优先级
        total_duration = sum(block.total_duration for block in blocks)
        priority += int(total_duration)

        return priority

    def _optimize_merge_strategy(self, groups: List[MergedGroup]) -> List[MergedGroup]:
        """优化合并策略"""
        optimized_groups = []

        for group in groups:
            # 检查是否需要进一步拆分
            if self._should_split_group(group):
                split_groups = self._split_large_group(group)
                optimized_groups.extend(split_groups)
            else:
                optimized_groups.append(group)

        # 按优先级排序
        optimized_groups.sort(key=lambda x: x.priority, reverse=True)

        return optimized_groups

    def _should_split_group(self, group: MergedGroup) -> bool:
        """判断是否需要拆分组"""
        # 数字人时长过长需要拆分
        if group.block_type == BlockType.TALKING_HEAD:
            if group.total_duration > self.max_talking_duration:
                return True

        # AI视频时长过长需要拆分
        elif group.block_type == BlockType.AI_VIDEO:
            if group.total_duration > self.max_ai_video_duration:
                return True

        # 块数量过多需要拆分
        if len(group.blocks) > self.max_segments_per_group:
            return True

        return False

    def _split_large_group(self, group: MergedGroup) -> List[MergedGroup]:
        """拆分大组"""
        if group.block_type == BlockType.TALKING_HEAD:
            return self._split_talking_head_group(group)
        elif group.block_type == BlockType.AI_VIDEO:
            return self._split_ai_video_group(group)
        else:
            return [group]

    def _split_talking_head_group(self, group: MergedGroup) -> List[MergedGroup]:
        """拆分数字人组"""
        # 按意思单元拆分（基于脚本内容）
        split_groups = []
        current_blocks = []
        current_duration = 0
        group_counter = 0

        for block in group.blocks:
            # 检查是否可以添加到当前组
            if (current_duration + block.total_duration <= self.max_talking_duration and
                len(current_blocks) < self.max_segments_per_group):
                current_blocks.append(block)
                current_duration += block.total_duration
            else:
                # 创建当前组
                if current_blocks:
                    split_group = self._create_merged_group(current_blocks, group_counter)
                    split_groups.append(split_group)

                # 开始新组
                current_blocks = [block]
                current_duration = block.total_duration
                group_counter += 1

        # 处理最后一组
        if current_blocks:
            split_group = self._create_merged_group(current_blocks, group_counter)
            split_groups.append(split_group)

        return split_groups

    def _split_ai_video_group(self, group: MergedGroup) -> List[MergedGroup]:
        """拆分AI视频组"""
        # 按场景切换点拆分
        split_groups = []
        current_blocks = []
        current_duration = 0
        group_counter = 0

        for block in group.blocks:
            # 检查是否可以添加到当前组
            if (current_duration + block.total_duration <= self.max_ai_video_duration and
                len(current_blocks) < self.max_segments_per_group):
                current_blocks.append(block)
                current_duration += block.total_duration
            else:
                # 创建当前组
                if current_blocks:
                    split_group = self._create_merged_group(current_blocks, group_counter)
                    split_groups.append(split_group)

                # 开始新组
                current_blocks = [block]
                current_duration = block.total_duration
                group_counter += 1

        # 处理最后一组
        if current_blocks:
            split_group = self._create_merged_group(current_blocks, group_counter)
            split_groups.append(split_group)

        return split_groups

    def _generate_merge_statistics(
        self,
        original_shots: List[Dict],
        merged_groups: List[MergedGroup]
    ) -> Dict[str, Any]:
        """生成合并统计信息"""

        # 基础统计
        total_shots = len(original_shots)
        total_groups = len(merged_groups)

        # 按类型统计
        type_stats = {}
        for group in merged_groups:
            block_type = group.block_type.value
            if block_type not in type_stats:
                type_stats[block_type] = {
                    "count": 0,
                    "total_duration": 0,
                    "total_blocks": 0
                }

            type_stats[block_type]["count"] += 1
            type_stats[block_type]["total_duration"] += group.total_duration
            type_stats[block_type]["total_blocks"] += len(group.blocks)

        # 计算合并效率
        original_generation_tasks = sum(1 for shot in original_shots
                                      if shot.get("needs_generation", False))

        merge_efficiency = 1.0 - (total_groups / max(original_generation_tasks, 1))

        # 估算性能提升
        estimated_time_saved = self._estimate_time_savings(merged_groups)
        estimated_consistency_improvement = self._estimate_consistency_improvement(merged_groups)

        return {
            "original_shots": total_shots,
            "merged_groups": total_groups,
            "merge_efficiency": merge_efficiency,
            "type_distribution": type_stats,
            "estimated_time_saved_minutes": estimated_time_saved,
            "estimated_consistency_improvement": estimated_consistency_improvement,
            "average_group_size": sum(len(g.blocks) for g in merged_groups) / total_groups if total_groups else 0,
            "largest_group_size": max(len(g.blocks) for g in merged_groups) if merged_groups else 0,
            "optimization_recommendations": self._generate_optimization_recommendations(merged_groups)
        }

    def _estimate_time_savings(self, groups: List[MergedGroup]) -> float:
        """估算时间节省"""
        total_saved = 0

        for group in groups:
            # 合并减少的初始化时间
            initialization_time_per_task = 2.0  # 分钟
            saved_initializations = len(group.blocks) - 1
            total_saved += saved_initializations * initialization_time_per_task

            # 连续生成的效率提升
            if group.block_type == BlockType.TALKING_HEAD:
                # 数字人连续生成可以节省模型加载时间
                total_saved += len(group.blocks) * 0.5

            elif group.block_type == BlockType.AI_VIDEO:
                # AI视频连续生成可以复用风格设置
                total_saved += len(group.blocks) * 1.0

        return total_saved

    def _estimate_consistency_improvement(self, groups: List[MergedGroup]) -> float:
        """估算一致性提升"""
        improvement_score = 0

        for group in groups:
            if len(group.blocks) > 1:
                # 连续块的一致性提升
                if group.block_type == BlockType.TALKING_HEAD:
                    # 同一数字人保证完全一致
                    improvement_score += 0.9 * len(group.blocks)
                elif group.block_type == BlockType.AI_VIDEO:
                    # 连续生成保证风格一致
                    improvement_score += 0.7 * len(group.blocks)

        # 标准化到0-1
        max_possible_improvement = sum(len(g.blocks) for g in groups)
        return improvement_score / max(max_possible_improvement, 1)

    def _generate_optimization_recommendations(self, groups: List[MergedGroup]) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 检查是否有过小的组
        small_groups = [g for g in groups if len(g.blocks) == 1 and g.total_duration < self.min_merge_duration]
        if small_groups:
            recommendations.append(f"发现{len(small_groups)}个过小的独立块，建议合并相邻块")

        # 检查是否有过大的组
        large_groups = [g for g in groups if len(g.blocks) > self.max_segments_per_group * 0.8]
        if large_groups:
            recommendations.append(f"发现{len(large_groups)}个较大的组，建议进一步细分")

        # 检查数字人使用情况
        talking_groups = [g for g in groups if g.block_type == BlockType.TALKING_HEAD]
        if len(talking_groups) > 1:
            recommendations.append("多个数字人组，建议检查是否可以使用统一数字人")

        # 检查AI视频碎片化
        ai_video_groups = [g for g in groups if g.block_type == BlockType.AI_VIDEO]
        if len(ai_video_groups) > 3:
            recommendations.append("AI视频较为碎片化，建议检查是否可以合并相似场景")

        return recommendations

    async def generate_merged_content(self, group: MergedGroup) -> Dict[str, Any]:
        """生成合并组的内容"""
        if group.block_type == BlockType.TALKING_HEAD:
            return await self._generate_talking_head_group(group)
        elif group.block_type == BlockType.AI_VIDEO:
            return await self._generate_ai_video_group(group)
        else:
            return {"error": "Unsupported block type for generation"}

    async def _generate_talking_head_group(self, group: MergedGroup) -> Dict[str, Any]:
        """生成数字人组内容"""
        # 合并所有脚本
        combined_script = self._merge_scripts(group)

        # 获取统一的数字人配置
        digital_human_config = self._get_unified_digital_human_config(group)

        # 生成连续的数字人视频
        return {
            "type": "talking_head_group",
            "group_id": group.group_id,
            "script": combined_script,
            "digital_human": digital_human_config,
            "total_duration": group.total_duration,
            "segments": self._create_segment_markers(group),
            "generation_strategy": group.generation_strategy
        }

    async def _generate_ai_video_group(self, group: MergedGroup) -> Dict[str, Any]:
        """生成AI视频组内容"""
        # 合并描述
        combined_description = self._merge_descriptions(group)

        # 获取统一风格
        unified_style = self._get_unified_style_config(group)

        # 生成连续的AI视频
        return {
            "type": "ai_video_group",
            "group_id": group.group_id,
            "description": combined_description,
            "style": unified_style,
            "total_duration": group.total_duration,
            "segments": self._create_segment_markers(group),
            "generation_strategy": group.generation_strategy
        }

    def _merge_scripts(self, group: MergedGroup) -> str:
        """合并脚本"""
        scripts = []
        for block in group.blocks:
            for script in block.scripts:
                if script.strip():
                    scripts.append(script.strip())

        # 使用适当的连接符
        return "。".join(scripts)

    def _merge_descriptions(self, group: MergedGroup) -> str:
        """合并描述"""
        descriptions = []
        for block in group.blocks:
            for desc in block.descriptions:
                if desc.strip():
                    descriptions.append(desc.strip())

        return " → ".join(descriptions)

    def _get_unified_digital_human_config(self, group: MergedGroup) -> Dict[str, Any]:
        """获取统一数字人配置"""
        # 从全局元素中获取或选择默认
        if self.style_manager.global_elements.main_character:
            return {
                "avatar_id": self.style_manager.global_elements.main_character.get("avatar_id", "default"),
                "voice_id": self.style_manager.global_elements.main_character.get("voice_id", "zh-CN-professional"),
                "style": "professional"
            }
        else:
            return {
                "avatar_id": "default_professional",
                "voice_id": "zh-CN-professional",
                "style": "professional"
            }

    def _get_unified_style_config(self, group: MergedGroup) -> Dict[str, Any]:
        """获取统一风格配置"""
        if self.style_manager.style_anchor:
            return {
                "style_type": self.style_manager.style_anchor.style_type.value,
                "style_prompt": self.style_manager.get_style_prompt(),
                "consistency_elements": self.style_manager.get_consistency_requirements()
            }
        else:
            return {"style_type": "realistic", "style_prompt": "realistic, natural lighting"}

    def _create_segment_markers(self, group: MergedGroup) -> List[Dict[str, Any]]:
        """创建片段标记"""
        segments = []
        current_time = 0

        for i, block in enumerate(group.blocks):
            segment = {
                "segment_id": f"{group.group_id}_seg_{i}",
                "start_time": current_time,
                "end_time": current_time + block.total_duration,
                "duration": block.total_duration,
                "original_indices": block.original_indices,
                "descriptions": block.descriptions
            }
            segments.append(segment)
            current_time += block.total_duration

        return segments