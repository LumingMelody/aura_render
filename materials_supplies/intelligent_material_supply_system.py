"""
智能素材供给系统 - 主控制器
整合所有组件，提供统一的素材供给服务
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import time

from .style_anchor_manager import StyleAnchorManager, StyleVector
from .three_level_supply_strategy import ThreeLevelSupplyStrategy, SupplyRequest, MaterialCandidate
from .smart_clip_selector import SmartClipSelector, ClipRequest
from .global_consistency_decision_tree import (
    GlobalConsistencyDecisionTree,
    ConsistencyStrategy,
    DecisionContext
)
from .continuous_block_merger import ContinuousBlockMerger, MergedGroup
from .digital_human_manager import DigitalHumanManager, SelectionCriteria
from .vl_cost_controller import VLCostController, CostBudget, VLCallRequest, VLCallPriority


@dataclass
class MaterialSupplyRequest:
    """素材供给总请求"""
    shots: List[Dict[str, Any]]
    project_config: Dict[str, Any]
    user_preferences: Dict[str, Any] = None
    budget_constraints: Dict[str, Any] = None


@dataclass
class SupplyResult:
    """供给结果"""
    final_materials: List[Dict[str, Any]]
    generation_tasks: List[MergedGroup]
    style_anchor: StyleVector
    decision_strategy: ConsistencyStrategy
    cost_report: Dict[str, Any]
    processing_time: float
    quality_metrics: Dict[str, Any]


class IntelligentMaterialSupplySystem:
    """智能素材供给系统主控制器"""

    def __init__(self, budget: Optional[CostBudget] = None):
        # 初始化所有组件
        self.style_manager = StyleAnchorManager()
        self.supply_strategy = ThreeLevelSupplyStrategy(self.style_manager)
        self.clip_selector = SmartClipSelector()
        self.decision_tree = GlobalConsistencyDecisionTree(self.style_manager)
        self.block_merger = ContinuousBlockMerger(self.style_manager)
        self.digital_human_manager = DigitalHumanManager()

        # 成本控制
        if budget is None:
            budget = CostBudget(total_budget=10.0, vl_call_limit=50)
        self.cost_controller = VLCostController(budget)

        # 系统状态
        self.processing_stats = {
            "total_projects": 0,
            "successful_supplies": 0,
            "total_processing_time": 0.0,
            "average_quality_score": 0.0
        }

    async def supply_materials(self, request: MaterialSupplyRequest) -> SupplyResult:
        """
        主要的素材供给接口
        """
        start_time = time.time()

        try:
            # Step 1: 建立风格锚点
            style_anchor = await self.style_manager.establish_style_anchor(request.shots)

            # Step 2: 提取全局元素
            self.style_manager.extract_global_elements(request.shots)

            # Step 3: 执行三级素材供给
            material_candidates = await self._execute_three_level_supply(request)

            # Step 4: 全局一致性决策
            decision_context = self._create_decision_context(request)
            strategy, decision_info = self.decision_tree.make_consistency_decision(
                request.shots, material_candidates, decision_context
            )

            # Step 5: 处理视频剪辑（如果需要）
            processed_materials = await self._process_video_clips(
                material_candidates, request.shots
            )

            # Step 6: 连续块合并
            merged_groups, merge_stats = self.block_merger.merge_continuous_blocks(
                request.shots, strategy
            )

            # Step 7: 数字人统一管理
            await self._setup_digital_human(request.shots, merged_groups)

            # Step 8: 生成最终结果
            final_materials = await self._generate_final_materials(
                processed_materials, merged_groups, strategy
            )

            # Step 9: 质量评估
            quality_metrics = await self._evaluate_quality(final_materials, request.shots)

            processing_time = time.time() - start_time

            # 更新统计
            self._update_system_stats(processing_time, quality_metrics)

            return SupplyResult(
                final_materials=final_materials,
                generation_tasks=merged_groups,
                style_anchor=style_anchor,
                decision_strategy=strategy,
                cost_report=self.cost_controller.get_cost_report(),
                processing_time=processing_time,
                quality_metrics=quality_metrics
            )

        except Exception as e:
            print(f"[素材供给系统] 处理失败: {e}")
            raise

    async def _execute_three_level_supply(self, request: MaterialSupplyRequest) -> List[MaterialCandidate]:
        """执行三级素材供给策略"""
        candidates = []

        for i, shot in enumerate(request.shots):
            # 只对需要素材的分镜执行供给
            if shot.get("asset_status") != "matched":
                supply_request = SupplyRequest(
                    shot_id=shot.get("shot_id", f"shot_{i}"),
                    description=shot.get("description", ""),
                    duration=shot.get("duration", 3.0),
                    required_style=self.style_manager.style_anchor,
                    shot_type=shot.get("shot_type", "video"),
                    context=shot.get("context", {})
                )

                candidate = await self.supply_strategy.supply_material(supply_request)
                if candidate:
                    candidates.append(candidate)

        return candidates

    def _create_decision_context(self, request: MaterialSupplyRequest) -> DecisionContext:
        """创建决策上下文"""
        project_config = request.project_config
        user_prefs = request.user_preferences or {}

        # 判断视频类型
        total_duration = sum(shot.get("duration", 3.0) for shot in request.shots)
        if total_duration <= 60:
            video_type = "short_video"
        elif total_duration <= 300:
            video_type = "medium_video"
        else:
            video_type = "long_video"

        # 提取优先级要求
        priority_requirements = []
        if user_prefs.get("quality_first"):
            priority_requirements.append("quality")
        if user_prefs.get("cost_conscious"):
            priority_requirements.append("cost")
        if user_prefs.get("fast_delivery"):
            priority_requirements.append("speed")
        if user_prefs.get("brand_consistency"):
            priority_requirements.append("consistency")

        return DecisionContext(
            video_type=video_type,
            priority_requirements=priority_requirements,
            user_budget=request.budget_constraints.get("total_budget") if request.budget_constraints else None,
            deadline=request.budget_constraints.get("deadline_hours") if request.budget_constraints else None,
            brand_guidelines=project_config.get("brand_guidelines")
        )

    async def _process_video_clips(
        self,
        candidates: List[MaterialCandidate],
        shots: List[Dict]
    ) -> List[MaterialCandidate]:
        """处理需要剪辑的视频素材"""
        processed = []

        for candidate in candidates:
            # 检查是否需要剪辑
            if (candidate.duration > candidate.verification_result.get("target_duration", 0) * 1.5 and
                candidate.supply_level.value != "ai_generation"):

                # 执行智能剪辑
                clip_request = ClipRequest(
                    video_url=candidate.url,
                    target_duration=candidate.duration,
                    target_description=candidate.description,
                    style_requirements=self.style_manager.get_consistency_requirements()
                )

                clips = await self.clip_selector.select_optimal_clips(clip_request)

                if clips:
                    # 使用最佳剪辑片段
                    best_clip = clips[0]
                    candidate.url = f"{candidate.url}#t={best_clip.start_time},{best_clip.end_time}"
                    candidate.verification_result["clipped"] = True
                    candidate.verification_result["clip_info"] = {
                        "start_time": best_clip.start_time,
                        "end_time": best_clip.end_time,
                        "original_duration": candidate.duration,
                        "clipped_duration": best_clip.duration
                    }

            processed.append(candidate)

        return processed

    async def _setup_digital_human(self, shots: List[Dict], merged_groups: List[MergedGroup]):
        """设置数字人"""
        # 检查是否有数字人需求
        talking_groups = [g for g in merged_groups if g.block_type.value == "talking_head"]

        if talking_groups:
            # 收集所有脚本
            all_scripts = []
            for group in talking_groups:
                for block in group.blocks:
                    all_scripts.extend(block.scripts)

            # 创建选择标准
            criteria = SelectionCriteria(
                language_requirement="zh-CN",
                content_type="general"
            )

            # 选择最优数字人
            await self.digital_human_manager.select_optimal_avatar(all_scripts, criteria)

    async def _generate_final_materials(
        self,
        processed_materials: List[MaterialCandidate],
        merged_groups: List[MergedGroup],
        strategy: ConsistencyStrategy
    ) -> List[Dict[str, Any]]:
        """生成最终素材列表"""
        final_materials = []

        # 处理现有素材
        for material in processed_materials:
            final_material = {
                "material_id": material.material_id,
                "type": "existing_material",
                "url": material.url,
                "thumbnail": material.thumbnail_url,
                "duration": material.duration,
                "confidence": material.confidence_score,
                "supply_method": material.supply_level.value,
                "verification_info": material.verification_result
            }
            final_materials.append(final_material)

        # 处理需要生成的组
        for group in merged_groups:
            if group.block_type.value in ["talking_head", "ai_video"]:
                generated_content = await self.block_merger.generate_merged_content(group)

                final_material = {
                    "material_id": group.group_id,
                    "type": "generated_material",
                    "generation_info": generated_content,
                    "duration": group.total_duration,
                    "confidence": 0.95,  # AI生成的置信度很高
                    "supply_method": "ai_generation",
                    "block_type": group.block_type.value
                }
                final_materials.append(final_material)

        return final_materials

    async def _evaluate_quality(self, materials: List[Dict], shots: List[Dict]) -> Dict[str, Any]:
        """评估最终质量"""
        total_materials = len(materials)

        # 计算各种指标
        ai_generated_count = len([m for m in materials if m.get("supply_method") == "ai_generation"])
        library_count = len([m for m in materials if m.get("supply_method") in ["library_search", "vl_verification"]])

        # 计算平均置信度
        confidences = [m.get("confidence", 0) for m in materials]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # 风格一致性评估
        style_consistency = self._evaluate_style_consistency(materials)

        # 完整性评估
        completeness = len(materials) / len(shots) if shots else 0

        return {
            "total_materials": total_materials,
            "ai_generated_ratio": ai_generated_count / max(total_materials, 1),
            "library_usage_ratio": library_count / max(total_materials, 1),
            "average_confidence": avg_confidence,
            "style_consistency_score": style_consistency,
            "completeness_score": completeness,
            "overall_quality_score": (avg_confidence * 0.4 +
                                    style_consistency * 0.3 +
                                    completeness * 0.3)
        }

    def _evaluate_style_consistency(self, materials: List[Dict]) -> float:
        """评估风格一致性"""
        if not self.style_manager.style_anchor:
            return 0.5

        # 简化的一致性评估
        consistent_count = 0
        total_count = len(materials)

        for material in materials:
            # AI生成的材料默认一致
            if material.get("supply_method") == "ai_generation":
                consistent_count += 1
            else:
                # 检查验证信息中的风格匹配度
                verification = material.get("verification_info", {})
                style_match = verification.get("style_match", 0)
                if style_match >= 70:
                    consistent_count += 1

        return consistent_count / max(total_count, 1)

    def _update_system_stats(self, processing_time: float, quality_metrics: Dict):
        """更新系统统计"""
        self.processing_stats["total_projects"] += 1
        self.processing_stats["total_processing_time"] += processing_time

        if quality_metrics.get("overall_quality_score", 0) > 0.6:
            self.processing_stats["successful_supplies"] += 1

        # 更新平均质量分数
        current_avg = self.processing_stats["average_quality_score"]
        new_score = quality_metrics.get("overall_quality_score", 0)
        total_projects = self.processing_stats["total_projects"]

        self.processing_stats["average_quality_score"] = (
            (current_avg * (total_projects - 1) + new_score) / total_projects
        )

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        avg_processing_time = (
            self.processing_stats["total_processing_time"] /
            max(self.processing_stats["total_projects"], 1)
        )

        success_rate = (
            self.processing_stats["successful_supplies"] /
            max(self.processing_stats["total_projects"], 1)
        )

        return {
            "system_stats": self.processing_stats,
            "performance_metrics": {
                "average_processing_time": avg_processing_time,
                "success_rate": success_rate,
                "average_quality": self.processing_stats["average_quality_score"]
            },
            "component_status": {
                "style_manager": "active" if self.style_manager.style_anchor else "ready",
                "digital_human": "configured" if self.digital_human_manager.selected_avatar else "ready",
                "cost_controller": self.cost_controller.get_cost_report()["budget_info"]
            }
        }

    async def optimize_system_performance(self) -> Dict[str, Any]:
        """优化系统性能"""
        # 获取各组件的优化建议
        cost_optimization = self.cost_controller.optimize_future_calls()

        # 系统级优化建议
        recommendations = []

        # 基于历史性能提供建议
        if self.processing_stats["average_quality_score"] < 0.7:
            recommendations.append("考虑提高AI生成比例以改善质量")

        if self.processing_stats["total_processing_time"] / max(self.processing_stats["total_projects"], 1) > 60:
            recommendations.append("处理时间过长，建议优化并发策略")

        success_rate = self.processing_stats["successful_supplies"] / max(self.processing_stats["total_projects"], 1)
        if success_rate < 0.8:
            recommendations.append("成功率较低，建议检查素材库质量")

        return {
            "cost_optimization": cost_optimization,
            "system_recommendations": recommendations,
            "optimization_actions": [
                "清理VL缓存以释放内存",
                "重置成本预算计数器",
                "更新风格匹配阈值"
            ]
        }

    def reset_system_state(self):
        """重置系统状态"""
        self.style_manager = StyleAnchorManager()
        self.digital_human_manager.reset_selection()
        self.cost_controller.clear_cache()

    async def batch_supply_materials(self, requests: List[MaterialSupplyRequest]) -> List[SupplyResult]:
        """批量处理素材供给请求"""
        tasks = []
        for request in requests:
            task = self.supply_materials(request)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[批量供给] 请求 {i} 失败: {result}")
                # 创建失败结果
                failed_result = SupplyResult(
                    final_materials=[],
                    generation_tasks=[],
                    style_anchor=None,
                    decision_strategy=ConsistencyStrategy.USE_LIBRARY_MATERIALS,
                    cost_report={},
                    processing_time=0.0,
                    quality_metrics={"overall_quality_score": 0.0}
                )
                final_results.append(failed_result)
            else:
                final_results.append(result)

        return final_results