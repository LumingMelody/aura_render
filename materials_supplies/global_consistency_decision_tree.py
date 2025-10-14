"""
全局一致性决策树 - 智能决策何时使用素材库 vs AI生成
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .style_anchor_manager import StyleAnchorManager, StyleVector
from .three_level_supply_strategy import MaterialCandidate, SupplyLevel


class ConsistencyStrategy(Enum):
    """一致性策略"""
    ALL_AI_GENERATION = "all_ai_generation"  # 全部AI生成
    AI_FILL_MISSING = "ai_fill_missing"  # AI补全缺失部分
    USE_LIBRARY_MATERIALS = "use_library_materials"  # 使用素材库
    HYBRID_WITH_STYLE_TRANSFER = "hybrid_with_style_transfer"  # 混合+风格迁移


@dataclass
class ConsistencyMetrics:
    """一致性指标"""
    style_consistency_ratio: float  # 风格一致性比例
    total_duration: float  # 总时长
    missing_duration: float  # 缺失时长
    user_material_count: int  # 用户素材数量
    library_material_count: int  # 素材库素材数量
    quality_variance: float  # 质量方差
    cost_estimate: float  # 成本估算


@dataclass
class DecisionContext:
    """决策上下文"""
    video_type: str  # "short_video", "long_video", "commercial", "educational"
    priority_requirements: List[str]  # ["consistency", "cost", "speed", "quality"]
    user_budget: Optional[float] = None
    deadline: Optional[float] = None  # 小时
    brand_guidelines: Optional[Dict] = None


class GlobalConsistencyDecisionTree:
    """全局一致性决策树"""

    def __init__(self, style_manager: StyleAnchorManager):
        self.style_manager = style_manager

        # 决策阈值配置
        self.short_video_threshold = 120  # 2分钟
        self.consistency_threshold = 0.7  # 一致性阈值
        self.missing_duration_threshold = 120  # 缺失时长阈值
        self.quality_variance_threshold = 0.3  # 质量方差阈值

        # 成本配置
        self.cost_per_ai_second = 0.5  # AI生成每秒成本
        self.cost_per_vl_call = 0.02  # VL调用成本
        self.cost_per_style_transfer = 0.1  # 风格迁移成本

    def make_consistency_decision(
        self,
        shots: List[Dict],
        material_candidates: List[MaterialCandidate],
        context: DecisionContext
    ) -> Tuple[ConsistencyStrategy, Dict[str, Any]]:
        """
        做出全局一致性决策
        """
        # 计算一致性指标
        metrics = self._calculate_consistency_metrics(shots, material_candidates)

        # 基于决策树选择策略
        strategy = self._apply_decision_tree(metrics, context)

        # 生成决策解释
        explanation = self._generate_decision_explanation(strategy, metrics, context)

        return strategy, {
            "strategy": strategy.value,
            "metrics": metrics,
            "explanation": explanation,
            "estimated_cost": self._estimate_cost(strategy, metrics),
            "estimated_time": self._estimate_time(strategy, metrics),
            "quality_prediction": self._predict_quality(strategy, metrics)
        }

    def _calculate_consistency_metrics(
        self,
        shots: List[Dict],
        candidates: List[MaterialCandidate]
    ) -> ConsistencyMetrics:
        """计算一致性指标"""

        total_duration = sum(shot.get("duration", 3.0) for shot in shots)

        # 分析素材来源
        user_materials = []
        library_materials = []
        missing_materials = []

        for i, shot in enumerate(shots):
            if shot.get("asset_status") == "matched":
                asset = shot.get("scheduled_asset", {})
                if asset.get("source") == "user_upload":
                    user_materials.append(shot)
                else:
                    library_materials.append(shot)
            else:
                missing_materials.append(shot)

        # 计算缺失时长
        missing_duration = sum(shot.get("duration", 3.0) for shot in missing_materials)

        # 计算风格一致性
        style_consistency = self._calculate_style_consistency(shots, candidates)

        # 计算质量方差
        quality_variance = self._calculate_quality_variance(candidates)

        return ConsistencyMetrics(
            style_consistency_ratio=style_consistency,
            total_duration=total_duration,
            missing_duration=missing_duration,
            user_material_count=len(user_materials),
            library_material_count=len(library_materials),
            quality_variance=quality_variance,
            cost_estimate=0.0  # 稍后计算
        )

    def _calculate_style_consistency(
        self,
        shots: List[Dict],
        candidates: List[MaterialCandidate]
    ) -> float:
        """计算风格一致性比例"""
        if not self.style_manager.style_anchor:
            return 0.5  # 无锚点时返回中性值

        total_shots = len(shots)
        consistent_shots = 0

        for shot in shots:
            if shot.get("asset_status") == "matched":
                # 检查已匹配素材的风格一致性
                asset = shot.get("scheduled_asset", {})
                if self._is_style_consistent_with_anchor(asset):
                    consistent_shots += 1

        # 加上候选素材的一致性
        for candidate in candidates:
            if candidate.confidence_score >= 0.7:  # 高置信度的候选
                if self._is_candidate_style_consistent(candidate):
                    consistent_shots += 1

        return consistent_shots / max(total_shots, 1)

    def _is_style_consistent_with_anchor(self, asset: Dict) -> bool:
        """检查素材与锚点风格的一致性"""
        if not asset or not self.style_manager.style_anchor:
            return False

        asset_style = asset.get("style")
        if not asset_style:
            return False

        try:
            # 简化的风格匹配
            anchor_style = self.style_manager.style_anchor.style_type.value
            return asset_style.lower() == anchor_style.lower()
        except:
            return False

    def _is_candidate_style_consistent(self, candidate: MaterialCandidate) -> bool:
        """检查候选素材的风格一致性"""
        if not self.style_manager.style_anchor:
            return False

        # 基于验证结果判断
        if candidate.verification_result:
            style_match = candidate.verification_result.get("style_match", 0)
            return style_match >= 70  # 70分以上认为一致

        return False

    def _calculate_quality_variance(self, candidates: List[MaterialCandidate]) -> float:
        """计算质量方差"""
        if not candidates:
            return 0.0

        scores = [c.confidence_score for c in candidates]
        if len(scores) < 2:
            return 0.0

        return float(np.var(scores))

    def _apply_decision_tree(
        self,
        metrics: ConsistencyMetrics,
        context: DecisionContext
    ) -> ConsistencyStrategy:
        """
        应用决策树逻辑
        """
        is_short_video = metrics.total_duration <= self.short_video_threshold

        # 根节点：检查是否有用户素材
        if metrics.user_material_count > 0:
            return self._user_material_branch(metrics, context, is_short_video)
        else:
            return self._no_user_material_branch(metrics, context, is_short_video)

    def _user_material_branch(
        self,
        metrics: ConsistencyMetrics,
        context: DecisionContext,
        is_short_video: bool
    ) -> ConsistencyStrategy:
        """有用户素材的决策分支"""

        # 检查风格一致性
        if metrics.style_consistency_ratio < self.consistency_threshold:
            # 风格不一致

            if is_short_video:
                # 短视频：优先保证一致性
                if "consistency" in context.priority_requirements:
                    return ConsistencyStrategy.ALL_AI_GENERATION
                else:
                    return ConsistencyStrategy.HYBRID_WITH_STYLE_TRANSFER

            else:
                # 长视频：考虑成本
                if metrics.missing_duration < self.missing_duration_threshold:
                    # 缺失部分较少，AI补全
                    return ConsistencyStrategy.AI_FILL_MISSING
                else:
                    # 缺失部分较多，考虑成本
                    if "cost" in context.priority_requirements:
                        return ConsistencyStrategy.HYBRID_WITH_STYLE_TRANSFER
                    else:
                        return ConsistencyStrategy.AI_FILL_MISSING

        else:
            # 风格一致性好
            if metrics.missing_duration > 0:
                return ConsistencyStrategy.AI_FILL_MISSING
            else:
                return ConsistencyStrategy.USE_LIBRARY_MATERIALS

    def _no_user_material_branch(
        self,
        metrics: ConsistencyMetrics,
        context: DecisionContext,
        is_short_video: bool
    ) -> ConsistencyStrategy:
        """无用户素材的决策分支"""

        # 检查素材库质量
        if metrics.quality_variance > self.quality_variance_threshold:
            # 质量差异大
            if is_short_video or "quality" in context.priority_requirements:
                return ConsistencyStrategy.ALL_AI_GENERATION
            else:
                return ConsistencyStrategy.HYBRID_WITH_STYLE_TRANSFER

        # 检查风格一致性
        if metrics.style_consistency_ratio < self.consistency_threshold:
            if is_short_video:
                return ConsistencyStrategy.ALL_AI_GENERATION
            else:
                if metrics.missing_duration < self.missing_duration_threshold:
                    return ConsistencyStrategy.AI_FILL_MISSING
                else:
                    return ConsistencyStrategy.HYBRID_WITH_STYLE_TRANSFER
        else:
            # 素材库质量和一致性都不错
            if metrics.missing_duration > 0:
                return ConsistencyStrategy.AI_FILL_MISSING
            else:
                return ConsistencyStrategy.USE_LIBRARY_MATERIALS

    def _estimate_cost(self, strategy: ConsistencyStrategy, metrics: ConsistencyMetrics) -> float:
        """估算成本"""
        base_cost = 0.0

        if strategy == ConsistencyStrategy.ALL_AI_GENERATION:
            base_cost = metrics.total_duration * self.cost_per_ai_second

        elif strategy == ConsistencyStrategy.AI_FILL_MISSING:
            base_cost = metrics.missing_duration * self.cost_per_ai_second

        elif strategy == ConsistencyStrategy.HYBRID_WITH_STYLE_TRANSFER:
            # 素材库成本 + 风格迁移成本
            transfer_count = metrics.library_material_count
            base_cost = transfer_count * self.cost_per_style_transfer
            base_cost += metrics.missing_duration * self.cost_per_ai_second

        elif strategy == ConsistencyStrategy.USE_LIBRARY_MATERIALS:
            # 主要是VL验证成本
            base_cost = metrics.library_material_count * self.cost_per_vl_call

        return base_cost

    def _estimate_time(self, strategy: ConsistencyStrategy, metrics: ConsistencyMetrics) -> float:
        """估算处理时间（分钟）"""
        if strategy == ConsistencyStrategy.ALL_AI_GENERATION:
            return metrics.total_duration * 2.0  # AI生成大约每秒需要2分钟

        elif strategy == ConsistencyStrategy.AI_FILL_MISSING:
            return metrics.missing_duration * 2.0 + 5.0  # 额外的合并时间

        elif strategy == ConsistencyStrategy.HYBRID_WITH_STYLE_TRANSFER:
            ai_time = metrics.missing_duration * 2.0
            transfer_time = metrics.library_material_count * 0.5
            return ai_time + transfer_time + 10.0

        elif strategy == ConsistencyStrategy.USE_LIBRARY_MATERIALS:
            return 5.0  # 主要是下载和处理时间

        return 0.0

    def _predict_quality(self, strategy: ConsistencyStrategy, metrics: ConsistencyMetrics) -> float:
        """预测最终质量（0-1）"""
        if strategy == ConsistencyStrategy.ALL_AI_GENERATION:
            return 0.9  # AI生成质量稳定且高

        elif strategy == ConsistencyStrategy.AI_FILL_MISSING:
            # 基于现有素材质量
            existing_quality = max(0.7, 1.0 - metrics.quality_variance)
            return existing_quality * 0.8 + 0.9 * 0.2  # 加权平均

        elif strategy == ConsistencyStrategy.HYBRID_WITH_STYLE_TRANSFER:
            base_quality = max(0.6, 1.0 - metrics.quality_variance)
            return base_quality * 0.85  # 风格迁移可能略微降低质量

        elif strategy == ConsistencyStrategy.USE_LIBRARY_MATERIALS:
            return max(0.6, 1.0 - metrics.quality_variance)

        return 0.5

    def _generate_decision_explanation(
        self,
        strategy: ConsistencyStrategy,
        metrics: ConsistencyMetrics,
        context: DecisionContext
    ) -> List[str]:
        """生成决策解释"""
        explanations = []

        # 基础分析
        is_short = metrics.total_duration <= self.short_video_threshold
        video_type_desc = "短视频" if is_short else "长视频"
        explanations.append(f"视频类型: {video_type_desc} ({metrics.total_duration:.1f}秒)")

        # 一致性分析
        consistency_desc = "高" if metrics.style_consistency_ratio >= self.consistency_threshold else "低"
        explanations.append(f"风格一致性: {consistency_desc} ({metrics.style_consistency_ratio:.1%})")

        # 素材分析
        if metrics.user_material_count > 0:
            explanations.append(f"包含 {metrics.user_material_count} 个用户上传素材")

        if metrics.missing_duration > 0:
            explanations.append(f"缺失素材时长: {metrics.missing_duration:.1f}秒")

        # 策略原因
        if strategy == ConsistencyStrategy.ALL_AI_GENERATION:
            if is_short:
                explanations.append("短视频优先保证风格统一，选择全部AI生成")
            else:
                explanations.append("风格差异过大，选择全部AI生成以保证一致性")

        elif strategy == ConsistencyStrategy.AI_FILL_MISSING:
            explanations.append("大部分素材质量良好，仅对缺失部分进行AI生成")

        elif strategy == ConsistencyStrategy.HYBRID_WITH_STYLE_TRANSFER:
            explanations.append("使用素材库+风格迁移，平衡成本与质量")

        elif strategy == ConsistencyStrategy.USE_LIBRARY_MATERIALS:
            explanations.append("素材库质量和一致性良好，直接使用")

        # 优先级考虑
        if context.priority_requirements:
            priority_desc = ", ".join(context.priority_requirements)
            explanations.append(f"考虑优先级: {priority_desc}")

        return explanations

    def evaluate_alternative_strategies(
        self,
        shots: List[Dict],
        candidates: List[MaterialCandidate],
        context: DecisionContext
    ) -> Dict[str, Dict[str, Any]]:
        """评估所有可能的策略"""
        metrics = self._calculate_consistency_metrics(shots, candidates)

        results = {}
        for strategy in ConsistencyStrategy:
            results[strategy.value] = {
                "estimated_cost": self._estimate_cost(strategy, metrics),
                "estimated_time": self._estimate_time(strategy, metrics),
                "predicted_quality": self._predict_quality(strategy, metrics),
                "pros": self._get_strategy_pros(strategy),
                "cons": self._get_strategy_cons(strategy)
            }

        return results

    def _get_strategy_pros(self, strategy: ConsistencyStrategy) -> List[str]:
        """获取策略优点"""
        pros_map = {
            ConsistencyStrategy.ALL_AI_GENERATION: [
                "风格完全统一", "质量稳定", "可控性强", "符合品牌要求"
            ],
            ConsistencyStrategy.AI_FILL_MISSING: [
                "成本相对较低", "保留用户素材", "一致性较好", "处理时间适中"
            ],
            ConsistencyStrategy.HYBRID_WITH_STYLE_TRANSFER: [
                "成本最低", "利用现有素材", "风格可调整", "处理速度快"
            ],
            ConsistencyStrategy.USE_LIBRARY_MATERIALS: [
                "成本极低", "速度最快", "素材丰富", "即用即得"
            ]
        }
        return pros_map.get(strategy, [])

    def _get_strategy_cons(self, strategy: ConsistencyStrategy) -> List[str]:
        """获取策略缺点"""
        cons_map = {
            ConsistencyStrategy.ALL_AI_GENERATION: [
                "成本最高", "处理时间长", "AI限制性", "可能过于相似"
            ],
            ConsistencyStrategy.AI_FILL_MISSING: [
                "部分风格差异", "处理复杂", "质量不均", "衔接可能不自然"
            ],
            ConsistencyStrategy.HYBRID_WITH_STYLE_TRANSFER: [
                "风格迁移质量不稳定", "仍有一致性风险", "处理流程复杂"
            ],
            ConsistencyStrategy.USE_LIBRARY_MATERIALS: [
                "风格可能不统一", "质量参差不齐", "难以满足特定需求"
            ]
        }
        return cons_map.get(strategy, [])

    def update_decision_based_on_feedback(
        self,
        original_decision: ConsistencyStrategy,
        feedback: Dict[str, Any],
        metrics: ConsistencyMetrics
    ) -> ConsistencyStrategy:
        """基于反馈更新决策"""

        # 如果用户反馈成本过高
        if feedback.get("cost_too_high"):
            if original_decision == ConsistencyStrategy.ALL_AI_GENERATION:
                return ConsistencyStrategy.HYBRID_WITH_STYLE_TRANSFER
            elif original_decision == ConsistencyStrategy.AI_FILL_MISSING:
                return ConsistencyStrategy.USE_LIBRARY_MATERIALS

        # 如果用户反馈质量要求更高
        if feedback.get("quality_priority"):
            if original_decision == ConsistencyStrategy.USE_LIBRARY_MATERIALS:
                return ConsistencyStrategy.AI_FILL_MISSING
            elif original_decision == ConsistencyStrategy.HYBRID_WITH_STYLE_TRANSFER:
                return ConsistencyStrategy.ALL_AI_GENERATION

        # 如果用户反馈时间紧急
        if feedback.get("time_urgent"):
            if original_decision == ConsistencyStrategy.ALL_AI_GENERATION:
                return ConsistencyStrategy.HYBRID_WITH_STYLE_TRANSFER
            elif original_decision == ConsistencyStrategy.AI_FILL_MISSING:
                return ConsistencyStrategy.USE_LIBRARY_MATERIALS

        return original_decision