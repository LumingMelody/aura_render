"""
Multimodal Fusion Engine - Day 3 Implementation
多模态融合引擎 - 智能融合多模态输入
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class WeightCalculator:
    """权重计算器"""

    @staticmethod
    def calculate_normalized_weights(media_items: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算标准化权重"""
        if not media_items:
            return {}

        # 提取权重值
        weights = {}
        total_weight = 0

        for item in media_items:
            item_id = f"{item.get('type', 'unknown')}_{item.get('url', 'unknown')}"
            weight = item.get('weight', 1.0)
            weights[item_id] = weight
            total_weight += weight

        # 标准化权重
        if total_weight > 0:
            for item_id in weights:
                weights[item_id] = weights[item_id] / total_weight

        return weights

    @staticmethod
    def apply_type_priority(weights: Dict[str, float], type_priorities: Dict[str, float]) -> Dict[str, float]:
        """应用类型优先级调整"""
        adjusted_weights = {}

        for item_id, weight in weights.items():
            item_type = item_id.split('_')[0]
            priority_multiplier = type_priorities.get(item_type, 1.0)
            adjusted_weights[item_id] = weight * priority_multiplier

        # 重新标准化
        total = sum(adjusted_weights.values())
        if total > 0:
            for item_id in adjusted_weights:
                adjusted_weights[item_id] = adjusted_weights[item_id] / total

        return adjusted_weights


class ConflictResolver:
    """冲突解决器"""

    @staticmethod
    def resolve_style_conflicts(style_analyses: List[Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, Any]:
        """解决风格冲突"""
        if not style_analyses:
            return {}

        # 收集所有风格属性
        all_colors = []
        all_moods = []
        all_styles = []

        for analysis in style_analyses:
            analysis_result = analysis.get('analysis_result', {})
            style_info = analysis_result.get('style_analysis', {})

            # 提取颜色信息
            colors = style_info.get('dominant_colors', [])
            if colors:
                item_id = f"{analysis.get('analysis_type', 'unknown')}_{analysis.get('image_path', 'unknown')}"
                weight = weights.get(item_id, 0.0)
                all_colors.extend([(color, weight) for color in colors])

            # 提取情绪信息
            mood = style_info.get('mood', '')
            if mood:
                item_id = f"{analysis.get('analysis_type', 'unknown')}_{analysis.get('image_path', 'unknown')}"
                weight = weights.get(item_id, 0.0)
                all_moods.append((mood, weight))

            # 提取风格信息
            style = style_info.get('style', '')
            if style:
                item_id = f"{analysis.get('analysis_type', 'unknown')}_{analysis.get('image_path', 'unknown')}"
                weight = weights.get(item_id, 0.0)
                all_styles.append((style, weight))

        # 权重投票选择主导风格
        resolved_style = {
            'dominant_colors': ConflictResolver._weighted_color_selection(all_colors),
            'primary_mood': ConflictResolver._weighted_text_selection(all_moods),
            'style_direction': ConflictResolver._weighted_text_selection(all_styles)
        }

        return resolved_style

    @staticmethod
    def _weighted_color_selection(color_weight_pairs: List[Tuple[str, float]], top_k: int = 3) -> List[str]:
        """基于权重选择主导颜色"""
        color_scores = {}

        for color, weight in color_weight_pairs:
            if color in color_scores:
                color_scores[color] += weight
            else:
                color_scores[color] = weight

        # 按分数排序并返回前K个
        sorted_colors = sorted(color_scores.items(), key=lambda x: x[1], reverse=True)
        return [color for color, _ in sorted_colors[:top_k]]

    @staticmethod
    def _weighted_text_selection(text_weight_pairs: List[Tuple[str, float]]) -> str:
        """基于权重选择主导文本"""
        if not text_weight_pairs:
            return ""

        text_scores = {}

        for text, weight in text_weight_pairs:
            if text in text_scores:
                text_scores[text] += weight
            else:
                text_scores[text] = weight

        # 返回得分最高的文本
        return max(text_scores.items(), key=lambda x: x[1])[0] if text_scores else ""


class IntelligentGapFiller:
    """智能补全器"""

    @staticmethod
    def fill_missing_style_info(fusion_result: Dict[str, Any]) -> Dict[str, Any]:
        """智能填充缺失的风格信息"""
        style_info = fusion_result.get('unified_style', {})

        # 如果缺少颜色信息，基于情绪推断
        if not style_info.get('dominant_colors') and style_info.get('primary_mood'):
            mood = style_info['primary_mood'].lower()
            inferred_colors = IntelligentGapFiller._infer_colors_from_mood(mood)
            style_info['dominant_colors'] = inferred_colors
            fusion_result['gap_filling_applied'] = fusion_result.get('gap_filling_applied', [])
            fusion_result['gap_filling_applied'].append('color_from_mood')

        # 如果缺少情绪信息，基于颜色推断
        if not style_info.get('primary_mood') and style_info.get('dominant_colors'):
            colors = style_info['dominant_colors']
            inferred_mood = IntelligentGapFiller._infer_mood_from_colors(colors)
            style_info['primary_mood'] = inferred_mood
            fusion_result['gap_filling_applied'] = fusion_result.get('gap_filling_applied', [])
            fusion_result['gap_filling_applied'].append('mood_from_colors')

        # 如果缺少技术规格，提供默认值
        tech_specs = fusion_result.get('technical_requirements', {})
        if not tech_specs.get('resolution'):
            tech_specs['resolution'] = '1920x1080'
            tech_specs['fps'] = 30
            tech_specs['duration'] = '30s'
            fusion_result['gap_filling_applied'] = fusion_result.get('gap_filling_applied', [])
            fusion_result['gap_filling_applied'].append('default_tech_specs')

        fusion_result['unified_style'] = style_info
        fusion_result['technical_requirements'] = tech_specs

        return fusion_result

    @staticmethod
    def _infer_colors_from_mood(mood: str) -> List[str]:
        """从情绪推断颜色"""
        mood_color_map = {
            'energetic': ['#FF6B35', '#F7931E', '#FFD23F'],
            'calm': ['#4A90E2', '#7ED321', '#B8E986'],
            'elegant': ['#2C3E50', '#95A5A6', '#ECF0F1'],
            'warm': ['#E74C3C', '#F39C12', '#F1C40F'],
            'cool': ['#3498DB', '#9B59B6', '#1ABC9C'],
            'professional': ['#34495E', '#2C3E50', '#95A5A6'],
            'creative': ['#E91E63', '#9C27B0', '#673AB7'],
            'natural': ['#4CAF50', '#8BC34A', '#CDDC39']
        }

        for mood_key, colors in mood_color_map.items():
            if mood_key in mood.lower():
                return colors

        return ['#333333', '#666666', '#999999']  # 默认中性色

    @staticmethod
    def _infer_mood_from_colors(colors: List[str]) -> str:
        """从颜色推断情绪"""
        # 简化的颜色情绪映射
        warm_colors = ['#FF', '#E7', '#F3', '#F1', '#FF6', '#F79']
        cool_colors = ['#34', '#3498', '#9B59', '#1ABC', '#4A90']
        neutral_colors = ['#2C', '#95A5', '#ECF0', '#34495E']

        warm_count = sum(1 for color in colors if any(warm in color.upper() for warm in warm_colors))
        cool_count = sum(1 for color in colors if any(cool in color.upper() for cool in cool_colors))
        neutral_count = sum(1 for color in colors if any(neutral in color.upper() for neutral in neutral_colors))

        if warm_count > cool_count and warm_count > neutral_count:
            return "energetic"
        elif cool_count > warm_count and cool_count > neutral_count:
            return "calm"
        else:
            return "professional"


class MultiModalFusionEngine:
    """多模态融合引擎 - 智能整合视频、图片和产品信息"""

    def __init__(self):
        self.weight_calculator = WeightCalculator()
        self.conflict_resolver = ConflictResolver()
        self.gap_filler = IntelligentGapFiller()

        # 类型优先级配置
        self.type_priorities = {
            'main_product': 1.0,      # 产品主图最高优先级
            'style_guide': 0.9,       # 风格指导次优先级
            'style_reference': 0.8,   # 风格参考视频
            'mood_board': 0.7,        # 情绪板
            'detail_shot': 0.6,       # 产品细节图
            'comprehensive': 0.5      # 综合分析
        }

    async def fuse_multimodal_inputs(self, processed_media: Dict[str, Any]) -> Dict[str, Any]:
        """
        融合多模态输入

        Args:
            processed_media: 已处理的多媒体数据
            {
                'video_analyses': [...],
                'image_analyses': [...],
                'product_analyses': [...]
            }

        Returns:
            融合结果字典
        """
        logger.info("🔄 开始多模态融合处理")

        try:
            # 1. 收集所有分析结果
            all_analyses = self._collect_all_analyses(processed_media)

            if not all_analyses:
                logger.warning("⚠️  没有找到可融合的分析结果")
                return self._create_fallback_result()

            # 2. 计算权重
            weights = self._calculate_fusion_weights(all_analyses)

            # 3. 风格融合
            unified_style = self._fuse_style_information(all_analyses, weights)

            # 4. 内容融合
            unified_content = self._fuse_content_information(all_analyses, weights)

            # 5. 技术要求融合
            technical_requirements = self._fuse_technical_requirements(all_analyses, weights)

            # 6. 构建融合结果
            fusion_result = {
                'fusion_timestamp': datetime.now().isoformat(),
                'input_summary': {
                    'total_inputs': len(all_analyses),
                    'video_count': len(processed_media.get('video_analyses', [])),
                    'image_count': len(processed_media.get('image_analyses', [])),
                    'product_count': len(processed_media.get('product_analyses', []))
                },
                'unified_style': unified_style,
                'unified_content': unified_content,
                'technical_requirements': technical_requirements,
                'fusion_weights': weights,
                'confidence_score': self._calculate_confidence_score(all_analyses, weights)
            }

            # 7. 智能补全缺失信息
            fusion_result = self.gap_filler.fill_missing_style_info(fusion_result)

            logger.info(f"✅ 多模态融合完成，融合了 {len(all_analyses)} 个输入")
            return fusion_result

        except Exception as e:
            logger.error(f"❌ 多模态融合失败: {e}")
            return self._create_error_result(str(e))

    def _collect_all_analyses(self, processed_media: Dict[str, Any]) -> List[Dict[str, Any]]:
        """收集所有分析结果"""
        all_analyses = []

        # 收集视频分析
        video_analyses = processed_media.get('video_analyses', [])
        all_analyses.extend(video_analyses)

        # 收集图片分析
        image_analyses = processed_media.get('image_analyses', [])
        all_analyses.extend(image_analyses)

        # 收集产品图片分析
        product_analyses = processed_media.get('product_analyses', [])
        all_analyses.extend(product_analyses)

        return all_analyses

    def _calculate_fusion_weights(self, analyses: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算融合权重"""
        # 基础权重计算
        base_weights = self.weight_calculator.calculate_normalized_weights(analyses)

        # 应用类型优先级
        adjusted_weights = self.weight_calculator.apply_type_priority(
            base_weights, self.type_priorities
        )

        return adjusted_weights

    def _fuse_style_information(self, analyses: List[Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, Any]:
        """融合风格信息"""
        style_analyses = [analysis for analysis in analyses
                         if analysis.get('analysis_result', {}).get('style_analysis')]

        if not style_analyses:
            return {'style_source': 'default', 'confidence': 0.0}

        # 解决风格冲突
        unified_style = self.conflict_resolver.resolve_style_conflicts(style_analyses, weights)

        # 计算风格置信度
        style_confidence = sum(weights.get(f"{analysis.get('analysis_type', 'unknown')}_{analysis.get('image_path', analysis.get('video_path', 'unknown'))}", 0.0)
                              for analysis in style_analyses)

        unified_style['style_source'] = 'multimodal_fusion'
        unified_style['confidence'] = min(style_confidence, 1.0)

        return unified_style

    def _fuse_content_information(self, analyses: List[Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, Any]:
        """融合内容信息"""
        content_elements = []
        scene_descriptions = []

        for analysis in analyses:
            analysis_result = analysis.get('analysis_result', {})

            # 收集内容元素
            if 'content_analysis' in analysis_result:
                content = analysis_result['content_analysis']
                if 'key_elements' in content:
                    content_elements.extend(content['key_elements'])
                if 'scene_description' in content:
                    scene_descriptions.append(content['scene_description'])

        # 去重和权重排序
        unique_elements = list(set(content_elements))

        return {
            'key_elements': unique_elements[:10],  # 取前10个关键元素
            'scene_description': '. '.join(scene_descriptions[:3]) if scene_descriptions else "现代简约场景",
            'content_source': 'multimodal_fusion'
        }

    def _fuse_technical_requirements(self, analyses: List[Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, Any]:
        """融合技术要求"""
        tech_requirements = {
            'resolution': '1920x1080',
            'fps': 30,
            'duration': '30s',
            'format': 'mp4',
            'quality': 'high'
        }

        # 从分析结果中提取技术要求
        for analysis in analyses:
            analysis_result = analysis.get('analysis_result', {})
            if 'technical_analysis' in analysis_result:
                tech_info = analysis_result['technical_analysis']

                # 更新分辨率（取最高）
                if 'resolution' in tech_info:
                    current_res = tech_requirements.get('resolution', '1920x1080')
                    new_res = tech_info['resolution']
                    if self._compare_resolution(new_res, current_res) > 0:
                        tech_requirements['resolution'] = new_res

                # 更新帧率（取最高）
                if 'fps' in tech_info:
                    tech_requirements['fps'] = max(
                        tech_requirements.get('fps', 30),
                        tech_info['fps']
                    )

        return tech_requirements

    def _compare_resolution(self, res1: str, res2: str) -> int:
        """比较分辨率大小"""
        try:
            w1, h1 = map(int, res1.split('x'))
            w2, h2 = map(int, res2.split('x'))

            pixels1 = w1 * h1
            pixels2 = w2 * h2

            return 1 if pixels1 > pixels2 else (-1 if pixels1 < pixels2 else 0)
        except:
            return 0

    def _calculate_confidence_score(self, analyses: List[Dict[str, Any]], weights: Dict[str, float]) -> float:
        """计算整体置信度分数"""
        if not analyses:
            return 0.0

        total_confidence = 0.0
        total_weight = 0.0

        for analysis in analyses:
            analysis_metadata = analysis.get('analysis_metadata', {})
            confidence = analysis_metadata.get('confidence', 0.5)

            item_id = f"{analysis.get('analysis_type', 'unknown')}_{analysis.get('image_path', analysis.get('video_path', 'unknown'))}"
            weight = weights.get(item_id, 0.0)

            total_confidence += confidence * weight
            total_weight += weight

        return total_confidence / total_weight if total_weight > 0 else 0.0

    def _create_fallback_result(self) -> Dict[str, Any]:
        """创建回退结果"""
        return {
            'fusion_timestamp': datetime.now().isoformat(),
            'input_summary': {
                'total_inputs': 0,
                'video_count': 0,
                'image_count': 0,
                'product_count': 0
            },
            'unified_style': {
                'dominant_colors': ['#333333', '#666666', '#999999'],
                'primary_mood': 'professional',
                'style_direction': 'modern',
                'style_source': 'fallback'
            },
            'unified_content': {
                'key_elements': ['产品展示', '简约设计'],
                'scene_description': '现代简约的产品展示场景',
                'content_source': 'fallback'
            },
            'technical_requirements': {
                'resolution': '1920x1080',
                'fps': 30,
                'duration': '30s',
                'format': 'mp4',
                'quality': 'high'
            },
            'fusion_weights': {},
            'confidence_score': 0.3,
            'fallback_reason': 'no_input_analyses'
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """创建错误结果"""
        fallback_result = self._create_fallback_result()
        fallback_result['error'] = error_message
        fallback_result['fallback_reason'] = 'fusion_error'
        fallback_result['confidence_score'] = 0.1
        return fallback_result