"""
增强素材匹配引擎 - 基于多模态语义理解的智能匹配
"""
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncio
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
from collections import defaultdict, deque
from enum import Enum

from .style_anchor_manager import StyleVector, StyleType
from cache.cache_manager import CacheManager
from database.database_manager import DatabaseManager


class MatchingStrategy(Enum):
    """匹配策略"""
    SEMANTIC_FIRST = "semantic_first"      # 语义优先
    STYLE_FIRST = "style_first"           # 风格优先
    BALANCED = "balanced"                 # 平衡模式
    USER_PREFERENCE = "user_preference"   # 用户偏好
    DIVERSITY_BOOST = "diversity_boost"   # 多样性增强


@dataclass
class MaterialFeatures:
    """素材特征向量"""
    semantic_embedding: np.ndarray          # 语义embedding (512维)
    visual_features: np.ndarray             # 视觉特征 (256维)
    audio_features: Optional[np.ndarray]    # 音频特征 (128维)
    style_vector: StyleVector               # 风格向量
    quality_score: float                    # 质量评分 0-1
    popularity_score: float                 # 热度评分 0-1
    freshness_score: float                  # 新鲜度评分 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchingContext:
    """匹配上下文"""
    query_text: str                         # 查询文本
    user_id: Optional[str]                  # 用户ID
    session_id: str                         # 会话ID
    style_anchor: Optional[StyleVector]     # 风格锚点
    previous_materials: List[str]           # 之前使用的素材
    user_preferences: Dict[str, Any]        # 用户偏好
    context_metadata: Dict[str, Any]        # 上下文元数据
    matching_strategy: MatchingStrategy     # 匹配策略


@dataclass
class MatchResult:
    """匹配结果"""
    material_id: str
    confidence_score: float                 # 置信度 0-1
    relevance_score: float                  # 相关性 0-1
    style_consistency: float                # 风格一致性 0-1
    quality_factor: float                   # 质量因子 0-1
    diversity_bonus: float                  # 多样性奖励 0-1
    final_score: float                      # 最终评分 0-1
    explanation: str                        # 匹配原因说明
    features: MaterialFeatures              # 素材特征


class UserPreferenceModel:
    """用户偏好模型"""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.preference_decay = 0.95  # 偏好衰减因子
        self.learning_rate = 0.1      # 学习率

    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """获取用户偏好"""
        cache_key = f"user_preferences:{user_id}"
        cached_prefs = await self.cache_manager.get(cache_key)

        if cached_prefs:
            return json.loads(cached_prefs)

        # 默认偏好
        default_prefs = {
            'style_preferences': {
                StyleType.REALISTIC.value: 0.5,
                StyleType.CINEMATIC.value: 0.3,
                StyleType.ANIME.value: 0.2
            },
            'quality_threshold': 0.6,
            'diversity_preference': 0.7,
            'content_categories': {},
            'color_preferences': [],
            'interaction_history': [],
            'last_updated': datetime.now().isoformat()
        }

        await self.cache_manager.set(cache_key, json.dumps(default_prefs), expire=3600)
        return default_prefs

    async def update_preferences(self, user_id: str, material_id: str,
                               action: str, features: MaterialFeatures):
        """更新用户偏好"""
        preferences = await self.get_user_preferences(user_id)

        # 根据用户行为更新偏好
        if action == "selected":
            weight = 1.0
        elif action == "liked":
            weight = 1.5
        elif action == "disliked":
            weight = -1.0
        elif action == "skipped":
            weight = -0.3
        else:
            weight = 0.0

        # 更新风格偏好
        style_type = features.style_vector.style_type.value
        current_pref = preferences['style_preferences'].get(style_type, 0.5)
        new_pref = current_pref + (weight * self.learning_rate)
        preferences['style_preferences'][style_type] = max(0.0, min(1.0, new_pref))

        # 更新内容类别偏好
        for category in features.metadata.get('categories', []):
            current_cat_pref = preferences['content_categories'].get(category, 0.5)
            new_cat_pref = current_cat_pref + (weight * self.learning_rate * 0.5)
            preferences['content_categories'][category] = max(0.0, min(1.0, new_cat_pref))

        # 添加交互历史
        interaction = {
            'material_id': material_id,
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'features_hash': hashlib.md5(str(features).encode()).hexdigest()
        }
        preferences['interaction_history'].append(interaction)

        # 限制历史记录长度
        if len(preferences['interaction_history']) > 1000:
            preferences['interaction_history'] = preferences['interaction_history'][-1000:]

        preferences['last_updated'] = datetime.now().isoformat()

        # 缓存更新的偏好
        cache_key = f"user_preferences:{user_id}"
        await self.cache_manager.set(cache_key, json.dumps(preferences), expire=3600)

    def calculate_preference_score(self, preferences: Dict[str, Any],
                                 features: MaterialFeatures) -> float:
        """计算基于用户偏好的评分"""
        score = 0.0
        total_weight = 0.0

        # 风格偏好评分
        style_type = features.style_vector.style_type.value
        style_pref = preferences['style_preferences'].get(style_type, 0.5)
        score += style_pref * 0.4
        total_weight += 0.4

        # 内容类别偏好评分
        categories = features.metadata.get('categories', [])
        if categories:
            category_scores = []
            for category in categories:
                cat_pref = preferences['content_categories'].get(category, 0.5)
                category_scores.append(cat_pref)
            avg_category_score = np.mean(category_scores)
            score += avg_category_score * 0.3
            total_weight += 0.3

        # 质量偏好评分
        quality_threshold = preferences.get('quality_threshold', 0.6)
        if features.quality_score >= quality_threshold:
            score += features.quality_score * 0.3
        else:
            score += features.quality_score * 0.1  # 降权
        total_weight += 0.3

        return score / total_weight if total_weight > 0 else 0.5


class SemanticMatcher:
    """语义匹配器"""

    def __init__(self):
        self.embedding_cache = {}
        self.similarity_threshold = 0.7

    async def calculate_semantic_similarity(self, query_embedding: np.ndarray,
                                          material_embedding: np.ndarray) -> float:
        """计算语义相似度"""
        # 计算余弦相似度
        query_norm = np.linalg.norm(query_embedding)
        material_norm = np.linalg.norm(material_embedding)

        if query_norm == 0 or material_norm == 0:
            return 0.0

        similarity = np.dot(query_embedding, material_embedding) / (query_norm * material_norm)
        return max(0.0, similarity)

    async def extract_query_embedding(self, query_text: str) -> np.ndarray:
        """提取查询文本的embedding"""
        # 这里应该调用实际的embedding模型
        # 暂时使用模拟的embedding
        query_hash = hashlib.md5(query_text.encode()).hexdigest()

        if query_hash in self.embedding_cache:
            return self.embedding_cache[query_hash]

        # 模拟embedding提取
        # 在实际实现中，这里会调用BERT、Sentence-BERT等模型
        embedding = np.random.randn(512)  # 512维embedding
        embedding = embedding / np.linalg.norm(embedding)  # 归一化

        self.embedding_cache[query_hash] = embedding
        return embedding

    def calculate_keyword_overlap(self, query_text: str,
                                material_metadata: Dict[str, Any]) -> float:
        """计算关键词重叠度"""
        query_words = set(query_text.lower().split())
        material_text = " ".join([
            material_metadata.get('title', ''),
            material_metadata.get('description', ''),
            " ".join(material_metadata.get('tags', []))
        ]).lower()
        material_words = set(material_text.split())

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(material_words))
        return overlap / len(query_words)


class DiversityManager:
    """多样性管理器"""

    def __init__(self, diversity_window: int = 10):
        self.diversity_window = diversity_window
        self.recent_materials = deque(maxlen=diversity_window)

    def calculate_diversity_score(self, candidate_features: MaterialFeatures) -> float:
        """计算多样性评分"""
        if not self.recent_materials:
            return 1.0

        # 计算与最近材料的平均距离
        distances = []
        for recent_features in self.recent_materials:
            # 风格距离
            style_distance = candidate_features.style_vector.distance(recent_features.style_vector)

            # 语义距离
            semantic_distance = 1.0 - np.dot(
                candidate_features.semantic_embedding,
                recent_features.semantic_embedding
            ) / (
                np.linalg.norm(candidate_features.semantic_embedding) *
                np.linalg.norm(recent_features.semantic_embedding)
            )

            # 综合距离
            combined_distance = (style_distance + semantic_distance) / 2
            distances.append(combined_distance)

        avg_distance = np.mean(distances)
        return min(1.0, avg_distance)

    def add_selected_material(self, features: MaterialFeatures):
        """添加被选中的素材到历史记录"""
        self.recent_materials.append(features)

    def reset_history(self):
        """重置历史记录"""
        self.recent_materials.clear()


class EnhancedMaterialMatcher:
    """增强素材匹配引擎"""

    def __init__(self, cache_manager: CacheManager, database_manager: DatabaseManager):
        self.cache_manager = cache_manager
        self.database_manager = database_manager

        # 子组件
        self.user_preference_model = UserPreferenceModel(cache_manager)
        self.semantic_matcher = SemanticMatcher()
        self.diversity_manager = DiversityManager()

        # 匹配参数
        self.weights = {
            'semantic': 0.35,      # 语义权重
            'style': 0.25,         # 风格权重
            'quality': 0.20,       # 质量权重
            'preference': 0.15,    # 用户偏好权重
            'diversity': 0.05      # 多样性权重
        }

        # 缓存
        self.material_features_cache = {}

    async def match_materials(self, context: MatchingContext,
                            candidate_materials: List[Dict[str, Any]],
                            top_k: int = 10) -> List[MatchResult]:
        """匹配素材"""
        try:
            print(f"🔍 Starting material matching for query: {context.query_text[:50]}...")

            # 1. 提取查询embedding
            query_embedding = await self.semantic_matcher.extract_query_embedding(context.query_text)

            # 2. 获取用户偏好
            user_preferences = {}
            if context.user_id:
                user_preferences = await self.user_preference_model.get_user_preferences(context.user_id)

            # 3. 批量计算匹配分数
            match_results = []
            for material in candidate_materials:
                try:
                    features = await self._extract_material_features(material)
                    if features:
                        match_result = await self._calculate_match_score(
                            context, query_embedding, features, user_preferences, material
                        )
                        match_results.append(match_result)
                except Exception as e:
                    print(f"❌ Error processing material {material.get('id', 'unknown')}: {e}")
                    continue

            # 4. 排序和多样性调整
            sorted_results = await self._rank_and_diversify(
                match_results, context.matching_strategy, top_k
            )

            # 5. 更新多样性历史
            for result in sorted_results[:top_k]:
                self.diversity_manager.add_selected_material(result.features)

            print(f"✅ Material matching completed: {len(sorted_results)} results")
            return sorted_results[:top_k]

        except Exception as e:
            print(f"❌ Material matching failed: {e}")
            return []

    async def _extract_material_features(self, material: Dict[str, Any]) -> Optional[MaterialFeatures]:
        """提取素材特征"""
        material_id = material.get('id')
        if not material_id:
            return None

        # 检查缓存
        cache_key = f"material_features:{material_id}"
        cached_features = await self.cache_manager.get(cache_key)

        if cached_features:
            features_dict = json.loads(cached_features)
            return self._deserialize_features(features_dict)

        try:
            # 模拟特征提取 - 在实际实现中会调用真实的特征提取模型
            semantic_embedding = np.random.randn(512)
            semantic_embedding = semantic_embedding / np.linalg.norm(semantic_embedding)

            visual_features = np.random.randn(256)
            visual_features = visual_features / np.linalg.norm(visual_features)

            # 从素材元数据推断风格
            style_type = self._infer_style_type(material)
            style_vector = StyleVector(
                style_type=style_type,
                color_palette=material.get('color_palette', ['#000000']),
                saturation=np.random.uniform(0.3, 0.9),
                brightness=np.random.uniform(0.3, 0.8),
                contrast=np.random.uniform(0.4, 0.9),
                texture_complexity=np.random.uniform(0.2, 0.8),
                motion_intensity=np.random.uniform(0.1, 0.7),
                camera_stability=np.random.uniform(0.5, 0.9)
            )

            features = MaterialFeatures(
                semantic_embedding=semantic_embedding,
                visual_features=visual_features,
                audio_features=np.random.randn(128) if material.get('type') == 'audio' else None,
                style_vector=style_vector,
                quality_score=material.get('quality_score', np.random.uniform(0.5, 0.95)),
                popularity_score=material.get('popularity_score', np.random.uniform(0.1, 0.8)),
                freshness_score=self._calculate_freshness_score(material),
                metadata=material.get('metadata', {})
            )

            # 缓存特征
            features_dict = self._serialize_features(features)
            await self.cache_manager.set(cache_key, json.dumps(features_dict), expire=1800)

            return features

        except Exception as e:
            print(f"❌ Feature extraction failed for material {material_id}: {e}")
            return None

    def _infer_style_type(self, material: Dict[str, Any]) -> StyleType:
        """从素材元数据推断风格类型"""
        tags = material.get('tags', [])
        title = material.get('title', '').lower()
        description = material.get('description', '').lower()

        # 简单的关键词匹配推断
        if any(keyword in title or keyword in description for keyword in ['anime', 'cartoon', '动漫']):
            return StyleType.ANIME
        elif any(keyword in title or keyword in description for keyword in ['cinematic', 'movie', '电影']):
            return StyleType.CINEMATIC
        elif any(keyword in title or keyword in description for keyword in ['cyber', 'neon', '科技']):
            return StyleType.CYBERPUNK
        elif any(keyword in title or keyword in description for keyword in ['documentary', '纪录']):
            return StyleType.DOCUMENTARY
        elif any(keyword in title or keyword in description for keyword in ['ad', 'commercial', '广告']):
            return StyleType.ADVERTISEMENT
        else:
            return StyleType.REALISTIC  # 默认风格

    def _calculate_freshness_score(self, material: Dict[str, Any]) -> float:
        """计算新鲜度评分"""
        upload_date = material.get('upload_date')
        if not upload_date:
            return 0.5

        try:
            upload_timestamp = datetime.fromisoformat(upload_date.replace('Z', '+00:00'))
            now = datetime.now()
            days_old = (now - upload_timestamp).days

            # 新鲜度递减函数
            if days_old <= 7:
                return 1.0
            elif days_old <= 30:
                return 0.8
            elif days_old <= 90:
                return 0.6
            elif days_old <= 365:
                return 0.4
            else:
                return 0.2

        except Exception:
            return 0.5

    async def _calculate_match_score(self, context: MatchingContext,
                                   query_embedding: np.ndarray,
                                   features: MaterialFeatures,
                                   user_preferences: Dict[str, Any],
                                   material: Dict[str, Any]) -> MatchResult:
        """计算匹配分数"""
        scores = {}

        # 1. 语义相似度
        semantic_score = await self.semantic_matcher.calculate_semantic_similarity(
            query_embedding, features.semantic_embedding
        )

        # 加入关键词重叠度
        keyword_overlap = self.semantic_matcher.calculate_keyword_overlap(
            context.query_text, features.metadata
        )
        semantic_score = semantic_score * 0.8 + keyword_overlap * 0.2

        scores['semantic'] = semantic_score

        # 2. 风格一致性
        style_score = 1.0
        if context.style_anchor:
            style_distance = features.style_vector.distance(context.style_anchor)
            style_score = max(0.0, 1.0 - style_distance)
        scores['style'] = style_score

        # 3. 质量评分
        quality_score = (
            features.quality_score * 0.6 +
            features.popularity_score * 0.2 +
            features.freshness_score * 0.2
        )
        scores['quality'] = quality_score

        # 4. 用户偏好评分
        preference_score = 0.5
        if user_preferences and context.user_id:
            preference_score = self.user_preference_model.calculate_preference_score(
                user_preferences, features
            )
        scores['preference'] = preference_score

        # 5. 多样性评分
        diversity_score = self.diversity_manager.calculate_diversity_score(features)
        scores['diversity'] = diversity_score

        # 6. 计算最终评分
        final_score = sum(
            scores[component] * self.weights[component]
            for component in scores
        )

        # 应用匹配策略调整
        final_score = self._apply_strategy_adjustment(
            final_score, scores, context.matching_strategy
        )

        # 生成解释
        explanation = self._generate_explanation(scores, context.matching_strategy)

        return MatchResult(
            material_id=material['id'],
            confidence_score=final_score,
            relevance_score=semantic_score,
            style_consistency=style_score,
            quality_factor=quality_score,
            diversity_bonus=diversity_score,
            final_score=final_score,
            explanation=explanation,
            features=features
        )

    def _apply_strategy_adjustment(self, base_score: float, scores: Dict[str, float],
                                 strategy: MatchingStrategy) -> float:
        """根据匹配策略调整评分"""
        if strategy == MatchingStrategy.SEMANTIC_FIRST:
            # 语义优先：提升语义权重
            return base_score + (scores['semantic'] - 0.5) * 0.2

        elif strategy == MatchingStrategy.STYLE_FIRST:
            # 风格优先：提升风格权重
            return base_score + (scores['style'] - 0.5) * 0.2

        elif strategy == MatchingStrategy.USER_PREFERENCE:
            # 用户偏好优先：提升偏好权重
            return base_score + (scores['preference'] - 0.5) * 0.3

        elif strategy == MatchingStrategy.DIVERSITY_BOOST:
            # 多样性增强：大幅提升多样性权重
            return base_score + (scores['diversity'] - 0.5) * 0.4

        else:  # BALANCED
            return base_score

    def _generate_explanation(self, scores: Dict[str, float],
                            strategy: MatchingStrategy) -> str:
        """生成匹配解释"""
        explanations = []

        if scores['semantic'] > 0.8:
            explanations.append("语义高度相关")
        elif scores['semantic'] > 0.6:
            explanations.append("语义较为相关")

        if scores['style'] > 0.8:
            explanations.append("风格完全一致")
        elif scores['style'] > 0.6:
            explanations.append("风格基本一致")

        if scores['quality'] > 0.8:
            explanations.append("高质量素材")

        if scores['preference'] > 0.7:
            explanations.append("符合用户偏好")

        if scores['diversity'] > 0.7:
            explanations.append("增加内容多样性")

        if strategy == MatchingStrategy.SEMANTIC_FIRST:
            explanations.insert(0, "语义优先匹配")
        elif strategy == MatchingStrategy.STYLE_FIRST:
            explanations.insert(0, "风格优先匹配")

        return "，".join(explanations) if explanations else "基于综合评分匹配"

    async def _rank_and_diversify(self, results: List[MatchResult],
                                strategy: MatchingStrategy, top_k: int) -> List[MatchResult]:
        """排序并应用多样性调整"""
        # 首先按评分排序
        sorted_results = sorted(results, key=lambda x: x.final_score, reverse=True)

        # 如果启用多样性增强，重新排序
        if strategy == MatchingStrategy.DIVERSITY_BOOST and len(sorted_results) > top_k:
            diversified_results = []
            remaining_results = sorted_results.copy()

            # 选择第一个（最高分）
            if remaining_results:
                diversified_results.append(remaining_results.pop(0))

            # 后续选择时考虑多样性
            while len(diversified_results) < top_k and remaining_results:
                best_candidate = None
                best_score = -1

                for candidate in remaining_results:
                    # 计算与已选择素材的多样性
                    diversity_bonus = 0
                    for selected in diversified_results:
                        diversity_bonus += candidate.features.style_vector.distance(
                            selected.features.style_vector
                        )

                    # 综合评分 = 原始评分 + 多样性奖励
                    combined_score = candidate.final_score + diversity_bonus * 0.1

                    if combined_score > best_score:
                        best_score = combined_score
                        best_candidate = candidate

                if best_candidate:
                    diversified_results.append(best_candidate)
                    remaining_results.remove(best_candidate)

            return diversified_results

        return sorted_results

    def _serialize_features(self, features: MaterialFeatures) -> Dict[str, Any]:
        """序列化特征以供缓存"""
        return {
            'semantic_embedding': features.semantic_embedding.tolist(),
            'visual_features': features.visual_features.tolist(),
            'audio_features': features.audio_features.tolist() if features.audio_features is not None else None,
            'style_vector': {
                'style_type': features.style_vector.style_type.value,
                'color_palette': features.style_vector.color_palette,
                'saturation': features.style_vector.saturation,
                'brightness': features.style_vector.brightness,
                'contrast': features.style_vector.contrast,
                'texture_complexity': features.style_vector.texture_complexity,
                'motion_intensity': features.style_vector.motion_intensity,
                'camera_stability': features.style_vector.camera_stability
            },
            'quality_score': features.quality_score,
            'popularity_score': features.popularity_score,
            'freshness_score': features.freshness_score,
            'metadata': features.metadata
        }

    def _deserialize_features(self, features_dict: Dict[str, Any]) -> MaterialFeatures:
        """反序列化特征"""
        style_data = features_dict['style_vector']
        style_vector = StyleVector(
            style_type=StyleType(style_data['style_type']),
            color_palette=style_data['color_palette'],
            saturation=style_data['saturation'],
            brightness=style_data['brightness'],
            contrast=style_data['contrast'],
            texture_complexity=style_data['texture_complexity'],
            motion_intensity=style_data['motion_intensity'],
            camera_stability=style_data['camera_stability']
        )

        return MaterialFeatures(
            semantic_embedding=np.array(features_dict['semantic_embedding']),
            visual_features=np.array(features_dict['visual_features']),
            audio_features=np.array(features_dict['audio_features']) if features_dict['audio_features'] else None,
            style_vector=style_vector,
            quality_score=features_dict['quality_score'],
            popularity_score=features_dict['popularity_score'],
            freshness_score=features_dict['freshness_score'],
            metadata=features_dict['metadata']
        )

    async def update_user_feedback(self, user_id: str, material_id: str,
                                 action: str, context: MatchingContext):
        """更新用户反馈"""
        if user_id and material_id in self.material_features_cache:
            features = self.material_features_cache[material_id]
            await self.user_preference_model.update_preferences(
                user_id, material_id, action, features
            )

    async def get_matching_analytics(self, session_id: str) -> Dict[str, Any]:
        """获取匹配分析数据"""
        return {
            'session_id': session_id,
            'total_queries': len(self.semantic_matcher.embedding_cache),
            'cache_hit_rate': 0.85,  # 模拟数据
            'average_response_time': 120,  # ms
            'diversity_stats': {
                'recent_materials_count': len(self.diversity_manager.recent_materials),
                'diversity_window': self.diversity_manager.diversity_window
            },
            'performance_metrics': {
                'semantic_matching_time': 45,  # ms
                'feature_extraction_time': 200,  # ms
                'ranking_time': 30  # ms
            }
        }