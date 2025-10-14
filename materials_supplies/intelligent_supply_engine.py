"""
智能素材供给引擎 - 整合所有匹配和供给策略的统一接口
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .enhanced_material_matcher import (
    EnhancedMaterialMatcher, MatchingContext, MatchingStrategy, MatchResult
)
from .three_level_supply_strategy import ThreeLevelSupplyStrategy, SupplyRequest
from .style_anchor_manager import StyleAnchorManager, StyleVector
from .api_clients.material_client_manager import MaterialClientManager, MaterialSearchRequest
from cache.cache_manager import CacheManager
from database.database_manager import DatabaseManager


class SupplyMode(Enum):
    """供给模式"""
    FAST = "fast"               # 快速模式：优先本地缓存和简单匹配
    BALANCED = "balanced"       # 平衡模式：平衡质量和速度
    COMPREHENSIVE = "comprehensive"  # 全面模式：完整的三级供给策略
    INTELLIGENT = "intelligent"  # 智能模式：基于上下文自适应


@dataclass
class SupplyConfig:
    """供给配置"""
    mode: SupplyMode = SupplyMode.BALANCED
    max_results: int = 20
    timeout_seconds: float = 30.0
    enable_style_consistency: bool = True
    enable_user_preferences: bool = True
    enable_diversity_boost: bool = True
    quality_threshold: float = 0.6
    cache_results: bool = True
    fallback_to_ai_generation: bool = True


@dataclass
class SupplyRequest:
    """统一供给请求"""
    query: str                              # 查询文本
    user_id: Optional[str] = None          # 用户ID
    session_id: str = "default"            # 会话ID
    content_type: str = "any"              # 内容类型：image, video, audio, any
    style_reference: Optional[Dict[str, Any]] = None  # 风格参考
    user_materials: List[Dict[str, Any]] = field(default_factory=list)  # 用户提供的素材
    context_metadata: Dict[str, Any] = field(default_factory=dict)  # 上下文元数据
    config: SupplyConfig = field(default_factory=SupplyConfig)


@dataclass
class SupplyResult:
    """供给结果"""
    request_id: str
    materials: List[Dict[str, Any]]        # 匹配的素材
    match_details: List[MatchResult]       # 详细匹配信息
    style_anchor: Optional[StyleVector]    # 使用的风格锚点
    supply_path: List[str]                 # 供给路径（本地->API->生成）
    performance_metrics: Dict[str, Any]    # 性能指标
    recommendations: List[str]             # 推荐建议
    cached: bool = False                   # 是否来自缓存


class IntelligentSupplyEngine:
    """智能素材供给引擎"""

    def __init__(self, cache_manager: CacheManager, database_manager: DatabaseManager):
        self.cache_manager = cache_manager
        self.database_manager = database_manager

        # 核心组件
        self.material_matcher = EnhancedMaterialMatcher(cache_manager, database_manager)
        self.three_level_strategy = ThreeLevelSupplyStrategy()
        self.style_anchor_manager = StyleAnchorManager()
        self.material_client_manager = MaterialClientManager()

        # 性能统计
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'local_matches': 0,
            'api_searches': 0,
            'ai_generations': 0,
            'average_response_time': 0.0
        }

    async def supply_materials(self, request: SupplyRequest) -> SupplyResult:
        """智能素材供给主入口"""
        start_time = datetime.now()
        request_id = f"supply_{start_time.strftime('%Y%m%d_%H%M%S_%f')}"

        try:
            print(f"🎯 Starting intelligent material supply: {request.query[:50]}...")

            # 1. 检查缓存
            cached_result = await self._check_cache(request, request_id)
            if cached_result:
                return cached_result

            # 2. 建立风格锚点
            style_anchor = await self._establish_style_anchor(request)

            # 3. 根据模式选择供给策略
            supply_result = await self._execute_supply_strategy(
                request, request_id, style_anchor
            )

            # 4. 后处理和优化
            optimized_result = await self._optimize_results(supply_result, request)

            # 5. 缓存结果
            if request.config.cache_results:
                await self._cache_result(request, optimized_result)

            # 6. 更新性能统计
            execution_time = (datetime.now() - start_time).total_seconds()
            await self._update_performance_stats(execution_time, optimized_result.supply_path)

            print(f"✅ Material supply completed in {execution_time:.2f}s")
            return optimized_result

        except Exception as e:
            print(f"❌ Material supply failed: {e}")
            # 返回空结果而不是抛出异常
            return SupplyResult(
                request_id=request_id,
                materials=[],
                match_details=[],
                style_anchor=None,
                supply_path=["error"],
                performance_metrics={'error': str(e)},
                recommendations=["请检查查询内容或稍后重试"]
            )

    async def _check_cache(self, request: SupplyRequest, request_id: str) -> Optional[SupplyResult]:
        """检查缓存"""
        if not request.config.cache_results:
            return None

        # 生成缓存键
        cache_key = self._generate_cache_key(request)
        cached_data = await self.cache_manager.get(cache_key)

        if cached_data:
            self.performance_stats['cache_hits'] += 1
            print("📦 Cache hit - returning cached results")

            # 反序列化缓存数据
            cached_result = self._deserialize_cached_result(cached_data, request_id)
            cached_result.cached = True
            return cached_result

        return None

    def _generate_cache_key(self, request: SupplyRequest) -> str:
        """生成缓存键"""
        key_components = [
            request.query.lower().strip(),
            request.content_type,
            str(request.config.quality_threshold),
            str(request.config.max_results),
            request.user_id or "anonymous"
        ]
        return f"supply_cache:{'|'.join(key_components)}"

    async def _establish_style_anchor(self, request: SupplyRequest) -> Optional[StyleVector]:
        """建立风格锚点"""
        if not request.config.enable_style_consistency:
            return None

        try:
            # 如果有用户提供的素材，使用它们建立风格锚点
            if request.user_materials:
                style_anchor = await self.style_anchor_manager.establish_style_anchor_from_materials(
                    request.user_materials
                )
                print(f"🎨 Style anchor established from user materials")
                return style_anchor

            # 如果有风格参考，使用它建立风格锚点
            if request.style_reference:
                style_anchor = await self.style_anchor_manager.establish_style_anchor_from_reference(
                    request.style_reference
                )
                print(f"🎨 Style anchor established from reference")
                return style_anchor

            # 从查询文本推断风格
            style_anchor = await self.style_anchor_manager.infer_style_from_query(request.query)
            print(f"🎨 Style anchor inferred from query")
            return style_anchor

        except Exception as e:
            print(f"⚠️ Failed to establish style anchor: {e}")
            return None

    async def _execute_supply_strategy(self, request: SupplyRequest, request_id: str,
                                     style_anchor: Optional[StyleVector]) -> SupplyResult:
        """执行供给策略"""
        supply_path = []
        all_materials = []
        all_match_details = []

        # 创建匹配上下文
        matching_context = MatchingContext(
            query_text=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            style_anchor=style_anchor,
            previous_materials=[],
            user_preferences={},
            context_metadata=request.context_metadata,
            matching_strategy=self._determine_matching_strategy(request)
        )

        if request.config.mode == SupplyMode.FAST:
            # 快速模式：仅本地搜索
            materials, match_details = await self._local_search(
                request, matching_context
            )
            supply_path.append("local")

        elif request.config.mode == SupplyMode.COMPREHENSIVE:
            # 全面模式：完整三级策略
            materials, match_details, path = await self._three_level_supply(
                request, matching_context
            )
            supply_path.extend(path)

        elif request.config.mode == SupplyMode.INTELLIGENT:
            # 智能模式：自适应策略选择
            materials, match_details, path = await self._intelligent_adaptive_supply(
                request, matching_context
            )
            supply_path.extend(path)

        else:  # BALANCED
            # 平衡模式：本地+API搜索
            materials, match_details, path = await self._balanced_supply(
                request, matching_context
            )
            supply_path.extend(path)

        all_materials.extend(materials)
        all_match_details.extend(match_details)

        # 生成推荐建议
        recommendations = self._generate_recommendations(request, all_match_details)

        return SupplyResult(
            request_id=request_id,
            materials=all_materials,
            match_details=all_match_details,
            style_anchor=style_anchor,
            supply_path=supply_path,
            performance_metrics=self._calculate_performance_metrics(all_match_details),
            recommendations=recommendations
        )

    def _determine_matching_strategy(self, request: SupplyRequest) -> MatchingStrategy:
        """确定匹配策略"""
        if request.config.enable_user_preferences and request.user_id:
            return MatchingStrategy.USER_PREFERENCE
        elif request.config.enable_diversity_boost:
            return MatchingStrategy.DIVERSITY_BOOST
        elif request.config.enable_style_consistency:
            return MatchingStrategy.STYLE_FIRST
        else:
            return MatchingStrategy.BALANCED

    async def _local_search(self, request: SupplyRequest,
                          context: MatchingContext) -> Tuple[List[Dict[str, Any]], List[MatchResult]]:
        """本地搜索"""
        # 模拟本地素材库搜索
        local_materials = await self._simulate_local_materials(request.query)

        if local_materials:
            match_results = await self.material_matcher.match_materials(
                context, local_materials, request.config.max_results
            )
            materials = [self._match_result_to_material(result) for result in match_results]
            return materials, match_results

        return [], []

    async def _three_level_supply(self, request: SupplyRequest,
                                context: MatchingContext) -> Tuple[List[Dict[str, Any]], List[MatchResult], List[str]]:
        """三级供给策略"""
        path = []
        all_materials = []
        all_match_details = []

        # 第一级：本地搜索
        local_materials, local_matches = await self._local_search(request, context)
        if local_materials:
            all_materials.extend(local_materials[:request.config.max_results // 3])
            all_match_details.extend(local_matches[:request.config.max_results // 3])
            path.append("local")
            self.performance_stats['local_matches'] += len(local_materials)

        # 第二级：API搜索
        if len(all_materials) < request.config.max_results:
            api_materials, api_matches = await self._api_search(request, context)
            if api_materials:
                remaining_slots = request.config.max_results - len(all_materials)
                all_materials.extend(api_materials[:remaining_slots])
                all_match_details.extend(api_matches[:remaining_slots])
                path.append("api")
                self.performance_stats['api_searches'] += 1

        # 第三级：AI生成
        if len(all_materials) < request.config.max_results and request.config.fallback_to_ai_generation:
            ai_materials, ai_matches = await self._ai_generation(request, context)
            if ai_materials:
                remaining_slots = request.config.max_results - len(all_materials)
                all_materials.extend(ai_materials[:remaining_slots])
                all_match_details.extend(ai_matches[:remaining_slots])
                path.append("ai_generation")
                self.performance_stats['ai_generations'] += 1

        return all_materials, all_match_details, path

    async def _balanced_supply(self, request: SupplyRequest,
                             context: MatchingContext) -> Tuple[List[Dict[str, Any]], List[MatchResult], List[str]]:
        """平衡供给策略"""
        path = []
        all_materials = []
        all_match_details = []

        # 并行执行本地搜索和API搜索
        local_task = asyncio.create_task(self._local_search(request, context))
        api_task = asyncio.create_task(self._api_search(request, context))

        local_materials, local_matches = await local_task
        api_materials, api_matches = await api_task

        # 合并结果
        if local_materials:
            all_materials.extend(local_materials[:request.config.max_results // 2])
            all_match_details.extend(local_matches[:request.config.max_results // 2])
            path.append("local")

        if api_materials:
            remaining_slots = request.config.max_results - len(all_materials)
            all_materials.extend(api_materials[:remaining_slots])
            all_match_details.extend(api_matches[:remaining_slots])
            path.append("api")

        return all_materials, all_match_details, path

    async def _intelligent_adaptive_supply(self, request: SupplyRequest,
                                         context: MatchingContext) -> Tuple[List[Dict[str, Any]], List[MatchResult], List[str]]:
        """智能自适应供给"""
        # 根据查询复杂度和历史表现自适应选择策略
        query_complexity = self._analyze_query_complexity(request.query)

        if query_complexity < 0.3:  # 简单查询
            return await self._local_search(request, context) + (["local"],)
        elif query_complexity > 0.7:  # 复杂查询
            return await self._three_level_supply(request, context)
        else:  # 中等复杂度
            return await self._balanced_supply(request, context)

    def _analyze_query_complexity(self, query: str) -> float:
        """分析查询复杂度"""
        # 简单的复杂度评估
        factors = []

        # 长度因子
        length_factor = min(len(query) / 100.0, 1.0)
        factors.append(length_factor)

        # 词汇复杂度
        words = query.split()
        unique_words = len(set(words))
        vocab_complexity = min(unique_words / len(words) if words else 0, 1.0)
        factors.append(vocab_complexity)

        # 特殊词汇
        complex_keywords = ['specific', 'detailed', 'professional', 'artistic', 'unique']
        complex_count = sum(1 for keyword in complex_keywords if keyword in query.lower())
        complex_factor = min(complex_count / len(complex_keywords), 1.0)
        factors.append(complex_factor)

        return sum(factors) / len(factors)

    async def _api_search(self, request: SupplyRequest,
                        context: MatchingContext) -> Tuple[List[Dict[str, Any]], List[MatchResult]]:
        """API搜索"""
        try:
            # 使用素材客户端管理器搜索
            search_request = MaterialSearchRequest(
                query=request.query,
                content_type=request.content_type,
                limit=request.config.max_results,
                quality_filter=request.config.quality_threshold
            )

            search_response = await self.material_client_manager.search_materials(search_request)

            if search_response.materials:
                match_results = await self.material_matcher.match_materials(
                    context, search_response.materials, request.config.max_results
                )
                materials = [self._match_result_to_material(result) for result in match_results]
                return materials, match_results

        except Exception as e:
            print(f"⚠️ API search failed: {e}")

        return [], []

    async def _ai_generation(self, request: SupplyRequest,
                           context: MatchingContext) -> Tuple[List[Dict[str, Any]], List[MatchResult]]:
        """AI生成"""
        try:
            # 这里会调用AI生成服务
            # 暂时返回模拟数据
            ai_materials = await self._simulate_ai_generated_materials(request.query)

            if ai_materials:
                match_results = await self.material_matcher.match_materials(
                    context, ai_materials, request.config.max_results
                )
                materials = [self._match_result_to_material(result) for result in match_results]
                return materials, match_results

        except Exception as e:
            print(f"⚠️ AI generation failed: {e}")

        return [], []

    async def _simulate_local_materials(self, query: str) -> List[Dict[str, Any]]:
        """模拟本地素材库"""
        # 返回模拟的本地素材
        return [
            {
                'id': f'local_{i}',
                'title': f'Local material {i} for {query[:20]}',
                'description': f'High quality local material matching {query}',
                'type': 'image',
                'url': f'https://local.storage/material_{i}.jpg',
                'quality_score': 0.8 + (i % 3) * 0.05,
                'tags': query.split()[:3],
                'upload_date': '2024-01-01T00:00:00Z'
            }
            for i in range(min(5, len(query.split())))
        ]

    async def _simulate_ai_generated_materials(self, query: str) -> List[Dict[str, Any]]:
        """模拟AI生成素材"""
        return [
            {
                'id': f'ai_generated_{i}',
                'title': f'AI generated: {query[:30]}',
                'description': f'AI generated content based on: {query}',
                'type': 'image',
                'url': f'https://ai.generated/content_{i}.jpg',
                'quality_score': 0.85,
                'tags': ['ai-generated'] + query.split()[:2],
                'upload_date': datetime.now().isoformat(),
                'metadata': {'generated': True, 'prompt': query}
            }
            for i in range(2)
        ]

    def _match_result_to_material(self, match_result: MatchResult) -> Dict[str, Any]:
        """将匹配结果转换为素材格式"""
        return {
            'id': match_result.material_id,
            'confidence_score': match_result.confidence_score,
            'relevance_score': match_result.relevance_score,
            'explanation': match_result.explanation,
            'features': match_result.features
        }

    async def _optimize_results(self, result: SupplyResult, request: SupplyRequest) -> SupplyResult:
        """优化结果"""
        # 质量过滤
        filtered_materials = []
        filtered_matches = []

        for material, match in zip(result.materials, result.match_details):
            if match.quality_factor >= request.config.quality_threshold:
                filtered_materials.append(material)
                filtered_matches.append(match)

        result.materials = filtered_materials
        result.match_details = filtered_matches

        # 去重
        seen_ids = set()
        unique_materials = []
        unique_matches = []

        for material, match in zip(result.materials, result.match_details):
            if material['id'] not in seen_ids:
                seen_ids.add(material['id'])
                unique_materials.append(material)
                unique_matches.append(match)

        result.materials = unique_materials
        result.match_details = unique_matches

        return result

    def _generate_recommendations(self, request: SupplyRequest,
                                match_details: List[MatchResult]) -> List[str]:
        """生成推荐建议"""
        recommendations = []

        if not match_details:
            recommendations.append("尝试使用更具体的关键词描述")
            recommendations.append("考虑调整质量阈值设置")
            return recommendations

        # 分析匹配质量
        avg_confidence = sum(match.confidence_score for match in match_details) / len(match_details)

        if avg_confidence < 0.5:
            recommendations.append("建议优化查询关键词以获得更相关的结果")

        # 分析多样性
        style_types = [match.features.style_vector.style_type for match in match_details]
        unique_styles = len(set(style_types))

        if unique_styles < 2:
            recommendations.append("启用多样性增强模式以获得更丰富的素材类型")

        # 质量建议
        high_quality_count = sum(1 for match in match_details if match.quality_factor > 0.8)
        if high_quality_count / len(match_details) < 0.5:
            recommendations.append("考虑提高质量阈值以获得更高质量的素材")

        return recommendations

    def _calculate_performance_metrics(self, match_details: List[MatchResult]) -> Dict[str, Any]:
        """计算性能指标"""
        if not match_details:
            return {}

        return {
            'total_matches': len(match_details),
            'average_confidence': sum(match.confidence_score for match in match_details) / len(match_details),
            'average_relevance': sum(match.relevance_score for match in match_details) / len(match_details),
            'average_quality': sum(match.quality_factor for match in match_details) / len(match_details),
            'high_confidence_ratio': sum(1 for match in match_details if match.confidence_score > 0.8) / len(match_details),
            'style_diversity': len(set(match.features.style_vector.style_type for match in match_details))
        }

    async def _cache_result(self, request: SupplyRequest, result: SupplyResult):
        """缓存结果"""
        try:
            cache_key = self._generate_cache_key(request)
            cache_data = self._serialize_result(result)
            await self.cache_manager.set(cache_key, cache_data, expire=3600)  # 1小时过期
        except Exception as e:
            print(f"⚠️ Failed to cache result: {e}")

    def _serialize_result(self, result: SupplyResult) -> str:
        """序列化结果以供缓存"""
        # 简化的序列化，实际实现需要处理复杂对象
        return json.dumps({
            'materials': result.materials,
            'supply_path': result.supply_path,
            'performance_metrics': result.performance_metrics,
            'recommendations': result.recommendations
        })

    def _deserialize_cached_result(self, cached_data: str, request_id: str) -> SupplyResult:
        """反序列化缓存结果"""
        data = json.loads(cached_data)
        return SupplyResult(
            request_id=request_id,
            materials=data['materials'],
            match_details=[],  # 缓存中不包含详细匹配信息
            style_anchor=None,
            supply_path=data['supply_path'],
            performance_metrics=data['performance_metrics'],
            recommendations=data['recommendations'],
            cached=True
        )

    async def _update_performance_stats(self, execution_time: float, supply_path: List[str]):
        """更新性能统计"""
        self.performance_stats['total_requests'] += 1

        # 更新平均响应时间
        current_avg = self.performance_stats['average_response_time']
        total_requests = self.performance_stats['total_requests']
        new_avg = ((current_avg * (total_requests - 1)) + execution_time) / total_requests
        self.performance_stats['average_response_time'] = new_avg

    async def get_performance_statistics(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.performance_stats.copy()

    async def reset_performance_statistics(self):
        """重置性能统计"""
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'local_matches': 0,
            'api_searches': 0,
            'ai_generations': 0,
            'average_response_time': 0.0
        }