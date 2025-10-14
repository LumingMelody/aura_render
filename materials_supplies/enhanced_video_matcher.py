"""
Enhanced Intelligent Video Matcher
优化版智能视频匹配器 - 基于新的分类体系和AI增强匹配算法
"""
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import math
from collections import defaultdict, Counter

from llm.qwen import QwenLLM
from .material_taxonomy import (
    MaterialMetadata, MediaType, ContentCategory, StyleTag,
    MaterialTagManager, MaterialTaxonomy
)
from .material_download_manager import MaterialStorage


@dataclass
class MatchingContext:
    """匹配上下文"""
    shot_description: str
    shot_duration: float
    content_category: Optional[ContentCategory] = None
    style_preferences: List[StyleTag] = field(default_factory=list)
    quality_requirement: str = "standard"
    user_constraints: Dict[str, Any] = field(default_factory=dict)
    project_theme: str = ""
    target_audience: str = ""


@dataclass
class MatchResult:
    """匹配结果"""
    material_id: str
    local_path: str
    metadata: MaterialMetadata
    match_score: float
    match_reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0
    processing_time: float = 0.0


class SemanticMatcher:
    """语义匹配器 - 基于AI理解的语义匹配"""

    def __init__(self):
        self.qwen = QwenLLM()
        self.cache = {}  # 语义分析缓存

    async def calculate_semantic_similarity(self, description1: str, description2: str) -> float:
        """计算语义相似度"""
        cache_key = f"{hash(description1)}_{hash(description2)}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""
        请分析以下两个描述的语义相似度。

        描述1: {description1}
        描述2: {description2}

        请从以下方面评估相似度：
        1. 主题内容相关性 (40%)
        2. 情感色调匹配度 (20%)
        3. 视觉元素相似性 (25%)
        4. 场景设定匹配度 (15%)

        请返回一个0-1之间的相似度分数，并简要说明理由。
        格式: 分数: 0.85
        理由: 主题相关度高，情感色调匹配
        """

        try:
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            response = await loop.run_in_executor(
                executor,
                lambda: self.qwen.generate(prompt=prompt, max_retries=2)
            )

            if response:
                response_text = str(response).strip()

                # 解析分数
                score = 0.0
                if "分数:" in response_text:
                    score_line = response_text.split("分数:")[1].split("\n")[0]
                    try:
                        score = float(score_line.strip())
                    except:
                        score = 0.5  # 默认中等相似度

                # 缓存结果
                self.cache[cache_key] = score
                return score

        except Exception as e:
            print(f"Semantic similarity calculation failed: {e}")

        return 0.5  # 默认中等相似度


class StyleMatcher:
    """风格匹配器 - 智能风格识别和匹配"""

    def __init__(self):
        self.style_compatibility_matrix = self._build_compatibility_matrix()

    def _build_compatibility_matrix(self) -> Dict[str, Dict[str, float]]:
        """构建风格兼容性矩阵"""
        return {
            StyleTag.CINEMATIC.value: {
                StyleTag.CINEMATIC.value: 1.0,
                StyleTag.REALISTIC.value: 0.8,
                StyleTag.DOCUMENTARY.value: 0.7,
                StyleTag.VINTAGE.value: 0.6,
                StyleTag.MODERN.value: 0.5,
                StyleTag.ANIME.value: 0.2,
                StyleTag.WATERCOLOR.value: 0.3
            },
            StyleTag.REALISTIC.value: {
                StyleTag.REALISTIC.value: 1.0,
                StyleTag.CINEMATIC.value: 0.8,
                StyleTag.DOCUMENTARY.value: 0.9,
                StyleTag.MODERN.value: 0.7,
                StyleTag.VINTAGE.value: 0.5,
                StyleTag.ANIME.value: 0.1,
                StyleTag.CYBERPUNK.value: 0.3
            },
            StyleTag.ANIME.value: {
                StyleTag.ANIME.value: 1.0,
                StyleTag.WATERCOLOR.value: 0.7,
                StyleTag.MODERN.value: 0.6,
                StyleTag.CYBERPUNK.value: 0.5,
                StyleTag.REALISTIC.value: 0.1,
                StyleTag.DOCUMENTARY.value: 0.1,
                StyleTag.CINEMATIC.value: 0.2
            },
            StyleTag.CYBERPUNK.value: {
                StyleTag.CYBERPUNK.value: 1.0,
                StyleTag.MODERN.value: 0.8,
                StyleTag.CINEMATIC.value: 0.6,
                StyleTag.ANIME.value: 0.5,
                StyleTag.REALISTIC.value: 0.3,
                StyleTag.VINTAGE.value: 0.2,
                StyleTag.DOCUMENTARY.value: 0.1
            }
        }

    def calculate_style_compatibility(self, requested_styles: List[StyleTag],
                                    material_styles: List[StyleTag]) -> float:
        """计算风格兼容性"""
        if not requested_styles or not material_styles:
            return 0.5  # 默认中等兼容性

        compatibility_scores = []

        for req_style in requested_styles:
            req_style_value = req_style.value if isinstance(req_style, StyleTag) else req_style
            best_match = 0.0

            for mat_style in material_styles:
                mat_style_value = mat_style.value if isinstance(mat_style, StyleTag) else mat_style

                # 获取兼容性分数
                compatibility = self.style_compatibility_matrix.get(
                    req_style_value, {}
                ).get(mat_style_value, 0.0)

                best_match = max(best_match, compatibility)

            compatibility_scores.append(best_match)

        # 返回平均兼容性
        return sum(compatibility_scores) / len(compatibility_scores)


class ContextualMatcher:
    """上下文匹配器 - 考虑项目整体背景的智能匹配"""

    def __init__(self):
        self.project_theme_keywords = {
            "科技": ["technology", "innovation", "digital", "future", "AI", "robot"],
            "商务": ["business", "office", "meeting", "professional", "corporate"],
            "教育": ["education", "learning", "school", "knowledge", "study"],
            "生活": ["lifestyle", "home", "family", "daily", "personal"],
            "自然": ["nature", "landscape", "outdoor", "environment", "natural"],
            "艺术": ["art", "creative", "design", "aesthetic", "artistic"]
        }

    def calculate_contextual_relevance(self, material_metadata: MaterialMetadata,
                                     context: MatchingContext) -> Tuple[float, List[str]]:
        """计算上下文相关性"""
        relevance_score = 0.0
        reasons = []

        # 1. 项目主题匹配
        theme_score = self._calculate_theme_relevance(material_metadata, context.project_theme)
        relevance_score += theme_score * 0.3
        if theme_score > 0.7:
            reasons.append(f"项目主题高度匹配 ({theme_score:.2f})")

        # 2. 目标受众匹配
        audience_score = self._calculate_audience_relevance(material_metadata, context.target_audience)
        relevance_score += audience_score * 0.2
        if audience_score > 0.6:
            reasons.append(f"目标受众适配 ({audience_score:.2f})")

        # 3. 时长匹配度
        duration_score = self._calculate_duration_fitness(material_metadata, context.shot_duration)
        relevance_score += duration_score * 0.2
        if duration_score > 0.8:
            reasons.append(f"时长高度适配 ({duration_score:.2f})")

        # 4. 质量要求匹配
        quality_score = self._calculate_quality_fitness(material_metadata, context.quality_requirement)
        relevance_score += quality_score * 0.3
        if quality_score > 0.7:
            reasons.append(f"质量要求匹配 ({quality_score:.2f})")

        return min(relevance_score, 1.0), reasons

    def _calculate_theme_relevance(self, metadata: MaterialMetadata, theme: str) -> float:
        """计算主题相关性"""
        if not theme:
            return 0.5

        theme_lower = theme.lower()
        material_keywords = [tag.value.lower() for tag in metadata.tags] + \
                          [kw.lower() for kw in metadata.keywords]

        # 检查主题关键词匹配
        for theme_key, keywords in self.project_theme_keywords.items():
            if theme_key in theme or any(kw in theme_lower for kw in keywords):
                # 计算匹配度
                matches = sum(1 for kw in keywords if any(kw in mk for mk in material_keywords))
                if matches > 0:
                    return min(matches / len(keywords), 1.0)

        return 0.3  # 默认低相关性

    def _calculate_audience_relevance(self, metadata: MaterialMetadata, audience: str) -> float:
        """计算受众匹配度"""
        if not audience:
            return 0.5

        audience_mapping = {
            "专业": ["business", "professional", "corporate"],
            "年轻": ["modern", "trendy", "energetic"],
            "家庭": ["family", "warm", "lifestyle"],
            "学生": ["education", "learning", "academic"],
            "创意": ["creative", "artistic", "innovative"]
        }

        audience_lower = audience.lower()
        material_keywords = [tag.value.lower() for tag in metadata.tags]

        for aud_key, keywords in audience_mapping.items():
            if aud_key in audience:
                matches = sum(1 for kw in keywords if any(kw in mk for mk in material_keywords))
                if matches > 0:
                    return min(matches / len(keywords), 1.0)

        return 0.4

    def _calculate_duration_fitness(self, metadata: MaterialMetadata, required_duration: float) -> float:
        """计算时长适配度"""
        if not metadata.duration or required_duration <= 0:
            return 0.5

        ratio = min(metadata.duration, required_duration) / max(metadata.duration, required_duration)

        # 时长越接近，适配度越高
        if ratio >= 0.9:
            return 1.0
        elif ratio >= 0.7:
            return 0.8
        elif ratio >= 0.5:
            return 0.6
        else:
            return 0.3

    def _calculate_quality_fitness(self, metadata: MaterialMetadata, quality_req: str) -> float:
        """计算质量适配度"""
        quality_hierarchy = {
            "low": 1,
            "standard": 2,
            "high": 3,
            "premium": 4
        }

        req_level = quality_hierarchy.get(quality_req.lower(), 2)
        mat_level = quality_hierarchy.get(metadata.quality_level.value.lower(), 2)

        # 质量等级匹配或超出要求
        if mat_level >= req_level:
            return 1.0
        else:
            return mat_level / req_level


class EnhancedVideoMatcher:
    """增强版智能视频匹配器"""

    def __init__(self, storage: MaterialStorage):
        self.storage = storage
        self.taxonomy = MaterialTaxonomy()
        self.tag_manager = MaterialTagManager()

        # 匹配组件
        self.semantic_matcher = SemanticMatcher()
        self.style_matcher = StyleMatcher()
        self.contextual_matcher = ContextualMatcher()

        # 匹配统计
        self.match_stats = {
            "total_requests": 0,
            "successful_matches": 0,
            "average_match_score": 0.0,
            "processing_time_total": 0.0
        }

        # 匹配缓存
        self.match_cache = {}

    async def find_best_matches(self, context: MatchingContext,
                               max_results: int = 10) -> List[MatchResult]:
        """寻找最佳匹配素材"""
        start_time = time.time()
        self.match_stats["total_requests"] += 1

        try:
            # 检查缓存
            cache_key = self._generate_cache_key(context)
            if cache_key in self.match_cache:
                return self.match_cache[cache_key]

            # 获取候选素材
            candidates = await self._get_candidate_materials(context)

            if not candidates:
                return []

            # 并发计算匹配分数
            match_tasks = [
                self._calculate_match_score(candidate, context)
                for candidate in candidates
            ]

            results = await asyncio.gather(*match_tasks, return_exceptions=True)

            # 过滤有效结果并排序
            valid_results = [
                result for result in results
                if isinstance(result, MatchResult) and result.match_score > 0.3
            ]

            # 按匹配分数排序
            valid_results.sort(key=lambda x: x.match_score, reverse=True)

            # 限制结果数量
            final_results = valid_results[:max_results]

            # 更新统计
            processing_time = time.time() - start_time
            self.match_stats["processing_time_total"] += processing_time

            if final_results:
                self.match_stats["successful_matches"] += 1
                avg_score = sum(r.match_score for r in final_results) / len(final_results)
                self.match_stats["average_match_score"] = (
                    (self.match_stats["average_match_score"] * (self.match_stats["successful_matches"] - 1) + avg_score)
                    / self.match_stats["successful_matches"]
                )

                # 缓存结果
                self.match_cache[cache_key] = final_results

            # 设置处理时间
            for result in final_results:
                result.processing_time = processing_time / len(final_results)

            return final_results

        except Exception as e:
            print(f"Match finding failed: {e}")
            return []

    async def _get_candidate_materials(self, context: MatchingContext) -> List[MaterialMetadata]:
        """获取候选素材"""
        # 获取所有视频素材
        all_materials = self.storage.list_materials(
            media_type=MediaType.VIDEO,
            limit=500  # 限制候选数量以提升性能
        )

        candidates = []
        for material_data in all_materials:
            try:
                if 'parsed_metadata' in material_data:
                    metadata_dict = material_data['parsed_metadata']

                    # 重构MaterialMetadata对象
                    from .material_taxonomy import ContentCategory, StyleTag, QualityLevel, UsageRights
                    metadata = MaterialMetadata(
                        material_id=material_data['material_id'],
                        filename=material_data['filename'],
                        media_type=MediaType(metadata_dict['media_type']),
                        file_size=metadata_dict['file_size'],
                        primary_category=ContentCategory(metadata_dict['primary_category']),
                        quality_level=QualityLevel(metadata_dict['quality_level']),
                        usage_rights=UsageRights(metadata_dict['usage_rights']),
                        duration=metadata_dict.get('duration'),
                        keywords=metadata_dict.get('keywords', []),
                        created_at=datetime.fromisoformat(metadata_dict['created_at'])
                    )

                    # 基础过滤
                    if self._passes_basic_filter(metadata, context):
                        candidates.append(metadata)

            except Exception as e:
                print(f"Error processing material {material_data.get('material_id', 'unknown')}: {e}")
                continue

        return candidates

    def _passes_basic_filter(self, metadata: MaterialMetadata, context: MatchingContext) -> bool:
        """基础过滤条件"""
        # 内容类别过滤
        if context.content_category and metadata.primary_category != context.content_category:
            # 检查是否在次要类别中
            if context.content_category not in metadata.secondary_categories:
                return False

        # 时长过滤 (允许±50%的弹性)
        if metadata.duration and context.shot_duration > 0:
            duration_ratio = metadata.duration / context.shot_duration
            if duration_ratio < 0.5 or duration_ratio > 2.0:
                return False

        # 质量过滤
        quality_hierarchy = {"low": 1, "standard": 2, "high": 3, "premium": 4}
        required_level = quality_hierarchy.get(context.quality_requirement.lower(), 2)
        material_level = quality_hierarchy.get(metadata.quality_level.value.lower(), 2)

        if material_level < required_level:
            return False

        return True

    async def _calculate_match_score(self, metadata: MaterialMetadata,
                                   context: MatchingContext) -> MatchResult:
        """计算匹配分数"""
        try:
            scores = {}
            all_reasons = []

            # 1. 语义相似度 (35%)
            material_description = " ".join(metadata.keywords + [tag.value for tag in metadata.tags])
            semantic_score = await self.semantic_matcher.calculate_semantic_similarity(
                context.shot_description, material_description
            )
            scores["semantic"] = semantic_score * 0.35
            if semantic_score > 0.7:
                all_reasons.append(f"语义高度匹配 ({semantic_score:.2f})")

            # 2. 风格匹配度 (25%)
            style_score = self.style_matcher.calculate_style_compatibility(
                context.style_preferences, metadata.style_tags
            )
            scores["style"] = style_score * 0.25
            if style_score > 0.6:
                all_reasons.append(f"风格匹配 ({style_score:.2f})")

            # 3. 上下文相关性 (30%)
            contextual_score, contextual_reasons = self.contextual_matcher.calculate_contextual_relevance(
                metadata, context
            )
            scores["contextual"] = contextual_score * 0.30
            all_reasons.extend(contextual_reasons)

            # 4. 使用频率加权 (10%)
            popularity_score = self._calculate_popularity_score(metadata)
            scores["popularity"] = popularity_score * 0.10
            if popularity_score > 0.8:
                all_reasons.append(f"热门素材 ({popularity_score:.2f})")

            # 计算总分
            total_score = sum(scores.values())

            # 计算置信度 (基于分数分布的一致性)
            score_values = list(scores.values())
            confidence = 1.0 - (max(score_values) - min(score_values))

            # 获取本地路径
            local_path = self.storage.get_material_path(metadata.material_id)

            return MatchResult(
                material_id=metadata.material_id,
                local_path=local_path or "",
                metadata=metadata,
                match_score=total_score,
                match_reasons=all_reasons,
                confidence=confidence
            )

        except Exception as e:
            print(f"Error calculating match score for {metadata.material_id}: {e}")
            return MatchResult(
                material_id=metadata.material_id,
                local_path="",
                metadata=metadata,
                match_score=0.0,
                match_reasons=[f"计算错误: {str(e)}"],
                confidence=0.0
            )

    def _calculate_popularity_score(self, metadata: MaterialMetadata) -> float:
        """计算素材热门程度分数"""
        # 基于下载次数和评分计算热门程度
        base_score = 0.5

        # 根据访问次数调整
        if metadata.view_count > 100:
            base_score += 0.3
        elif metadata.view_count > 50:
            base_score += 0.2
        elif metadata.view_count > 10:
            base_score += 0.1

        # 根据评分调整
        if metadata.rating > 4.0:
            base_score += 0.2
        elif metadata.rating > 3.0:
            base_score += 0.1

        return min(base_score, 1.0)

    def _generate_cache_key(self, context: MatchingContext) -> str:
        """生成缓存键"""
        key_parts = [
            context.shot_description,
            str(context.shot_duration),
            str(context.content_category),
            str(sorted([s.value for s in context.style_preferences])),
            context.quality_requirement,
            context.project_theme,
            context.target_audience
        ]
        return hash(str(key_parts))

    def get_match_statistics(self) -> Dict[str, Any]:
        """获取匹配统计"""
        avg_processing_time = (
            self.match_stats["processing_time_total"] / max(1, self.match_stats["total_requests"])
        )

        success_rate = (
            self.match_stats["successful_matches"] / max(1, self.match_stats["total_requests"])
        ) * 100

        return {
            "total_requests": self.match_stats["total_requests"],
            "successful_matches": self.match_stats["successful_matches"],
            "success_rate": success_rate,
            "average_match_score": self.match_stats["average_match_score"],
            "average_processing_time": avg_processing_time,
            "cache_size": len(self.match_cache)
        }

    def clear_cache(self):
        """清除匹配缓存"""
        self.match_cache.clear()
        self.semantic_matcher.cache.clear()


# 使用示例和测试
async def test_enhanced_matcher():
    """测试增强版匹配器"""
    print("🧪 测试增强版智能视频匹配器")
    print("=" * 50)

    # 初始化存储和匹配器
    from .material_download_manager import MaterialStorage
    storage = MaterialStorage("/tmp/aura_render_outputs/materials")
    matcher = EnhancedVideoMatcher(storage)

    # 创建测试匹配上下文
    test_contexts = [
        MatchingContext(
            shot_description="现代办公室中的商务会议场景",
            shot_duration=8.0,
            content_category=ContentCategory.BUSINESS,
            style_preferences=[StyleTag.MODERN, StyleTag.REALISTIC],
            quality_requirement="high",
            project_theme="企业宣传",
            target_audience="商务专业人士"
        ),
        MatchingContext(
            shot_description="美丽的自然风景，山峦和湖泊",
            shot_duration=10.0,
            content_category=ContentCategory.NATURE,
            style_preferences=[StyleTag.CINEMATIC, StyleTag.REALISTIC],
            quality_requirement="premium",
            project_theme="自然纪录片",
            target_audience="自然爱好者"
        ),
        MatchingContext(
            shot_description="科技感十足的AI机器人场景",
            shot_duration=6.0,
            content_category=ContentCategory.TECHNOLOGY,
            style_preferences=[StyleTag.CYBERPUNK, StyleTag.MODERN],
            quality_requirement="high",
            project_theme="科技创新",
            target_audience="科技专业人士"
        )
    ]

    # 执行匹配测试
    all_results = []
    for i, context in enumerate(test_contexts):
        print(f"\n🎯 测试场景 {i+1}: {context.shot_description[:30]}...")

        start_time = time.time()
        results = await matcher.find_best_matches(context, max_results=5)
        processing_time = time.time() - start_time

        print(f"   找到 {len(results)} 个匹配结果 (耗时: {processing_time:.2f}s)")

        for j, result in enumerate(results):
            print(f"   {j+1}. {result.material_id}")
            print(f"      匹配分数: {result.match_score:.3f}")
            print(f"      置信度: {result.confidence:.3f}")
            print(f"      匹配原因: {', '.join(result.match_reasons[:2])}")
            if j >= 2:  # 只显示前3个结果
                break

        all_results.extend(results)

    # 显示总体统计
    stats = matcher.get_match_statistics()
    print(f"\n📊 匹配统计:")
    print(f"   总请求数: {stats['total_requests']}")
    print(f"   成功匹配: {stats['successful_matches']}")
    print(f"   成功率: {stats['success_rate']:.1f}%")
    print(f"   平均匹配分数: {stats['average_match_score']:.3f}")
    print(f"   平均处理时间: {stats['average_processing_time']:.3f}s")

    print("\n🎉 增强版匹配器测试完成！")

    return {
        "test_results": all_results,
        "statistics": stats
    }


if __name__ == "__main__":
    asyncio.run(test_enhanced_matcher())