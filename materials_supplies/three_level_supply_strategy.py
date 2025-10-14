"""
三级素材供给策略
优先级：素材库检索 → VL验证 → AI生成补充
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor

from .style_anchor_manager import StyleAnchorManager, StyleVector
from llm.qwen import QwenLLM


class SupplyLevel(Enum):
    """供给级别"""
    LIBRARY_SEARCH = "library_search"  # 素材库检索
    VL_VERIFICATION = "vl_verification"  # VL验证
    AI_GENERATION = "ai_generation"  # AI生成


@dataclass
class MaterialCandidate:
    """素材候选项"""
    material_id: str
    url: str
    thumbnail_url: str
    description: str
    tags: List[str]
    duration: float
    style_info: Optional[Dict] = None
    confidence_score: float = 0.0
    supply_level: SupplyLevel = SupplyLevel.LIBRARY_SEARCH
    verification_result: Optional[Dict] = None


@dataclass
class SupplyRequest:
    """素材供给请求"""
    shot_id: str
    description: str
    duration: float
    required_style: StyleVector
    shot_type: str  # "video", "talking_head", "image"
    context: Dict[str, Any]  # 上下文信息


class ThreeLevelSupplyStrategy:
    """三级素材供给策略"""

    def __init__(self, style_manager: StyleAnchorManager):
        self.style_manager = style_manager
        self.qwen = QwenLLM()

        # API配置
        self.library_api_url = "https://api.material-library.com/v1/search"
        self.vl_verification_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

        # 供给策略配置
        self.max_candidates_per_level = 10
        self.verification_threshold = 0.7
        self.style_compatibility_threshold = 0.6

    async def supply_material(self, request: SupplyRequest) -> Optional[MaterialCandidate]:
        """
        执行三级供给策略
        """
        # Level 1: 素材库检索
        candidates = await self._level1_library_search(request)

        if candidates:
            # Level 2: VL验证
            verified_candidates = await self._level2_vl_verification(request, candidates)

            # 选择最佳候选
            best_candidate = self._select_best_candidate(verified_candidates)
            if best_candidate and best_candidate.confidence_score >= self.verification_threshold:
                return best_candidate

        # Level 3: AI生成
        return await self._level3_ai_generation(request)

    async def _level1_library_search(self, request: SupplyRequest) -> List[MaterialCandidate]:
        """
        Level 1: 素材库检索
        基于标签和描述进行初步匹配
        """
        # 构建搜索查询
        search_params = self._build_search_params(request)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.library_api_url,
                    json=search_params,
                    timeout=15.0
                )

                if response.status_code == 200:
                    results = response.json().get("results", [])
                    candidates = []

                    for result in results[:self.max_candidates_per_level]:
                        candidate = MaterialCandidate(
                            material_id=result["id"],
                            url=result["url"],
                            thumbnail_url=result["thumbnail"],
                            description=result["description"],
                            tags=result.get("tags", []),
                            duration=result.get("duration", request.duration),
                            style_info=result.get("style_info"),
                            supply_level=SupplyLevel.LIBRARY_SEARCH
                        )
                        candidates.append(candidate)

                    return candidates

        except Exception as e:
            print(f"[Level1] 素材库搜索失败: {e}")

        return []

    def _build_search_params(self, request: SupplyRequest) -> Dict[str, Any]:
        """构建搜索参数"""
        # 提取关键词
        keywords = self._extract_keywords(request.description)

        # 风格要求
        style_filters = {
            "style_type": request.required_style.style_type.value,
            "min_saturation": max(0, request.required_style.saturation - 0.2),
            "max_saturation": min(1, request.required_style.saturation + 0.2),
            "min_brightness": max(0, request.required_style.brightness - 0.2),
            "max_brightness": min(1, request.required_style.brightness + 0.2)
        }

        return {
            "query": request.description,
            "keywords": keywords,
            "duration_range": [request.duration * 0.5, request.duration * 2.0],
            "content_type": request.shot_type,
            "style_filters": style_filters,
            "limit": self.max_candidates_per_level,
            "sort_by": "relevance"
        }

    def _extract_keywords(self, description: str) -> List[str]:
        """从描述中提取关键词"""
        # 简单的关键词提取，可以用更复杂的NLP方法
        import re

        # 移除停用词
        stop_words = {"的", "了", "在", "是", "和", "与", "或", "但是", "然后", "接着"}

        # 分词（简单版本）
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', description)
        keywords = [w for w in words if len(w) > 1 and w not in stop_words]

        return keywords[:10]  # 最多10个关键词

    async def _level2_vl_verification(self, request: SupplyRequest, candidates: List[MaterialCandidate]) -> List[MaterialCandidate]:
        """
        Level 2: VL验证
        使用视觉语言模型验证素材匹配度
        """
        verified_candidates = []

        # 并发验证候选素材
        tasks = []
        for candidate in candidates:
            task = self._verify_single_candidate(request, candidate)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for candidate, result in zip(candidates, results):
            if isinstance(result, Exception):
                print(f"[Level2] 验证失败 {candidate.material_id}: {result}")
                # 降级到文本匹配
                candidate.confidence_score = self._fallback_text_matching(request, candidate)
            else:
                candidate.verification_result = result
                candidate.confidence_score = result.get("confidence_score", 0.0)

            if candidate.confidence_score > 0:
                verified_candidates.append(candidate)

        # 按置信度排序
        verified_candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        return verified_candidates

    async def _verify_single_candidate(self, request: SupplyRequest, candidate: MaterialCandidate) -> Dict[str, Any]:
        """
        验证单个候选素材
        """
        # 构建VL验证prompt
        verification_prompt = f"""
        请分析这个视频/图片素材是否符合以下要求：

        【需求描述】: {request.description}
        【要求时长】: {request.duration}秒
        【风格要求】: {request.required_style.style_type.value}

        【候选素材信息】:
        - 描述: {candidate.description}
        - 标签: {', '.join(candidate.tags)}
        - 时长: {candidate.duration}秒

        请分析：
        1. 内容匹配度 (0-100分)
        2. 风格一致性 (0-100分)
        3. 画面质量 (0-100分)
        4. 可用性评估 (是否推荐使用)
        5. 问题说明 (如果有不匹配的地方)

        输出JSON格式：
        {{
            "content_match": 85,
            "style_match": 90,
            "quality_score": 80,
            "recommended": true,
            "confidence_score": 0.85,
            "issues": []
        }}
        """

        try:
            # 使用线程池调用VL模型
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=3)

            response = await loop.run_in_executor(
                executor,
                lambda: self.qwen.generate(
                    prompt=verification_prompt,
                    images=[candidate.thumbnail_url],
                    max_retries=2
                )
            )

            if response:
                import json
                result = json.loads(response)

                # 风格兼容性检查
                if candidate.style_info:
                    style_compatibility = self._check_style_compatibility(
                        candidate.style_info,
                        request.required_style
                    )
                    result["style_compatibility"] = style_compatibility

                    # 调整总分
                    if style_compatibility < self.style_compatibility_threshold:
                        result["confidence_score"] *= 0.7

                return result

        except Exception as e:
            print(f"[VL验证] {candidate.material_id} 失败: {e}")

        # 失败时返回低分
        return {
            "content_match": 0,
            "style_match": 0,
            "quality_score": 0,
            "recommended": False,
            "confidence_score": 0.0,
            "issues": ["VL验证失败"]
        }

    def _check_style_compatibility(self, candidate_style: Dict, required_style: StyleVector) -> float:
        """检查风格兼容性"""
        if not candidate_style:
            return 0.5  # 中性分

        # 简单的风格匹配逻辑
        style_type_match = 1.0 if candidate_style.get("type") == required_style.style_type.value else 0.3

        # 其他特征匹配
        saturation_diff = abs(candidate_style.get("saturation", 0.5) - required_style.saturation)
        brightness_diff = abs(candidate_style.get("brightness", 0.5) - required_style.brightness)

        feature_match = 1.0 - (saturation_diff + brightness_diff) / 2.0

        return (style_type_match * 0.6 + feature_match * 0.4)

    def _fallback_text_matching(self, request: SupplyRequest, candidate: MaterialCandidate) -> float:
        """降级文本匹配"""
        # 简单的关键词匹配
        request_words = set(self._extract_keywords(request.description))
        candidate_words = set(self._extract_keywords(candidate.description))
        candidate_tags = set(candidate.tags)

        # 计算交集
        content_overlap = len(request_words & candidate_words) / max(len(request_words), 1)
        tag_overlap = len(request_words & candidate_tags) / max(len(request_words), 1)

        # 时长匹配
        duration_match = 1.0 - min(abs(candidate.duration - request.duration) / request.duration, 1.0)

        # 综合评分
        score = (content_overlap * 0.4 + tag_overlap * 0.4 + duration_match * 0.2)
        return min(score, 0.6)  # 文本匹配最高0.6

    def _select_best_candidate(self, candidates: List[MaterialCandidate]) -> Optional[MaterialCandidate]:
        """选择最佳候选素材"""
        if not candidates:
            return None

        # 按置信度排序，选择最佳
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)

        best = candidates[0]
        if best.confidence_score >= self.verification_threshold:
            best.supply_level = SupplyLevel.VL_VERIFICATION
            return best

        return None

    async def _level3_ai_generation(self, request: SupplyRequest) -> MaterialCandidate:
        """
        Level 3: AI生成
        当素材库无法满足需求时，触发AI生成
        """
        # 判断生成类型
        if self._is_talking_head_request(request):
            generated = await self._generate_talking_head(request)
        else:
            generated = await self._generate_ai_video(request)

        return MaterialCandidate(
            material_id=f"ai_generated_{request.shot_id}",
            url=generated["url"],
            thumbnail_url=generated.get("thumbnail", ""),
            description=f"AI生成: {request.description}",
            tags=["ai_generated"],
            duration=request.duration,
            confidence_score=0.95,  # AI生成的置信度很高
            supply_level=SupplyLevel.AI_GENERATION,
            verification_result={
                "content_match": 100,
                "style_match": 100,
                "quality_score": 90,
                "recommended": True,
                "generation_info": generated
            }
        )

    def _is_talking_head_request(self, request: SupplyRequest) -> bool:
        """判断是否为数字人口播请求"""
        description = request.description.lower()
        talking_keywords = [
            "讲解", "介绍", "说明", "演讲", "主持", "播报",
            "解说", "阐述", "叙述", "讲述", "表达", "传达"
        ]
        return any(keyword in description for keyword in talking_keywords)

    async def _generate_talking_head(self, request: SupplyRequest) -> Dict[str, Any]:
        """生成数字人视频"""
        # 这里调用数字人生成API
        # 实际实现中会调用真实的API

        script = self._extract_script_from_description(request.description)

        # 模拟生成结果
        return {
            "url": f"https://generated-video.com/talking_{request.shot_id}.mp4",
            "thumbnail": f"https://generated-video.com/talking_{request.shot_id}_thumb.jpg",
            "type": "talking_head",
            "script": script,
            "avatar_used": "default_professional",
            "voice_used": "zh-CN-professional",
            "generation_time": 30.0
        }

    async def _generate_ai_video(self, request: SupplyRequest) -> Dict[str, Any]:
        """生成AI视频"""
        # 获取风格提示词
        style_prompt = self.style_manager.get_style_prompt()

        # 构建生成prompt
        generation_prompt = f"{request.description}, {style_prompt}"

        # 模拟AI视频生成
        return {
            "url": f"https://generated-video.com/ai_{request.shot_id}.mp4",
            "thumbnail": f"https://generated-video.com/ai_{request.shot_id}_thumb.jpg",
            "type": "ai_video",
            "prompt": generation_prompt,
            "style": request.required_style.style_type.value,
            "generation_time": 45.0
        }

    def _extract_script_from_description(self, description: str) -> str:
        """从描述中提取脚本内容"""
        # 简单的脚本提取，实际中可以用更复杂的NLP
        if "说" in description or "讲" in description:
            # 尝试提取引号内容
            import re
            quotes = re.findall(r'["""](.*?)["""]', description)
            if quotes:
                return quotes[0]

        # 如果没有明确的脚本，返回描述本身
        return description

    async def batch_supply(self, requests: List[SupplyRequest]) -> List[Optional[MaterialCandidate]]:
        """
        批量供给素材
        """
        tasks = []
        for request in requests:
            task = self.supply_material(request)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[批量供给] 请求 {requests[i].shot_id} 失败: {result}")
                final_results.append(None)
            else:
                final_results.append(result)

        return final_results

    def get_supply_statistics(self, results: List[Optional[MaterialCandidate]]) -> Dict[str, Any]:
        """获取供给统计信息"""
        if not results:
            return {}

        valid_results = [r for r in results if r is not None]

        # 按供给级别统计
        level_counts = {}
        for result in valid_results:
            level = result.supply_level.value
            level_counts[level] = level_counts.get(level, 0) + 1

        # 计算平均置信度
        avg_confidence = sum(r.confidence_score for r in valid_results) / len(valid_results)

        return {
            "total_requests": len(results),
            "successful_supplies": len(valid_results),
            "success_rate": len(valid_results) / len(results),
            "supply_level_distribution": level_counts,
            "average_confidence": avg_confidence,
            "library_usage_rate": level_counts.get("library_search", 0) / len(valid_results) if valid_results else 0,
            "ai_generation_rate": level_counts.get("ai_generation", 0) / len(valid_results) if valid_results else 0
        }