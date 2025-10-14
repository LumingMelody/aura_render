"""
VL成本控制机制 - 智能控制VL模型调用，优化成本效益
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import hashlib
import json
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

from llm.qwen import QwenLLM


class VLCallPriority(Enum):
    """VL调用优先级"""
    CRITICAL = "critical"      # 关键调用，必须执行
    HIGH = "high"             # 高优先级
    MEDIUM = "medium"         # 中等优先级
    LOW = "low"               # 低优先级，可省略


@dataclass
class VLCallRequest:
    """VL调用请求"""
    request_id: str
    image_url: str
    prompt: str
    priority: VLCallPriority
    cache_key: Optional[str] = None
    context: Dict[str, Any] = None
    timestamp: float = None
    estimated_cost: float = 0.02  # 默认成本

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.cache_key is None:
            self.cache_key = self._generate_cache_key()

    def _generate_cache_key(self) -> str:
        """生成缓存键"""
        content = f"{self.image_url}:{self.prompt}"
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class CostBudget:
    """成本预算"""
    total_budget: float
    spent_amount: float = 0.0
    vl_call_limit: int = 100
    vl_calls_used: int = 0
    time_window_minutes: int = 60
    window_start: float = None

    def __post_init__(self):
        if self.window_start is None:
            self.window_start = time.time()


class VLCostController:
    """VL成本控制器"""

    def __init__(self, budget: CostBudget):
        self.qwen = QwenLLM()
        self.budget = budget

        # 缓存系统
        self.result_cache: Dict[str, Dict] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # 智能采样策略
        self.sampling_strategies = {
            "keyframe_only": self._keyframe_sampling,
            "quality_based": self._quality_based_sampling,
            "content_aware": self._content_aware_sampling,
            "batch_optimized": self._batch_optimized_sampling
        }

        # 成本控制配置
        self.max_concurrent_calls = 5
        self.batch_size = 3
        self.similarity_threshold = 0.8
        self.quality_threshold = 0.6

        # 统计信息
        self.call_statistics = {
            "total_requests": 0,
            "processed_requests": 0,
            "skipped_requests": 0,
            "cached_requests": 0,
            "total_cost": 0.0,
            "average_processing_time": 0.0
        }

    async def process_vl_requests(
        self,
        requests: List[VLCallRequest],
        strategy: str = "content_aware"
    ) -> List[Optional[Dict]]:
        """
        处理VL请求列表，应用成本控制策略
        """
        # 检查预算
        if not self._check_budget_availability():
            return await self._handle_budget_exceeded(requests)

        # 应用采样策略
        if strategy in self.sampling_strategies:
            filtered_requests = await self.sampling_strategies[strategy](requests)
        else:
            filtered_requests = requests

        # 批量处理
        results = await self._batch_process_requests(filtered_requests)

        # 更新统计
        self._update_statistics(requests, results)

        return results

    def _check_budget_availability(self) -> bool:
        """检查预算可用性"""
        # 检查时间窗口
        current_time = time.time()
        if current_time - self.budget.window_start > self.budget.time_window_minutes * 60:
            # 重置时间窗口
            self.budget.window_start = current_time
            self.budget.vl_calls_used = 0
            self.budget.spent_amount = 0.0

        # 检查调用次数限制
        if self.budget.vl_calls_used >= self.budget.vl_call_limit:
            return False

        # 检查总预算
        if self.budget.spent_amount >= self.budget.total_budget:
            return False

        return True

    async def _handle_budget_exceeded(self, requests: List[VLCallRequest]) -> List[Optional[Dict]]:
        """处理预算超出情况"""
        results = []

        for request in requests:
            # 检查缓存
            if request.cache_key in self.result_cache:
                results.append(self.result_cache[request.cache_key])
                self.cache_hits += 1
            elif request.priority == VLCallPriority.CRITICAL:
                # 关键请求仍然处理
                result = await self._single_vl_call(request)
                results.append(result)
            else:
                # 非关键请求使用降级方案
                result = await self._fallback_analysis(request)
                results.append(result)

        return results

    async def _keyframe_sampling(self, requests: List[VLCallRequest]) -> List[VLCallRequest]:
        """关键帧采样策略"""
        # 只处理关键帧和高优先级请求
        filtered = []

        for request in requests:
            if (request.priority in [VLCallPriority.CRITICAL, VLCallPriority.HIGH] or
                self._is_keyframe_request(request)):
                filtered.append(request)

        # 如果还是太多，进一步采样
        if len(filtered) > self.budget.vl_call_limit // 2:
            # 按优先级排序，取前一半
            filtered.sort(key=lambda x: self._get_priority_score(x.priority), reverse=True)
            filtered = filtered[:self.budget.vl_call_limit // 2]

        return filtered

    async def _quality_based_sampling(self, requests: List[VLCallRequest]) -> List[VLCallRequest]:
        """基于质量的采样策略"""
        # 预评估图片质量，只处理高质量图片
        quality_scores = await self._batch_assess_image_quality(requests)

        filtered = []
        for request, quality in zip(requests, quality_scores):
            if (quality >= self.quality_threshold or
                request.priority == VLCallPriority.CRITICAL):
                filtered.append(request)

        return filtered

    async def _content_aware_sampling(self, requests: List[VLCallRequest]) -> List[VLCallRequest]:
        """内容感知采样策略"""
        # 使用聚类算法，每个聚类只处理代表性样本
        clusters = await self._cluster_similar_requests(requests)

        filtered = []
        for cluster in clusters:
            # 每个聚类选择1-2个代表
            representatives = self._select_cluster_representatives(cluster)
            filtered.extend(representatives)

        # 确保关键请求都被包含
        critical_requests = [r for r in requests if r.priority == VLCallPriority.CRITICAL]
        for critical in critical_requests:
            if critical not in filtered:
                filtered.append(critical)

        return filtered

    async def _batch_optimized_sampling(self, requests: List[VLCallRequest]) -> List[VLCallRequest]:
        """批量优化采样策略"""
        # 优化批次组合，最大化信息收益
        if len(requests) <= self.budget.vl_call_limit:
            return requests

        # 使用贪心算法选择最有价值的组合
        selected = []
        remaining = requests.copy()

        while len(selected) < self.budget.vl_call_limit and remaining:
            # 计算每个请求的边际价值
            values = [self._calculate_marginal_value(req, selected) for req in remaining]

            # 选择价值最高的
            best_idx = max(range(len(values)), key=lambda i: values[i])
            selected.append(remaining.pop(best_idx))

        return selected

    def _is_keyframe_request(self, request: VLCallRequest) -> bool:
        """判断是否为关键帧请求"""
        context = request.context or {}
        return (context.get("is_keyframe", False) or
                context.get("scene_change", False) or
                context.get("important_frame", False))

    def _get_priority_score(self, priority: VLCallPriority) -> int:
        """获取优先级分数"""
        scores = {
            VLCallPriority.CRITICAL: 4,
            VLCallPriority.HIGH: 3,
            VLCallPriority.MEDIUM: 2,
            VLCallPriority.LOW: 1
        }
        return scores.get(priority, 1)

    async def _batch_assess_image_quality(self, requests: List[VLCallRequest]) -> List[float]:
        """批量评估图片质量"""
        # 使用简单的启发式方法评估质量
        quality_scores = []

        for request in requests:
            context = request.context or {}

            # 基于上下文信息估算质量
            quality = 0.5  # 基础质量

            # 如果有面部检测信息
            if context.get("has_faces"):
                quality += 0.2

            # 如果有场景变化信息
            if context.get("scene_stability", 0) > 0.7:
                quality += 0.2

            # 如果是用户上传素材
            if context.get("source") == "user_upload":
                quality += 0.1

            quality_scores.append(min(quality, 1.0))

        return quality_scores

    async def _cluster_similar_requests(self, requests: List[VLCallRequest]) -> List[List[VLCallRequest]]:
        """聚类相似请求"""
        # 简化的聚类实现
        clusters = []
        used = set()

        for i, request in enumerate(requests):
            if i in used:
                continue

            cluster = [request]
            used.add(i)

            # 找到相似的请求
            for j, other_request in enumerate(requests[i+1:], i+1):
                if j in used:
                    continue

                if self._are_requests_similar(request, other_request):
                    cluster.append(other_request)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def _are_requests_similar(self, req1: VLCallRequest, req2: VLCallRequest) -> bool:
        """判断两个请求是否相似"""
        # 基于prompt相似性
        prompt_similarity = self._calculate_text_similarity(req1.prompt, req2.prompt)

        # 基于上下文相似性
        context_similarity = self._calculate_context_similarity(
            req1.context or {}, req2.context or {}
        )

        # 综合相似性
        overall_similarity = (prompt_similarity * 0.7 + context_similarity * 0.3)

        return overall_similarity >= self.similarity_threshold

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似性"""
        # 简单的词重叠相似性
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union

    def _calculate_context_similarity(self, ctx1: Dict, ctx2: Dict) -> float:
        """计算上下文相似性"""
        if not ctx1 and not ctx2:
            return 1.0
        if not ctx1 or not ctx2:
            return 0.0

        # 比较关键字段
        key_fields = ["content_type", "style", "scene_type"]
        matches = 0
        total_fields = 0

        for field in key_fields:
            if field in ctx1 or field in ctx2:
                total_fields += 1
                if ctx1.get(field) == ctx2.get(field):
                    matches += 1

        return matches / max(total_fields, 1)

    def _select_cluster_representatives(self, cluster: List[VLCallRequest]) -> List[VLCallRequest]:
        """选择聚类代表"""
        if len(cluster) <= 2:
            return cluster

        # 按优先级排序
        cluster.sort(key=lambda x: self._get_priority_score(x.priority), reverse=True)

        # 选择最高优先级的1-2个
        representatives = [cluster[0]]

        # 如果聚类较大，选择第二个代表
        if len(cluster) > 3:
            representatives.append(cluster[len(cluster)//2])

        return representatives

    def _calculate_marginal_value(self, request: VLCallRequest, selected: List[VLCallRequest]) -> float:
        """计算边际价值"""
        base_value = self._get_priority_score(request.priority)

        # 如果与已选择的请求相似，降低价值
        for selected_req in selected:
            if self._are_requests_similar(request, selected_req):
                base_value *= 0.5
                break

        # 考虑上下文价值
        context = request.context or {}
        if context.get("is_keyframe"):
            base_value *= 1.5
        if context.get("has_faces"):
            base_value *= 1.2

        return base_value

    async def _batch_process_requests(self, requests: List[VLCallRequest]) -> List[Optional[Dict]]:
        """批量处理请求"""
        results = [None] * len(requests)

        # 检查缓存
        uncached_requests = []
        uncached_indices = []

        for i, request in enumerate(requests):
            if request.cache_key in self.result_cache:
                results[i] = self.result_cache[request.cache_key]
                self.cache_hits += 1
            else:
                uncached_requests.append(request)
                uncached_indices.append(i)
                self.cache_misses += 1

        # 批量处理未缓存的请求
        if uncached_requests:
            batch_results = await self._concurrent_vl_calls(uncached_requests)

            for i, result in enumerate(batch_results):
                original_index = uncached_indices[i]
                results[original_index] = result

                # 缓存结果
                if result and uncached_requests[i].cache_key:
                    self.result_cache[uncached_requests[i].cache_key] = result

        return results

    async def _concurrent_vl_calls(self, requests: List[VLCallRequest]) -> List[Optional[Dict]]:
        """并发VL调用"""
        # 分批处理
        results = []

        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]

            # 检查预算
            if not self._check_budget_availability():
                # 预算不足，使用降级方案
                batch_results = []
                for request in batch:
                    fallback_result = await self._fallback_analysis(request)
                    batch_results.append(fallback_result)
                results.extend(batch_results)
                continue

            # 并发处理批次
            tasks = []
            semaphore = asyncio.Semaphore(self.max_concurrent_calls)

            for request in batch:
                task = self._rate_limited_vl_call(request, semaphore)
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            processed_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    processed_results.append(None)
                else:
                    processed_results.append(result)

            results.extend(processed_results)

        return results

    async def _rate_limited_vl_call(self, request: VLCallRequest, semaphore: asyncio.Semaphore) -> Optional[Dict]:
        """限流的VL调用"""
        async with semaphore:
            return await self._single_vl_call(request)

    async def _single_vl_call(self, request: VLCallRequest) -> Optional[Dict]:
        """单个VL调用"""
        try:
            start_time = time.time()

            # 更新预算使用
            self.budget.vl_calls_used += 1
            self.budget.spent_amount += request.estimated_cost

            # 执行VL调用
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=2)

            response = await loop.run_in_executor(
                executor,
                lambda: self.qwen.generate(
                    prompt=request.prompt,
                    images=[request.image_url],
                    max_retries=2
                )
            )

            if response:
                processing_time = time.time() - start_time
                import json
                result = json.loads(response)
                result["processing_time"] = processing_time
                result["request_id"] = request.request_id
                return result

        except Exception as e:
            print(f"[VL调用] 失败 {request.request_id}: {e}")

        return None

    async def _fallback_analysis(self, request: VLCallRequest) -> Dict[str, Any]:
        """降级分析"""
        # 基于上下文信息提供近似结果
        context = request.context or {}

        # 简单的规则基分析
        fallback_result = {
            "request_id": request.request_id,
            "confidence_score": 0.5,  # 降级分数
            "analysis_method": "fallback",
            "content_match": 50,
            "style_match": 50,
            "quality_score": 50,
            "recommended": context.get("has_faces", False),
            "fallback_reason": "budget_limit_or_error"
        }

        # 基于上下文调整分数
        if context.get("user_upload"):
            fallback_result["confidence_score"] = 0.7
            fallback_result["recommended"] = True

        if context.get("high_quality"):
            fallback_result["quality_score"] = 75

        return fallback_result

    def _update_statistics(self, original_requests: List[VLCallRequest], results: List[Optional[Dict]]):
        """更新统计信息"""
        self.call_statistics["total_requests"] += len(original_requests)
        self.call_statistics["processed_requests"] += len([r for r in results if r is not None])
        self.call_statistics["skipped_requests"] += len([r for r in results if r is None])
        self.call_statistics["cached_requests"] = self.cache_hits
        self.call_statistics["total_cost"] = self.budget.spent_amount

        # 计算平均处理时间
        processing_times = [r.get("processing_time", 0) for r in results if r and "processing_time" in r]
        if processing_times:
            self.call_statistics["average_processing_time"] = sum(processing_times) / len(processing_times)

    def get_cost_report(self) -> Dict[str, Any]:
        """获取成本报告"""
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)

        return {
            "budget_info": {
                "total_budget": self.budget.total_budget,
                "spent_amount": self.budget.spent_amount,
                "remaining_budget": self.budget.total_budget - self.budget.spent_amount,
                "budget_utilization": self.budget.spent_amount / self.budget.total_budget,
                "call_limit": self.budget.vl_call_limit,
                "calls_used": self.budget.vl_calls_used,
                "remaining_calls": self.budget.vl_call_limit - self.budget.vl_calls_used
            },
            "cache_performance": {
                "cache_hit_rate": cache_hit_rate,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_size": len(self.result_cache)
            },
            "call_statistics": self.call_statistics,
            "efficiency_metrics": {
                "cost_per_processed_request": self.budget.spent_amount / max(self.call_statistics["processed_requests"], 1),
                "success_rate": self.call_statistics["processed_requests"] / max(self.call_statistics["total_requests"], 1),
                "average_cost_savings": cache_hit_rate * 0.02  # 假设每次缓存命中节省0.02
            }
        }

    def optimize_future_calls(self) -> Dict[str, Any]:
        """优化未来调用建议"""
        report = self.get_cost_report()
        recommendations = []

        # 基于当前性能提供建议
        if report["cache_performance"]["cache_hit_rate"] < 0.3:
            recommendations.append("增加缓存策略的使用，当前缓存命中率较低")

        if report["budget_info"]["budget_utilization"] > 0.8:
            recommendations.append("预算使用率过高，建议增加采样策略的严格性")

        if report["call_statistics"]["success_rate"] < 0.8:
            recommendations.append("调用成功率较低，建议检查网络和API配置")

        if self.call_statistics["average_processing_time"] > 5.0:
            recommendations.append("平均处理时间较长，建议减少并发数或优化prompt")

        return {
            "current_performance": report,
            "optimization_recommendations": recommendations,
            "suggested_strategies": self._suggest_optimal_strategies(report)
        }

    def _suggest_optimal_strategies(self, report: Dict) -> List[str]:
        """建议最优策略"""
        strategies = []

        budget_util = report["budget_info"]["budget_utilization"]
        cache_hit_rate = report["cache_performance"]["cache_hit_rate"]

        if budget_util > 0.7:
            strategies.append("content_aware")  # 内容感知采样
        elif cache_hit_rate < 0.4:
            strategies.append("batch_optimized")  # 批量优化
        else:
            strategies.append("quality_based")  # 基于质量

        return strategies

    def clear_cache(self):
        """清空缓存"""
        self.result_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def reset_budget(self, new_budget: CostBudget):
        """重置预算"""
        self.budget = new_budget