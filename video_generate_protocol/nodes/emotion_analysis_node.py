# nodes/emotion_analysis_node.py

from typing import Dict, List, Any, Optional, Tuple
import random
import json
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import time
from functools import wraps

from video_generate_protocol import BaseNode
from video_generate_protocol.prompt_manager import get_prompt_manager
from llm import QwenLLM

# 情感类型定义（可扩展）
EMOTION_CATEGORIES = [
    "激昂", "温馨", "悬疑", "幽默", "悲伤", "励志", "冷静", "浪漫", "恐惧", "感动"
]

# 扩展的情感维度
EMOTION_DIMENSIONS = {
    # 能量维度 (Energy)
    "energy": {
        "high": ["激昂", "幽默", "恐惧"],
        "medium": ["励志", "悬疑", "感动"],
        "low": ["温馨", "悲伤", "冷静", "浪漫"]
    },
    # 情感价值 (Valence)
    "valence": {
        "positive": ["激昂", "温馨", "幽默", "励志", "浪漫", "感动"],
        "neutral": ["冷静", "悬疑"],
        "negative": ["悲伤", "恐惧"]
    },
    # 紧张度 (Tension)
    "tension": {
        "high": ["悬疑", "恐惧", "激昂"],
        "medium": ["励志", "感动"],
        "low": ["温馨", "幽默", "冷静", "浪漫", "悲伤"]
    }
}

# 情感转移矩阵 - 定义情感之间的兼容性
EMOTION_COMPATIBILITY = {
    "激昂": {"励志": 0.8, "感动": 0.6, "幽默": 0.4},
    "温馨": {"浪漫": 0.9, "感动": 0.8, "冷静": 0.7},
    "悬疑": {"恐惧": 0.7, "冷静": 0.6},
    "幽默": {"温馨": 0.6, "激昂": 0.4},
    "悲伤": {"感动": 0.8, "冷静": 0.6},
    "励志": {"激昂": 0.8, "感动": 0.7, "冷静": 0.5},
    "冷静": {"温馨": 0.7, "悲伤": 0.6, "励志": 0.5},
    "浪漫": {"温馨": 0.9, "感动": 0.6},
    "恐惧": {"悬疑": 0.7, "悲伤": 0.5},
    "感动": {"温馨": 0.8, "悲伤": 0.8, "励志": 0.7}
}

@dataclass
class EmotionCurve:
    """情感曲线数据结构"""
    timeline: List[float]  # 时间点
    emotion_values: Dict[str, List[float]]  # 每个情感在各时间点的强度
    peak_moments: List[Tuple[float, str, float]]  # (时间, 情感, 强度)
    transitions: List[Tuple[float, str, str]]  # (时间, 从情感, 到情感)

@dataclass
class EmotionAnalysisResult:
    """情感分析结果"""
    primary_emotions: Dict[str, float]  # 主要情感及权重
    emotion_curve: Optional[EmotionCurve] = None  # 情感曲线
    emotion_tags: List[str] = None  # 情感标签
    confidence_score: float = 0.0  # 置信度
    analysis_method: str = "llm"  # 分析方法
    recommendations: Dict[str, Any] = None  # 建议

# 视频类型 → 情感倾向增强权重（示例配置）
VIDEO_TYPE_EMOTION_BIAS = {
    "产品广告": {"激昂": 0.2, "幽默": 0.15},
    "品牌宣传": {"激昂": 0.25, "励志": 0.2},
    "促销视频": {"激昂": 0.3, "幽默": 0.1},
    "知识讲解": {"冷静": 0.2, "温馨": 0.1},
    "技能教学": {"冷静": 0.15},
    "在线课程": {"冷静": 0.2, "温馨": 0.1},
    "微电影": {"感动": 0.2, "悲伤": 0.15},
    "短视频故事": {"幽默": 0.2, "感动": 0.1},
    "动画短片": {"幽默": 0.25, "浪漫": 0.1},
    "VLOG": {"温馨": 0.3, "幽默": 0.15},
    "社交媒体内容": {"幽默": 0.3, "激昂": 0.1},
    "直播回放": {"幽默": 0.2, "温馨": 0.1},
    "新闻播报": {"冷静": 0.4},
    "访谈节目": {"冷静": 0.2, "感动": 0.15},
    "纪录片": {"冷静": 0.3, "感动": 0.2}
}

# 缓存和重试机制
class EmotionCache:
    """情感分析结果缓存"""
    def __init__(self, max_size: int = 100, ttl: int = 3600):  # TTL: 1小时
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl

    def _generate_key(self, text: str, video_type: str) -> str:
        """生成缓存键"""
        content = f"{text}_{video_type}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, video_type: str) -> Optional[Dict[str, float]]:
        """获取缓存结果"""
        key = self._generate_key(text, video_type)

        if key not in self.cache:
            return None

        # 检查TTL
        if time.time() - self.timestamps[key] > self.ttl:
            self._remove(key)
            return None

        return self.cache[key]

    def set(self, text: str, video_type: str, result: Dict[str, float]):
        """设置缓存"""
        key = self._generate_key(text, video_type)

        # 如果缓存满了，删除最老的条目
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            self._remove(oldest_key)

        self.cache[key] = result.copy()
        self.timestamps[key] = time.time()

    def _remove(self, key: str):
        """删除缓存条目"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)

def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """异步重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        print(f"⚠️ {func.__name__} 第{attempt + 1}次尝试失败: {e}")
                        print(f"⏳ 等待 {wait_time:.1f}s 后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"❌ {func.__name__} 经过{max_attempts}次尝试后最终失败")

            raise last_exception
        return wrapper
    return decorator


class EmotionAnalysisNode(BaseNode):
    required_inputs = [
        {
            "name": "user_description_id",
            "label": "用户原始输入",
            "type": str,
            "required": True,
            "default": "",
            "desc": "用户原始输入",
            "field_type": "textarea"
        },
        {
            "name": "video_type_id",
            "label": "视频类型",
            "type": str,
            "required": True,
            "default": "",
            "desc": "视频类型，如“宣传片”、“VLOG”等"
        },
       
    ]

    output_schema=[
         {
            "name": "emotions_id",
            "label": "情感标签及权重",
            "type": str,
            "required": True,
            "default": "",
            "desc": "情感标签及权重，如 {'励志': 50, '冷静': 30}",
            "field_type": "text"
        },
        {
            "name": "primary_emotion",
            "label": "主要情感",
            "type": str,
            "required": True,
            "default": "冷静",
            "desc": "得分最高的主要情感类别",
            "field_type": "text"
        }

    ]

    file_upload_config = {
        "image": {"enabled": False},
        "video": {"enabled": False},
        "audio": {"enabled": False}
    }

    system_parameters = {
        "max_emotion_labels": 3,
        "min_confidence": 0.05,
        "cache_enabled": True,
        "cache_size": 100,
        "cache_ttl": 3600,  # 1 hour
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "retry_backoff": 2.0
    }

    def __init__(self, node_id: str, name: str = "情感基调分析"):
        super().__init__(node_id=node_id, node_type="emotion_analysis", name=name)
        self._qwen = None  # 懒加载 QwenLLM 实例

        # 初始化缓存
        cache_params = self.system_parameters
        self.cache = EmotionCache(
            max_size=cache_params["cache_size"],
            ttl=cache_params["cache_ttl"]
        ) if cache_params["cache_enabled"] else None

        # 增强配置（临时禁用复杂功能以确保基本功能可用）
        self.enable_curve_analysis = False
        self.enable_music_matching = False
        self.enable_multi_dimensional_analysis = False

        # 性能统计
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "fallback_calls": 0,
            "avg_response_time": 0.0
        }

    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """增强的情感分析生成方法"""
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            self.validate_context(context)  # 异步调用父类方法

            user_description = context["user_description_id"]
            video_type = context["video_type_id"]

            # 获取可选的时长和分镜信息
            total_duration = context.get("total_duration", 60.0)
            shots_info = context.get("shots_info", [])

            # 执行多维情感分析
            analysis_result = await self.analyze_emotions_comprehensive(
                user_description,
                video_type,
                total_duration,
                shots_info
            )
        except Exception as e:
            # 记录错误并使用fallback
            print(f"❌ EmotionAnalysisNode.generate 失败: {e}")
            fallback_emotions = self._fallback_emotion_analysis(context.get("user_description_id", ""))
            analysis_result = EmotionAnalysisResult(
                primary_emotions=fallback_emotions,
                emotion_tags=["冷静"],
                confidence_score=0.3,
                analysis_method="fallback",
                recommendations={}
            )
            self.stats["fallback_calls"] += 1

            # ✅ 输出降级分析结果
            print(f"⚠️ [Node 2] 使用降级情感分析结果:")
            print(f"   情感分布: {fallback_emotions}")
        finally:
            # 更新性能统计
            response_time = time.time() - start_time
            self._update_stats(response_time)

        # 生成情感曲线（如果启用）
        if self.enable_curve_analysis and total_duration > 10:
            analysis_result.emotion_curve = await self.generate_emotion_curve(
                analysis_result.primary_emotions,
                total_duration,
                shots_info
            )

        # 生成音乐匹配建议
        if self.enable_music_matching:
            analysis_result.recommendations = await self.generate_music_recommendations(
                analysis_result.primary_emotions
            )

        # 提取主要情感（得分最高的）
        primary_emotion = max(analysis_result.primary_emotions, key=analysis_result.primary_emotions.get) if analysis_result.primary_emotions else "冷静"

        # ✅ 输出分析结果到日志
        print(f"🎭 [Node 2] 情感基调分析结果:")
        print(f"   主情感: {primary_emotion}")
        print(f"   情感分布: {analysis_result.primary_emotions}")
        print(f"   情感标签: {analysis_result.emotion_tags}")
        print(f"   置信度: {analysis_result.confidence_score:.2f}")
        print(f"   分析方法: {analysis_result.analysis_method}")

        return {
            "emotions_id": analysis_result.primary_emotions,
            "primary_emotion": primary_emotion,  # ✅ 添加主要情感字段
            "emotion_analysis_result": analysis_result,
            "emotion_curve": analysis_result.emotion_curve,
            "music_recommendations": analysis_result.recommendations
        }

    async def analyze_emotions_comprehensive(
        self,
        text: str,
        video_type: str,
        duration: float = 60.0,
        shots_info: List[Dict] = None
    ) -> EmotionAnalysisResult:
        """综合情感分析"""

        # Step 1: 基础情感分析
        base_emotions = await self._enhanced_emotion_analysis(text, video_type)

        # Step 2: 视频类型调整
        adjusted_emotions = self._adjust_emotions_by_video_type(base_emotions, video_type)

        # Step 3: 多维度分析
        if self.enable_multi_dimensional_analysis:
            dimensional_scores = self._calculate_dimensional_scores(adjusted_emotions)
            adjusted_emotions = self._apply_dimensional_constraints(adjusted_emotions, dimensional_scores)

        # Step 4: 分镜级别分析（如果有分镜信息）
        if shots_info:
            shot_emotions = await self._analyze_shot_emotions(shots_info, adjusted_emotions)
        else:
            shot_emotions = None

        # Step 5: 归一化和限制
        final_emotions = self._normalize_and_limit(adjusted_emotions)

        # Step 6: 计算置信度（简化版）
        confidence = 0.8  # 固定置信度

        # Step 7: 生成情感标签（简化版）
        emotion_tags = list(final_emotions.keys())[:3]  # 取前3个情感作为标签

        return EmotionAnalysisResult(
            primary_emotions=final_emotions,
            emotion_tags=emotion_tags,
            confidence_score=confidence,
            analysis_method="comprehensive_llm",
            recommendations={}
        )

    async def _enhanced_emotion_analysis(self, text: str, video_type: str = "") -> Dict[str, float]:
        """增强的情感分析，包含缓存和重试机制"""
        # 检查缓存
        if self.cache:
            cached_result = self.cache.get(text, video_type)
            if cached_result:
                self.stats["cache_hits"] += 1
                print(f"✅ 缓存命中，跳过LLM调用")
                return cached_result

        try:
            # 使用LLM分析（主要方法）
            llm_result = await self._llm_emotion_analysis_with_retry(text)

            # 存储到缓存
            if self.cache:
                self.cache.set(text, video_type, llm_result)

            self.stats["llm_calls"] += 1
            return llm_result

        except Exception as e:
            print(f"❌ Enhanced emotion analysis 失败: {e}")
            fallback_result = self._fallback_emotion_analysis(text)
            self.stats["fallback_calls"] += 1
            return fallback_result

    @async_retry(max_attempts=3, delay=1.0, backoff=2.0)
    async def _llm_emotion_analysis_with_retry(self, text: str) -> Dict[str, float]:
        """带重试机制的LLM情感分析"""
        return await self._llm_emotion_analysis(text)

    async def _llm_emotion_analysis(self, text: str) -> Dict[str, float]:
        """使用LLM进行情感分析，集成专业时代偏好指导"""
        try:
            qwen = self._get_qwen()

            # 获取专业时代偏好指导提示词
            prompt_manager = get_prompt_manager()

            emotion_list = ", ".join(EMOTION_CATEGORIES)
            specific_task = f"""
请深度分析以下文本的情感倾向，不仅要识别明显的情感，还要捕捉隐含的情绪、语境暗示和情感层次。

情感类别：{emotion_list}

分析要求：
1. 每个情感类别都要评分（0-1分，保留2位小数）
2. 考虑文本的深层含义和隐含情感
3. 注意情感的强度和持续性
4. 识别可能的情感转换和层次
5. 输出标准JSON格式，无其他内容

输出格式示例：
{{"激昂": 0.75, "温馨": 0.20, "悬疑": 0.05, "幽默": 0.40, "悲伤": 0.10, "励志": 0.65, "冷静": 0.30, "浪漫": 0.15, "恐惧": 0.00, "感动": 0.80}}

分析文本：
"{text.strip()}"
"""

            # 使用PromptManager增强提示词，融入产品时代偏好分析
            enhanced_prompt = prompt_manager.enhance_prompt(
                specific_task,
                "emotion_analysis",
                context={
                    "product": text,
                    "input": text
                }
            )

            # 使用线程池执行同步方法
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)

            response = await loop.run_in_executor(
                executor,
                lambda: qwen.generate(prompt=enhanced_prompt)
            )

            cleaned_response = self._extract_json_from_text(response)

            if not cleaned_response:
                raise ValueError("未能从响应中提取有效 JSON")

            result = json.loads(cleaned_response)

            # 验证并清洗输出
            scores = {}
            for emotion in EMOTION_CATEGORIES:
                score = result.get(emotion, 0.1)
                try:
                    score = float(score)
                    scores[emotion] = max(0.0, min(1.0, score))
                except (TypeError, ValueError):
                    scores[emotion] = 0.1

            return scores

        except Exception as e:
            print(f"⚠️ LLM 情感分析失败: {e}")
            return self._fallback_emotion_analysis(text)

    def _get_qwen(self):
        """懒加载 QwenLLM 实例"""
        if self._qwen is None:
            try:
                
                self._qwen = QwenLLM()
            except ImportError as e:
                raise ImportError("无法导入 QwenLLM，请确保 qwenllm.py 存在并正确实现。") from e
        return self._qwen

    def _mock_nlp_emotion_analysis(self, text: str) -> Dict[str, float]:
        """
        使用 QwenLLM 进行情感分析，输出 JSON 格式的分数。
        若失败，则降级为关键词匹配 + 随机扰动。
        """
        try:
            qwen = self._get_qwen()

            emotion_list = ", ".join(EMOTION_CATEGORIES)
            prompt = f"""
            你是一个专业的情感分析模型。请分析以下文本的情感倾向，并为以下情感类别分别打分（0-1分，保留1位小数）：
            {emotion_list}

            要求：
            1. 每个情感类别都要评分，不能遗漏。
            2. 分数是独立的，不需要归一化。
            3. 仅输出一个标准的 JSON 对象，不要任何解释、前缀或后缀。
            4. 如果不确定，给0.1-0.3的低分。

            输出格式示例：
            {{"激昂": 0.7, "温馨": 0.2, "悬疑": 0.1, "幽默": 0.5, "悲伤": 0.1, "励志": 0.4, "冷静": 0.3, "浪漫": 0.2, "恐惧": 0.0, "感动": 0.8}}

            文本内容：
            "{text.strip()}"
            """

            response = qwen.generate(prompt=prompt)
            cleaned_response = self._extract_json_from_text(response)

            if not cleaned_response:
                raise ValueError("未能从响应中提取有效 JSON")

            result = json.loads(cleaned_response)

            # 验证并清洗输出
            scores = {}
            for emotion in EMOTION_CATEGORIES:
                score = result.get(emotion, 0.1)
                try:
                    score = float(score)
                    scores[emotion] = max(0.0, min(1.0, score))
                except (TypeError, ValueError):
                    scores[emotion] = 0.1

            return scores

        except Exception as e:
            print(f"⚠️ Qwen 情感分析失败: {e}")
            print("➡️ 使用降级策略：关键词匹配 + 随机扰动")

            # 降级逻辑
            return self._fallback_emotion_analysis(text)

    def _extract_json_from_text(self, text: str) -> str:
        """从可能包含代码块、换行、注释的文本中提取 JSON 字符串"""
        # 去除首尾空白
        text = text.strip()

        # 尝试匹配 ```json ... ``` 或 ``` ... ```
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        # 查找第一个 { 到最后一个 } 之间的内容
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            return ""

        try:
            # 尝试解析截取的内容
            candidate = text[start:end + 1]
            json.loads(candidate)  # 验证是否合法
            return candidate
        except json.JSONDecodeError:
            return ""

    def _fallback_emotion_analysis(self, text: str) -> Dict[str, float]:
        """降级用的情感分析：关键词匹配 + 随机扰动"""
        lower_text = text.lower()
        scores = {emotion: 0.1 for emotion in EMOTION_CATEGORIES}

        rules = [
            (["激动", "热血", "燃", "爆发"], "激昂", 0.5),
            (["温暖", "家", "爱", "陪伴"], "温馨", 0.5),
            (["神秘", "未知", "背后", "真相"], "悬疑", 0.6),
            (["搞笑", "笑死", "段子"], "幽默", 0.4),
            (["感人", "泪目", "坚持"], "感动", 0.5),
            (["悲伤", "难过", "失落"], "悲伤", 0.5),
            (["励志", "奋斗", "梦想"], "励志", 0.4),
            (["冷静", "理性", "分析"], "冷静", 0.5),
            (["浪漫", "爱情", "甜蜜"], "浪漫", 0.4),
            (["恐惧", "害怕", "惊悚"], "恐惧", 0.6),
        ]

        for keywords, emotion, boost in rules:
            if any(word in lower_text for word in keywords):
                scores[emotion] += boost

        # 添加随机扰动
        for k in scores:
            scores[k] += random.uniform(-0.1, 0.2)
            scores[k] = max(0.0, scores[k])

        return scores

    def _adjust_emotions_by_video_type(self, emotions: Dict[str, float], video_type: str) -> Dict[str, float]:
        adjusted = emotions.copy()
        bias = VIDEO_TYPE_EMOTION_BIAS.get(video_type, {})
        for emotion, boost in bias.items():
            if emotion in adjusted:
                adjusted[emotion] *= (1 + boost)
            else:
                adjusted[emotion] = 0.3 * boost
        return adjusted

    def _normalize_and_limit(self, emotions: Dict[str, float]) -> Dict[str, int]:
        total = sum(emotions.values())
        if total == 0:
            return {"冷静": 100}

        percentages = {k: round((v / total) * 100) for k, v in emotions.items() if v > 0}
        min_confidence = self.system_parameters["min_confidence"] * 100
        filtered = {k: v for k, v in percentages.items() if v >= min_confidence}

        max_labels = self.system_parameters["max_emotion_labels"]
        sorted_emotions = sorted(filtered.items(), key=lambda x: -x[1])
        top_emotions = dict(sorted_emotions[:max_labels])

        new_total = sum(top_emotions.values())
        if new_total == 0:
            return {"冷静": 100}
        final = {k: round((v / new_total) * 100) for k, v in top_emotions.items()}

        # 修正浮点误差
        while sum(final.values()) != 100:
            diff = 100 - sum(final.values())
            key = max(final, key=final.get)
            final[key] += diff

        return final

    def _update_stats(self, response_time: float):
        """更新性能统计"""
        # 更新平均响应时间
        if self.stats["total_requests"] > 1:
            current_avg = self.stats["avg_response_time"]
            n = self.stats["total_requests"] - 1
            self.stats["avg_response_time"] = (current_avg * n + response_time) / self.stats["total_requests"]
        else:
            self.stats["avg_response_time"] = response_time

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        total = self.stats["total_requests"]
        if total == 0:
            return {"message": "暂无统计数据"}

        cache_hit_rate = (self.stats["cache_hits"] / total) * 100 if total > 0 else 0
        return {
            "total_requests": total,
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "llm_calls": self.stats["llm_calls"],
            "fallback_calls": self.stats["fallback_calls"],
            "avg_response_time": f"{self.stats['avg_response_time']:.3f}s"
        }

    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.cache.clear()
            self.cache.timestamps.clear()
            print("✅ 情感分析缓存已清空")

    def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        super().regenerate(context, user_intent)
        override = user_intent.get("emotion_override")
        if override and isinstance(override, dict):
            return {"emotions": {k: v for k, v in override.items() if k in EMOTION_CATEGORIES}}
        return self.generate(context)
    
if __name__ == "__main__":
    import asyncio

    async def test_enhanced_emotion_analysis():
        # 1. 创建情感分析节点实例
        emotion_node = EmotionAnalysisNode(node_id="emotion_1", name="增强情感分析节点")
        print("🚀 测试增强版情感分析节点 (带缓存和重试)")

        # 2. 模拟上游节点传来的上下文（context）
        context = {
            "user_description_id": "我想做一个python的教学视频",
            "video_type_id": "知识讲解"
        }

        # 3. 调用 generate 方法进行情感分析 (第一次)
        try:
            print("\n--- 第一次调用 (将调用LLM) ---")
            result = await emotion_node.generate(context)
            print("✅ 情感分析结果：")
            print(result["emotions_id"])
        except Exception as e:
            print(f"❌ 执行失败：{e}")

        # 4. 第二次调用同样内容 (应该命中缓存)
        try:
            print("\n--- 第二次调用 (应该命中缓存) ---")
            result2 = await emotion_node.generate(context)
            print("✅ 情感分析结果 (缓存)：")
            print(result2["emotions_id"])
        except Exception as e:
            print(f"❌ 执行失败：{e}")

        # 5. 测试不同内容
        print("\n--- 测试产品广告内容 ---")
        context_ad = {
            "user_description_id": "这款手机性能超强，运行速度飞快，游戏体验爆棚，热血沸腾！",
            "video_type_id": "产品广告"
        }
        try:
            ad_result = await emotion_node.generate(context_ad)
            print("🎯 广告情感分析结果：")
            print(ad_result["emotions_id"])
        except Exception as e:
            print(f"❌ 广告分析失败：{e}")

        # 6. 显示性能统计
        print("\n--- 性能统计 ---")
        stats = emotion_node.get_performance_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

        # 7. 测试缓存清理
        print("\n--- 清理缓存 ---")
        emotion_node.clear_cache()

    # 运行异步测试
    asyncio.run(test_enhanced_emotion_analysis())
