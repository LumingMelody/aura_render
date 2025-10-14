# video_type_identification_node.py

import json
from typing import Dict, Any, List, Tuple, Optional
import logging
import hashlib
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from video_generate_protocol import BaseNode
from video_generate_protocol.prompt_manager import get_prompt_manager
from llm import QwenLLM  # 假设这是你封装好的 Qwen 调用模块

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义所有支持的视频类型，用于校验
SUPPORTED_VIDEO_TYPES = {
    # 商业类
    "产品广告", "品牌宣传", "促销视频",
    # 教育类
    "知识讲解", "技能教学", "在线课程",
    # 娱乐类
    "微电影", "短视频故事", "动画短片",
    # 社交类
    "VLOG", "社交媒体内容", "直播回放",
    # 专业类
    "新闻播报", "访谈节目", "纪录片"
}

# 缓存和重试机制
class VideoTypeCache:
    """视频类型识别结果缓存"""
    def __init__(self, max_size: int = 100, ttl: int = 3600):  # TTL: 1小时
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl

    def _generate_key(self, theme: str, keywords: List[str], duration: int) -> str:
        """生成缓存键"""
        content = f"{theme}_{','.join(sorted(keywords))}_{duration}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, theme: str, keywords: List[str], duration: int) -> Optional[Tuple[str, Dict[str, str]]]:
        """获取缓存结果"""
        key = self._generate_key(theme, keywords, duration)

        if key not in self.cache:
            return None

        # 检查TTL
        if time.time() - self.timestamps[key] > self.ttl:
            self._remove(key)
            return None

        return self.cache[key]

    def set(self, theme: str, keywords: List[str], duration: int, result: Tuple[str, Dict[str, str]]):
        """设置缓存"""
        key = self._generate_key(theme, keywords, duration)

        # 如果缓存满了，删除最老的条目
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            self._remove(oldest_key)

        self.cache[key] = result
        self.timestamps[key] = time.time()

    def _remove(self, key: str):
        """删除缓存条目"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)

def async_retry_video_type(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
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


class VideoTypeIdentificationNode(BaseNode):
    required_inputs = [
        {
            "name": "theme_id",
            "label": "视频主题",
            "type": str,
            "required": True,
            "desc": "视频的主要话题或背景"
        },
        {
            "name": "keywords_id",
            "label": "关键词",
            "type": list,
            "required": True,
            "desc": "与视频内容相关的关键词列表"
        },
        {
            "name": "target_duration_id",
            "label": "目标时长",
            "type": int,
            "required": True,
            "desc": "视频的目标长度（秒）"
        },
        {
            "name": "user_description_id",
            "label": "用户原始输入",
            "type": str,
            "required": True,
            "desc": "用户原始输入"
        }
    ]

    output_schema=[
        {
            "name": "video_type_id",
            "label": "视频类型",
            "type": str,
            "desc": "视频类型，如“宣传片”、“VLOG”等"
        },
        {
            "name": "structure_template_id",
            "label": "视频结构模板",
            "type": str,
            "desc": "视频结构模板"
        }
    ]

    def __init__(self, node_id: str, name: str = "视频类型识别"):
        super().__init__(node_id=node_id, node_type="video_type_identification", name=name)

        # 初始化缓存
        self.cache = VideoTypeCache(max_size=100, ttl=3600)

        # 性能统计
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "fallback_calls": 0,
            "avg_response_time": 0.0
        }

    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行视频类型识别的主入口（增强版）。
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            self.validate_context(context)

            theme = context.get('theme_id')
            keywords = context.get('keywords_id')
            target_duration = context.get('target_duration_id')
            user_description = context.get('user_description_id')

            # DEBUG: Print parameter values
            print(f"🔍 [VIDEO_TYPE_DEBUG] theme={theme} (type: {type(theme)})")
            print(f"🔍 [VIDEO_TYPE_DEBUG] keywords={keywords} (type: {type(keywords)})")
            print(f"🔍 [VIDEO_TYPE_DEBUG] target_duration={target_duration} (type: {type(target_duration)})")
            print(f"🔍 [VIDEO_TYPE_DEBUG] user_description={user_description} (type: {type(user_description)})")

            # 更精确的验证，避免空列表和零值的误判
            missing = []
            if not theme or theme == "":
                missing.append('theme_id')
            if keywords is None:
                missing.append('keywords_id')
            if target_duration is None or target_duration <= 0:
                missing.append('target_duration_id')
            if not user_description or user_description == "":
                missing.append('user_description_id')

            if missing:
                raise ValueError(f"缺少必要输入参数: {missing}")

            # 检查缓存
            cached_result = self.cache.get(theme, keywords, target_duration)
            if cached_result:
                self.stats["cache_hits"] += 1
                print(f"✅ 缓存命中，跳过LLM调用")
                video_type, structure_template = cached_result
            else:
                # 调用大模型分析（带重试）
                video_type, structure_template = await self._analyze_video_theme_with_retry(
                    theme, keywords, target_duration
                )

                # 存储到缓存
                self.cache.set(theme, keywords, target_duration, (video_type, structure_template))
                self.stats["llm_calls"] += 1

            # 输出结果
            return {
                "video_type_id": video_type,
                "structure_template_id": structure_template
            }

        except Exception as e:
            logger.error(f"视频类型识别失败: {str(e)}")
            self.stats["fallback_calls"] += 1
            return {
                "video_type_id": "未知类型",
                "structure_template_id": {
                    "intro": "通用开场",
                    "body": "内容展开",
                    "conclusion": "总结"
                },
                "error": str(e)
            }
        finally:
            # 更新性能统计
            response_time = time.time() - start_time
            self._update_stats(response_time)

    @async_retry_video_type(max_attempts=3, delay=1.0, backoff=2.0)
    async def _analyze_video_theme_with_retry(self, theme: str, keywords: List[str], target_duration: int) -> Tuple[str, Dict[str, str]]:
        """带重试机制的视频主题分析"""
        return await self._analyze_video_theme_async(theme, keywords, target_duration)

    async def _analyze_video_theme_async(self, theme: str, keywords: List[str], target_duration: int) -> Tuple[str, Dict[str, str]]:
        """异步版本的视频主题分析"""
        # 构建提示词
        prompt = f"""
        请根据以下信息分析视频内容，并严格以 JSON 格式输出视频类型和基础结构模板：

        ### 输入信息：
        - **主题**：{theme}
        - **关键词**：{', '.join(keywords)}
        - **目标时长**：{target_duration} 秒

        ### 可选视频类型（请从中选择最合适的）：
        - 商业类：产品广告、品牌宣传、促销视频
        - 教育类：知识讲解、技能教学、在线课程
        - 娱乐类：微电影、短视频故事、动画短片
        - 社交类：VLOG、社交媒体内容、直播回放
        - 专业类：新闻播报、访谈节目、纪录片

        ### 常见结构模板参考：
        - 教程类：问题引入 → 步骤演示 → 总结
        - 宣传片：高潮前置 → 故事展开 → 品牌露出
        - VLOG：日常开场 → 生活记录 → 结尾感想
        - 知识讲解：提出问题 → 分析解释 → 结论

        ### 要求：
        1. 从上述类型中选择一个最匹配的视频类型。
        2. 设计一个适合目标时长的三段式结构：intro（开头）、body（主体）、conclusion（结尾）。
        3. 输出必须是合法 JSON，格式如下：
        {{
        "video_type": "具体类型",
        "structure_template": {{
            "intro": "开头设计",
            "body": "主体内容",
            "conclusion": "结尾设计"
        }}
        }}

        请开始你的分析：
        """

        try:
            # 调用 Qwen 模型（异步）
            qwen = QwenLLM()

            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)

            response = await loop.run_in_executor(
                executor,
                lambda: qwen.generate(prompt=prompt)
            )

            # 解析响应
            raw_text = response.strip()
            print(f"模型原始输出: {raw_text}")

            # 尝试提取 JSON（防止模型输出包含额外文本）
            json_str = self._extract_json(raw_text)
            if not json_str:
                raise ValueError("无法从模型输出中提取有效 JSON")

            result = json.loads(json_str)

            # 校验字段
            video_type = result.get("video_type")
            structure_template = result.get("structure_template")

            if not video_type or not isinstance(structure_template, dict):
                raise ValueError("模型返回格式错误：缺少 video_type 或 structure_template")

            return video_type, structure_template

        except Exception as e:
            logger.error(f"调用 Qwen 模型失败或解析失败: {e}")
            # 返回默认值
            return "未知类型", {
                "intro": "通用开场",
                "body": "内容展开",
                "conclusion": "总结"
            }

    def analyze_video_theme(self, theme: str, keywords: List[str], target_duration: int) -> Tuple[str, Dict[str, str]]:
        """
        调用 Qwen 大模型分析视频主题，返回视频类型和结构模板。
        """
        # 构建提示词
        prompt = f"""
        请根据以下信息分析视频内容，并严格以 JSON 格式输出视频类型和基础结构模板：

        ### 输入信息：
        - **主题**：{theme}
        - **关键词**：{', '.join(keywords)}
        - **目标时长**：{target_duration} 秒

        ### 可选视频类型（请从中选择最合适的）：
        - 商业类：产品广告、品牌宣传、促销视频
        - 教育类：知识讲解、技能教学、在线课程
        - 娱乐类：微电影、短视频故事、动画短片
        - 社交类：VLOG、社交媒体内容、直播回放
        - 专业类：新闻播报、访谈节目、纪录片

        ### 常见结构模板参考：
        - 教程类：问题引入 → 步骤演示 → 总结
        - 宣传片：高潮前置 → 故事展开 → 品牌露出
        - VLOG：日常开场 → 生活记录 → 结尾感想
        - 知识讲解：提出问题 → 分析解释 → 结论

        ### 要求：
        1. 从上述类型中选择一个最匹配的视频类型。
        2. 设计一个适合目标时长的三段式结构：intro（开头）、body（主体）、conclusion（结尾）。
        3. 输出必须是合法 JSON，格式如下：
        {{
        "video_type": "具体类型",
        "structure_template": {{
            "intro": "开头设计",
            "body": "主体内容",
            "conclusion": "结尾设计"
        }}
        }}

        请开始你的分析：
        """

        try:
            # 调用 Qwen 模型
            qwen=QwenLLM()
            response = qwen.generate(
                prompt=prompt
            )

            # 解析响应
            raw_text = response.strip()
            print(f"模型原始输出: {raw_text}")
            # 尝试提取 JSON（防止模型输出包含额外文本）
            json_str = self._extract_json(raw_text)
            if not json_str:
                raise ValueError("无法从模型输出中提取有效 JSON")

            result = json.loads(json_str)

            # 校验字段
            video_type = result.get("video_type")
            structure_template = result.get("structure_template")

            if not video_type or not isinstance(structure_template, dict):
                raise ValueError("模型返回格式错误：缺少 video_type 或 structure_template")

            # （可选）校验类型是否在支持列表中
            # if video_type not in SUPPORTED_VIDEO_TYPES:
            #     logger.warning(f"识别出非标准视频类型: {video_type}，将设为'未知类型'")
            #     video_type = "未知类型"
            #     structure_template = {
            #         "intro": "通用开场",
            #         "body": "内容展开",
            #         "conclusion": "总结"
            #     }

            return video_type, structure_template

        except Exception as e:
            logger.error(f"调用 Qwen 模型失败或解析失败: {e}")
            # 返回默认值
            return "未知类型", {
                "intro": "通用开场",
                "body": "内容展开",
                "conclusion": "总结"
            }

    def _extract_json(self, text: str) -> str:
        """从模型输出中提取 JSON 字符串"""
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            return None
        try:
            # 尝试解析提取的部分
            json_str = text[start:end]
            json.loads(json_str)  # 验证是否合法
            return json_str
        except json.JSONDecodeError:
            return None

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
        self.cache.cache.clear()
        self.cache.timestamps.clear()
        print("✅ 视频类型识别缓存已清空")

    def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据用户意图重新生成内容。
        可在此加入用户反馈，比如“不要广告风格”等。
        当前简化处理，直接重新 generate。
        """
        # TODO: 可根据 user_intent 调整 prompt，例如排除某些类型
        return self.generate(context)
    

if __name__ == "__main__":
    import asyncio

    async def test_enhanced_video_type_identification():
        # 1. 创建视频类型识别节点实例
        node = VideoTypeIdentificationNode("test_node")
        print("🚀 测试增强版视频类型识别节点 (带缓存和重试)")

        # 2. 模拟输入上下文
        context = {
            "theme_id": "教育",
            "keywords_id": ["Python", "机器学习"],
            "target_duration_id": 60,
            "user_description_id": "我想做一个Python机器学习的教育视频"
        }

        # 3. 第一次调用 (将调用LLM)
        try:
            print("\n--- 第一次调用 (将调用LLM) ---")
            result = await node.generate(context)
            print("✅ 视频类型识别结果：")
            print(f"视频类型: {result['video_type_id']}")
            print(f"结构模板: {result['structure_template_id']}")
        except Exception as e:
            print(f"❌ 执行失败：{e}")

        # 4. 第二次调用同样内容 (应该命中缓存)
        try:
            print("\n--- 第二次调用 (应该命中缓存) ---")
            result2 = await node.generate(context)
            print("✅ 视频类型识别结果 (缓存)：")
            print(f"视频类型: {result2['video_type_id']}")
            print(f"结构模板: {result2['structure_template_id']}")
        except Exception as e:
            print(f"❌ 执行失败：{e}")

        # 5. 测试不同内容
        print("\n--- 测试产品广告内容 ---")
        context_ad = {
            "theme_id": "产品宣传",
            "keywords_id": ["手机", "性能", "游戏"],
            "target_duration_id": 30,
            "user_description_id": "这款手机性能超强，运行速度飞快，游戏体验爆棚"
        }
        try:
            ad_result = await node.generate(context_ad)
            print("🎯 广告视频类型识别结果：")
            print(f"视频类型: {ad_result['video_type_id']}")
            print(f"结构模板: {ad_result['structure_template_id']}")
        except Exception as e:
            print(f"❌ 广告分析失败：{e}")

        # 6. 显示性能统计
        print("\n--- 性能统计 ---")
        stats = node.get_performance_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

        # 7. 测试缓存清理
        print("\n--- 清理缓存 ---")
        node.clear_cache()

    # 运行异步测试
    asyncio.run(test_enhanced_video_type_identification())
