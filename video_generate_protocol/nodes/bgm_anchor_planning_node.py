# nodes/bgm_anchor_planning_node.py

from video_generate_protocol import BaseNode
from typing import Dict, List, Any, Optional
import math
import json
import re
import hashlib
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from llm import QwenLLM  # 假设这是你封装好的 Qwen 调用模块

# 音乐类型推荐库（情感 → 音乐风格）
MUSIC_GENRE_MAPPING = {
    "激昂": {
        "genre": "史诗交响乐",
        "bpm_range": (120, 160),
        "instruments": ["鼓组", "铜管", "弦乐群"],
        "example": "Two Steps From Hell 风格"
    },
    "温馨": {
        "genre": "轻音乐 / 钢琴曲",
        "bpm_range": (60, 80),
        "instruments": ["钢琴", "木吉他", "弦乐"],
        "example": "Yiruma 风格"
    },
    "悬疑": {
        "genre": "氛围电子 / 环境音效",
        "bpm_range": (50, 70),
        "instruments": ["低频嗡鸣", "金属敲击", "回声"],
        "example": "Hans Zimmer 悬疑风格"
    },
    "幽默": {
        "genre": "轻快电子 / 搞怪音效",
        "bpm_range": (100, 130),
        "instruments": ["合成器", "卡通音效", "打击乐"],
        "example": "喜剧节目背景音乐"
    },
    "感动": {
        "genre": "抒情弦乐",
        "bpm_range": (70, 90),
        "instruments": ["大提琴", "小提琴", "钢琴"],
        "example": "电影感人片段配乐"
    },
    "励志": {
        "genre": "励志流行",
        "bpm_range": (90, 110),
        "instruments": ["鼓点", "电吉他", "人声和声"],
        "example": "广告常用激励音乐"
    },
    "冷静": {
        "genre": "极简电子 / Lo-fi",
        "bpm_range": (75, 95),
        "instruments": ["合成贝斯", "节拍器", "环境白噪"],
        "example": "科技类视频常用"
    }
}

# 节奏标记 → 剪辑节奏 BPM 映射
PACING_TO_BPM = {
    "fast": 120,
    "normal": 90,
    "slow": 60,
    "freeze": 40
}

# 简化的缓存和重试机制
class BGMCache:
    def __init__(self, max_size: int = 50, ttl: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key: str) -> Optional[List[Dict]]:
        if key not in self.cache:
            return None
        if time.time() - self.timestamps[key] > self.ttl:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
            return None
        return self.cache[key]

    def set(self, key: str, value: List[Dict]):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            self.cache.pop(oldest_key, None)
            self.timestamps.pop(oldest_key, None)
        self.cache[key] = value.copy()
        self.timestamps[key] = time.time()

def async_retry_bgm(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_attempts - 1:
                        print(f"⚠️ BGM规划第{attempt + 1}次尝试失败: {e}")
                        await asyncio.sleep(delay)
                    else:
                        print(f"❌ BGM规划经过{max_attempts}次尝试后失败")
                        raise
            return None
        return wrapper
    return decorator


class BGMAanchorPlanningNode(BaseNode):
    required_inputs = [
        {
        "name": "shot_blocks_id",
        "label": "分镜块列表",
        "type": list,
        "required": True,
        "desc": "分镜块列表，包含视觉描述、字幕、节奏等信息，用于提取情感与音乐锚点",
        "field_type": "json"
        }
    ]

    output_schema=[
         {
            "name":"bgm_tracks_id",
            "label": "BGM分镜块列表",
            "type": "list[dict]",
            "desc": "BGM分镜块列表，每个元素包含 start_time, end_time, mood, genre, instruments, narrative_role 等",
            "required": True,
            "schema": {
                "start_time": {"type": "float", "description": "片段起始时间（秒）"},
                "end_time": {"type": "float", "description": "片段结束时间（秒）"},
                "mood": {"type": "str", "description": "情绪，如 温馨、励志、冷静"},
                "genre": {"type": "str", "description": "音乐类型，如 轻音乐 / 钢琴曲"},
                "bpm": {"type": "int", "description": "节奏速度"},
                "instruments": {"type": "list[str]", "description": "主要乐器"},
                "transition": {"type": "str", "description": "过渡方式：淡入、渐强、交叉淡化等"},
                "narrative_role": {"type": "str", "description": "该段音乐在叙事中的作用"},
                "segment_index": {"type": "int", "description": "片段索引"},
                "recommended_track": {
                    "type": "dict",
                    "description": "推荐曲目信息",
                    "fields": {
                        "title": {"type": "str"},
                        "artist": {"type": "str"},
                        "reason": {"type": "str"}
                    }
                }
            }
        },
        {
            "name":"narrative_arc_id",
            "type": str,
            "desc": "整体叙事结构分析，如 英雄之旅、三幕剧 等",
            "required": False,
            "default": ""
        }
    ]

    file_upload_config = {
        "image": {"enabled": False},
        "video": {"enabled": False},
        "audio": {"enabled": False}  # 当前节点不上传音频
    }

    system_parameters = {
        "highlight_threshold": 0.7,  # 情感强度超过此值视为“高潮”
        "transition_window": 1.5,    # 转场前后时间窗口（秒）
        "bpm_tolerance": 10          # BPM 匹配容差
    }

    # output_schema= {
    #         "bgm_recommendation": {
    #             "primary_emotion": str,
    #             "genre": str,
    #             "bpm_range": list,
    #             "instruments": list,
    #             "example": str,
    #             "confidence": float,
    #             "reason": str
    #         },
    #         "music_anchors": [
    #             {
    #                 "time": float,
    #                 "type": str,
    #                 "label": str,
    #                 "duration": float,
    #                 "confidence": float
    #             }
    #         ],
    #         "rhythm_match_score": float  # 0-100
    #     }

    def __init__(self, node_id: str, name: str = "BGM锚点规划"):
        super().__init__(node_id=node_id, node_type="bgm_anchor_planning", name=name)

        # 初始化缓存和统计
        self.cache = BGMCache(max_size=50, ttl=3600)
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "fallback_calls": 0,
            "avg_response_time": 0.0
        }

    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_context(context)

        shot_blocks: List[Dict] = context["shot_blocks_id"]

        # 1. 生成 BGM 分镜（带时间轴）
        storyboard = self._generate_bgm_storyboard(shot_blocks)

        # 2. 为每个段落注入 Qwen 推荐的真实曲目
        final_tracks = self._enrich_with_reference_tracks(storyboard)

        return {
            "bgm_tracks_id": final_tracks,
            "narrative_arc_id": self._infer_narrative_arc(shot_blocks)
        }
    
    def _generate_bgm_storyboard(self, shot_blocks: List[Dict]) -> List[Dict]:
        """
        使用 Qwen 分析每个 shot 的语义，并生成对应的 BGM 建议段落
        """
        prompt = (
            "你是一个影视配乐策划专家。请根据以下视频分镜的视觉描述和字幕，"
            "为每个镜头段落生成一段 BGM 建议，包括：情绪、音乐风格、BPM、主要乐器、过渡方式和叙事作用。\n\n"
            "输出格式为 JSON 列表，每个元素包含字段：\n"
            "- start_time: 开始时间（秒）\n"
            "- end_time: 结束时间\n"
            "- mood: 情绪（从：激昂、温馨、悬疑、幽默、感动、励志、冷静 中选择）\n"
            "- genre: 音乐风格\n"
            "- bpm: 建议 BPM\n"
            "- instruments: 乐器列表\n"
            "- transition: 与上一段的过渡方式（如：淡入、渐强、突变、交叉淡化）\n"
            "- narrative_role: 此段音乐在叙事中的作用（一句话）\n\n"
            "分镜数据：\n"
        )

        for block in shot_blocks:
            start_time = sum(b["duration"] for b in shot_blocks if b is block)  # 累计时间
            # 实际中你可能需要先计算累计时间
            pass

        # 先计算时间轴
        current_time = 0.0
        segments = []
        for block in shot_blocks:
            duration = block["duration"]
            end_time = current_time + duration
            segments.append({
                "start_time": current_time,
                "end_time": end_time,
                "visual_description": block["visual_description"],
                "caption": block["caption"],
                "pacing": block["pacing"]
            })
            current_time = end_time

        # 构造 prompt
        for seg in segments:
            prompt += (
                f"时间 {seg['start_time']:.1f}-{seg['end_time']:.1f}s:\n"
                f"视觉: {seg['visual_description']}\n"
                f"字幕: {seg['caption']}\n"
                f"节奏: {seg['pacing']}\n\n"
            )

        prompt += (
            "请输出一个 JSON 列表，严格按照上述字段，不要额外解释。"
        )

        # 调用 Qwen
        qwen = QwenLLM()
        try:
            response = qwen.generate(prompt=prompt)
            import json, re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if json_match:
                storyboard = json.loads(json_match.group())
                # 补全乐器和 BPM 范围（查表）
                for item in storyboard:
                    mood = item.get("mood", "冷静")
                    genre_info = MUSIC_GENRE_MAPPING.get(mood, MUSIC_GENRE_MAPPING["冷静"])
                    item["genre"] = genre_info["genre"]
                    item["instruments"] = genre_info["instruments"]
                    item["bpm"] = sum(genre_info["bpm_range"]) // 2
                return storyboard
            else:
                raise ValueError("未提取到 JSON")
        except Exception as e:
            print(f"[警告] Qwen 生成失败，使用默认 BGM 结构: {e}")
            # 降级方案：统一用温馨风格
            total_duration = sum(b["duration"] for b in shot_blocks)
            return [{
                "start_time": 0.0,
                "end_time": total_duration,
                "mood": "温馨",
                "genre": "轻音乐 / 钢琴曲",
                "bpm": 70,
                "instruments": ["钢琴", "弦乐"],
                "transition": "淡入",
                "narrative_role": "通用背景音乐"
            }]
        
    def _enrich_with_reference_tracks(self, storyboard: List[Dict]) -> List[Dict]:
        """
        为每个 BGM 段落添加 recommended_track 字段（由 Qwen 推荐）
        如果失败，recommended_track 为 None，便于后续降级
        """
        if not storyboard:
            return []

        prompt = (
            "你是一个资深音乐策划，熟悉大量影视、广告、独立音乐作品。\n"
            "请为以下每个 BGM 段落推荐一首最匹配的 **真实存在** 的背景音乐。\n\n"
            "要求：\n"
            "- 必须是真实歌曲或知名作曲家作品（如 Ludovico Einaudi）\n"
            "- 输出字段：title, artist, reason\n"
            "- 不要虚构曲目！\n"
            "- 如果不确定，reason 写 '风格通用，暂无具体推荐'\n\n"
            "输出格式为 JSON 列表，每个元素包含原有字段 + recommended_track 字段：\n"
            "{\n"
            '  "segment_index": 0,\n'
            '  "start_time": 0.0,\n'
            '  ...所有原有字段...,\n'
            '  "recommended_track": {\n'
            '    "title": "River Flows in You",\n'
            '    "artist": "Yiruma",\n'
            '    "reason": "温柔钢琴契合温馨情绪"\n'
            '  }\n'
            '}\n\n'
            "BGM 段落列表：\n"
        )

        # 构建输入（带索引）
        for i, seg in enumerate(storyboard):
            seg_copy = seg.copy()
            seg_copy["segment_index"] = i
            prompt += json.dumps(seg_copy, ensure_ascii=False, indent=2) + ",\n"

        prompt += "\n请输出完整的 JSON 列表，保持原有字段不变，只增加 recommended_track 字段。"

        qwen = QwenLLM()
        try:
            response = qwen.generate(prompt=prompt)
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # 验证字段完整性
                for item in result:
                    if "recommended_track" not in item:
                        item["recommended_track"] = None
                return result
            else:
                # 降级：所有段落 recommended_track = None
                for seg in storyboard:
                    seg["segment_index"] = storyboard.index(seg)
                    seg["recommended_track"] = None
                return storyboard
        except Exception as e:
            print(f"[警告] 曲目注入失败: {e}")
            for seg in storyboard:
                seg["segment_index"] = storyboard.index(seg)
                seg["recommended_track"] = None
            return storyboard
    def _suggest_reference_tracks(self, storyboard: List[Dict]) -> List[Dict]:
        """
        调用 Qwen 为每个 BGM 段落推荐 1-2 首真实存在的、风格匹配的参考曲目
        返回结构化信息（歌曲名、艺术家、理由）
        """
        if not storyboard:
            return []

        prompt = (
            "你是一个资深音乐总监，熟悉大量影视、广告、独立音乐作品。\n"
            "请根据以下 BGM 段落描述（情绪、风格、BPM、叙事作用），为每个段落推荐 1-2 首最合适的 **真实存在** 的背景音乐。\n"
            "要求：\n"
            "- 必须是真实歌曲或知名作曲家作品（如 Hans Zimmer）\n"
            "- 包含歌曲名、艺术家/作曲家\n"
            "- 一句话说明推荐理由（结合情绪与场景）\n"
            "- 不要虚构曲目！\n\n"
            "输出格式为 JSON 列表，每个元素如下：\n"
            "{\n"
            '  "segment_index": 0,\n'
            '  "mood": "温馨",\n'
            '  "recommended_tracks": [\n'
            '    {\n'
            '      "title": "River Flows in You",\n'
            '      "artist": "Yiruma",\n'
            '      "reason": "温柔钢琴旋律契合回顾时刻的温情"\n'
            '    }\n'
            '  ]\n'
            '}\n\n'
            "BGM 分镜数据：\n"
        )

        # 构建输入
        for i, seg in enumerate(storyboard):
            prompt += (
                f"段落 {i+1}（{seg['start_time']:.1f}s - {seg['end_time']:.1f}s）:\n"
                f"情绪: {seg['mood']}\n"
                f"风格: {seg['genre']}\n"
                f"BPM: {seg['bpm']}\n"
                f"叙事作用: {seg['narrative_role']}\n"
                f"过渡方式: {seg.get('transition', '无')}\n\n"
            )

        prompt += "请严格按照上述 JSON 格式输出，不要额外解释。"

        qwen = QwenLLM()
        try:
            response = qwen.generate(prompt=prompt)
            import json, re
            # 提取 JSON 数组
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # 降级：使用原规则推荐
                fallback = []
                for i, seg in enumerate(storyboard):
                    mood = seg["mood"]
                    example = MUSIC_GENRE_MAPPING[mood]["example"]
                    fallback.append({
                        "segment_index": i,
                        "mood": mood,
                        "recommended_tracks": [{
                            "title": "示例曲目",
                            "artist": "未知",
                            "reason": f"基于 {example} 风格的通用推荐"
                        }]
                    })
                return fallback
        except Exception as e:
            print(f"[警告] Qwen 曲目推荐失败: {e}")
            return []


    def _infer_narrative_arc(self, shot_blocks: List[Dict]) -> str:
        captions = " ".join(b.get("caption", "") for b in shot_blocks)
        prompt = f"根据以下字幕序列，判断视频的叙事结构类型：\n{captions}\n\n选项：起承转合、问题-解决、英雄之旅、线性叙述、情感递进"
        qwen = QwenLLM()
        try:
            return qwen.generate(prompt=prompt).strip()
        except:
            return "线性叙述"
        
    def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        支持用户通过自然语言或结构化指令干预 BGM 生成结果
        示例 user_intent:
        - {"rewrite": "把第二段音乐换成更激昂的风格"}
        - {"adjust_segment": {"index": 1, "mood": "激昂", "transition": "突然爆发"}}
        - {"overall_style": "全部用电子乐"}
        """
        super().regenerate(context, user_intent)

        # 获取原始生成结果（用于修改）
        # 注意：我们不能直接调用 self.generate，因为可能已被覆盖
        # 所以我们重新走一遍流程，但在生成后进行 patch
        shot_blocks = context["shot_blocks_id"]

        # 先生成原始 storyboard
        storyboard = self._generate_bgm_storyboard(shot_blocks)

        # === 用户干预处理 ===
        if not user_intent:
            pass  # 无干预
        elif "rewrite" in user_intent:
            # 自然语言重写
            rewrite_prompt = (
                "请根据用户指令调整以下 BGM 分镜建议。保持 JSON 格式不变。\n\n"
                f"用户指令: {user_intent['rewrite']}\n\n"
                "当前 BGM 分镜:\n"
                f"{json.dumps(storyboard, ensure_ascii=False, indent=2)}\n\n"
                "请输出修改后的 BGM 分镜 JSON 列表："
            )
            try:
                qwen = QwenLLM()
                response = qwen.generate(prompt=rewrite_prompt)
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                if json_match:
                    storyboard = json.loads(json_match.group())
            except Exception as e:
                print(f"[警告] 自然语言重写失败，使用原方案: {e}")

        elif "adjust_segment" in user_intent:
            # 结构化修改某一段
            adj = user_intent["adjust_segment"]
            idx = adj["index"]
            if 0 <= idx < len(storyboard):
                seg = storyboard[idx]
                # 更新字段
                if "mood" in adj:
                    mood = adj["mood"]
                    if mood in MUSIC_GENRE_MAPPING:
                        genre_info = MUSIC_GENRE_MAPPING[mood]
                        seg["mood"] = mood
                        seg["genre"] = genre_info["genre"]
                        seg["instruments"] = genre_info["instruments"]
                        seg["bpm"] = sum(genre_info["bpm_range"]) // 2
                if "transition" in adj:
                    seg["transition"] = adj["transition"]
                if "narrative_role" in adj:
                    seg["narrative_role"] = adj["narrative_role"]

        elif "overall_style" in user_intent:
            # 全局风格覆盖
            style_hint = user_intent["overall_style"]
            prompt = f"用户希望整体 BGM 风格改为：{style_hint}。请重写以下 BGM 分镜以符合该风格。\n\n" + \
                    f"原分镜：{json.dumps(storyboard, ensure_ascii=False)}\n\n" + \
                    "输出修改后的 JSON 列表："
            try:
                qwen = QwenLLM()
                response = qwen.generate(prompt=prompt)
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                if json_match:
                    storyboard = json.loads(json_match.group())
            except Exception as e:
                print(f"[警告] 全局风格调整失败: {e}")

        # === 重新生成推荐曲目（可选）


        # 直接复用 _enrich_with_reference_tracks
        final_tracks = self._enrich_with_reference_tracks(storyboard)

        return {
            "bgm_tracks": final_tracks,
            "narrative_arc": self._infer_narrative_arc(context["shot_blocks_id"]),
            "regeneration_applied": True
        }

        # === 返回最终结果
        # return {
        #     "bgm_storyboard": storyboard,
        #     "recommended_tracks": recommended_tracks,
        #     "narrative_arc": self._infer_narrative_arc(shot_blocks),
        #     "regeneration_applied": True  # 标记为再生结果
        # }
    # def _recommend_music_genre(self, shot_blocks: List[Dict]) -> Dict[str, Any]:
    #     """
    #     使用 Qwen 大模型分析 shot_blocks 中的 visual_description 和 caption，
    #     推理整体情感倾向，并推荐匹配的音乐类型。
    #     """
    #     # 构造 prompt
    #     prompt_segments = []
    #     for i, block in enumerate(shot_blocks):
    #         desc = block.get("visual_description", "")
    #         caption = block.get("caption", "")
    #         pacing = block.get("pacing", "常规")
    #         prompt_segments.append(
    #             f"镜头{i+1}:\n"
    #             f"视觉描述: {desc}\n"
    #             f"字幕: {caption}\n"
    #             f"节奏: {pacing}\n"
    #         )

    #     full_prompt = (
    #         "你是一个视频情感与音乐风格分析专家。请根据以下镜头的视觉描述、字幕和节奏，"
    #         "分析整个视频的主情感基调（从以下中选择最匹配的一项：激昂、温馨、悬疑、幽默、感动、励志、冷静），"
    #         "并推荐最适合的音乐风格。\n\n"
    #         + "\n".join(prompt_segments) +
    #         "\n\n请按以下 JSON 格式输出：\n"
    #         "{\n"
    #         '  "primary_emotion": "情感标签",\n'
    #         '  "genre": "推荐音乐风格",\n'
    #         '  "reason": "推荐理由（一句话）"\n'
    #         "}"
    #     )

    #     # 调用 Qwen
    #     qwen = QwenLLM()
    #     try:
    #         raw_response = qwen.generate(prompt=full_prompt)
    #         import json
    #         # 尝试提取 JSON
    #         try:
    #             parsed = json.loads(raw_response)
    #         except json.JSONDecodeError:
    #             # 如果响应包含 JSON 但有额外文本，尝试提取
    #             import re
    #             json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
    #             if json_match:
    #                 parsed = json.loads(json_match.group())
    #             else:
    #                 raise ValueError("无法解析 JSON 响应")

    #         primary_emotion = parsed.get("primary_emotion", "冷静")

    #         # 查表获取详细信息
    #         genre_info = MUSIC_GENRE_MAPPING.get(primary_emotion, MUSIC_GENRE_MAPPING["冷静"])

    #         return {
    #             "primary_emotion": primary_emotion,
    #             "genre": genre_info["genre"],
    #             "bpm_range": genre_info["bpm_range"],
    #             "instruments": genre_info["instruments"],
    #             "example": genre_info["example"],
    #             "confidence": 80,  # 模型推荐置信度
    #             "reason": parsed.get("reason", "AI 根据视觉与文本内容分析得出")
    #         }

    #     except Exception as e:
    #         print(f"[警告] Qwen 推荐失败，使用默认音乐风格: {e}")
    #         default = MUSIC_GENRE_MAPPING["冷静"]
    #         return {
    #             "primary_emotion": "冷静",
    #             "genre": default["genre"],
    #             "bpm_range": default["bpm_range"],
    #             "instruments": default["instruments"],
    #             "example": default["example"],
    #             "confidence": 50,
    #             "reason": "AI 分析失败，使用默认风格"
    #         }
    # def _extract_music_anchors(self, shot_blocks: List[Dict]) -> List[Dict]:
    #     """提取音乐锚点：高潮、转场、节奏变化点"""
    #     anchors = []
    #     threshold = self.system_parameters["highlight_threshold"]
    #     window = self.system_parameters["transition_window"]

    #     for i, block in enumerate(shot_blocks):
    #         start_time = block["start_time"]
    #         end_time = block["end_time"]
    #         segment = block["segment"]
    #         pacing = block.get("pacing", "normal")
    #         emotion_hint = block.get("emotion_hint", "")

    #         # 1. 高潮点：情感为“激昂”“感动”且时长足够
    #         if emotion_hint in ["激昂", "感动", "励志"] and (end_time - start_time) >= 3.0:
    #             strength = 0.8 if emotion_hint == "激昂" else 0.7
    #             if strength >= threshold:
    #                 anchors.append({
    #                     "time": round(start_time, 1),
    #                     "type": "高潮进入",
    #                     "label": f"{emotion_hint}情绪爆发",
    #                     "duration": round(end_time - start_time, 1),
    #                     "confidence": strength
    #                 })

    #         # 2. 转场点：段落切换处
    #         if i > 0 and shot_blocks[i-1]["segment"] != segment:
    #             transition_time = start_time
    #             anchors.append({
    #                 "time": round(transition_time, 1),
    #                 "type": "转场点",
    #                 "label": f"从 '{shot_blocks[i-1]['segment']}' 切换到 '{segment}'",
    #                 "duration": 0,
    #                 "confidence": 0.9
    #             })

    #         # 3. 快剪开始/结束
    #         if pacing == "快剪" and (i == 0 or shot_blocks[i-1].get("pacing") != "快剪"):
    #             anchors.append({
    #                 "time": round(start_time, 1),
    #                 "type": "节奏加速",
    #                 "label": "快剪开始",
    #                 "confidence": 0.8
    #             })

    #         if pacing == "慢镜头" and (i == 0 or shot_blocks[i-1].get("pacing") != "慢镜头"):
    #             anchors.append({
    #                 "time": round(start_time, 1),
    #                 "type": "节奏放缓",
    #                 "label": "慢镜头开始",
    #                 "confidence": 0.75
    #             })

    #     # 去重并按时间排序
    #     unique_anchors = {a["time"]: a for a in anchors}
    #     return sorted(unique_anchors.values(), key=lambda x: x["time"])

    # def _calculate_bpm_match(self, shot_blocks: List[Dict], bgm_rec: Dict[str, Any]) -> float:
    #     """计算剪辑节奏与推荐BPM的匹配度（0-100）"""
    #     recommended_bpm = sum(bgm_rec["bpm_range"]) / 2  # 取中值
    #     total_weight = 0
    #     weighted_deviation = 0

    #     for block in shot_blocks:
    #         duration = block["end_time"] - block["start_time"]
    #         pacing = block.get("pacing", "normal")
    #         clip_bpm = PACING_TO_BPM.get(pacing, 90)

    #         # 偏差（绝对值）
    #         deviation = abs(clip_bpm - recommended_bpm)
    #         tolerance = self.system_parameters["bpm_tolerance"]

    #         # 匹配度评分：在容差内为100，线性下降
    #         if deviation <= tolerance:
    #             score = 100
    #         else:
    #             score = max(0, 100 - (deviation - tolerance) * 5)

    #         weighted_deviation += score * duration
    #         total_weight += duration

    #     if total_weight == 0:
    #         return 50

    #     final_score = weighted_deviation / total_weight
    #     return final_score

    # def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
    #     """支持用户干预，如：“高潮提前到15秒”"""
    #     super().regenerate(context, user_intent)

    #     override = user_intent.get("bgm_override")
    #     if override and isinstance(override, dict):
    #         result = self.generate(context)
    #         anchors = result["music_anchors"]

    #         # 支持时间偏移
    #         if "shift_peak" in override:
    #             shift_time = override["shift_peak"]  # 如：{"shift_peak": 15.0}
    #             for anchor in anchors:
    #                 if anchor["type"] == "高潮进入":
    #                     anchor["time"] = shift_time
    #                     anchor["label"] += "（用户指定）"

    #         result["music_anchors"] = sorted(anchors, key=lambda x: x["time"])
    #         return result


    # def _extract_music_anchors(self, shot_blocks: List[Dict]) -> List[Dict]:
    #     """提取音乐锚点：高潮、转场、节奏变化点"""
    #     anchors = []
    #     threshold = self.system_parameters["highlight_threshold"]
    #     PACING_MAPPING = {
    #         "常规": "normal",
    #         "快剪": "fast",
    #         "慢镜头": "slow",
    #         "冻结": "freeze"
    #     }

    #     # 关键词库
    #     excitement_keywords = ["开启", "开始你的", "轮到你了", "突破", "成功", "激动", "终于", "出发", "启程"]
    #     warm_keywords = ["回顾", "旅程", "感谢", "陪伴", "成长"]

    #     current_time = 0.0
    #     prev_segment = None
    #     prev_pacing = None

    #     for i, block in enumerate(shot_blocks):
    #         duration = block.get("duration", 2.0)
    #         visual_description = block.get("visual_description", "")
    #         caption = block.get("caption", "")
    #         segment = block.get("segment", f"段落{i+1}")
    #         pacing_zh = block.get("pacing", "常规")
    #         pacing = PACING_MAPPING.get(pacing_zh, "normal")

    #         start_time = current_time
    #         end_time = current_time + duration

    #         # === 1. 高潮点：基于关键词判断激励性内容 ===
    #         text_content = (visual_description + " " + caption).lower()
    #         is_peak = any(kw in text_content for kw in excitement_keywords)

    #         if is_peak and duration >= 3.0:
    #             strength = 0.8
    #             if strength >= threshold:
    #                 anchors.append({
    #                     "time": round(start_time, 1),
    #                     "type": "高潮进入",
    #                     "label": f"激励情绪爆发：{caption[:20]}...",
    #                     "duration": round(duration, 1),
    #                     "confidence": strength
    #                 })

    #         # === 2. 转场点 ===
    #         if i > 0 and prev_segment != segment:
    #             anchors.append({
    #                 "time": round(start_time, 1),
    #                 "type": "转场点",
    #                 "label": f"从 '{prev_segment}' 切换到 '{segment}'",
    #                 "duration": 0,
    #                 "confidence": 0.9
    #             })

    #         # === 3. 节奏变化点 ===
    #         if pacing == "fast" and (i == 0 or prev_pacing != "fast"):
    #             anchors.append({
    #                 "time": round(start_time, 1),
    #                 "type": "节奏加速",
    #                 "label": "快剪开始",
    #                 "confidence": 0.8
    #             })

    #         if pacing == "slow" and (i == 0 or prev_pacing != "slow"):
    #             anchors.append({
    #                 "time": round(start_time, 1),
    #                 "type": "节奏放缓",
    #                 "label": "慢镜头开始",
    #                 "confidence": 0.75
    #             })

    #         current_time = end_time
    #         prev_segment = segment
    #         prev_pacing = pacing

    #     # 去重并排序
    #     unique_anchors = {a["time"]: a for a in anchors}
    #     return sorted(unique_anchors.values(), key=lambda x: x["time"])

    # def _calculate_bpm_match(self, shot_blocks: List[Dict], bgm_rec: Dict[str, Any]) -> float:
    #     """计算剪辑节奏与推荐BPM的匹配度（0-100）"""
    #     recommended_bpm = sum(bgm_rec["bpm_range"]) / 2
    #     total_weight = 0.0
    #     weighted_deviation = 0.0

    #     # 中文 pacing → BPM 映射
    #     PACING_TO_BPM = {
    #         "快剪": 120,
    #         "常规": 90,
    #         "正常": 90,
    #         "慢镜头": 60,
    #         "冻结": 40
    #     }

    #     tolerance = self.system_parameters["bpm_tolerance"]

    #     for block in shot_blocks:
    #         duration = block.get("duration", 2.0)
    #         pacing_zh = block.get("pacing", "常规")
    #         clip_bpm = PACING_TO_BPM.get(pacing_zh, 90)

    #         deviation = abs(clip_bpm - recommended_bpm)
    #         if deviation <= tolerance:
    #             score = 100
    #         else:
    #             score = max(0, 100 - (deviation - tolerance) * 5)

    #         weighted_deviation += score * duration
    #         total_weight += duration

    #     return weighted_deviation / total_weight if total_weight > 0 else 50.0
    # def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
    #     """支持用户干预，如：“高潮提前到15秒”"""
    #     super().regenerate(context, user_intent)

    #     override = user_intent.get("bgm_override")
    #     if override and isinstance(override, dict):
    #         result = self.generate(context)
    #         anchors = result["music_anchors"]

    #         # 支持时间偏移
    #         if "shift_peak" in override:
    #             shift_time = override["shift_peak"]  # 如：{"shift_peak": 15.0}
    #             for anchor in anchors:
    #                 if anchor["type"] == "高潮进入":
    #                     anchor["time"] = shift_time
    #                     anchor["label"] += "（用户指定）"

    #         result["music_anchors"] = sorted(anchors, key=lambda x: x["time"])
    #         return result

    #     return self.generate(context)
if __name__ == "__main__":
    shot_blocks = [
        {
            "shot_type": "中景",
            "duration": 8,
            "visual_description": "讲师站在白板前微笑，手指向屏幕上的课程总结要点；背景为明亮温馨的教室。",
            "pacing": "常规",
            "caption": "我们已经走过了这段旅程的关键点。"
        },
        {
            "shot_type": "特写",
            "duration": 4,
            "visual_description": "讲师的手指轻轻触碰笔记本电脑触控板，屏幕上显示‘开始你的项目’字样。",
            "pacing": "慢镜头",
            "caption": "现在轮到你了！"
        },
        {
            "shot_type": "全景",
            "duration": 8,
            "visual_description": "画面切换至一位学生在家中设置好的工作区认真操作电脑，周围环境整洁有序，墙上挂着激励性的海报。",
            "pacing": "常规",
            "caption": "开启你的机器学习之旅吧。"
        }
    ]

    context = {
        "shot_blocks": shot_blocks
        # 不再需要 emotions
    }

    node = BGMAanchorPlanningNode(node_id="bgm_plan_1")
    result = node.generate(context)

    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))