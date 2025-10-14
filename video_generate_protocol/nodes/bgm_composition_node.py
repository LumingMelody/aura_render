# nodes/bgm_composition_node.py

from video_generate_protocol import BaseNode
from typing import Dict, List, Any
import random
import math
from dataclasses import dataclass
import requests
from materials_supplies import match_bgm,BGMRequest,BGMResponse
import asyncio


# ==================== 情感强度权重（用于评分）====================
MOOD_WEIGHTS = {
    "激昂": 1.0,
    "励志": 0.9,
    "感动": 0.85,
    "温馨": 0.8,
    "幽默": 0.6,
    "冷静": 0.5,
    "悬疑": 0.75,
    "科技": 0.6
}

# BPM 容差范围
BPM_TOLERANCE = 8  # ±8 BPM

# 默认淡入淡出时间（秒）
DEFAULT_FADE_IN = 3.0
DEFAULT_FADE_OUT = 5.0

# 音乐 API 地址（示例）
MUSIC_SEARCH_API = "https://api.yourbgmservice.com/v1/tracks/search"


@dataclass
class Track:
    id: str
    title: str
    file_path: str
    bpm: float
    duration: float
    genre: List[str]
    mood: List[str]
    key: str
    has_bass_drop: bool

class BGMCompositionNode(BaseNode):
    # 声明 generate 所需的输入结构
    required_inputs = [
        {
            "name":"bgm_tracks_id",
            "label": "BGM分镜块列表",
            "type": List[Dict],
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

    output_schema=[
        {
            "name": "bgm_composition_id",
            "label": "BGM合成结果列表",
            "type": list,
            "required": True,
            "desc": "包含每段匹配的音乐资源，如 [{'segment_index': 0, 'start_time': 0.0, 'end_time': 10.0, 'mood': '温馨', 'genre': '轻音乐', 'narrative_role': '开场', 'transition': '淡入', 'music_suggestion': {'title': '轻松的早晨', 'artist': '轻音乐大师', 'reason': '适合开场的温馨氛围'}, 'matched_audio': {...}, 'alternatives': [...] }]",
            "field_type": "json"
        },
        {
            "name": "total_music_duration_id",
            "label": "总音乐时长",
            "type": float,
            "required": False,
            "desc": "音乐总时长，单位为秒",
            "field_type": "text"
        }
    ]

    def __init__(self, node_id: str, name: str = "BGM合成（分镜驱动）"):
        self.node_id = node_id
        self.name = name

    # async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     主生成函数：输入为 bgm_tracks 分镜块，输出为每段匹配的音乐资源
    #     """
    #     bgm_tracks = context.get("bgm_tracks")
    #     narrative_arc = context.get("narrative_arc", "")

    #     if not bgm_tracks:
    #         raise ValueError("缺少 bgm_tracks 分镜数据")

    #     result_segments = []

    #     # 并发地为每个分镜段请求音乐
    #     tasks = [self._fetch_music_for_segment(segment) for segment in bgm_tracks]
    #     music_results = await asyncio.gather(*tasks)

    #     for segment, matches in zip(bgm_tracks, music_results):
    #         primary_match = matches[0] if matches else None

    #         result_segment = {
    #             "segment_index": segment["segment_index"],
    #             "start_time": segment["start_time"],
    #             "end_time": segment["end_time"],
    #             "mood": segment["mood"],
    #             "genre": segment["genre"],
    #             "narrative_role": segment["narrative_role"],
    #             "transition": segment["transition"],
    #             "music_suggestion": {
    #                 "title": segment["recommended_track"]["title"],
    #                 "artist": segment["recommended_track"]["artist"],
    #                 "reason": segment["recommended_track"]["reason"]
    #             },
    #             "matched_audio": primary_match.dict() if primary_match else None,
    #             "alternatives": [m.dict() for m in matches[1:]] if matches and len(matches) > 1 else []
    #         }
    #         result_segments.append(result_segment)

    #     return {
    #         # "status": "success",
    #         # "narrative_arc": narrative_arc,
    #         "bgm_composition_id": result_segments,
    #         "total_music_duration_id": max((seg["end_time"] for seg in bgm_tracks), default=0),
    #         # "timestamp": asyncio.get_event_loop().time()
    #     }

    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        主生成函数：输入为 bgm_tracks 分镜块，输出为符合标准格式的 BGM 音轨
        """
        self.validate_context(context)
        bgm_tracks = context.get("bgm_tracks_id")
        narrative_arc = context.get("narrative_arc", "")

        if not bgm_tracks:
            raise ValueError("缺少 bgm_tracks 分镜数据")

        # 并发匹配音乐
        tasks = [self._fetch_music_for_segment(segment) for segment in bgm_tracks]
        music_results = await asyncio.gather(*tasks)

        clips = []
        for idx, (segment, matches) in enumerate(zip(bgm_tracks, music_results)):
            duration = segment["end_time"] - segment["start_time"]

            primary_match = matches[0] if matches else None
            alternatives = [m.dict() for m in matches[1:]] if matches and len(matches) > 1 else []

            # 如果没有匹配，使用静音或默认音效占位
            if not primary_match:
                fallback_url = "https://assets.example.com/silence.mp3"
                primary_match = {
                    "url": fallback_url,
                    "cut_start": 0.0,
                    "cut_end": duration,
                    "duration": duration
                }
            else:
                primary_match = primary_match.dict()

            clip = {
                "clip_id": f"bgm_{idx}",
                "start": segment["start_time"],
                "end": segment["end_time"],
                "duration": duration,
                "mood": segment["mood"],
                "genre": segment["genre"],
                "narrative_role": segment["narrative_role"],
                "transition": segment["transition"],
                "music_suggestion": {
                    "title": segment["recommended_track"]["title"],
                    "artist": segment["recommended_track"]["artist"],
                    "reason": segment["recommended_track"]["reason"]
                },
                "audio": {
                    "url": primary_match["url"],
                    "in_point": primary_match["cut_start"],
                    "out_point": primary_match["cut_end"],
                    "duration": primary_match["duration"]
                },
                "alternatives": alternatives,
                "volume_db": self._suggest_volume_db(segment["mood"]),  # 自动建议音量
                "pan": 0.0  # 居中
            }
            clips.append(clip)

        total_duration = max((seg["end_time"] for seg in bgm_tracks), default=0)

        # ✅ 返回标准音轨格式
        return {
            "track_id": "bgm_track",
            "track_name": "背景音乐",
            "track_type": "background_music",
            "total_duration": total_duration,
            "clips": clips,
            "metadata": {
                "narrative_arc": narrative_arc,
                "generated_by": self.node_id,
                "timestamp": asyncio.get_event_loop().time()
            }
        }

    async def _fetch_music_for_segment(self, segment: Dict[str, Any]) -> List[BGMResponse]:
        """
        为单个分镜段调用 match_bgm 获取匹配音乐
        """
        duration = segment["end_time"] - segment["start_time"]

        # 构造请求描述
        description = (
            f"{segment['mood']}情绪，使用{', '.join(segment['instruments'])}，"
            f"节奏{segment['bpm']} BPM，用于{segment['narrative_role']}"
        )

        category = segment["genre"].split("/")[0].strip()  # 取主类型

        request = BGMRequest(
            description=description,
            category=category,
            duration=duration
        )

        try:
            matches = await match_bgm(request)
            return matches
        except Exception as e:
            print(f"⚠️ 匹配音乐失败 [{segment['segment_index']}]: {str(e)}")
            return []
        
    def _suggest_volume_db(self, mood: str) -> float:
        """根据情绪建议默认音量"""
        volume_map = {
            "励志": -16.0,
            "激动": -15.0,
            "紧张": -17.0,
            "温馨": -18.0,
            "平静": -20.0,
            "冷静": -20.0,
            "舒缓": -20.0,
            "悲伤": -19.0
        }
        return volume_map.get(mood, -18.0)
    async def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        支持用户干预的重新生成函数
        支持的干预类型：
        - 更换某段的情绪或描述
        - 强制重新匹配某段
        - 指定某段使用特定音乐 URL
        - 添加自定义提示语优化匹配
        """
        bgm_tracks = context.get("bgm_tracks")
        if not bgm_tracks:
            raise ValueError("缺少 bgm_tracks 分镜数据")

        # 创建可修改的分镜副本
        modified_segments = [dict(segment) for segment in bgm_tracks]

        override = user_intent.get("bgm_override")
        if not override:
            # 如果没有覆盖指令，只是重新生成（比如重新随机选曲）
            return await self.generate(context)

        updated_indices = set()

        for cmd in override:
            idx = cmd.get("segment_index")

            if idx is None or idx >= len(modified_segments):
                continue

            segment = modified_segments[idx]
            updated_indices.add(idx)

            # === 1. 更换情绪/风格 ===
            if "mood" in cmd:
                segment["mood"] = cmd["mood"]
                segment["narrative_role"] = cmd.get("narrative_role", segment["narrative_role"])
                print(f"🔁 用户干预：段落 {idx} 情绪更改为 '{cmd['mood']}'")

            if "genre" in cmd:
                segment["genre"] = cmd["genre"]

            # === 2. 自定义描述增强匹配 ===
            if "description_hint" in cmd:
                # 附加到 narrative_role 或 instruments
                extra = cmd["description_hint"]
                segment["narrative_role"] += f"。特别注意：{extra}"
                print(f"🔍 段落 {idx} 添加描述提示：{extra}")

            # === 3. 强制使用指定音乐 URL ===
            if "use_url" in cmd:
                url = cmd["use_url"]
                duration = segment["end_time"] - segment["start_time"]
                cut_start = cmd.get("cut_start", 0.0)
                cut_end = cmd.get("cut_end", cut_start + duration)

                # 直接注入 matched_audio，跳过 match_bgm
                segment["_forced_audio"] = {
                    "url": url,
                    "cut_start": cut_start,
                    "cut_end": cut_end,
                    "duration": cut_end - cut_start
                }
                print(f"🎵 强制指定音乐：段落 {idx} → {url}")

            # === 4. 强制重新匹配（带新参数）===
            if "reroll" in cmd and cmd["reroll"]:
                # 可结合 hint 一起使用
                hint = cmd.get("hint", "")
                if hint:
                    segment["narrative_role"] += f"。优先考虑：{hint}"
                print(f"🔄 重新匹配段落 {idx}（带新提示）")

        # 重新生成：对被修改的段落重新请求，其余保留原结果？
        # 注意：当前设计是全量重新生成。若要增量更新，需更复杂的状态管理

        # 这里我们选择：全量重新生成（简单可靠）
        # 但你可以扩展为：仅对 updated_indices 重新请求

        # 临时打标强制音频，在 generate 后注入
        context_with_forced = {"bgm_tracks": modified_segments, "narrative_arc": context.get("narrative_arc", "")}
        result = await self.generate(context_with_forced)

        # 注入用户强制指定的音频（绕过 match_bgm）
        for seg_result in result["bgm_composition"]:
            idx = seg_result["segment_index"]
            orig_segment = modified_segments[idx]
            # if "_forced_audio" in orig_segment:
            #     forced = orig_segment["_forced_audio"]
            #     seg_result["matched_audio"] = forced
            #     seg_result["alternatives"] = []
            #     seg_result["music_suggestion"] = {
            #         "title": "用户指定曲目",
            #         "artist": "Custom",
            #         "reason": f"来自用户指令: {forced['url']}"
            #     }
            #     print(f"✅ 已注入用户指定音频到段落 {idx}")
            # 在 regenerate 中注入用户指定音频时，保持结构一致
            if "_forced_audio" in orig_segment:
                forced = orig_segment["_forced_audio"]
                seg_result["audio"] = {
                    "url": forced["url"],
                    "in_point": forced["cut_start"],
                    "out_point": forced["cut_end"],
                    "duration": forced["cut_end"] - forced["cut_start"]
                }
                seg_result["music_suggestion"] = {
                    "title": "用户指定曲目",
                    "artist": "Custom",
                    "reason": f"来自用户指令: {forced['url']}"
                }
                seg_result["alternatives"] = []
                print(f"✅ 已注入用户指定音频到段落 {idx}")
        # 记录用户意图
        result["regeneration_reason"] = str(user_intent)
        return result


    def validate_inputs(self, context: Dict[str, Any]) -> (bool, List[str]):
        """
        根据 required_inputs 校验输入
        返回: (is_valid, error_messages)
        """
        errors = []

        for key, spec in self.required_inputs.items():
            required = spec.get("required", False)
            if required and key not in context:
                errors.append(f"缺少必需输入: {key}")
                continue

            value = context.get(key)
            if value is None and required:
                errors.append(f"输入不能为空: {key}")
                continue

            # 类型检查（简化版）
            if value is not None:
                expected_type = spec["type"]
                if expected_type == "list[dict]" and not isinstance(value, list):
                    errors.append(f"输入 '{key}' 应为列表类型，实际为 {type(value)}")
                elif expected_type == "str" and not isinstance(value, str):
                    errors.append(f"输入 '{key}' 应为字符串类型，实际为 {type(value)}")
                elif expected_type == "list[str]" and not (
                    isinstance(value, list) and all(isinstance(i, str) for i in value)
                ):
                    errors.append(f"输入 '{key}' 应为字符串列表")

        return len(errors) == 0, errors
if __name__ == "__main__":
    print("🎬 分镜驱动BGM合成系统启动...\n")

    # 创建节点
    node = BGMCompositionNode(node_id="bgm_001")

    # 输入分镜块
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
        "shot_blocks": shot_blocks,
        "target_duration": 60.0
    }

    print("📌 分镜数量:", len(shot_blocks))
    print("🎯 目标时长:", context["target_duration"], "秒\n")

    # 执行生成
    result = node.generate(context)
    print("🎉 BGM合成结果:"+str(result))

    if "bgm_track" in result and result["bgm_track"].get("file_path"):
        track = result["bgm_track"]
        print("✅ 推荐BGM:")
        print(f"   曲名: {track['title']}")
        print(f"   来源: {track['source']}")
        print(f"   时长: {track['duration']}s")
        print(f"   BPM: {track['original_bpm']}")
        print(f"   情绪: {track['metadata']['mood']}")
        print(f"   文件: {track['file_path']}")
    else:
        print("❌ 未能生成BGM。")