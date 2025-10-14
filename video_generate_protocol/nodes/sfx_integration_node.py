# sfx_integration_node.py

from typing import Dict, List, Optional,Any
import asyncio
import json

 # 假设你使用 Qwen-Agent 框架
from materials_supplies import match_sfx, SFXRequest  # 替换为你的真实匹配模块
from llm.qwen import QwenLLM  # 假设这是你封装好的 Qwen 调用模块
from video_generate_protocol import BaseNode  # 假设这是你的视频生成协议基类


# 配置常量
AVOIDANCE_MARGIN_DB = 12.0  # 人声重叠时降 12dB
KEYWORD_ALIGNMENT_WINDOW = 0.3
MIN_GAP_FOR_INSERT = 0.5
SFX_FADE_DURATION = 0.1


class SFXIntegrationNode(BaseNode):

    required_inputs = [
            {   "name":"shot_blocks_id",
                "label": "镜头序列",
                "type": List[Dict],
                "desc": "视频分镜序列，包含镜头类型、时长、描述、字幕等",
                "required": False,
                "default": [],
                "fields": {
                    "shot_type": {"type": "str", "desc": "镜头类型，如 close-up, wide"},
                    "duration": {"type": "float", "desc": "镜头时长（秒）"},
                    "visual_description": {"type": "str", "desc": "画面内容描述"},
                    "pacing": {"type": "str", "desc": "节奏：快/中/慢"},
                    "caption": {"type": "str", "desc": "可选：字幕文本"}
                }
            },
            {   "name":"voice_track_id",
                "label": "人声轨道信息",
                "type": "Dict",
                "desc": "人声轨道信息，包含 ASR 文本与时间戳",
                "required": False,
                "default": {},
                "fields": {
                    "clips": {
                        "type": "List[Dict]",
                        "desc": "语音片段列表",
                        "item_fields": {
                            "start": {"type": "float", "desc": "起始时间（秒）"},
                            "duration": {"type": "float", "desc": "持续时间"},
                            "text": {"type": "str", "desc": "转录文本"},
                            "words": {
                                "type": "List[Dict]",
                                "desc": "可选：字/词级时间戳",
                                "item_fields": {
                                    "word": {"type": "str", "desc": "单个字或词"},
                                    "start": {"type": "float", "desc": "起始时间"},
                                    "end": {"type": "float", "desc": "结束时间"}
                                }
                            }
                        }
                    }
                }
            },
            {   "name":"existing_sfx",
                "label": "已有音效轨道",
                "type": "Dict",
                "desc": "可选：已有音效轨道，用于 regenerate 时参考",
                "required": False,
                "default": None
            },
            {   "name":"system_parameters",
                "label": "系统参数",
                "type": "Dict",
                "desc": "系统参数，如是否避让人声、混响等",
                "required": False,
                "default": {},
                "example": {
                    "avoid_voice": True,
                    "reverb_apply": True,
                    "max_density_per_minute": 6
                }
            },
        ]
    
    output_schema=[
        {
            "name": "sfx_track_id",
            "label": "音效片段列表",
            "type": list,
            "required": True,
            "desc": "音效片段列表”包含每个片段的开始时间和结束时间，以及音效的描述信息",
            "field_type": "json"
        },
    ]
    
    def __init__(self,node_id: str, name: str = "音效添加",system_parameters: Dict = None):
        self.node_id = node_id
        self.name = name
        # self.required_inputs =  # 输入镜头列表
        self.qwen = QwenLLM()
        self.system_parameters = system_parameters or {}
        self.last_sfx_time = 0.0  # 用于密度控制

    def _is_overlapping(self, start1: float, end1: float, start2: float, end2: float) -> bool:
        """判断两个时间段是否重叠"""
        return max(start1, start2) < min(end1, end2)

    def _get_category_stats(self, history: List[Dict]) -> Dict[str, int]:
        """统计已添加音效的类别分布"""
        stats = {}
        for item in history:
            cat = item["category"]
            stats[cat] = stats.get(cat, 0) + 1
        return stats

    def _fallback_narrative_arc(self, total: int) -> List[str]:
        """简单基于进度的 fallback 情绪曲线"""
        arc = []
        for i in range(total):
            progress = i / total
            if progress < 0.3:
                arc.append("平静 / 总结")
            elif progress < 0.6:
                arc.append("期待 / 启动")
            elif progress < 0.8:
                arc.append("投入 / 高潮")
            else:
                arc.append("激励 / 行动")
        return arc

    def _analyze_emotion_curve(self, shots: List[Dict]) -> List[str]:
        """
        使用 Qwen 对完整分镜序列分析，生成情绪曲线标签
        """
        scene_descriptions = "\n".join([
            f"镜头 {i+1} ({shot['shot_type']}, {shot['duration']}秒): "
            f"{shot['visual_description']} "
            f"[字幕: {shot.get('caption', '')}]"
            for i, shot in enumerate(shots)
        ])

        prompt = f"""
        你是一位影视叙事分析师。请分析以下视频分镜序列的情感发展和叙事节奏。
        为每一个镜头判断其所属的叙事阶段或情绪类型，从以下类别中选择最合适的：

        - 平静 / 总结
        - 铺垫 / 引入
        - 期待 / 启动
        - 投入 / 高潮
        - 激励 / 行动
        - 收尾 / 升华

        请以 JSON 数组格式输出，数组长度必须等于镜头数量，顺序对应：

        [
          "平静 / 总结",
          "期待 / 启动"
        ]

        【分镜序列】
        {scene_descriptions}

        注意：只输出 JSON 数组，不要额外说明。
        """

        try:
            response = self.qwen.generate(prompt=prompt,parse_json=True)
            # result = json.loads(response.strip())
            result = response
            if isinstance(result, list) and len(result) == len(shots):
                return result
            else:
                print("⚠️ 情绪曲线分析失败：格式错误，使用默认")
        except Exception as e:
            print(f"⚠️ 情绪曲线分析失败: {e}")

        return self._fallback_narrative_arc(len(shots))

    def should_add_sfx(
        self,
        shot: Dict,
        history: List[Dict],
        index: int,
        total: int,
        emotion_label: str
    ) -> Optional[Dict[str, str]]:
        """使用 Qwen 判断是否添加音效，并推荐类别"""
        history = history or []
        category_stats = self._get_category_stats(history)
        category_summary = ", ".join([f"{k}:{v}" for k, v in category_stats.items()]) if category_stats else "无"
        recent_sfx = "无" if not history else "; ".join([
            f"{h['title']}@{h['time']:.1f}s" for h in history[-2:]
        ])

        prompt = f"""
        你是一位资深影视音效设计师。请结合叙事节奏和音效平衡，决定是否添加音效。

        【AI 分析的情绪阶段】
        {emotion_label}

        【全局信息】
        - 视频共 {total} 个镜头，当前是第 {index + 1} 个。
        - 已添加音效类别统计：{category_summary}
        - 最近添加音效：{recent_sfx}

        【设计原则】
        1. 平静/总结：保持安静，避免干扰。
        2. 期待/启动：可加 UI 音效引导。
        3. 投入/高潮：适合交互音、打字声。
        4. 激励/行动：可加 impact 音效增强动力。
        5. 避免同类音效连续出现。

        【当前镜头】
        - 类型：{shot['shot_type']}
        - 时长：{shot['duration']}秒
        - 节奏：{shot['pacing']}
        - 描述：{shot['visual_description']}
        - 字幕：{shot.get('caption', '')}

        请输出 JSON：
        {{
          "add_sfx": true/false,
          "category": "ui|foley|ambience|impact|typing",
          "reason": "决策理由，包含对情绪和历史的分析"
        }}
        """

        try:
            response = self.qwen.generate(prompt=prompt,parse_json=True)

            # 确保response是字典类型
            if isinstance(response, str):
                import json
                result = json.loads(response)
            elif isinstance(response, dict):
                result = response
            else:
                print(f"⚠️ Qwen返回了意外的类型: {type(response)}")
                return None

            if result.get("add_sfx"):
                return {
                    "category": result["category"],
                    "reason": result["reason"]
                }
        except Exception as e:
            print(f"⚠️ Qwen 决策失败: {e}")
        return None

    def _extract_voice_timeline(self, voice_track: Dict) -> List[Dict]:
        """提取带文本的人声时间线"""
        timeline = []
        for clip in voice_track.get("clips", []):
            start = clip.get("start")
            duration = clip.get("duration", 0.0)
            text = clip.get("text", "").strip().lower()
            if start is not None and duration > 0 and text:
                timeline.append({
                    "start": start,
                    "end": start + duration,
                    "text": text
                })
        return sorted(timeline, key=lambda x: x["start"])

    def _find_keyword_timestamp(self, voice_timeline: List[Dict], keyword: str) -> Optional[float]:
        """查找关键词出现的大致时间"""
        for segment in voice_timeline:
            if keyword in segment["text"]:
                words = segment["text"].split()
                try:
                    idx = words.index(keyword)
                    ratio = idx / len(words)
                    return segment["start"] + ratio * (segment["end"] - segment["start"])
                except:
                    continue
        return None

    def _find_available_gaps(self, voice_timeline: List[Dict], min_gap: float, sfx_duration: float) -> List[Dict]:
        """查找可用于插入音效的静音间隙"""
        gaps = []
        last_end = 0.0
        for seg in sorted(voice_timeline, key=lambda x: x["start"]):
            gap_start = last_end
            gap_end = seg["start"]
            if gap_end - gap_start >= max(min_gap, sfx_duration):
                gaps.append({"start": gap_start, "end": gap_end})
            last_end = seg["end"]
        return gaps

    def _align_sfx_to_voice(self, sfx_track: Dict, voice_track: Dict):
        """
        智能调整音效：关键词对齐 → 间隙插入 → 音量避让 → 淡入淡出
        """
        if not self.system_parameters.get("avoid_voice", True):
            return

        sfx_clips = sfx_track.get("clips", [])
        voice_clips = voice_track.get("clips", [])
        if not sfx_clips or not voice_clips:
            return

        voice_timeline = self._extract_voice_timeline(voice_track)

        for sfx_clip in sfx_clips:
            original_duration = sfx_clip["duration"]
            category = sfx_clip["category"]
            title = sfx_clip["title"]

            # 1. 关键词对齐（UI 类）
            # 1. 使用 Qwen 智能语义对齐（替代 keyword_map）
            aligned = False
            if category in ["ui", "click", "impact", "typing", "foley"]:
                word_timestamps = []
                for clip in voice_track.get("clips", []):
                    if "words" in clip:  # 假设 words 包含字/词级时间戳
                        for word_info in clip["words"]:
                            word_timestamps.append({
                                "word": word_info["word"],
                                "start": word_info["start"],
                                "end": word_info["end"]
                            })

                if word_timestamps:
                    transcript = "".join([w["word"] for w in word_timestamps])
                    words_json = json.dumps(word_timestamps, ensure_ascii=False, indent=2)

                    prompt = f"""
                    你是一位音效同步专家。请根据音效语义和人声文本的时间戳，判断音效最应“对齐”到哪一个字或词。

                    【音效信息】
                    - 名称：{title}
                    - 类别：{category}
                    - 时长：{original_duration:.2f}秒

                    【人声文本（带时间戳）】
                    {transcript}

                    详细时间戳（单位：秒）：
                    {words_json}

                    【任务】
                    1. 分析音效语义（如“点击”应匹配“点击”一词）
                    2. 找出最应同步的**目标词或字**
                    3. 返回该词的**建议对齐时间**：
                    - 如果是“瞬时音效”（如 click），对齐到词的**起始时间**
                    - 如果是“持续音效”，可对齐到词中间
                    4. 输出 JSON：
                    {{
                        "target_word": "点击",
                        "alignment_time": 10.9,
                        "reason": "‘点击’音效应与‘点击’动词同步，增强操作反馈"
                    }}

                    注意：只输出 JSON，不要额外说明。
                    """

                    try:
                        response = self.qwen.generate(prompt=prompt,parse_json=True)
                        # result = json.loads(response.strip())
                        result = response
                        target_time = result.get("alignment_time")
                        target_word = result.get("target_word", "未知")

                        if target_time is not None:
                            new_start = max(0.0, target_time - original_duration * 0.5)
                            sfx_clip["start"] = new_start
                            # 在 _align_sfx_to_voice 中，当你修改了 start 或 duration 后，加上：
                            sfx_clip["end"] = sfx_clip["start"] + sfx_clip["duration"]
                            sfx_clip["audio"]["duration"] = sfx_clip["duration"]
                            sfx_clip["audio"]["out_point"] = sfx_clip["duration"]
                            sfx_clip["processing_note"] = (
                                f"AI 智能对齐 '{target_word}' @ {target_time:.2f}s | {result.get('reason', '')}"
                            )
                            aligned = True
                    except Exception as e:
                        print(f"⚠️ Qwen 对齐推理失败: {e}")

            # 2. 间隙插入（环境音等）
            if not aligned:
                gaps = self._find_available_gaps(voice_timeline, MIN_GAP_FOR_INSERT, original_duration)
                for gap in gaps:
                    if gap["end"] - gap["start"] >= original_duration:
                        insert_time = gap["start"] + (gap["end"] - gap["start"] - original_duration) / 2
                        sfx_clip["start"] = insert_time
                        sfx_clip["end"] = sfx_clip["start"] + sfx_clip["duration"]
                        sfx_clip["audio"]["duration"] = sfx_clip["duration"]
                        sfx_clip["audio"]["out_point"] = sfx_clip["duration"]
                        sfx_clip["processing_note"] = f"插入静音间隙 [{gap['start']:.1f}s-{gap['end']:.1f}s]"
                        aligned = True
                        break

            # 3. 音量避让（兜底）
            sfx_start = sfx_clip["start"]
            sfx_end = sfx_start + sfx_clip["duration"]


            for voice_clip in voice_clips:
                v_start = voice_clip.get("start")
                v_dur = voice_clip.get("duration", 0.0)
                v_end = v_start + v_dur
                if v_start is None or v_dur <= 0:
                    continue
                if self._is_overlapping(sfx_start, sfx_end, v_start, v_end):
                    orig_vol = sfx_clip.get("volume_db", -12.0)
                    sfx_clip["volume_db"] = orig_vol - AVOIDANCE_MARGIN_DB
                    note = f"因人声 [{v_start:.1f}s-{v_end:.1f}s] 降 {AVOIDANCE_MARGIN_DB}dB"
                    if "processing_note" in sfx_clip:
                        sfx_clip["processing_note"] += "; " + note
                    else:
                        sfx_clip["processing_note"] = note
                    break
                sfx_clip["end"] = sfx_clip["start"] + sfx_clip["duration"]
                sfx_clip["audio"]["duration"] = sfx_clip["duration"]
                sfx_clip["audio"]["out_point"] = sfx_clip["duration"]

            # 4. 添加淡入淡出
            if "fade_in" not in sfx_clip:
                sfx_clip["fade_in"] = SFX_FADE_DURATION
            if "fade_out" not in sfx_clip:
                sfx_clip["fade_out"] = SFX_FADE_DURATION

    # async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     主入口：生成音效轨道
    #     """
    #     shots: List[Dict] = context["shot_blocks_id"]
    #     voice_track: Dict = context.get("voice_track", {})
    #     current_time = 0.0

    #     sfx_track = {
    #         "track_name": "sfx",
    #         "track_type": "sound_effects",
    #         "clips": []
    #     }
    #     sfx_history = []

    #     # 🔥 第一步：AI 分析情绪曲线
    #     emotion_curve = self._analyze_emotion_curve(shots)
    #     print("📊 情绪曲线:", emotion_curve)

    #     # 处理每个镜头
    #     for idx, shot in enumerate(shots):
    #         duration = shot["duration"]
    #         start_time = current_time

    #         # 决策是否加音效
    #         decision =  self.should_add_sfx(
    #             shot=shot,
    #             history=sfx_history,
    #             index=idx,
    #             total=len(shots),
    #             emotion_label=emotion_curve[idx]
    #         )
    #         if not decision:
    #             current_time += duration
    #             continue

    #         # 匹配音效
    #         request = SFXRequest(
    #             description=shot["visual_description"],
    #             category=decision["category"]
    #         )
    #         candidates = await match_sfx(request)
    #         if not candidates:
    #             current_time += duration
    #             continue

    #         sfx = candidates[0]
    #         final_duration = min(sfx.duration, duration * 0.8)
    #         insert_time = start_time + max(0, (duration - final_duration) / 2)

    #         # 创建音效片段
    #         clip = {
    #             "sfx_id": f"auto_{len(sfx_track['clips']) + 1}",
    #             "title": sfx.title,
    #             "file_path": sfx.url,
    #             "start": insert_time,
    #             "duration": final_duration,
    #             "in_point": 0.0,
    #             "volume_db": -12.0,
    #             "pan": 0.0,
    #             "reverb_level": 0.3 if self.system_parameters.get("reverb_apply") else 0.0,
    #             "category": sfx.category,
    #             "tags": [],
    #             "source": "ai_match",
    #             "reason": decision["reason"]
    #         }
    #         sfx_track["clips"].append(clip)

    #         # 更新历史
    #         sfx_history.append({
    #             "time": insert_time,
    #             "category": sfx.category,
    #             "title": sfx.title,
    #             "shot_index": idx
    #         })

    #         print(f"✅ 添加音效: {sfx.title} @ {insert_time:.1f}s | {sfx.category}")
    #         current_time += duration

    #     # 🔊 最后一步：智能对齐与避让
    #     self._align_sfx_to_voice(sfx_track, voice_track)

    #     return {"sfx_track": sfx_track}
    

  
    #新generate，修改了格式，没改逻辑
    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        主入口：生成符合标准格式的音效轨道
        """
        self.validate_context(context)
        shots: List[Dict] = context["shot_blocks_id"]
        voice_track: Dict = context.get("voice_track", {})
        system_params = context.get("system_parameters", {})
        current_time = 0.0

        # 初始化标准音轨结构
        sfx_track = {
            "track_id": "sfx_track",
            "track_name": "音效",
            "track_type": "sound_effects",
            "total_duration": sum(shot["duration"] for shot in shots),
            "clips": []
        }
        sfx_history = []

        # 🔥 第一步：AI 分析情绪曲线
        emotion_curve = self._analyze_emotion_curve(shots)
        print("📊 情绪曲线:", emotion_curve)

        # 处理每个镜头
        for idx, shot in enumerate(shots):
            duration = shot["duration"]
            start_time = current_time

            # 决策是否加音效
            decision = self.should_add_sfx(
                shot=shot,
                history=sfx_history,
                index=idx,
                total=len(shots),
                emotion_label=emotion_curve[idx]
            )
            if not decision:
                current_time += duration
                continue

            # 匹配音效
            request = SFXRequest(
                description=shot["visual_description"],
                category=decision["category"]
            )
            candidates = await match_sfx(request)
            if not candidates:
                current_time += duration
                continue

            sfx = candidates[0]
            final_duration = min(sfx.duration, duration * 0.8)
            insert_time = start_time + max(0, (duration - final_duration) / 2)

            # ✅ 构建标准 clip 结构
            clip = {
                "clip_id": f"sfx_{len(sfx_track['clips'])}",
                "start": insert_time,
                "end": insert_time + final_duration,
                "duration": final_duration,
                "title": sfx.title,
                "category": sfx.category,
                "tags": [],
                "source": "ai_match",
                "reason": decision["reason"],
                "audio": {
                    "url": sfx.url,
                    "in_point": 0.0,
                    "out_point": final_duration,
                    "duration": final_duration
                },
                "volume_db": -12.0,
                "pan": 0.0
            }
            sfx_track["clips"].append(clip)

            # 更新历史（用于类别统计）
            sfx_history.append({
                "time": insert_time,
                "category": sfx.category,
                "title": sfx.title,
                "shot_index": idx
            })

            print(f"✅ 添加音效: {sfx.title} @ {insert_time:.1f}s | {sfx.category}")
            current_time += duration

        # 🔊 最后一步：智能对齐与避让（会修改 clip 的 start/end/duration/audio）
        self._align_sfx_to_voice(sfx_track, voice_track)

        # ✅ 重新计算 end 和 audio duration（对齐后可能变化）
        for clip in sfx_track["clips"]:
            new_duration = clip["duration"]
            clip["end"] = clip["start"] + new_duration
            clip["audio"]["out_point"] = new_duration
            clip["audio"]["duration"] = new_duration

        return {"sfx_track": sfx_track}
    async def regenerate(
        self,
        context: Dict[str, Any],
        existing_result: Optional[Dict] = None,
        feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        基于已有结果和反馈，重新生成音效轨道。
        适用于：用户手动调整后要求“再优化一次”或提供反馈。

        :param context: 原始上下文（shot_blocks_id, voice_track 等）
        :param existing_result: 上一次 generate 的输出，可选
        :param feedback: 用户或系统的反馈，如 "UI音效太多"、"点击音效要更靠前"
        :return: 新的 sfx_track
        """
        print("🔄 执行 regenerate，接收反馈:", feedback or "无")

        # 构造增强上下文
        enhanced_context = context.copy()

        if existing_result and feedback:
            # 将上次结果和反馈注入 context，供 AI 决策时参考
            enhanced_context["previous_sfx_track"] = existing_result.get("sfx_track")
            enhanced_context["regeneration_feedback"] = feedback

        # 调用主生成逻辑（可在此加入反馈感知的 prompt 修改）
        return await self.generate(enhanced_context)