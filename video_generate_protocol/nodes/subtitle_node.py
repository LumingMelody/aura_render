# nodes/subtitle_node.py

from video_generate_protocol import BaseNode
from typing import Dict, List, Any
import re
import json
from datetime import datetime
from llm import QwenLLM
import asyncio
from materials_supplies import TTSRequest, TTSResponse
from materials_supplies.matcher.tts_matcher import match_tts

# 字幕样式预设
SUBTITLE_STYLES = {
    "default": {
        "font": "Microsoft YaHei, sans-serif",
        "font_size": 24,
        "color": "#FFFFFF",
        "background": "rgba(0, 0, 0, 0.6)",
        "stroke": "#000000",
        "stroke_width": 2,
        "align": "center",
        "position": "bottom-center",
        "padding": "10px 16px",
        "border_radius": "6px",
        "shadow": "0 1px 4px rgba(0,0,0,0.5)"
    },
    "minimal": {
        "font": "Arial, sans-serif",
        "font_size": 20,
        "color": "#FFFFFF",
        "background": "none",
        "stroke": "#000000",
        "stroke_width": 1.5,
        "align": "center",
        "position": "bottom-center",
        "padding": "0",
        "border_radius": "0",
        "shadow": "0 1px 3px rgba(0,0,0,0.4)"
    },
    "karaoke": {
        "font": "Comic Sans MS, cursive",
        "font_size": 28,
        "color": "#FFFF00",
        "background": "rgba(0, 0, 0, 0.5)",
        "stroke": "#FF0000",
        "stroke_width": 2,
        "align": "center",
        "position": "bottom-center",
        "padding": "12px 20px",
        "border_radius": "8px",
        "shadow": "0 2px 6px rgba(0,0,0,0.6)"
    }
}
class SubtitleNode(BaseNode):
    required_inputs = [
        {
            "name": "shot_blocks_id",
            "label": "镜头序列",
            "type": List[Dict],
            "required": True,
            "desc": "包含视觉描述、字幕文本和可选 frame_image 的镜头列表",
            "field_type": "json"
        },
        {
            "name": "video_width",
            "label": "视频宽度",
            "type": int,
            "required": False,
            "default": 1920,
            "desc": "用于字幕换行计算",
            "field_type": "number"
        },
        {
            "name": "video_height",
            "label": "视频高度",
            "type": int,
            "required": False,
            "default": 1080,
            "desc": "用于定位",
            "field_type": "number"
        }
    ]

    output_schema=[
        {
            "name": "subtitle_sequence_id",
            "label": "字幕轨道",
            "type": list,
            "required": True,
            "desc": "字幕轨道，包含字幕片段列表，如 [{'start_time': 0.0, 'end_time': 10.0, 'text': '欢迎来到机器学习世界！'}]",
            "field_type": "json"
        },
        {
            "name": "audio_tracks_id",
            "label": "语音音轨",
            "type": float,
            "required": False,
            "desc": "生成的 TTS 语音音轨列表，如 [{'start': 0.0, 'end': 10.0, 'url': 'https://...'}]",
            "field_type": "text"
        }
    ]

    system_parameters = {
        "max_chars_per_line": 20,
        "max_lines": 2,
        "style_preset": "default",
        "auto_align": True,
        "min_duration": 1.0,
        "word_wrap": True,
        "language": "zh-CN",
        "tts_voice": "zh-CN-XiaoyiNeural",  # 默认音色
        "tts_speed": 1.1
    }

    def __init__(self, node_id: str, name: str = "视频下方的字幕"):
        super().__init__(node_id=node_id, node_type="subtitle", name=name)
        self.qwen = QwenLLM()
    
    
        
    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_context(context)

        shots: List[Dict] = context["shot_blocks_id"]
        video_width = context.get("video_width", 1920)
        video_height = context.get("video_height", 1080)

        # === 决策 1: 是否生成 TTS？以及使用什么音色？===
        should_tts, voice_description = self._should_generate_tts(shots)

        # === 决策 2: 字幕颜色 + 描边颜色（三级策略）===
        color_result = self._decide_subtitle_colors(shots)
        final_color = color_result["text_color"]
        final_stroke = color_result["stroke_color"]

        # 构建样式配置
        style_config = SUBTITLE_STYLES["default"].copy()
        style_config["color"] = final_color
        style_config["stroke"] = final_stroke

        # 生成字幕片段
        subtitle_clips = []
        current_time = 0.0

        for idx, shot in enumerate(shots):
            caption = shot.get("caption", "").strip()
            duration = shot.get("duration", 2.0)

            if not caption:
                current_time += duration
                continue

            start = current_time
            end = current_time + duration
            current_time = end

            if end - start < self.system_parameters["min_duration"]:
                continue

            # 换行处理
            wrapped_lines = self._wrap_text(caption, video_width)
            if not wrapped_lines:
                continue

            clip = {
                "start": start,
                "end": end,
                "text": "\n".join(wrapped_lines),
                "lines": wrapped_lines,
                "duration": duration,
                "position": self._get_position(video_width, video_height),
                "style": "default",
                "metadata": {
                    "source": "visual_caption",
                    "shot_type": shot["shot_type"],
                    "pacing": shot["pacing"],
                    "should_tts": should_tts,
                    "text_color": final_color,
                    "stroke_color": final_stroke
                }
            }
            subtitle_clips.append(clip)

        # 构建字幕轨道
        subtitle_track = {
            "track_name": "subtitle",
            "track_type": "text",
            "format": "srt_compatible",
            "style": "default",
            "style_config": style_config,
            "clips": subtitle_clips,
            "timing": {
                "first_start": subtitle_clips[0]["start"] if subtitle_clips else 0,
                "last_end": subtitle_clips[-1]["end"] if subtitle_clips else 0
            }
        }

        # === 生成 TTS 音轨（如果需要）===
        # audio_tracks = []
        # if should_tts:
        #     print(f"🔊 开始生成 TTS 语音，使用音色风格：{voice_description}")
        #     tts_requests = []
        #     for clip in subtitle_clips:
        #         tts_requests.append(TTSRequest(
        #             text=clip["text"],
        #             voice=voice_description,  # ✅ 使用音色描述（如“温暖女声”）
        #             speed=self.system_parameters["tts_speed"],
        #             duration=clip["duration"]
        #         ))

        #     # 并行调用 TTS
        #     tts_responses = await asyncio.gather(
        #         *[match_tts(req) for req in tts_requests],
        #         return_exceptions=True
        #     )

        #     # 构建音频轨道
        #     for idx, response_list in enumerate(tts_responses):
        #         if isinstance(response_list, Exception):
        #             print(f"[TTS Error] Clip {idx}: {response_list}")
        #             continue
        #         response = response_list[0]
        #         audio_clip = {
        #             "start": subtitle_clips[idx]["start"],
        #             "end": response.duration + subtitle_clips[idx]["start"],
        #             "url": response.url,
        #             "voice": response.voice,  # 实际使用的音色（可能与描述略有不同）
        #             "duration": response.duration,
        #             "text": response.text,
        #             "type": "speech",
        #             "voice_preference": voice_description  # 记录原始偏好
        #         }
        #         audio_tracks.append(audio_clip)


        # === 生成 TTS 音轨（如果需要）===
        tts_track = None
        if should_tts:
            print(f"🔊 开始生成 TTS 语音，使用音色风格：{voice_description}")
            tts_requests = []
            for clip in subtitle_clips:
                tts_requests.append(TTSRequest(
                    text=clip["text"],
                    voice=voice_description,
                    speed=self.system_parameters["tts_speed"],
                    duration=clip["duration"]
                ))

            # 并行调用 TTS
            tts_responses = await asyncio.gather(
                *[match_tts(req) for req in tts_requests],
                return_exceptions=True
            )

            # 构建 clips 列表
            tts_clips = []
            total_duration = 0.0

            for idx, response_list in enumerate(tts_responses):
                if isinstance(response_list, Exception):
                    print(f"[TTS Error] Clip {idx}: {response_list}")
                    continue

                response = response_list[0]
                clip_info = subtitle_clips[idx]

                # 实际语音时长（可用于对齐）
                actual_duration = response.duration
                expected_duration = clip_info["duration"]

                # 如果 TTS 返回的语音太短，可做拉伸或保留原时长（这里我们以实际语音为准）
                # 但为了兼容，我们保留原始时间轴的 start/end
                start_time = clip_info["start"]
                end_time = start_time + expected_duration  # 保持与字幕同步

                tts_clip = {
                    "clip_id": f"tts_{idx}",
                    "start": start_time,
                    "end": end_time,
                    "duration": expected_duration,  # 保持与字幕一致
                    "text": response.text,
                    "voice": response.voice,  # 实际使用的音色（如 zh-CN-XiaoyiNeural）
                    "voice_preference": voice_description,  # 原始偏好（如“年轻活力女声”）
                    "audio": {
                        "url": response.url,
                        "in_point": 0.0,
                        "out_point": actual_duration,
                        "duration": actual_duration
                    },
                    "volume_db": -10.0,  # 默认音量
                    "pan": 0.0           # 居中
                }
                tts_clips.append(tts_clip)
                total_duration = max(total_duration, end_time)

            # 构建完整音轨
            tts_track = {
                "track_id": "tts_track",
                "track_name": "语音",
                "track_type": "speech",
                "total_duration": total_duration,
                "clips": tts_clips
            }
        # === 返回完整结果 ===

        result = {
            "subtitle_sequence_id": subtitle_track,
            # "audio_tracks_id": audio_tracks if audio_tracks else None,#老格式
            "tts_track_id": tts_track,  # ✅ 使用新格式
            # "tts_required": should_tts,
            # "tts_voice_preference": voice_description  # ✅ 新增：记录推荐音色
        }

        return result

    def _should_generate_tts(self, shots: List[Dict]) -> tuple[bool, str]:
        """
        判断是否需要生成 TTS，并推荐合适的音色描述（如“温暖女声”、“沉稳男声”）
        返回: (should_generate: bool, voice_description: str)
        """
        prompt = """
        请分析以下教学视频的镜头内容，判断是否需要为字幕生成语音（TTS），并推荐合适的音色风格。

        考虑因素：
        - 视频是否以讲解、引导为主？
        - 是否适合用语音增强情感或教学效果？
        - 目标受众是学生、成人还是儿童？
        - 整体氛围是温馨、激励、专业还是轻松？

        返回 JSON 格式：
        {{
            "should_generate_tts": true/false,
            "voice_preference": "如：温暖女声、沉稳男声、年轻活力女声、专业男声、亲切童声等",
            "reason": "简要说明"
        }}

        镜头内容：
        {}
        """.format(json.dumps(shots, ensure_ascii=False, indent=2))

        result = self.qwen.generate(
            prompt=prompt,
            parse_json=True,
            json_schema={
                "type": "object",
                "properties": {
                    "should_generate_tts": {"type": "boolean"},
                    "voice_preference": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["should_generate_tts", "voice_preference"]
            }
        )

        if result and isinstance(result, dict):
            should_tts = result.get("should_generate_tts", False)
            voice_desc = result.get("voice_preference", "标准女声")
            reason = result.get("reason", "")
            print(f"[TTS决策] {reason} | 音色建议: {voice_desc}")
            return should_tts, voice_desc

        return False, "标准女声"
    def _decide_subtitle_colors(self, shots: List[Dict]) -> Dict[str, str]:
        """三级策略决定字幕颜色与描边"""
        images = [shot["frame_image"] for shot in shots if "frame_image" in shot]
        if images:
            prompt = "分析这些教学视频帧的画面氛围，并建议字幕文本颜色和描边颜色（十六进制）。要求高可读性。"
            result = self.qwen.generate(
                prompt=prompt,
                images=images,
                parse_json=True,
                json_schema={
                    "type": "object",
                    "properties": {
                        "text_color": {"type": "string", "format": "color"},
                        "stroke_color": {"type": "string", "format": "color"}
                    },
                    "required": ["text_color", "stroke_color"]
                }
            )
            if result and isinstance(result, dict):
                print(f"[颜色决策] VL → 文字: {result['text_color']}, 描边: {result['stroke_color']}")
                return {
                    "text_color": result["text_color"],
                    "stroke_color": result["stroke_color"]
                }

        descriptions = [shot["visual_description"] for shot in shots if "visual_description" in shot]
        if descriptions:
            prompt = f"""
            根据以下视觉描述判断教学视频的整体氛围，推荐适合的字幕颜色和描边颜色（十六进制）：
            {json.dumps(descriptions, ensure_ascii=False, indent=2)}
            要求：颜色需与常见背景有对比，适合长期阅读。
            返回 JSON：{{"text_color": str, "stroke_color": str}}
            """
            result = self.qwen.generate(
                prompt=prompt,
                parse_json=True,
                json_schema={
                    "type": "object",
                    "properties": {
                        "text_color": {"type": "string", "format": "color"},
                        "stroke_color": {"type": "string", "format": "color"}
                    },
                    "required": ["text_color", "stroke_color"]
                }
            )
            if result and isinstance(result, dict):
                print(f"[颜色决策] LLM → 文字: {result['text_color']}, 描边: {result['stroke_color']}")
                return {
                    "text_color": result["text_color"],
                    "stroke_color": result["stroke_color"]
                }

        print("[颜色决策] 使用默认颜色")
        return {
            "text_color": "#FFFFFF",
            "stroke_color": "#000000"
        }

    def _wrap_text(self, text: str, video_width: int) -> List[str]:
        max_chars = self.system_parameters["max_chars_per_line"]
        max_lines = self.system_parameters["max_lines"]
        if len(text) <= max_chars:
            return [text]
        lines = []
        words = text.split()
        current = ""
        for word in words:
            if len(current + word) <= max_chars:
                current += (word + " ")
            else:
                if current:
                    lines.append(current.strip())
                    if len(lines) >= max_lines:
                        break
                    current = word + " "
                else:
                    current = word + " "
        if current and len(lines) < max_lines:
            lines.append(current.strip())
        return lines

    def _get_position(self, width: int, height: int) -> Dict[str, Any]:
        margin = 40
        return {
            "x": "50%",
            "y": height - margin,
            "anchor": "bottom",
            "align": "center"
        }

    async def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        result = await self.generate(context)
        override = user_intent.get("subtitle_override")
        if not override:
            return result

        track = result["subtitle_sequence"]

        if "change_color" in override:
            new_color = override["change_color"]
            new_stroke = override.get("stroke_color", "#000000")
            track["style_config"]["color"] = new_color
            track["style_config"]["stroke"] = new_stroke
            for clip in track["clips"]:
                clip["metadata"]["text_color"] = new_color
                clip["metadata"]["stroke_color"] = new_stroke

        if "add_manual_caption" in override:
            manual = override["add_manual_caption"]
            clip = {
                "start": manual["start"],
                "end": manual["end"],
                "text": manual["text"],
                "position": self._get_position(context.get("video_width", 1920), context.get("video_height", 1080)),
                "style": "default",
                "metadata": {
                    "source": "manual",
                    "text_color": track["style_config"]["color"],
                    "stroke_color": track["style_config"]["stroke"]
                }
            }
            track["clips"].append(clip)
            track["clips"] = sorted(track["clips"], key=lambda x: x["start"])

        return result

