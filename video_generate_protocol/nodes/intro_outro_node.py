# nodes/intro_outro_node.py


from typing import Dict, List, Any
import uuid
import random
from datetime import datetime
import json
import asyncio

from video_generate_protocol import BaseNode
from llm import QwenLLM
from materials_supplies import match_introoutro,IntroOutroRequest,IntroOutroResponse

# 片头动画模板库（可扩展为JSON配置或模板系统）
INTRO_TEMPLATES = {
    "minimalist": {
        "name": "极简风格",
        "duration": 4.0,
        "elements": [
            {"type": "fade_in_text", "text": "{title}", "delay": 0.5, "duration": 2.0, "font": "Helvetica", "size": 48, "color": "#FFFFFF"},
            {"type": "logo_fade", "asset": "brand_logo.png", "position": "center", "delay": 1.5, "duration": 2.5}
        ],
        "background": {"type": "solid", "color": "#000000"},
        "transition": "fade"
    },
    "dynamic": {
        "name": "动感科技",
        "duration": 5.0,
        "elements": [
            {"type": "text_scale_in", "text": "{title}", "delay": 0.8, "duration": 2.5, "font": "Orbitron", "size": 56, "color": "#00FFFF"},
            {"type": "logo_pulse", "asset": "brand_logo.png", "delay": 1.0, "duration": 3.0}
        ],
        "background": {"type": "particle", "preset": "digital_grid"},
        "transition": "slide_left"
    },
    "elegant": {
        "name": "优雅文艺",
        "duration": 4.5,
        "elements": [
            {"type": "typewriter_text", "text": "{title}", "delay": 1.0, "duration": 2.8, "font": "Georgia", "size": 44, "color": "#F5F5DC"},
            {"type": "logo_slide_up", "asset": "brand_logo.png", "delay": 1.5, "duration": 2.5}
        ],
        "background": {"type": "blur_video", "ref": "bg_scenic.mp4"},
        "transition": "cross_dissolve"
    },
    "vlog": {
        "name": "Vlog风格",
        "duration": 3.5,
        "elements": [
            {"type": "handwriting_text", "text": "Hey guys! Today...", "delay": 0.3, "duration": 2.2, "font": "Comic Sans MS", "size": 36, "color": "#FF6B6B"},
            {"type": "sticker_pop", "asset": "mic_icon.png", "position": "right", "delay": 1.0, "duration": 1.8}
        ],
        "background": {"type": "solid", "color": "#FFE66D"},
        "transition": "zoom"
    }
}

# 片尾模板
OUTRO_TEMPLATES = {
    "standard": {
        "name": "标准片尾",
        "duration": 8.0,
        "elements": [
            {"type": "scrolling_credits", "data_key": "credits", "speed": 40, "font": "Arial", "size": 28, "color": "#CCCCCC"},
            {"type": "social_icons", "platforms": ["YouTube", "Instagram", "Twitter"], "position": "bottom", "icons": True, "delay": 2.0}
        ],
        "background": {"type": "solid", "color": "#000000"},
        "music": "outro_music.mp3"
    },
    "minimal": {
        "name": "极简致谢",
        "duration": 5.0,
        "elements": [
            {"type": "fade_text", "text": "感谢观看", "delay": 0.5, "duration": 2.0, "font": "Helvetica", "size": 36, "color": "#AAAAAA"},
            {"type": "text", "text": "@{username}", "delay": 2.5, "duration": 3.0, "font": "Courier", "size": 24, "color": "#66CCFF"}
        ],
        "background": {"type": "blur", "level": 10}
    }
}

# 视频类型 → 风格映射
GENRE_TO_STYLE = {
    "educational": "minimalist",
    "tech_review": "dynamic",
    "vlog": "vlog",
    "documentary": "elegant",
    "storytelling": "elegant",
    "general": "minimalist"
}

class IntroOutroNode(BaseNode):
    required_inputs = [
        {
            "name": "shot_blocks_id",
            "label": "分镜脚本",
            "type": List[Dict],
            "required": True,
            "desc": "包含 shot_type, duration, visual_description, caption 的镜头列表",
            "field_type": "json"
        },
        {
            "name": "video_topic",
            "label": "视频主题",
            "type": str,
            "required": False,
            "default": "",
            "desc": "如 'Python教学', '产品开箱'",
            "field_type": "text"
        },
        {
            "name": "force_intro",
            "label": "强制添加片头",
            "type": bool,
            "required": False,
            "default": False,
            "desc": "忽略AI判断，强制添加片头",
            "field_type": "checkbox"
        },
        {
            "name": "force_outro",
            "label": "强制添加片尾",
            "type": bool,
            "required": False,
            "default": False,
            "desc": "忽略AI判断，强制添加片尾",
            "field_type": "checkbox"
        }
    ]

    output_schema=[
         {
            "name": "intro_outro_sequence_id",
            "label": "首尾序列",
            "type": list,
            "required": True,
            "desc": "首尾序列，包含 intro 和 outro 段的序列，如 {intro: {...}, outro: {...}}",
            "field_type": "json"
        }
    ]

    system_parameters = {
        "llm_decision_prompt": """
        你是一个视频结构分析专家。请根据以下分镜脚本和视频主题，判断：
        1. 是否需要添加片头（intro）？
        2. 是否需要添加片尾（outro）？
        3. 推荐的风格分类（如：教育、科技、情感、商业、动感）？
        4. 推荐的片头/片尾时长（秒）？

        分镜脚本：
        {shot_list_json}

        视频主题：{topic}

        请以 JSON 格式返回：
        {{
            "add_intro": true|false,
            "add_outro": true|false,
            "style_category": "风格",
            "suggested_duration": 5.0
        }}

        注意：
        - 如果第一个镜头是“开场引导”（如讲师说‘大家好，今天我们讲...’），则不需要片头。
        - 如果最后一个镜头是“结束语”（如‘感谢观看’），则不需要片尾。
        """,
        "username": "CreatorName"
    }

    def __init__(self, node_id: str, name: str = "智能首尾判断生成器"):
        super().__init__(node_id=node_id, node_type="intro_outro", name=name)
        self.qwen = QwenLLM()



    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_context(context)

        # 1. 提取输入
        shot_list: List[Dict] = context["shot_blocks_id"]
        video_topic: str = context.get("video_topic", "未知主题")
        force_intro: bool = context.get("force_intro", False)
        force_outro: bool = context.get("force_outro", False)

        # 2. 使用 Qwen 判断是否需要首尾 + 风格建议
        prompt = self.system_parameters["llm_decision_prompt"].format(
            shot_list_json=json.dumps(shot_list, ensure_ascii=False, indent=2),
            topic=video_topic
        )

        try:
            llm_raw = self.qwen.generate(prompt=prompt,parse_json=True)
            # decision = json.loads(llm_raw)
            decision=llm_raw
        except Exception as e:
            raise RuntimeError(f"Qwen 分析失败: {e}")

        add_intro = decision.get("add_intro", True)
        add_outro = decision.get("add_outro", True)
        style_category = decision.get("style_category", "default")
        suggested_duration = float(decision.get("suggested_duration", 5.0))

        # 强制覆盖
        if force_intro:
            add_intro = True
        if force_outro:
            add_outro = True

        # 3. 构造素材请求
        if not add_intro and not add_outro:
            # 无需调用服务
            intro_segment = None
            outro_segment = None
        else:
            request = IntroOutroRequest(
                duration=suggested_duration,
                category=style_category,
                intro_required=add_intro,
                outro_required=add_outro
            )

            try:
                responses: List[IntroOutroResponse] = await match_introoutro(request)
            except Exception as e:
                raise RuntimeError(f"素材匹配服务调用失败: {e}")

            intro_segment = None
            outro_segment = None

            for resp in responses:
                segment = {
                    "id": f"{resp.type}_{uuid.uuid4().hex[:8]}",
                    "type": resp.type,
                    "video_url": resp.video_url,
                    "duration": resp.cut_end - resp.cut_start,
                    "trim": {
                        "start": resp.cut_start,
                        "end": resp.cut_end
                    },
                    "audio_embedded": resp.audio_embedded,
                    "source_duration": resp.total_duration
                }
                if resp.type == "intro" and add_intro:
                    intro_segment = segment
                elif resp.type == "outro" and add_outro:
                    outro_segment = segment

        # 4. 构建输出
        intro_outro_sequence = {
            "intro": intro_segment,
            "outro": outro_segment,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "ai_decision": {
                    "should_add_intro": decision.get("add_intro"),
                    "should_add_outro": decision.get("add_outro"),
                    "reasoning": "Based on shot list analysis",
                    "style_category": style_category,
                    "suggested_duration": suggested_duration
                },
                "forced": {
                    "intro": force_intro,
                    "outro": force_outro
                },
                "video_topic": video_topic,
                "total_shots": len(shot_list)
            }
        }

        return {"intro_outro_sequence": intro_outro_sequence}

    async def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """支持用户干预"""
        override = user_intent.get("intro_outro_override")
        if not override:
            return await self.generate(context)

        # 支持手动强制添加
        if "force_intro" in override:
            context["force_intro"] = bool(override["force_intro"])
        if "force_outro" in override:
            context["force_outro"] = bool(override["force_outro"])

        return await self.generate(context)