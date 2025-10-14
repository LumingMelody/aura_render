# nodes/aux_text_insertion_node.py

from video_generate_protocol import BaseNode
from typing import Dict, List, Any,Optional
import re
import random
from datetime import datetime
import asyncio

from materials_supplies import match_font,FontRequest,FontResponse
from llm import QwenLLM  # 假设这是你封装好的 Qwen 调用模块

# 文字模板库（可扩展为数据库或JSON配置）
TEXT_TEMPLATES = [
    {
        "id": "fact_001",
        "keywords": "数据|统计|增长|百分比|分析",
        "type": "fact_label",
        "category": "data",
        "text": "{value}% 的用户表示满意",
        "style": "bold_badge",
        "position": "top-center",
        "duration": 3.0,
        "fade_in": 0.3,
        "fade_out": 0.3,
        "color": "#FFD700",
        "font_size": 24,
        "relevance": 0.9
    },
    {
        "id": "quote_002",
        "keywords": "思想|哲学|人生|意义|反思",
        "type": "quote",
        "category": "philosophy",
        "text": "“我们看到的不仅是光，更是时间的回响。”",
        "style": "elegant_center",
        "position": "center",
        "duration": 5.0,
        "fade_in": 0.8,
        "fade_out": 0.8,
        "color": "#FFFFFF",
        "font": "Georgia",
        "relevance": 0.85
    },
    {
        "id": "location_003",
        "keywords": "地点|城市|国家|位置|地图",
        "type": "location_tag",
        "category": "info",
        "text": "📍 {location}",
        "style": "minimalist",
        "position": "bottom-left",
        "duration": 4.0,
        "color": "#A0A0A0",
        "font_size": 18,
        "relevance": 0.9
    },
    {
        "id": "year_004",
        "keywords": "年份|时间|历史|过去|1990|2020",
        "type": "time_tag",
        "category": "info",
        "text": "📅 {year}",
        "style": "retro",
        "position": "top-left",
        "duration": 3.5,
        "color": "#C0C0C0",
        "relevance": 0.88
    },
    {
        "id": "mood_005",
        "keywords": "夜晚|孤独|寂静|森林|神秘",
        "type": "atmosphere_text",
        "category": "mood",
        "text": "寂静中，藏着未说的秘密…",
        "style": "fade_in_out",
        "position": "center-bottom",
        "duration": 6.0,
        "fade_in": 1.0,
        "fade_out": 1.0,
        "color": "#8888FF",
        "italic": True,
        "relevance": 0.8
    },
    {
        "id": "title_006",
        "keywords": "章节|第一部分|引言|序幕",
        "type": "chapter_title",
        "category": "navigation",
        "text": "第一章：起源",
        "style": "cinematic_intro",
        "position": "center",
        "duration": 4.0,
        "color": "#FFFFFF",
        "font_size": 36,
        "bold": True,
        "relevance": 0.92
    }
]

# 视频类型偏好
GENRE_TO_TEXT_HINT = {
    "documentary": ["fact_label", "location_tag", "time_tag", "quote"],
    "educational": ["fact_label", "chapter_title", "data"],
    "vlog": ["location_tag", "mood", "atmosphere_text"],
    "storytelling": ["quote", "mood", "chapter_title"],
    "tech_review": ["fact_label", "data"]
}

# 动态变量映射（从上下文提取）
DYNAMIC_VARS = {
    "location": ["巴黎", "东京", "纽约", "上海", "开罗"],
    "year": ["1920", "1969", "1995", "2020", "2049"],
    "value": ["78", "92", "65", "88", "100"]
}


class AuxTextInsertionNode(BaseNode):
    required_inputs = [
        {
            "name": "shot_blocks_id",
            "label": "分镜描述",
            "type": List[Dict],
            "required": True,
            "desc": "包含镜头类型、视觉描述、节奏等信息的分镜列表",
            "field_type": "json"
        }
    ]

    output_schema=[
        {
            "name": "text_overlay_track_id",
            "label": "额外文字轨道",
            "type": str,
            "desc": "额外文字轨道，如场景描述、提示语等"
        },
       
    ]

    system_parameters = {
        "max_texts_per_scene": 1,
        "min_caption_length": 3,
        "enable_qwen_decision": True,
        "default_font_style": {"color": "#FFFFFF", "stroke": "#000000", "size": 28, "bold": True}
    }

    def __init__(self, node_id: str, name: str = "额外插入的说明或装饰文字"):
        super().__init__(node_id=node_id, node_type="aux_text_insertion", name=name)
        self.qwen = QwenLLM()

    # async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     同步入口，内部运行异步逻辑
    #     """
    #     try:
    #         # 尝试获取当前事件循环
    #         loop = asyncio.get_running_loop()
    #     except RuntimeError:
    #         # 如果没有运行中的循环，则新建一个
    #         loop = asyncio.new_event_loop()
    #         asyncio.set_event_loop(loop)

    #     return loop.run_until_complete(self.generate_async(context))
    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_context(context)
        shot_blocks: List[Dict] = context["shot_blocks_id"]

        text_track = {
            "track_name": "text_overlay",
            "track_type": "text",
            "clips": []
        }

        cumulative_time = 0.0

        # 第一步：决定哪些分镜需要添加补充性说明文字
        selected_shots_for_text = await self._select_relevant_shots(shot_blocks)


        for block in shot_blocks:
            if block not in selected_shots_for_text:
                cumulative_time += block["duration"]
                continue

            duration = block["duration"]
            visual_desc = block["visual_description"]
            raw_caption = block.get("caption", "").strip()
            shot_type = block["shot_type"]
            pacing = block["pacing"]

            # 使用 Qwen 决策：是否显示？位置？样式？
            # 假设 shot_block 中可选包含 'frame_image' 字段
            frame_image = block.get("frame_image")  # 可能为 None

            decision = await self._decide_text_insertion(
                caption=raw_caption,
                visual_desc=visual_desc,
                shot_type=shot_type,
                pacing=pacing,
                frame_image_url=frame_image  # 传入图像（可为 None）
            )

            if not decision or not decision.get("should_display", False):
                cumulative_time += duration
                continue

            # 获取字体
            font_url = await self._get_font_url(visual_desc)

            # 构建字幕片段
            clip = {
                "text": decision["text"],
                "start": cumulative_time,
                "duration": duration,
                "position": decision["position"],
                "font": font_url,
                "style": decision["style"]  # 包含 color, size, bold 等
            }

            text_track["clips"].append(clip)
            cumulative_time += duration

        return {"text_overlay_track_id": text_track}

    async def _select_shots_for_text(self, shot_blocks: List[Dict]) -> List[Dict]:
        """使用Qwen决策选择哪些分镜需要添加补充性说明文字"""
        prompt = """
        根据提供的分镜列表，选出最适合添加补充性说明文字的分镜。请考虑画面的情感表达和氛围营造等因素。

        分镜列表：
        {shots}

        输出 JSON，格式如下：
        {{
            "selected_shots": [{{shot_indices}}]
        }}
        """.format(shots="\n".join([f"{i}. {shot}" for i, shot in enumerate(shot_blocks)]))

        json_schema = {
            "type": "object",
            "properties": {
                "selected_shots": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            },
            "required": ["selected_shots"]
        }

        response = self.qwen.generate(prompt=prompt, parse_json=True)
        selected_indices = response.get("selected_shots", [])
        return [shot_blocks[i] for i in selected_indices]

    async def _decide_text_insertion(
        self,
        caption: str,
        visual_desc: str,
        shot_type: str,
        pacing: str,
        frame_image_url: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        决定辅助文字内容、位置、样式
        优先使用 Qwen-VL（若有图像），否则降级为纯文本分析
        两个阶段都能独立生成完整字段
        """
        # === 共用 JSON Schema ===
        schema = self._get_decision_schema()

        # === 阶段1：如果有图像，使用 Qwen-VL 进行图文理解 ===
        if frame_image_url:
            vl_prompt = f"""
            你是一个专业的视频视觉设计师。请根据提供的画面截图和描述，决定是否添加一条辅助性文字，并完整设计其内容、位置和样式。

            【任务】
            设计一条简短的情绪/氛围词（如“太好吃了！”、“见山见性”、“这一刻”），用于增强感染力。

            【画面信息】
            - 镜头类型：{shot_type}
            - 节奏：{pacing}
            - 视觉描述：{visual_desc}
            - 原始字幕（参考）：{caption}

            【要求】
            1. 文字必须简短（2-6字为佳），有感染力，避免与字幕重复
            2. 位置必须避开人脸、主体、已有文字区域（请结合图像判断构图）
            3. 风格匹配画面氛围（如温馨→手写体感，震撼→大字粗体）
            4. 若画面过于复杂或无安全区域，可返回 should_display: false

            输出 JSON（必须包含所有字段）：
            {{"should_display": true, "text": "文字", "position": "top-center", "style": {{"color": "#FFFFFF", "stroke": "#000000", "size": 36, "bold": true}}}}
            """

            try:
                response = self.qwen.generate(
                    prompt=vl_prompt,
                    images=[frame_image_url],
                    parse_json=True,
                )
                if response:
                    print(f"[✅ Qwen-VL 成功] 生成文字: '{response['text']}' @ {response['position']}")
                    return response
            except Exception as e:
                print(f"[⚠️ Qwen-VL 失败] {e} → 降级到纯文本分析...")

        # === 阶段2：无图像 或 VL 失败 → 使用纯文本分析（必须能独立生成完整决策）===
        text_prompt = f"""
        你是一个视频文案设计师。当前无法获取画面截图，需仅根据文字描述，决定是否添加辅助性文字，并设计其内容、位置和样式。

        【画面描述】
        {visual_desc}

        - 镜头类型：{shot_type}
        - 节奏：{pacing}
        - 原始字幕：{caption}

        【任务】
        请设计一条简短的情绪/氛围词（如“太震撼了”、“静谧时光”、“突破自我”），增强画面感染力。

        【要求】
        1. 文字 2-6 字，有感染力，避免与字幕重复
        2. 推测构图并选择安全位置（如：人物居左 → 文字放右；顶部开阔 → 放 top-center）
        3. 风格匹配氛围（颜色、粗细等）
        4. 若描述模糊或信息密集，可返回 should_display: false

        推荐位置选项：
        top-left, top-center, top-right, center, center-bottom, bottom-left, bottom-center, bottom-right

        输出 JSON（必须完整，格式如下）：
        {{"should_display": true, "text": "文字", "position": "top-center", "style": {{"color": "#FFFFFF", "stroke": "#000000", "size": 36, "bold": true}}}}
        """

        try:
            response = self.qwen.generate(
                prompt=text_prompt,
                parse_json=True,
            )
            if response:
                print(f"[✅ 文本分析成功] 生成文字: '{response['text']}' @ {response['position']}")
                return response
        except Exception as e:
            print(f"[❌ 文本分析失败] {e}")

        return None
    async def _get_font_url(self, visual_desc: str) -> str:
        """调用外部服务获取字体 URL"""
        try:
            request = FontRequest(description=visual_desc)
            fonts: List[FontResponse] = await match_font(request)
            return fonts[0].url if fonts else "https://fonts.example.com/default.ttf"
        except Exception as e:
            print(f"[Font Error] {e}")
            return "https://fonts.example.com/default.ttf"
        
    
    async def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        重新生成接口，支持用户干预
        user_intent 示例:
        {
            "action": "reselect",  # 或 "modify_text", "adjust_position"
            "target_shot_index": 1,
            "new_text": "太震撼了！",
            "new_position": "top-center"
        }
        """
        shot_blocks: List[Dict] = context["shot_blocks"]
        user_action = user_intent.get("action")

        # 1. 先生成原始结果
        result = await self.generate(context)
        clips = result["aux_text_track"]["clips"]

        if not clips:
            return result  # 无法修改空结果

        if user_action == "reselect":
            # 用户希望重新选择哪些镜头加文字
            selected_blocks = await self._select_relevant_shots(shot_blocks, user_intent)
            # 重新生成（可加入用户偏好）
            return await self.generate({**context, "user_intent": user_intent})

        elif user_action == "modify_text" and "target_shot_index" in user_intent:
            idx = user_intent["target_shot_index"]
            if 0 <= idx < len(clips):
                new_text = user_intent.get("new_text", clips[idx]["text"])
                # 保持其他属性，只改文字
                clips[idx]["text"] = new_text

        elif user_action == "adjust_position" and "target_shot_index" in user_intent:
            idx = user_intent["target_shot_index"]
            if 0 <= idx < len(clips):
                new_pos = user_intent.get("new_position", clips[idx]["position"])
                clips[idx]["position"] = new_pos

        elif user_action == "regenerate_all":
            # 完全重新生成（可加入新提示）
            prompt_hint = user_intent.get("prompt_hint", "")
            # 可扩展：将 hint 传入 _select_relevant_shots 或 _decide_text_insertion
            return await self.generate(context)  # 简化版，实际可更智能

        return result

    async def _select_relevant_shots(self, shot_blocks: List[Dict], user_intent: Dict[str, Any] = None) -> List[Dict]:
        """
        全局决策：哪些镜头适合加辅助文字（避免每个都加）
        """
        prompt = f"""
        你是一个视频叙事设计师。请从以下分镜中，选择最适合添加**非字幕类辅助性文字**的镜头。
        这类文字是情绪性（如“太好吃了！”）、氛围性（如“见山见性”）、或强调性短语，用于增强感染力。

        请避免选择：
        - 人物说话/讲解的镜头（已有字幕）
        - 画面信息密集或文字/人脸遮挡风险高的镜头

        分镜列表：
        {self._format_shots_for_prompt(shot_blocks)}

        {f'用户偏好提示：{user_intent.get("prompt_hint", "")}' if user_intent else ''}

        输出 JSON：
        {{
            "selected_indices": [0, 2]  // 选中的分镜索引
        }}
        """

        json_schema = {
            "type": "object",
            "properties": {
                "selected_indices": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "uniqueItems": True
                }
            },
            "required": ["selected_indices"]
        }

        try:
            response = self.qwen.generate(
                prompt=prompt,
                parse_json=True,
            )
            indices = response.get("selected_indices", [])
            # 过滤合法索引
            return [shot_blocks[i] for i in indices if 0 <= i < len(shot_blocks)]
        except Exception as e:
            print(f"[Selection Error] {e}")
            return []  # 默认不加



    def _get_decision_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "should_display": {"type": "boolean"},
                "text": {"type": "string", "minLength": 1},
                "position": {
                    "type": "string",
                    "enum": ["top-left", "top-center", "top-right", "center", "center-bottom", 
                            "bottom-left", "bottom-center", "bottom-right"]
                },
                "style": {
                    "type": "object",
                    "properties": {
                        "color": {"type": "string"},
                        "stroke": {"type": "string"},
                        "size": {"type": "integer", "minimum": 12, "maximum": 72},
                        "bold": {"type": "boolean"},
                        "italic": {"type": "boolean"}
                    },
                    "required": ["color", "size"]
                }
            },
            "required": ["should_display", "text", "position", "style"]
        }
    

    def _format_shots_for_prompt(self, shot_blocks: List[Dict]) -> str:
        """格式化分镜用于 prompt"""
        lines = []
        for i, block in enumerate(shot_blocks):
            lines.append(f"{i}. [{block['shot_type']}] {block['visual_description']} (时长: {block['duration']}s)")
        return "\n".join(lines)
