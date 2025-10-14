# nodes/dynamic_effects_node.py

from video_generate_protocol import BaseNode
from typing import Dict, List, Any
import random
import math
import re
import json

from llm import QwenLLM  # 假设这是你封装好的 Qwen 调用模块

# 特效类型库
EFFECT_TEMPLATES = {
    "lens_flare": {
        "name": "镜头光晕",
        "type": "overlay",
        "layer": "top",
        "position": "auto",
        "animation": "fadeInOut",
        "opacity": 0.6,
        "scale": 1.0,
        "blend_mode": "screen",
        "trigger": ["high_emotion", "bright_scene"],
        "description": "模拟真实光学镜头光晕，增强画面戏剧性"
    },
    "particle_sparkle": {
        "name": "粒子星光",
        "type": "particle",
        "emitter": "point",
        "count": 50,
        "lifetime": 3.0,
        "speed": 0.8,
        "color": "gold",
        "trigger": ["high_emotion", "celebration"],
        "description": "细小闪烁粒子，营造梦幻或激动氛围"
    },
    "glow_pulse": {
        "name": "脉冲辉光",
        "type": "effect",
        "target": "subject",
        "radius": 15,
        "intensity": 1.2,
        "frequency": 2.0,
        "trigger": ["high_emotion", "focus_moment"],
        "description": "主体周围柔和脉动光晕，突出关键瞬间"
    },
    "animated_textbox": {
        "name": "动态文字框",
        "type": "text",
        "style": "modern",
        "animation": "slideUp",
        "position": "safe_bottom",
        "background_opacity": 0.7,
        "trigger": ["info_scene", "narration"],
        "description": "带动画进入的半透明文字框，用于说明性内容"
    },
    "border_glow": {
        "name": "边缘辉光",
        "type": "effect",
        "side": "all",
        "color": "blue",
        "width": 3,
        "opacity": 0.4,
        "trigger": ["suspense", "tech_scene"],
        "description": "画面边缘发光，增强科技感或紧张氛围"
    },
    "film_grain": {
        "name": "胶片颗粒",
        "type": "overlay",
        "intensity": 0.3,
        "trigger": ["cinematic", "retro"],
        "description": "轻微颗粒感，增强电影质感"
    }
}

# 情感强度映射
EMOTION_TO_INTENSITY = {
    "激昂": 0.9,
    "感动": 0.85,
    "励志": 0.8,
    "幽默": 0.6,
    "冷静": 0.3,
    "悬疑": 0.7,
    "温馨": 0.75
}

# 安全区域比例（相对于分辨率）
SAFE_AREA = {
    "title": 0.9,   # 标题安全区（90%居中）
    "action": 0.95  # 动作安全区（95%）
}

# 默认分辨率
DEFAULT_RESOLUTION = (1920, 1080)


class DynamicEffectsNode(BaseNode):
    required_inputs = [
        {
            "name": "filter_sequence_id",
            "label": "剪辑序列",
            "type": list[Dict],
            "required": False,
            "default": [],
            "desc": "包含镜头信息的剪辑序列，支持 emotion_hint, scene_type, faces 等字段",
            "field_type": "json"
        },
        {
            "name": "emotions_id",
            "label": "情感强度",
            "type": dict,
            "required": False,
            "desc": "情感标签及权重，如 {'励志': 50, '冷静': 30}",
            "field_type": "json"
        },
        {
            "name": "resolution",
            "label": "输出分辨率",
            "type": tuple,
            "required": False,
            "default": DEFAULT_RESOLUTION,
            "desc": "视频输出分辨率，如 (1920, 1080)",
            "field_type": "text"
        }
    ]


    output_schema=[
         {
            "name": "effects_sequence_id",
            "label": "添加特效的分镜块列表",
            "type": list,
            "required": True,
            "desc": "包含镜头信息及添加的特效，如 [{'shot_id': 's1', 'shot_type': '全景', 'pacing': '快剪', 'emotion_hint': '激昂', 'visual_effects': [{'effect_id': 'eff_lens_flare_1234', 'name': '镜头光晕', 'position': {'x': 960, 'y': 540}, 'safe_position': True, 'position_reason': '画面右上角空旷区域'}]}]",
            "field_type": "json"
        }
        
    ]
    file_upload_config = {
        "image": {"enabled": True, "accept": ".png,.jpg,.svg", "desc": "可上传自定义贴图（如光晕PNG）"},
        "video": {"enabled": False}
    }

    system_parameters = {
        "max_effects_per_clip": 2,
        "default_emotion": "冷静",
        "min_emotion_threshold": 0.6  # 触发高情感特效的阈值
    }


    def __init__(self, node_id: str, name: str = "动态特效添加"):
        self.node_id = node_id
        self.node_type = "dynamic_effects"
        self.name = name
        self.qwen_caller = QwenLLM() 
        self.resolution = DEFAULT_RESOLUTION
        self.uploaded_files = []

    def set_qwen_caller(self, caller: QwenLLM):
        """注入 Qwen 调用器"""
        self.qwen_caller = caller
        return self

    def set_resolution(self, width: int, height: int):
        self.resolution = (width, height)
        return self

    def _extract_text_from_clip(self, clip: Dict) -> str:
        desc = clip.get("metadata", {}).get("description", "")
        match = re.search(r"Placeholder for: (.+)", desc)
        return match.group(1).strip() if match else desc

    def _get_effect_suggestions_from_qwen(self, description: str) -> List[str]:
        """使用文本 Qwen 推荐特效（可改为 VL 统一接口）"""
        prompt = f"""
你是一个专业的视频剪辑导演...

可用特效：
{list(EFFECT_TEMPLATES.keys())}

镜头描述：
"{description}"

要求：只返回 JSON 数组，如 ["lens_flare", "film_grain"]
        """
        resp = self.qwen_caller.generate(prompt)
        if not resp:
            return []

        try:
            json_start = resp.find("[")
            json_end = resp.rfind("]") + 1
            if json_start == -1 or json_end <= 0:
                return []
            effects = json.loads(resp[json_start:json_end])
            return [e for e in effects if e in EFFECT_TEMPLATES][:2]
        except:
            return []

    def _get_position_from_image_vl(self, clip: Dict, effect_name: str) -> Dict:
        
        return {
            "position_px": {"x": int(1920), "y": int(1080)},
            "position_norm": {"x": 1, "y": 1},
            "safe": False,
            "reason": "图像分析失败，使用随机位置",
            "raw": None
        }
        
        image_url = clip["source_url"]
        description = self._extract_text_from_clip(clip)
        width, height = self.resolution

        # 🔹 由本类构造 prompt
        prompt = f"""
            请分析图像，为【{effect_name}】特效推荐一个合适的添加位置。

            【画面描述】
            {description}

            【要求】
            - 输出一个 JSON 对象
            - 包含字段：position (x, y)，范围 0~1；safe (boolean)；reason (string)
            - 位置应为空旷区域，避免遮挡人脸或主体

            示例：
            {{"position": {{"x": 0.8, "y": 0.2}}, "safe": true, "reason": "右上角是天空"}}
                    """

        # 🔹 调用通用接口
        result = self.qwen_caller.generate(
            prompt=prompt,
            images=[image_url],
            parse_json=True,
            json_schema={
                "position": "dict",
                "safe": "bool",
                "reason": "str"
            },
            max_retries=3
        )

        # 🔹 本类负责解析和 fallback
        if isinstance(result, dict):
            pos = result.get("position", {})
            x = max(0.0, min(1.0, pos.get("x", random.uniform(0.3, 0.7))))
            y = max(0.0, min(1.0, pos.get("y", random.uniform(0.3, 0.7))))
            return {
                "position_px": {"x": int(x * width), "y": int(y * height)},
                "position_norm": {"x": x, "y": y},
                "safe": bool(result.get("safe", False)),
                "reason": str(result.get("reason", "AI 推荐位置")),
                "raw": result
            }

        # 🔹 Fallback
        x_norm, y_norm = random.uniform(0.2, 0.8), random.uniform(0.2, 0.6)
        return {
            "position_px": {"x": int(x_norm * width), "y": int(y_norm * height)},
            "position_norm": {"x": x_norm, "y": y_norm},
            "safe": False,
            "reason": "图像分析失败，使用随机位置",
            "raw": None
        }
    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_context(context)
        if not self.qwen_caller:
            raise RuntimeError("必须先调用 set_qwen_caller() 设置 Qwen 调用器")

        sequence: List[Dict] = context.get("filter_sequence_id", [])
        self.resolution = context.get("resolution", DEFAULT_RESOLUTION)

        processed_sequence = []

        for clip in sequence:
            description = self._extract_text_from_clip(clip)
            if not description.strip():
                clip["visual_effects"] = []
                processed_sequence.append(clip)
                continue

            # Step 1: Qwen 推荐特效
            suggested_keys = self._get_effect_suggestions_from_qwen(description)
            if not suggested_keys:
                clip["visual_effects"] = []
                processed_sequence.append(clip)
                continue

            # Step 2: 为每个特效调用图像理解定位
            effects_to_add = []
            for key in suggested_keys:
                template = EFFECT_TEMPLATES[key]
                effect = {**template}

                # 调用 VL 获取智能位置
                pos_result = self._get_position_from_image_vl(clip, effect["name"])
                effect["position"] = pos_result["position_px"]  # 像素坐标
                effect["safe_position"] = pos_result["safe"]
                effect["position_reason"] = pos_result["reason"]

                # 唯一 ID
                effect["effect_id"] = f"eff_{key}_{random.randint(1000, 9999)}"

                effects_to_add.append(effect)

            # 自定义贴图支持
            if self.uploaded_files and "lens_flare" in suggested_keys:
                for file_info in self.uploaded_files:
                    if file_info["type"] == "image" and "flare" in file_info["filename"].lower():
                        for eff in effects_to_add:
                            if eff["name"] == "镜头光晕":
                                eff["texture_path"] = file_info["path"]
                                eff["custom"] = True
                        break

            clip["visual_effects"] = effects_to_add
            processed_sequence.append(clip)

        return {"effects_sequence_id": processed_sequence}

    def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """支持用户干预，如：“这个镜头加金色粒子” 或 “移除光晕”"""
        super().regenerate(context, user_intent)

        override = user_intent.get("effects_override")
        if not override:
            return self.generate(context)

        result = self.generate(context)
        sequence = result["edited_sequence"]

        if "clear_all" in override:
            for clip in sequence:
                if override["clear_all"]:
                    clip["visual_effects"] = []
        elif "per_clip" in override:
            for item in override["per_clip"]:
                shot_id = item["shot_id"]
                effects = item["effects"]  # list of effect keys
                for clip in sequence:
                    if clip.get("shot_id") == shot_id:
                        clip["visual_effects"] = []
                        for eff_key in effects:
                            if eff_key in EFFECT_TEMPLATES:
                                clip["visual_effects"].append(
                                    self._create_effect(eff_key, context.get("resolution", DEFAULT_RESOLUTION), clip.get("faces", []))
                                )
                        # 限制数量
                        clip["visual_effects"] = clip["visual_effects"][:self.system_parameters["max_effects_per_clip"]]

        return result
    


if __name__ == "__main__":
     # 加载你的输入数据
    input_data = {
  "edited_sequence": [
    {
      "id": "clip_1a2b3c4d",
      "index": 0,
      "asset_id": "placeholder_0",
      "source_url": "https://example.com/assets/placeholder.mp4",
      "start": 0.0,
      "end": 4.5,
      "duration": 4.5,
      "source": {
        "in": 0.0,
        "out": 4.5
      },
      "transition_in": {
        "type": "cross_dissolve",
        "duration": 0
      },
      "transition_out": {
        "type": "cross_dissolve",
        "name": "叠化",
        "duration": 1.2
      },
      "metadata": {
        "description": "Placeholder for: 一个阳光明媚的早晨，鸟儿在树上歌唱，城市 慢慢苏醒。...",
        "tags": [],
        "provider": "system_placeholder"
      },
      "transform": {
        "scale": 1.0,
        "position": "center"
      },
      "color_filter": {
        "preset": "cinematic",
        "name": "电影感",
        "lut_path": "luts/cinematic.cube",
        "intensity": 0.8,
        "applied_params": {
          "contrast": 1.792,
          "saturation": 1.664,
          "shadows": 0.98,
          "highlights": 1.69,
          "temperature": 0.24
        },
        "scene_adaptation": "户外晴天",
        "description": "低饱和、高对比，适合叙事类视频",
        "source": "ai_decision_with_consistency",
        "global_theme": "cinematic",
        "local_decision": "cinematic",
        "smooth_adjusted": False
      }
    },
    {
      "id": "clip_5e6f7g8h",
      "index": 1,
      "asset_id": "placeholder_1",
      "source_url": "https://example.com/assets/placeholder.mp4",
      "start": 5.0,
      "end": 8.0,
      "duration": 3.0,
      "source": {
        "in": 0.0,
        "out": 3.0
      },
      "transition_in": {
        "type": "cross_dissolve",
        "duration": 0.5
      },
      "transition_out": {
        "type": "cross_dissolve",
        "name": "叠化",
        "duration": 1.2
      },
      "metadata": {
        "description": "Placeholder for: 一位年轻人在咖啡馆里专注地敲着笔记本电脑 ，周围人来往。...",
        "tags": [],
        "provider": "system_placeholder"
      },
      "transform": {
        "scale": 1.0,
        "position": "center"
      },
      "color_filter": {
        "preset": "cinematic",
        "name": "电影感",
        "lut_path": "luts/cinematic.cube",
        "intensity": 0.8,
        "applied_params": {
          "contrast": 1.12,
          "saturation": 0.96,
          "shadows": 1.82,
          "highlights": 0.93,
          "temperature": 0.304
        },
        "scene_adaptation": "室内",
        "description": "低饱和、高对比，适合叙事类视频",
        "source": "ai_decision_with_consistency",
        "global_theme": "cinematic",
        "local_decision": "cinematic",
        "smooth_adjusted": False
      }
    },
    {
      "id": "clip_9i0j1k2l",
      "index": 2,
      "asset_id": "placeholder_2",
      "source_url": "https://example.com/assets/placeholder.mp4",
      "start": 8.5,
      "end": 11.5,
      "duration": 3.0,
      "source": {
        "in": 0.0,
        "out": 3.0
      },
      "transition_in": {
        "type": "cross_dissolve",
        "duration": 0.5
      },
      "transition_out": {
        "type": "none",
        "duration": 0.0
      },
      "metadata": {
        "description": "Placeholder for: 夕阳下，一对情侣在海边散步，背影温馨。...",
        "tags": [],
        "provider": "system_placeholder"
      },
      "transform": {
        "scale": 1.0,
        "position": "center"
      },
      "color_filter": {
        "preset": "cinematic",
        "name": "电影感",
        "lut_path": "luts/cinematic.cube",
        "intensity": 0.8,
        "applied_params": {
          "contrast": 1.12,
          "saturation": 1.696,
          "shadows": 0.98,
          "highlights": 1.81,
          "temperature": 0.368
        },
        "scene_adaptation": "黄昏",
        "description": "低饱和、高对比，适合叙事类视频",
        "source": "ai_decision_with_consistency",
        "global_theme": "cinematic",
        "local_decision": "cinematic",
        "smooth_adjusted": False
      }
    }
  ],
  "color_consistency_report": {
    "main_theme": "cinematic",
    "filter_chain": [
      "cinematic",
      "cinematic",
      "cinematic"
    ],
    "consistency_score": 0.83,
    "total_clips": 3
  }
}
    # 创建节点
    node = DynamicEffectsNode(node_id="dynamic_fx_01")

    # 调用 generate
    result = node.generate(input_data)

    # 输出结果
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))