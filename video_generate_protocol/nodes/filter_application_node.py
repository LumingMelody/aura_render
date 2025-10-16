# nodes/filter_application_node.py

from video_generate_protocol import BaseNode
import logging

logger = logging.getLogger(__name__)

from typing import Dict, List, Any
import os
import json
import re
from llm import QwenLLM


# 内置滤镜预设库（保持不变）
FILTER_PRESETS = {
    "cinematic": {
        "name": "电影感",
        "lut_path": "luts/cinematic.cube",
        "base_params": {
            "contrast": 1.15,
            "saturation": 0.95,
            "shadows": 0.98,
            "highlights": 0.93,
            "temperature": 0.05
        },
        "description": "低饱和、高对比，适合叙事类视频"
    },
    "vibrant": {
        "name": "鲜艳",
        "lut_path": "luts/vibrant.cube",
        "base_params": {
            "contrast": 1.1,
            "saturation": 1.2,
            "shadows": 1.0,
            "highlights": 0.95,
            "temperature": 0.0
        },
        "description": "增强色彩，适合旅游、VLOG"
    },
    "monochrome": {
        "name": "黑白",
        "lut_path": "luts/monochrome.cube",
        "base_params": {
            "contrast": 1.2,
            "saturation": 0.0,
            "shadows": 0.95,
            "highlights": 1.05,
            "temperature": 0.0
        },
        "description": "经典黑白胶片风格"
    },
    "dreamy": {
        "name": "梦幻",
        "lut_path": "luts/dreamy.cube",
        "base_params": {
            "contrast": 0.9,
            "saturation": 0.85,
            "shadows": 1.1,
            "highlights": 0.85,
            "temperature": 0.1,
            "blur": 0.3
        },
        "description": "柔光+低对比，适合情感类片段"
    },
    "cyberpunk": {
        "name": "赛博朋克",
        "lut_path": "luts/cyberpunk.cube",
        "base_params": {
            "contrast": 1.3,
            "saturation": 1.4,
            "shadows": 0.8,
            "highlights": 1.2,
            "temperature": -0.2,
            "teal_shift": 0.4,
            "glow": 0.5
        },
        "description": "高对比+青橙色，适合科技、未来感"
    },
    "natural": {
        "name": "自然",
        "lut_path": "luts/natural.cube",
        "base_params": {
            "contrast": 1.0,
            "saturation": 1.05,
            "shadows": 1.0,
            "highlights": 0.98,
            "temperature": 0.0
        },
        "description": "轻微优化，保留真实感"
    }
}

# 场景关键词 → 场景类型映射
SCENE_KEYWORDS = {
    "早晨": "户外晴天",
    "清晨": "户外晴天",
    "白天": "户外晴天",
    "阳光": "户外晴天",
    "城市": "户外晴天",
    "咖啡馆": "室内",
    "室内": "室内",
    "夜晚": "夜景",
    "夜景": "夜景",
    "傍晚": "黄昏",
    "黄昏": "黄昏",
    "夕阳": "黄昏",
    "海边": "黄昏",
    "水下": "水下",
    "森林": "户外晴天",
    "公园": "户外晴天"
}

# 场景类型 → 色彩微调规则（保持不变）
SCENE_TO_COLOR_ADJUST = {
    "夜景": {
        "temperature": -0.15,
        "shadows": 0.85,
        "contrast": 1.1,
        "brightness": -0.05
    },
    "黄昏": {
        "temperature": 0.2,
        "saturation": 1.15,
        "highlights": 1.1
    },
    "室内": {
        "temperature": 0.1,
        "shadows": 1.05
    },
    "户外晴天": {
        "saturation": 1.1,
        "contrast": 1.05,
        "highlights": 0.95
    },
    "阴天": {
        "temperature": -0.1,
        "saturation": 0.9,
        "brightness": -0.05
    },
    "水下": {
        "teal_shift": 0.3,
        "temperature": -0.2,
        "contrast": 1.1
    }
}

class FilterApplicationNode(BaseNode):
    # ✅ 修改输入字段名为 edited_sequence
    required_inputs = [
        {
            "name": "transition_sequence_id",
            "label": "添加转场的分镜块列表",
            "type": list[Dict],
            "required": False,
            "default": [],
            "desc": "包含镜头信息及转场的剪辑序列，如 [{'shot_id': 's1', 'shot_type': '全景', 'pacing': '快剪', 'emotion_hint': '激昂', 'transition_out': {'type': 'cross_dissolve', 'duration': 1.2}}]",
            "field_type": "json"
        },
        {
            "name": "style_config_id",
            "label": "艺术风格配置",
            "type": dict,
            "required": False,
            "desc": "指定滤镜风格，如 {'filter_preset': 'cinematic', 'intensity': 0.8}",
            "field_type": "json"
        }
    ]

    output_schema=[
         {
            "name": "filter_sequence_id",
            "label": "添加滤镜的分镜块列表",
            "type": list,
            "required": True,
            "desc": "包含镜头信息及滤镜应用的剪辑序列，如 [{'shot_id': 's1', 'shot_type': '全景', 'pacing': '快剪', 'emotion_hint': '激昂', 'color_filter': {'preset': 'cinematic', 'intensity': 0.8, 'applied_params': {...}}}]",
            "field_type": "json"
        }
        
    ]


    file_upload_config = {
        "image": {"enabled": False},
        "video": {"enabled": False},
        "file": {
            "enabled": True,
            "accept": ".cube,.png,.json",
            "desc": "可上传自定义 LUT 文件 (.cube) 或滤镜配置"
        }
    }

    system_parameters = {
        "default_filter": "natural",
        "max_intensity": 1.2,
        "min_intensity": 0.5
    }

    def __init__(self, node_id: str, name: str = "滤镜应用"):
        super().__init__(node_id=node_id, node_type="filter_application", name=name)

    def detect_scene_type(self, description: str) -> str:
        """从描述中检测场景类型"""
        for keyword, scene in SCENE_KEYWORDS.items():
            if keyword in description:
                return scene
        return "通用"

    # 在 FilterApplicationNode 类中替换 generate 方法

    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_context(context)

        sequence: List[Dict] = context["transition_sequence_id"]
        style_config: Dict = context.get("style_config", {})

        try:

            qwen = QwenLLM()
        except ImportError:
            qwen = None
            logger.info(f"Warning: QwenLLM not available, falling back to rule-based.")

        if not sequence:
            return {"filter_sequence_id": []}

        # ✅ Step 1: 提取所有描述，用于全局分析
        descriptions = []
        clean_descs = []
        for clip in sequence:
            desc = clip.get("metadata", {}).get("description", "")
            clean_desc = re.sub(r"^Placeholder for:\s*", "", desc).strip()
            clean_desc = clean_desc or "一个普通的视频片段"
            descriptions.append(desc)
            clean_descs.append(clean_desc)

        # ✅ Step 2: Qwen 全局分析 → 提取“主基调”
        main_theme = "natural"
        if qwen:
            theme_prompt = f"""
    你是一个影视调色总监。请分析以下视频片段序列的整体情绪和视觉风格，确定一个统一的主色调风格。

    【片段描述列表】
    {json.dumps(clean_descs, ensure_ascii=False, indent=2)}

    请从以下滤镜中选择一个作为主基调（只能返回一个名称）：
    cinematic, vibrant, monochrome, dreamy, cyberpunk, natural

    要求：
    - 如果整体偏情感、抒情 → dreamy 或 cinematic
    - 如果是城市、科技、未来 → cyberpunk
    - 如果是自然、旅行 → vibrant
    - 如果是纪实、访谈 → natural
    - 如果是故事性强 → cinematic

    只返回滤镜名称，不要解释。
    """.strip()
            try:
                response = qwen.generate(prompt=theme_prompt)
                main_theme = response.strip().lower()
                if main_theme not in FILTER_PRESETS:
                    main_theme = "natural"
                logger.info(f"[全局分析] 主基调: {main_theme}")
            except Exception as e:
                logger.info(f"Qwen 主基调分析失败: {e}")
                main_theme = "natural"
        else:
            main_theme = style_config.get("filter_preset", "natural")

        # ✅ Step 3: 逐镜头处理 + 连贯性控制
        result_sequence = []
        prev_filter = None  # 记录上一个镜头的滤镜

        for i, clip in enumerate(sequence):
            processed_clip = {**clip}
            clean_desc = clean_descs[i]
            intensity = style_config.get("intensity", 0.8)
            intensity = max(self.system_parameters["min_intensity"], min(self.system_parameters["max_intensity"], intensity))

            # 场景类型识别
            scene_type = self.detect_scene_type(clean_desc)

            # ✅ Qwen 局部推荐
            if qwen:
                prompt = f"""
                    你是一个视频调色师。请为以下片段选择最合适的滤镜（从列表中选）：
                    可用：cinematic, vibrant, monochrome, dreamy, cyberpunk, natural

                    描述：{clean_desc}

                    要求：
                    - 优先考虑与主基调 '{main_theme}' 的一致性
                    - 若情绪差异大可微调，但避免完全相反风格

                    只返回滤镜名称。
                    """.strip()
                try:
                    response = qwen.generate(prompt=prompt)
                    local_filter = response.strip().lower()
                    if local_filter not in FILTER_PRESETS:
                        local_filter = main_theme  # fallback 到主基调
                except:
                    local_filter = main_theme
            else:
                local_filter = main_theme

            # ✅ 连贯性策略：避免突兀切换
            recommended_filter = local_filter

            if prev_filter:
                # 定义“冲突风格对”
                conflicting_pairs = {
                    ("dreamy", "cyberpunk"),
                    ("cyberpunk", "dreamy"),
                    ("vibrant", "monochrome"),
                    ("monochrome", "vibrant"),
                    ("cinematic", "vibrant")  # 视情况可调
                }
                if (prev_filter["preset"], recommended_filter) in conflicting_pairs:
                    logger.info(f"检测到风格冲突: {prev_filter['preset']} → {recommended_filter}，平滑处理...")
                    # 平滑策略：保留主基调，或降低强度
                    recommended_filter = main_theme
                    intensity = intensity * 0.7  # 降低强度以缓冲

            base_filter = FILTER_PRESETS[recommended_filter]
            final_params = {**base_filter["base_params"]}

            # 场景微调
            if scene_type in SCENE_TO_COLOR_ADJUST:
                scene_adjust = SCENE_TO_COLOR_ADJUST[scene_type]
                for key, value in scene_adjust.items():
                    final_params[key] = final_params.get(key, 1.0) + value * intensity

            # 强度控制
            scalable_keys = ["contrast", "saturation", "temperature", "teal_shift", "glow", "blur"]
            for key in scalable_keys:
                if key in final_params:
                    center = 1.0 if key not in ["teal_shift", "glow", "blur"] else 0.0
                    final_params[key] = center + (final_params[key] - center) * intensity

            # 构建滤镜应用
            filter_application = {
                "preset": recommended_filter,
                "name": base_filter["name"],
                "lut_path": base_filter["lut_path"],
                "intensity": round(intensity, 2),
                "applied_params": {k: round(v, 3) for k, v in final_params.items()},
                "scene_adaptation": scene_type,
                "description": base_filter["description"],
                "source": "ai_decision_with_consistency",
                "global_theme": main_theme,
                "local_decision": recommended_filter,
                "smooth_adjusted": recommended_filter != local_filter if prev_filter else False
            }

            # 自定义 LUT 覆盖（最高优先级）
            # if self.uploaded_files:
            #     for file_info in self.uploaded_files:
            #         if file_info["type"] == "file" and file_info["filename"].endswith(".cube"):
            #             filter_application["lut_path"] = file_info["path"]
            #             filter_application["name"] = "自定义LUT"
            #             break

            processed_clip["color_filter"] = filter_application
            result_sequence.append(processed_clip)

            # 更新 prev_filter 用于下一帧
            prev_filter = filter_application

            logger.info(f"✅ 片段 {i} '{clip.get('id', 'unknown')}' 处理完成 → 滤镜: {main_theme}")  # 👈 确认每帧都处理

        # ✅ Step 4: 输出连贯性评分（可选，用于调试）
        filter_chain = [clip["color_filter"]["preset"] for clip in result_sequence]
        diversity = len(set(filter_chain))
        consistency_score = round(1.0 - (diversity / len(filter_chain)) * 0.5, 2)  # 简单评分
        logger.info(f"[连贯性分析] 滤镜链: {filter_chain} | 风格多样性: {diversity} | 一致性评分: {consistency_score}")
        print(json.dumps(result_sequence,indent=2, ensure_ascii=False))
        return {
            "filter_sequence_id": result_sequence,
            # "color_consistency_report": {
            #     "main_theme": main_theme,
            #     "filter_chain": filter_chain,
            #     "consistency_score": consistency_score,
            #     "total_clips": len(filter_chain)
            # }
        }

    def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        super().regenerate(context, user_intent)

        override = user_intent.get("filter_override")
        if override and isinstance(override, dict):
            result = self.generate(context)
            sequence = result["edited_sequence"]

            if "global_preset" in override:
                new_preset = override["global_preset"]
                if new_preset in FILTER_PRESETS:
                    for clip in sequence:
                        if "color_filter" in clip:
                            base = FILTER_PRESETS[new_preset]
                            clip["color_filter"]["preset"] = new_preset
                            clip["color_filter"]["name"] = base["name"]
                            clip["color_filter"]["lut_path"] = base["lut_path"]
                            clip["color_filter"]["description"] = base["description"]

            elif "per_clip" in override:
                for item in override["per_clip"]:
                    clip_id = item["clip_id"]
                    preset = item["preset"]
                    if preset in FILTER_PRESETS:
                        for clip in sequence:
                            if clip.get("id") == clip_id:
                                base = FILTER_PRESETS[preset]
                                clip["color_filter"] = {
                                    "preset": preset,
                                    "name": base["name"],
                                    "lut_path": base["lut_path"],
                                    "intensity": item.get("intensity", 1.0),
                                    "applied_params": base["base_params"],
                                    "scene_adaptation": self.detect_scene_type(
                                        clip.get("metadata", {}).get("description", "")
                                    ),
                                    "description": base["description"]
                                }

            return result

        return self.generate(context)
    


if __name__ == "__main__":
    # 你提供的输入数据
    INPUT_DATA = {
        "edited_sequence": [
            {
                "id": "clip_1a2b3c4d",
                "index": 0,
                "asset_id": "placeholder_0",
                "source_url": "https://example.com/assets/placeholder.mp4",
                "start": 0.0,
                "end": 4.5,
                "duration": 4.5,
                "source": {"in": 0.0, "out": 4.5},
                "transition_in": {"type": "cross_dissolve", "duration": 0},
                "transition_out": {"type": "cross_dissolve", "name": "叠化", "duration": 1.2},
                "metadata": {
                    "description": "Placeholder for: 一个阳光明媚的早晨，鸟儿在树上歌唱，城市慢慢苏醒。...",
                    "tags": [],
                    "provider": "system_placeholder"
                },
                "transform": {"scale": 1.0, "position": "center"}
            },
            {
                "id": "clip_5e6f7g8h",
                "index": 1,
                "asset_id": "placeholder_1",
                "source_url": "https://example.com/assets/placeholder.mp4",
                "start": 5.0,
                "end": 8.0,
                "duration": 3.0,
                "source": {"in": 0.0, "out": 3.0},
                "transition_in": {"type": "cross_dissolve", "duration": 0.5},
                "transition_out": {"type": "cross_dissolve", "name": "叠化", "duration": 1.2},
                "metadata": {
                    "description": "Placeholder for: 一位年轻人在咖啡馆里专注地敲着笔记本电脑，周围人来往。...",
                    "tags": [],
                    "provider": "system_placeholder"
                },
                "transform": {"scale": 1.0, "position": "center"}
            },
            {
                "id": "clip_9i0j1k2l",
                "index": 2,
                "asset_id": "placeholder_2",
                "source_url": "https://example.com/assets/placeholder.mp4",
                "start": 8.5,
                "end": 11.5,
                "duration": 3.0,
                "source": {"in": 0.0, "out": 3.0},
                "transition_in": {"type": "cross_dissolve", "duration": 0.5},
                "transition_out": {"type": "none", "duration": 0.0},
                "metadata": {
                    "description": "Placeholder for: 夕阳下，一对情侣在海边散步，背影温馨。...",
                    "tags": [],
                    "provider": "system_placeholder"
                },
                "transform": {"scale": 1.0, "position": "center"}
            }
        ],
        "style_config": {
            "filter_preset": "natural",
            "intensity": 0.8
        }
    }


    node = FilterApplicationNode(node_id="filter_01", name="滤镜应用节点")

    # 模拟上传自定义 LUT（可选）
    # custom_lut = "custom_lut.cube"
    # if not os.path.exists(custom_lut):
    #     with open(custom_lut, 'w') as f:
    #         f.write("# Test LUT\n")
    # node.uploaded_files = [
    #     {"type": "file", "filename": "custom.cube", "path": custom_lut}
    # ]

    # 调用 generate
    logger.info(f"=== Generate ===")
    result = node.generate(context=INPUT_DATA)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 用户干预：更换某个镜头
    # logger.info(f"\n=== Regenerate: 更换夕阳镜头为赛博朋克 ===")
    # intent = {
    #     "filter_override": {
    #         "per_clip": [
    #             {
    #                 "clip_id": "clip_9i0j1k2l",
    #                 "preset": "cyberpunk",
    #                 "intensity": 1.0
    #             }
    #         ]
    #     }
    # }
    # result2 = node.regenerate(context=INPUT_DATA, user_intent=intent)
    # print(json.dumps(result2, indent=2, ensure_ascii=False))