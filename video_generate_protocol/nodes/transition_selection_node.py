# nodes/transition_selection_node.py

from video_generate_protocol import BaseNode
from typing import Dict, List, Any

from llm import QwenLLM  # 假设这是你封装好的 Qwen 调用模块

# 转场类型库
TRANSITIONS = {
    "cut": {
        "name": "硬切",
        "duration": 0.0,
        "description": "直接切换，无特效，适用于快节奏或匹配剪辑",
        "suitable_for": ["快节奏", "动作连续"]
    },
    "fade_in_out": {
        "name": "淡入淡出",
        "duration": 1.0,
        "description": "画面渐黑再亮起，常用于段落分隔或时间跳跃",
        "suitable_for": ["情感转折", "时间过渡"]
    },
    "cross_dissolve": {
        "name": "叠化",
        "duration": 1.2,
        "description": "前画面渐隐，后画面渐显，适用于情绪延续或场景过渡",
        "suitable_for": ["情感延续", "同主题转场"]
    },
    "wipe_push": {
        "name": "推进式模糊",
        "duration": 0.8,
        "description": "前画面被“推”出，后画面聚焦进入，适合空间推进感",
        "suitable_for": ["全景→特写", "视角推进"]
    },
    "flash_white": {
        "name": "闪光白帧",
        "duration": 0.5,
        "description": "短促白闪，制造冲击或节奏突变",
        "suitable_for": ["快→慢节奏", "情绪突变"]
    },
    "match_cut": {
        "name": "匹配剪辑",
        "duration": 0.3,
        "description": "利用动作、构图或运动方向的相似性无缝转场",
        "suitable_for": ["同场景不同角度", "动作连续"]
    },
    "zoom_transition": {
        "name": "缩放转场",
        "duration": 1.0,
        "description": "通过快速缩放连接两个镜头，增强动感",
        "suitable_for": ["风格化视频", "VLOG"]
    },
    "slide": {
        "name": "滑动",
        "duration": 0.7,
        "description": "画面水平或垂直滑动切换，简洁现代",
        "suitable_for": ["信息类视频", "PPT式剪辑"]
    }
}

# 镜头类型抽象等级（用于判断推进/拉远）
SHOT_LEVEL = {
    "aerial": 5,        # 最远
    "wide": 4,
    "medium": 3,
    "tracking": 3,
    "dolly": 3,
    "close_up": 2,
    "extreme_close_up": 1,  # 最近
    "overhead": 4,
    "low_angle": 3
}

# 节奏映射
PACING_TO_SPEED = {
    "fast": 2,
    "normal": 1,
    "slow": 0,
    "freeze": -1
}


class TransitionSelectionNode(BaseNode):
    required_inputs = [
        {
            "name": "preliminary_sequence_id",
            "label": "初步剪辑序列",
            "type": list,
            "required": True,
            "desc": "包含镜头信息的剪辑序列，如 [{'shot_id': 's1', 'shot_type': '全景', 'pacing': '快剪', 'emotion_hint': '激昂'}]",
            "field_type": "json"
        }
    ]


    output_schema=[
         {
            "name": "transition_sequence_id",
            "label": "添加转场的分镜块列表",
            "type": list,
            "required": True,
            "desc": "包含镜头信息及转场的剪辑序列，如 [{'shot_id': 's1', 'shot_type': '全景', 'pacing': '快剪', 'emotion_hint': '激昂', 'transition_out': {'type': 'cross_dissolve', 'duration': 1.2}}]",
            "field_type": "json"
        }
        
    ]

    file_upload_config = {
        "image": {"enabled": False},
        "video": {"enabled": False}
    }

    system_parameters = {
        "default_transition": "cross_dissolve",
        "min_duration": 0.3,
        "max_duration": 2.0
    }

    def __init__(self, node_id: str, name: str = "转场选择"):
        super().__init__(node_id=node_id, node_type="transition_selection", name=name)
        # 初始化 Qwen 模型实例
        self.qwen = QwenLLM()

    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_context(context)

        sequence: List[Dict] = context["preliminary_sequence_id"]

        if len(sequence) < 2:
            # 单镜头无需转场
            for clip in sequence:
                clip["transition_out"] = {"type": "none", "duration": 0.0}
            return {"transition_sequence_id": sequence}

        result_sequence = [sequence[0]]  # 第一个镜头直接加入

        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_clip = sequence[i + 1]

            # 从 metadata.description 中提取描述
            current_desc = current.get("metadata", {}).get("description", "无描述")
            next_desc = next_clip.get("metadata", {}).get("description", "无描述")

            # 使用 Qwen 模型决定转场
            transition = self._decide_transition_with_llm(
                current_description=current_desc,
                next_description=next_desc,
                current_clip=current,
                next_clip=next_clip
            )

            # 添加到前一个镜头的输出转场
            current_with_transition = {**current}
            current_with_transition["transition_out"] = transition
            result_sequence[-1] = current_with_transition  # 替换最后一个

            # 添加下一个镜头（后续可能被继续修改）
            result_sequence.append(next_clip)

        # 最后一个镜头添加默认无转场
        result_sequence[-1]["transition_out"] = {"type": "none", "duration": 0.0}
        print(result_sequence)
        return {"transition_sequence_id": result_sequence}

    def _decide_transition_with_llm(self, current_description: str, next_description: str, current_clip: Dict, next_clip: Dict) -> Dict:
        """
        使用 Qwen 大模型根据前后镜头的画面描述和元数据决定转场。
        """
        # 构建提示词 (Prompt)
        # 将可用的转场类型以清晰的格式注入
        transitions_info = "\n".join([
            f"- **{key}** ({info['name']}): {info['description']} (时长: {info['duration']}秒)"
            for key, info in TRANSITIONS.items()
        ])

        prompt = f"""
        你是一个专业的视频剪辑师。你的任务是根据两个连续镜头的画面内容描述，选择最合适的转场效果。
        请从以下转场类型中选择最合适的一个，并只返回转场类型的英文键（例如 'cut', 'fade_in_out'）。不要解释。

        **可用的转场类型：**
        {transitions_info}

        **当前镜头 (前一个) 描述：**
        {current_description}

        **下一个镜头 (后一个) 描述：**
        {next_description}

        **额外信息（可选参考）：**
        - 当前镜头的镜头类型（shot_type）: {current_clip.get('shot_type', '未知')}
        - 下一个镜头的镜头类型（shot_type）: {next_clip.get('shot_type', '未知')}
        - 当前镜头的节奏（pacing）: {current_clip.get('pacing', '正常')}
        - 下一个镜头的节奏（pacing）: {next_clip.get('pacing', '正常')}

        **决策逻辑参考：**
        1.  如果镜头从远景/全景变为中景/特写（推进感），优先考虑 'wipe_push'。
        2.  如果镜头从中景/特写变为远景/全景（拉远感），优先考虑 'zoom_transition'。
        3.  如果节奏从快变慢，考虑使用 'flash_white' 制造冲击或突变。
        4.  如果两个镜头在同一个场景，且有动作或构图上的相似性，考虑 'match_cut'。
        5.  如果情感或氛围发生巨大变化（如从欢快到悲伤），考虑 'fade_in_out'。
        6.  其他情况下，'cross_dissolve' 是安全且优雅的选择。

        请仅返回你选择的转场类型的英文键。
        """

        try:
            # 调用 Qwen 模型
            response = self.qwen.generate(prompt=prompt)
            # 假设 response 是一个包含文本结果的对象，我们取其文本内容
            # 具体属性名取决于 QwenLLM 的实现，这里假设是 `.text`
            llm_output = response

            # 解析模型输出，确保是有效的转场键
            chosen_key = None
            for key in TRANSITIONS.keys():
                if key in llm_output:
                    chosen_key = key
                    break

            # 如果模型输出无效，则使用默认转场
            if chosen_key is None:
                chosen_key = "cross_dissolve"
                print(f"Warning: LLM output '{llm_output}' not recognized. Using default: {chosen_key}")

        except Exception as e:
            print(f"Error calling Qwen LLM: {e}. Using default transition.")
            chosen_key = "cross_dissolve"

        # 获取转场模板
        template = TRANSITIONS[chosen_key]

        # 动态调整时长 (可以基于描述的复杂度、节奏等，这里简化为使用基础时长)
        # 你也可以让 LLM 同时输出建议时长，但为了简单，我们先用基础时长
        base_duration = template["duration"]
        # 这里可以添加更复杂的时长调整逻辑，例如基于节奏
        # prev_speed = PACING_TO_SPEED.get(current_clip.get("pacing", "normal"), 1)
        # next_speed = PACING_TO_SPEED.get(next_clip.get("pacing", "normal"), 1)
        # if prev_speed == 2 and next_speed < 2 and chosen_key == "flash_white":
        #     base_duration = TRANSITIONS["flash_white"]["duration"]
        duration = base_duration

        # 限制范围
        duration = max(self.system_parameters["min_duration"], 
                      min(self.system_parameters["max_duration"], duration))

        return {
            "type": chosen_key,
            "name": template["name"],
            "duration": round(duration, 2),
            "description": template["description"]
        }
    

    # 以下方法在新逻辑中不再直接使用，但可以保留以备将来参考或作为备用逻辑
    # def _normalize_shot_type(self, shot_type: str) -> str: ...
    # def _is_same_scene_context(self, clip1: Dict, clip2: Dict) -> bool: ...
    # def _emotion_weight(self, emotion: str) -> int: ...
    # def _emotion_tempo_factor(self, emotion: str) -> float: ...

    def _normalize_shot_type(self, shot_type: str) -> str:
        """将中文镜头类型映射为标准英文键"""
        mapping = {
            "全景": "wide",
            "中景": "medium",
            "特写": "close_up",
            "大特写": "extreme_close_up",
            "航拍": "aerial",
            "推镜头": "dolly",
            "跟拍": "tracking",
            "俯拍": "overhead",
            "仰拍": "low_angle"
        }
        for cn, en in mapping.items():
            if cn in shot_type:
                return en
        return "medium"

    def _is_same_scene_context(self, clip1: Dict, clip2: Dict) -> bool:
        """判断是否为同场景不同角度（简化版）"""
        # 可扩展为基于视觉特征或元数据（如GPS、场景标签）
        seg1 = clip1.get("segment", "")
        seg2 = clip2.get("segment", "")
        return seg1 == seg2 and "特写" not in clip1.get("shot_type", "") and "特写" not in clip2.get("shot_type", "")

    def _emotion_weight(self, emotion: str) -> int:
        """情感抽象等级（用于计算变化）"""
        weights = {
            "激昂": 5,
            "励志": 4,
            "感动": 4,
            "幽默": 3,
            "冷静": 2,
            "悬疑": 3
        }
        return weights.get(emotion, 3)

    def _emotion_tempo_factor(self, emotion: str) -> float:
        """根据情感调整转场时长节奏"""
        factors = {
            "激昂": 0.8,    # 稍快
            "幽默": 0.9,
            "正常": 1.0,
            "温馨": 1.1,    # 稍慢
            "感动": 1.2,
            "悬疑": 1.0
        }
        return factors.get(emotion, 1.0)

    # regenerate 方法也需要相应调整，移除对 emotions 的依赖
    def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """支持用户干预"""
        super().regenerate(context, user_intent)

        override = user_intent.get("transition_override")
        if override and isinstance(override, dict):
            result = self.generate(context) # 注意：这里调用的是修改后的 generate
            sequence = result["edited_sequence"]

            if "global_type" in override:
                trans_key = override["global_type"]
                if trans_key in TRANSITIONS:
                    dur = TRANSITIONS[trans_key]["duration"]
                    for clip in sequence:
                        if "transition_out" in clip:
                            clip["transition_out"] = {
                                "type": trans_key,
                                "name": TRANSITIONS[trans_key]["name"],
                                "duration": dur,
                                "description": TRANSITIONS[trans_key]["description"] + "（用户指定）"
                            }

            elif "specific" in override:
                from_id, to_id = override["specific"].get("from"), override["specific"].get("to")
                trans_type = override["specific"].get("type")
                for clip in sequence:
                    if clip.get("id") == from_id and "transition_out" in clip: # 注意：使用 'id' 而不是 'shot_id'
                        if trans_type in TRANSITIONS:
                            template = TRANSITIONS[trans_type]
                            clip["transition_out"] = {
                                "type": trans_type,
                                "name": template["name"],
                                "duration": template["duration"],
                                "description": template["description"] + "（用户指定）"
                            }

            return result

        return self.generate(context)
    



if __name__ == "__main__":
        # 实例化转场选择节点
    node = TransitionSelectionNode(node_id="node_01", name="转场选择示例")

    # 构建输入上下文
    context = {
  "preliminary_sequence": [
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
        "duration": 0.5
      },
      "metadata": {
        "description": "Placeholder for: 一个阳光明媚的早晨，鸟儿在树上歌唱，城市慢慢苏醒。...",
        "tags": [],
        "provider": "system_placeholder"
      },
      "transform": {
        "scale": 1.0,
        "position": "center"
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
        "duration": 0.5
      },
      "metadata": {
        "description": "Placeholder for: 一位年轻人在咖啡馆里专注地敲着笔记本电脑，周围人来往。...",
        "tags": [],
        "provider": "system_placeholder"
      },
      "transform": {
        "scale": 1.0,
        "position": "center"
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
        "type": "cross_dissolve",
        "duration": 0.5
      },
      "metadata": {
        "description": "Placeholder for: 夕阳下，一对情侣在海边散步，背影温馨。...",
        "tags": [],
        "provider": "system_placeholder"
      },
      "transform": {
        "scale": 1.0,
        "position": "center"
      }
    }
  ],
  "total_duration": 12.0,
  "timestamp": "2025-08-05T16:22:30.123456"
}

    # 调用 generate 方法并获取结果
    result = node.generate(context=context)

    # 打印处理后的剪辑序列
    print(result)