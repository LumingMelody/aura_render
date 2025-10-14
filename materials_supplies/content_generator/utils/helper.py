# ai_content_pipeline/utils/helpers.py
from typing import Dict

def is_consecutive(block1: Dict, block2: Dict) -> bool:
    """
    判断两个分镜是否连续
    可根据 scene、subject、camera_motion 等字段判断
    """
    # 示例逻辑：同一场景且主语相同
    scene1 = block1.get("scene")
    scene2 = block2.get("scene")
    subject1 = block1.get("subject")
    subject2 = block2.get("subject")

    if not scene1 or not scene2:
        return False

    return (scene1 == scene2) and (subject1 == subject2)

def parse_duration(dur: Any) -> float:
    """
    安全解析时长
    """
    try:
        return float(dur)
    except (TypeError, ValueError):
        return 2.0