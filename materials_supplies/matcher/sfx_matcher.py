# matcher/sfx_matcher.py
from materials_supplies.models import SFXRequest, SFXResponse
from typing import List

async def match_sfx(request: SFXRequest) -> List[SFXResponse]:
    # 模拟音效匹配
    sfx_map = {
        "关门声": {"title":"激光","category":"激光声音","url": "https://sfx.com/door-close.mp3", "duration": 1.2},
        "激光": {"title":"激光","category":"激光声音","url": "https://sfx.com/laser.mp3", "duration": 0.8}
    }
    keyword = request.description.split()[0]  # 简单提取关键词
    info = sfx_map.get(keyword, sfx_map["激光"])

    return [SFXResponse(title=info["title"],category=info["category"],url=info["url"], duration=info["duration"])]