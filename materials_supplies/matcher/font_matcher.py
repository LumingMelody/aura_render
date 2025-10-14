# matcher/font_matcher.py
from materials_supplies.models import FontRequest, FontResponse
from typing import List

async def match_font(request: FontRequest) -> List[FontResponse]:
    font_map = {
        "未来感": "https://fonts.com/ali-future.ttf",
        "复古": "https://fonts.com/retro.ttf",
        "手写": "https://fonts.com/handwrite.ttf"
    }
    url = font_map.get(request.description, font_map["未来感"])
    return [FontResponse(url=url)]