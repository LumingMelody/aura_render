# main.py
from fastapi import FastAPI, HTTPException
from typing import List

from .models import VideoRequest, BGMRequest, SFXRequest, FontRequest, SupplementRequest
from .models import VideoResponse, BGMResponse, SFXResponse, FontResponse, SupplementResponse

from matcher.main_video_matcher import match_main_video
from matcher.bgm_matcher import match_bgm
from matcher.sfx_matcher import match_sfx
from matcher.font_matcher import match_font
from matcher.supplement_matcher import match_supplement

app = FastAPI(title="智能素材供给系统 - 四大接口")

@app.post("/supply-main-video", response_model=List[VideoResponse])
async def supply_main_video(request: VideoRequest):
    try:
        results = await match_main_video(request)
        return results[:3]  # 返回 Top 3
    except Exception as e:
        raise HTTPException(500, detail=f"主视频匹配失败: {str(e)}")

@app.post("/supply-bgm", response_model=List[BGMResponse])
async def supply_bgm(request: BGMRequest):
    try:
        results = await match_bgm(request)
        return results[:3]
    except Exception as e:
        raise HTTPException(500, detail=f"BGM 匹配失败: {str(e)}")

@app.post("/supply-sfx", response_model=List[SFXResponse])
async def supply_sfx(request: SFXRequest):
    try:
        results = await match_sfx(request)
        return results[:3]
    except Exception as e:
        raise HTTPException(500, detail=f"音效匹配失败: {str(e)}")

@app.post("/supply-fonts", response_model=List[FontResponse])
async def supply_fonts(request: FontRequest):
    try:
        results = await match_font(request)
        return results
    except Exception as e:
        raise HTTPException(500, detail=f"字体匹配失败: {str(e)}")

@app.post("/supply-extra-supplement", response_model=List[SupplementResponse])
async def supply_extra_supplement(request: SupplementRequest):
    try:
        results = await match_supplement(request)
        return results[:3]  # 返回 Top 3 辅助素材
    except Exception as e:
        raise HTTPException(500, detail=f"辅助素材匹配失败: {str(e)}")

# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)