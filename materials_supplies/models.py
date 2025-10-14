# models.py
from pydantic import BaseModel
from typing import List, Optional

class VideoRequest(BaseModel):
    description: str
    category: str
    duration: float  # 秒

class BGMRequest(BaseModel):
    description: str
    category: str
    duration: float

class SFXRequest(BaseModel):
    description: str
    category: str

class FontRequest(BaseModel):
    description: str

class VideoResponse(BaseModel):
    url: str
    thumbnail: str
    in_point: float
    out_point: float
    match_score: float

class BGMResponse(BaseModel):
    url: str
    cut_start: float
    cut_end: float
    duration: float

class SFXResponse(BaseModel):
    url: str
    title: str
    category: str
    duration: float

class FontResponse(BaseModel):
    url: str


class SupplementRequest(BaseModel):
    description: str           # 素材内容描述，如“火焰爆炸特效”
    category: str              # 类型：特效 / 转场 / 背景 / 贴图 等
    duration: float            # 所需时长（秒），图片类默认使用此值

class SupplementResponse(BaseModel):
    url: str                   # 素材原始文件 URL
    thumbnail: str             # 缩略图 URL
    media_type: str            # "video" 或 "image"
    in_point: float            # 入点（视频裁剪起点）
    out_point: float           # 出点（视频裁剪终点）
    duration: float            # 实际素材时长（图片视为 duration=请求时长）
    match_score: float         # 匹配度评分

class TTSRequest(BaseModel):
    text: str
    voice: str
    speed: float = 1.0
    duration: float = 0.0  # 期望的语音时长（秒），可用于匹配或调整

class TTSResponse(BaseModel):
    url: str           # 生成的语音文件地址
    text: str          # 对应的文本
    voice: str         # 使用的音色
    duration: float    # 实际语音时长
    speed: float       # 语速



class IntroOutroRequest(BaseModel):
    duration: float            # 所需片段的期望时长（秒）
    category: str = "default"  # 风格分类，如科技、情感、商业等
    intro_required: bool = True
    outro_required: bool = True

class IntroOutroResponse(BaseModel):
    type: str                  # "intro" 或 "outro"
    video_url: str             # 视频文件地址（含画面+内嵌音频）
    audio_embedded: bool = True  # 音频是否已嵌入视频（通常为True）
    total_duration: float      # 原始视频总时长
    cut_start: float           # 裁剪起始时间
    cut_end: float             # 裁剪结束时间


class MaterialRequest(BaseModel):
    """通用素材请求模型"""
    description: str
    category: str
    duration: Optional[float] = None
    style: Optional[str] = None
    keywords: Optional[List[str]] = None


class AITaskRequest(BaseModel):
    """AI任务请求模型"""
    task_type: str  # 任务类型：generate, validate, enhance等
    material_type: str  # 素材类型：video, audio, image等
    prompt: str  # AI提示词
    parameters: Optional[dict] = None  # 额外参数
    priority: int = 5  # 优先级（1-10）


class CandidateMaterial(BaseModel):
    """候选素材模型"""
    id: str
    url: str
    thumbnail: Optional[str] = None
    title: str
    description: str
    category: str
    media_type: str  # video, audio, image
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    metadata: Optional[dict] = None