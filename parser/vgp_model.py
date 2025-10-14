from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum

# --- 枚举类型 ---
class TrackType(Enum):
    VIDEO = "video"
    AUDIO = "audio"
    SUBTITLE = "subtitle"
    OVERLAY = "overlay"
    DATA = "data"

class EffectType(Enum):
    # 这里列出可能的效果类型，实际应更丰富
    COLOR_CORRECTION = "color_correction"
    BLUR = "blur"
    SCALE = "scale"
    POSITION = "position"
    CROP = "crop"
    VOLUME = "volume" # 音频
    OPACITY = "opacity" # 视频/叠加层
    # ... 其他

class TransitionType(Enum):
    FADE = "fade"
    DISSOLVE = "dissolve"
    WIPE = "wipe"
    SLIDE = "slide"
    # ... 其他

class InterpolationType(Enum):
    LINEAR = "linear"
    EASE_IN = "ease_out" # 注意：通常ease_in对应出点缓动
    EASE_OUT = "ease_in" # 注意：通常ease_out对应入点缓动
    HOLD = "hold"
    BEZIER = "bezier"

# --- 核心数据模型 ---
@dataclass
class Source:
    id: str
    type: str  # "video", "audio", "image", "color", "text"
    path: Optional[str] = None
    duration: Optional[float] = None
    resolution: Optional[Dict[str, int]] = None # {"width": w, "height": h}
    fps: Optional[float] = None

@dataclass
class EffectParameterKeyframe:
    time: float  # 相对于 clip start 的时间 (秒)
    parameters: Dict[str, Any]
    interpolation: InterpolationType = InterpolationType.LINEAR

@dataclass
class Effect:
    id: str
    type: str  # EffectType.value
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    keyframes: List[EffectParameterKeyframe] = field(default_factory=list)

@dataclass
class Transition:
    type: str  # TransitionType.value
    duration: float
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Marker:
    time: float  # 相对于 clip start 的时间 (秒)
    label: str
    color: Optional[str] = None
    note: Optional[str] = None

@dataclass
class Clip:
    id: str
    source: Source
    start: float  # 在时间线上的开始时间 (秒)
    duration: float  # 在时间线上占据的时长 (秒)
    in_point: Optional[float] = None  # 源媒体的入点 (秒), 默认0
    out_point: Optional[float] = None  # 源媒体的出点 (秒), 默认为源媒体时长
    speed: float = 1.0
    reverse: bool = False
    enabled: bool = True
    name: Optional[str] = None
    effects: List[Effect] = field(default_factory=list)
    transition_in: Optional[Transition] = None
    transition_out: Optional[Transition] = None
    markers: List[Marker] = field(default_factory=list)

    def __post_init__(self):
        # 设置默认的 in/out 点
        if self.in_point is None:
            self.in_point = 0.0
        # out_point 在解析后，如果未指定，需要在知道 source.duration 后设置

@dataclass
class Track:
    id: str
    type: TrackType
    clips: List[Clip]
    name: Optional[str] = None
    enabled: bool = True
    locked: bool = False
    volume: float = 1.0  # 音频: 音量 0.0-1.0, 视频/叠加: 不透明度 0.0-1.0

@dataclass
class Timeline:
    tracks: List[Track]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Project:
    version: str
    name: str = "Untitled Project"
    description: Optional[str] = None
    duration: Optional[float] = None
    fps: float = 24.0
    resolution: Dict[str, int] = field(default_factory=lambda: {"width": 1920, "height": 1080})
    timeline: Timeline = field(default_factory=Timeline)