from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

class MaterialType(Enum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"

class MaterialFormat(Enum):
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    JPG = "jpg"
    PNG = "png"
    MP3 = "mp3"
    WAV = "wav"

class LicenseType(Enum):
    FREE = "free"
    CREATIVE_COMMONS = "creative_commons"
    PAID = "paid"
    ROYALTY_FREE = "royalty_free"

@dataclass
class MaterialMetadata:
    title: str
    description: Optional[str] = None
    tags: List[str] = None
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[MaterialFormat] = None
    license: LicenseType = LicenseType.FREE
    source: str = "unknown"
    author: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []