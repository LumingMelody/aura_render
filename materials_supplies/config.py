# config.py
import os
from pydantic import BaseModel

class Settings(BaseModel):
    # Java 素材库 API 地址
    MATERIAL_API_URL: str = os.getenv("MATERIAL_API_URL", "http://your-java-api.com/api/v1")
    
    # 验证阈值
    TYPE_VALIDATION_THRESHOLD: float = 0.85
    STYLE_VALIDATION_THRESHOLD: float = 0.80

    # AI 模型配置（可从数据库或配置中心加载）
    AI_MODELS = [
        {
            "model_id": "sd_pro",
            "type": "image",
            "styles": ["科技", "现代", "极简", "商务"],
            "quality": 0.95,
            "latency": 8.0,
            "cost": 1.2,
            "endpoint": "https://api.example.com/sd-pro/generate",
            "enabled": True
        },
        {
            "model_id": "midjourney_lite",
            "type": "image",
            "styles": ["卡通", "手绘", "插画", "可爱"],
            "quality": 0.85,
            "latency": 12.0,
            "cost": 0.9,
            "endpoint": "https://api.mj-lite.com/generate",
            "enabled": True
        },
        {
            "model_id": "style_transfer_v2",
            "type": "style_transfer",
            "styles": "all",  # 支持所有风格迁移
            "quality": 0.90,
            "latency": 6.0,
            "cost": 0.8,
            "endpoint": "https://api.styler.com/v2/transfer",
            "enabled": True
        }
    ]

    # 调度策略（可外部配置）
    SCHEDULING_POLICY: str = "quality_first"  # 支持: cost_first, speed_first, quality_first, balanced

settings = Settings()