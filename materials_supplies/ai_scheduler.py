# ai_scheduler.py
import asyncio
import httpx
from typing import List, Dict, Any
from config import settings
from .models import MaterialRequest, AITaskRequest

class AIScheduler:
    def __init__(self):
        self.models = [m for m in settings.AI_MODELS if m["enabled"]]

    def filter_models_by_type(self, material_type: str) -> List[Dict]:
        """筛选支持该类型生成的模型"""
        return [m for m in self.models if m["type"] == "image" and material_type in ["image", "product_image", "banner"]]

    def filter_models_by_style(self, models: List[Dict], style: str) -> List[Dict]:
        """筛选支持该风格的模型"""
        filtered = []
        for m in models:
            if m["styles"] == "all" or style in m["styles"]:
                filtered.append(m)
        return filtered

    def calculate_score(self, model: Dict, policy: str) -> float:
        """根据策略计算模型得分（越高越好）"""
        q = model["quality"]
        t = model["latency"]
        c = model["cost"]

        if policy == "quality_first":
            return q
        elif policy == "cost_first":
            return -c
        elif policy == "speed_first":
            return -t
        elif policy == "balanced":
            # 标准化后加权（示例）
            norm_q = q
            norm_t = (15 - min(t, 15)) / 15  # 假设最大15秒
            norm_c = (2 - min(c, 2)) / 2     # 假设最大2元
            return 0.5 * norm_q + 0.3 * norm_t + 0.2 * norm_c
        else:
            return q  # 默认质量优先

    async def select_best_model(self, material_type: str, style: str, policy: str = None) -> Dict:
        """选择最优AI模型"""
        policy = policy or settings.SCHEDULING_POLICY

        candidates = self.filter_models_by_type(material_type)
        candidates = self.filter_models_by_style(candidates, style)

        if not candidates:
            raise ValueError(f"找不到支持类型='{material_type}' 风格='{style}' 的AI模型")

        scored = [(m, self.calculate_score(m, policy)) for m in candidates]
        best_model = max(scored, key=lambda x: x[1])[0]
        return best_model

    async def trigger_generation(self, model: Dict, request: MaterialRequest) -> Dict:
        """调用AI生成服务"""
        async with httpx.AsyncClient() as client:
            payload = {
                "prompt": f"{request.description}, 风格: {request.style}",
                "style": request.style,
                "size": "1024x1024"
            }
            try:
                resp = await client.post(
                    model["endpoint"],
                    json=payload,
                    timeout=30.0
                )
                resp.raise_for_status()
                result = resp.json()
                return {
                    "success": True,
                    "model_id": model["model_id"],
                    "generated_url": result.get("image_url"),
                    "task_id": result.get("task_id")
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }

    async def trigger_style_transfer(self, source_url: str, target_style: str) -> Dict:
        """触发风格迁移（当风格不匹配时）"""
        st_models = [m for m in self.models if m["type"] == "style_transfer"]
        if not st_models:
            return {"success": False, "error": "无可用风格迁移模型"}

        model = st_models[0]  # 通常只有一个
        async with httpx.AsyncClient() as client:
            payload = {
                "image_url": source_url,
                "style_prompt": target_style
            }
            try:
                resp = await client.post(model["endpoint"], json=payload)
                resp.raise_for_status()
                result = resp.json()
                return {
                    "success": True,
                    "model_id": model["model_id"],
                    "transferred_url": result.get("result_url")
                }
            except Exception as e:
                return {"success": False, "error": str(e)}