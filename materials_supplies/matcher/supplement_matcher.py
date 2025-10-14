# matcher/supplement_matcher.py
import httpx
import base64
from typing import List
from materials_supplies.models import SupplementRequest, SupplementResponse
# from materials_supplies.config import settings

# Qwen-VL 多模态验证（可选增强）
QWEN_VL_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

async def download_image_as_base64(url: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=10.0)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode('utf-8')

async def match_supplement(request: SupplementRequest) -> List[SupplementResponse]:
    """
    匹配辅助视觉素材（视频或图片）
    模拟从 Java 素材库获取候选，并支持图文联合验证
    """
    # 模拟 Java 返回的候选素材（实际应通过 HTTP 调用）
    candidates = [
        {
            "material_id": "supp_001",
            "url": "https://assets.com/fire-explosion.mp4",
            "thumbnail": "https://thumb.com/fire-explosion.jpg",
            "description": "红色火焰爆炸特效，慢动作",
            "duration": 5.0,
            "media_type": "video",
            "tags": ["特效", "火焰", "爆炸"]
        },
        {
            "material_id": "supp_002",
            "url": "https://assets.com/cyber-grid.jpg",
            "thumbnail": "https://thumb.com/cyber-grid-thumb.jpg",
            "description": "科技感蓝色网格背景",
            "duration": 0.0,  # 图片无时长
            "media_type": "image",
            "tags": ["背景", "科技感", "网格"]
        }
    ]

    c=candidates[0]
    return [SupplementResponse(
                    url=c["url"],
                    thumbnail=c["thumbnail"],
                    media_type=c["media_type"],
                    in_point=0,
                    out_point=1,
                    duration=1,
                    match_score=1
                )]

    results = []
    headers = {"Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}", "Content-Type": "application/json"}

    for c in candidates:
        try:
            # ✅ 使用 Qwen-VL 进行图文匹配验证（可选，提升准确性）
            image_base64 = await download_image_as_base64(c["thumbnail"])
            prompt = f"""
            判断该视觉素材是否符合用户需求。
            【用户需求】描述：{request.description}，类别：{request.category}，所需时长：{request.duration}s
            【素材描述】{c['description']}，类型：{c['media_type']}
            请输出 JSON：{{"match": true, "score": 90, "reason": "内容风格匹配"}}
            """

            payload = {
                "model": "qwen-vl-max",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"text": prompt},
                                {"image": f"data:image/jpeg;base64,{image_base64}"}
                            ]
                        }
                    ]
                },
                "parameters": {"response_format": {"type": "json_object"}}
            }

            async with httpx.AsyncClient() as client:
                resp = await client.post(QWEN_VL_API_URL, json=payload, headers=headers, timeout=30.0)
                result = resp.json()["output"]["text"]
                import json
                analysis = json.loads(result)

                if not analysis.get("match", False):
                    continue

                match_score = analysis["score"]

                # 处理视频：裁剪 in/out 点
                if c["media_type"] == "video":
                    actual_duration = c["duration"]
                    if actual_duration >= request.duration:
                        in_point = 0.0
                        out_point = request.duration
                    else:
                        in_point = 0.0
                        out_point = actual_duration
                    final_duration = out_point - in_point

                # 处理图片：视为静态素材，持续时间为请求时长
                else:  # image
                    in_point = 0.0
                    out_point = request.duration
                    final_duration = request.duration

                results.append(SupplementResponse(
                    url=c["url"],
                    thumbnail=c["thumbnail"],
                    media_type=c["media_type"],
                    in_point=in_point,
                    out_point=out_point,
                    duration=final_duration,
                    match_score=match_score
                ))

        except Exception as e:
            # 失败则跳过，不中断整体
            continue

    # 按匹配度排序
    results.sort(key=lambda x: x.match_score, reverse=True)
    return results