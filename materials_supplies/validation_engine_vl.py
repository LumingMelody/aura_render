# validation_engine_vl.py
import httpx
import base64
from typing import List, Dict, Any
from .models import CandidateMaterial, MaterialRequest
from config import settings

# Qwen-VL API 地址（百炼平台）
QWEN_VL_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

async def download_image_as_base64(image_url: str) -> str:
    """下载图片并转为 base64（供 Qwen-VL 使用）"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(image_url, timeout=10.0)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode('utf-8')

async def validate_with_qwen_vl(
    request: MaterialRequest,
    candidates: List[CandidateMaterial]
) -> List[Dict]:
    """
    使用 Qwen-VL 大模型进行图文联合验证
    输入：用户需求 + 候选素材（图 + 描述）
    输出：匹配度评分 + 推荐结果
    """
    results = []
    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    for candidate in candidates:
        try:
            # 下载缩略图并转为 base64
            image_base64 = await download_image_as_base64(candidate.thumbnail_url)

            # 构造多模态 prompt
            prompt = f"""
            你是一个专业的视觉素材审核专家，请结合图片和文字描述，判断该素材是否符合用户需求。

            【用户原始需求】
            - 主体描述：{request.description}
            - 风格要求：{request.style}
            - 附加说明：{request.additional_notes or '无'}

            【候选素材信息】
            - 文字描述：{candidate.description}
            - 标签：{', '.join(candidate.tags)}

            请分析图片内容，并回答以下问题：
            1. 图片内容是否与用户描述一致？
            2. 风格（如科技感、复古等）是否匹配？
            3. 是否存在明显不符的元素（如风格错乱、主体错误）？

            请输出 JSON 格式：
            {{
                "material_id": "{candidate.material_id}",
                "match_score": 85,
                "reason": "图片显示一辆未来飞行汽车在城市上空飞行，带有蓝色光轨，风格科技感强，与描述一致",
                "recommended": true
            }}
            """

            payload = {
                "model": "qwen-vl-max",  # 推荐使用 qwen-vl-max（精度高），qwen-vl-plus（性价比）
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
                "parameters": {
                    "response_format": {"type": "json_object"}
                }
            }

            async with httpx.AsyncClient() as client:
                resp = await client.post(QWEN_VL_API_URL, json=payload, headers=headers, timeout=30.0)
                if resp.status_code != 200:
                    raise Exception(f"Qwen-VL API 错误: {resp.status_code}, {resp.text}")

                result = resp.json()
                model_output = result["output"]["text"]
                # 安全解析 JSON（生产环境建议用 json.loads + try-catch）
                import json
                item = json.loads(model_output)
                item["url"] = candidate.url
                item["thumbnail_url"] = candidate.thumbnail_url
                results.append(item)

        except Exception as e:
            # 单个失败不中断整体，降级为文本匹配
            fallback = await _fallback_text_only(request, candidate)
            results.append(fallback)

    # 按 match_score 排序
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results

async def _fallback_text_only(request: MaterialRequest, candidate: CandidateMaterial) -> Dict:
    """当图片无法获取或 Qwen-VL 调用失败时，降级为文本匹配"""
    from validation_engine import validate_with_qwen  # 复用之前的文本验证
    # 模拟单候选列表
    single_candidate = [candidate]
    result = await validate_with_qwen(request, single_candidate)
    return result[0] if result else {
        "material_id": candidate.material_id,
        "url": candidate.url,
        "match_score": 0,
        "reason": f"验证失败，降级处理: {str(e)}",
        "recommended": False
    }