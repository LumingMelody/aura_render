# validation_engine.py
import httpx
from typing import List, Dict, Any
from .models import CandidateMaterial, MaterialRequest
from config import settings  # 存放 API_KEY 和 BASE_URL

# 百炼 Qwen API 地址
QWEN_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

async def validate_with_qwen(request: MaterialRequest, candidates: List[CandidateMaterial]) -> List[Dict]:
    """
    使用 Qwen 大模型对候选素材描述进行语义验证与排序
    返回：按匹配度排序的验证结果列表
    """
    results = []

    # 构造 prompt
    prompt = f"""
    你是一个专业的创意素材审核专家，请根据用户的原始需求，判断以下候选素材的描述是否高度匹配。
    
    【用户原始需求】
    - 主体描述：{request.description}
    - 风格要求：{request.style}
    - 附加说明：{request.additional_notes or '无'}

    请对每个候选素材进行打分（0-100），并判断是否推荐使用。
    输出格式为 JSON 列表，包含 material_id、match_score、reason、recommended 字段。
    
    候选素材列表：
    {[
        {
            "material_id": c.material_id,
            "description": c.description,
            "tags": c.tags
        } for c in candidates
    ]}
    
    请严格按照以下 JSON 格式输出，不要包含其他内容：
    [
      {{
        "material_id": "xxx",
        "match_score": 95,
        "reason": "描述完全符合未来飞行汽车在科技城市中飞行的场景，风格一致",
        "recommended": true
      }}
    ]
    """

    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen-max",  # 可选 qwen-plus, qwen-turbo
        "input": {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        "parameters": {
            "response_format": {"type": "json_object"}  # 强制 JSON 输出
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(QWEN_API_URL, json=payload, headers=headers, timeout=30.0)
            if response.status_code != 200:
                raise Exception(f"Qwen API 错误: {response.status_code}, {response.text}")

            result = response.json()
            # 解析模型输出的 JSON 内容
            llm_output = result["output"]["text"]
            validated_list = eval(llm_output)  # 注意：生产环境建议用 json.loads + 更安全解析

            # 补充原始素材信息
            material_map = {c.material_id: c for c in candidates}
            for item in validated_list:
                if item["material_id"] in material_map:
                    item["url"] = material_map[item["material_id"]].url
                    item["source_description"] = material_map[item["material_id"]].description

            # 按 match_score 排序
            validated_list.sort(key=lambda x: x["match_score"], reverse=True)
            return validated_list

        except Exception as e:
            # 失败降级：按 tags 与 description 简单关键词匹配
            return await _fallback_keyword_match(request, candidates)
        
async def _fallback_keyword_match(request: MaterialRequest, candidates: List[CandidateMaterial]):
    """降级：基于关键词匹配"""
    keywords = (request.description + " " + request.style + " " + (request.additional_notes or "")).lower()
    results = []
    for c in candidates:
        desc = " ".join(c.tags) + " " + c.description
        score = sum(1 for word in keywords.split() if word in desc.lower())
        results.append({
            "material_id": c.material_id,
            "url": c.url,
            "match_score": score * 10,
            "reason": "关键词匹配",
            "recommended": score > 2,
            "source_description": c.description
        })
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results