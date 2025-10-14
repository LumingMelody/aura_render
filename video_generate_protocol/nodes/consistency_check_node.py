# nodes/consistency_check_node.py
"""
一致性检查节点 (Node 4C in VGP Workflow)

这是视频一致性保障的第三步，负责：
1. 分析相邻镜头的场景、物体、人物相似度
2. 判断哪些镜头需要使用图生图（image-to-image）
3. 标记参考图来源（前一帧 or 产品原图）

工作流位置：
- 输入：来自 node_4b_keyframe_refinement (首帧细化)
- 输出：传递给 node_5_asset_request (素材请求节点)

重要说明：
- 优先判断是否为同一物体
- 若是同一物体 → 使用前一帧图生图
- 若不是但有产品 → 使用产品原图图生图
- 若完全不同 → 文生图
"""

from video_generate_protocol import BaseNode
from typing import Dict, List, Any
from datetime import datetime
import asyncio


class ConsistencyCheckNode(BaseNode):
    """
    一致性检查节点

    分析镜头间的连续性，决定生成策略。
    """

    required_inputs = [
        {
            "name": "refined_keyframes_id",
            "label": "细化后的首帧列表",
            "type": list,
            "required": True,
            "desc": "结构化的首帧描述列表",
            "field_type": "json"
        },
        {
            "name": "reference_media",
            "label": "参考媒体（产品图片/视频）",
            "type": dict,
            "required": False,
            "desc": "包含产品图片或视频的参考媒体",
            "field_type": "json"
        }
    ]

    output_schema = [
        {
            "name": "keyframes_with_strategy_id",
            "label": "带生成策略的首帧列表",
            "type": list,
            "required": True,
            "desc": "标记了生成策略的首帧列表",
            "field_type": "json"
        }
    ]

    system_parameters = {
        "llm_model": "qwen-plus",
        "temperature": 0.3,  # 较低温度，保证判断一致性
        "max_tokens": 2000
    }

    def __init__(self, node_id: str, name: str = "一致性检查"):
        super().__init__(node_id=node_id, node_type="consistency_analysis", name=name)

    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_context(context)

        # 提取输入
        refined_keyframes = context.get("refined_keyframes_id", [])
        reference_media = context.get("reference_media", {})

        if not refined_keyframes:
            raise ValueError("缺少细化后的首帧列表")

        # 检查是否有产品图片
        product_image_url = None
        if reference_media:
            product_images = reference_media.get("product_images", [])
            if product_images and len(product_images) > 0:
                product_image_url = product_images[0].get("url")

        if product_image_url:
            print(f"🔍 [Node 4C] 正在检查 {len(refined_keyframes)} 个镜头的一致性（使用产品参考图）...")
        else:
            print(f"🔍 [Node 4C] 正在检查 {len(refined_keyframes)} 个镜头的一致性...")

        # 分析一致性
        keyframes_with_strategy = await self._analyze_consistency(refined_keyframes, product_image_url)

        # 统计
        img2img_count = sum(1 for kf in keyframes_with_strategy if kf["generation_strategy"] == "image_to_image")
        txt2img_count = len(keyframes_with_strategy) - img2img_count
        product_ref_count = sum(1 for kf in keyframes_with_strategy if kf.get("reference_source") == "product_image")

        print(f"✅ [Node 4C] 一致性检查完成")
        print(f"   - 图生图: {img2img_count} 个")
        print(f"   - 文生图: {txt2img_count} 个")
        if product_image_url:
            print(f"   - 使用产品图参考: {product_ref_count} 个")

        return {
            "keyframes_with_strategy_id": keyframes_with_strategy
        }

    async def _analyze_consistency(self, refined_keyframes: List[Dict], product_image_url: str = None) -> List[Dict]:
        """分析镜头间的一致性"""

        # 构建提示词
        prompt = self._build_consistency_prompt(refined_keyframes, product_image_url)

        # 调用LLM
        strategies = await self._call_llm(prompt, len(refined_keyframes))

        # 组装结果
        keyframes_with_strategy = []
        for idx, (keyframe, strategy) in enumerate(zip(refined_keyframes, strategies)):
            kf = keyframe.copy()
            kf["generation_strategy"] = strategy["strategy"]
            kf["reference_source"] = strategy["reference"]
            kf["reason"] = strategy["reason"]
            keyframes_with_strategy.append(kf)

        return keyframes_with_strategy

    def _build_consistency_prompt(self, refined_keyframes: List[Dict], product_image_url: str = None) -> str:
        """构建一致性检查提示词"""

        # 提取镜头描述
        shot_descriptions = []
        for idx, kf in enumerate(refined_keyframes, 1):
            prompt_text = kf.get("refined_prompt", "")
            shot_descriptions.append(f"镜头{idx}: {prompt_text}")

        shots_text = "\n".join(shot_descriptions)

        # 如果有产品图片，添加特殊说明
        product_section = ""
        if product_image_url:
            product_section = f"""

## ⚠️ 重要：产品参考图可用
用户上传了产品参考图：{product_image_url}

**强制规则**：
- 镜头1（第一个镜头）**必须**使用产品参考图作为基础，策略为：`image_to_image` + `reference: product_image`
- 后续镜头如果是同一产品的不同角度/特写，使用 `previous_frame` 作为参考
- 这样可以确保所有镜头与真实产品保持一致性
"""

        prompt = f"""# 角色
你是一个专注于处理视频分镜场景一致性的助手。
{product_section}
# 任务
分析以下镜头，判断哪些需要使用图生图（image-to-image），哪些使用文生图（text-to-image）。

## 镜头列表
{shots_text}

# 判断规则
1. **如果有产品参考图（见上方说明）**：
   - 镜头1 **必须** 使用产品图片作为参考（`image_to_image` + `product_image`）
   - 后续镜头参考前一帧（`image_to_image` + `previous_frame`）

2. **如果没有产品参考图**：
   - 镜头1：文生图（`text_to_image` + `none`）
   - 相邻镜头若为同一物体 → 参考前一帧（`image_to_image` + `previous_frame`）
   - 完全不同场景 → 文生图（`text_to_image` + `none`）

# 输出格式
请按照以下JSON格式输出（必须是有效的JSON）：

{{
  "strategies": [
    {{
      "shot_index": 1,
      "strategy": "image_to_image",  // 有产品图时必须用 "image_to_image"
      "reference": "product_image",  // 有产品图时必须用 "product_image"
      "reason": "第一个镜头，使用产品参考图保持一致性"
    }},
    {{
      "shot_index": 2,
      "strategy": "image_to_image",
      "reference": "previous_frame",
      "reason": "与镜头1为同一产品，参考前一帧"
    }},
    ...
  ]
}}

重要：直接输出JSON，不要添加任何解释性文字。"""

        return prompt

    async def _call_llm(self, prompt: str, expected_count: int) -> List[Dict]:
        """调用LLM分析一致性"""
        import os
        import json
        import aiohttp

        api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('AI__DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("缺少千问API密钥")

        endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        request_body = {
            "model": self.system_parameters["llm_model"],
            "messages": [
                {"role": "system", "content": "你是一位专业的视频一致性分析师。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.system_parameters["temperature"],
            "max_tokens": self.system_parameters["max_tokens"]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=request_body) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"LLM API error {response.status}: {error_text}")

                    result = await response.json()
                    content = result["choices"][0]["message"]["content"].strip()

            # 解析JSON
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)
            strategies = result.get("strategies", [])

            # 确保数量匹配
            if len(strategies) < expected_count:
                print(f"⚠️ [Node 4C] LLM返回的策略数量不足，补充默认策略")
                for i in range(len(strategies), expected_count):
                    strategies.append({
                        "shot_index": i + 1,
                        "strategy": "text_to_image",
                        "reference": "none",
                        "reason": "默认策略"
                    })

            return strategies[:expected_count]

        except Exception as e:
            print(f"⚠️ [Node 4C] LLM调用失败: {e}")
            # 返回默认策略（第一个文生图，后续图生图）
            strategies = []
            for i in range(expected_count):
                if i == 0:
                    strategies.append({
                        "shot_index": i + 1,
                        "strategy": "text_to_image",
                        "reference": "none",
                        "reason": "第一帧"
                    })
                else:
                    strategies.append({
                        "shot_index": i + 1,
                        "strategy": "image_to_image",
                        "reference": "previous_frame",
                        "reason": "参考前一帧"
                    })
            return strategies

    def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """支持用户干预调整策略"""
        print("⚠️ regenerate 暂不支持，建议使用 async generate")
        return {"keyframes_with_strategy_id": []}
