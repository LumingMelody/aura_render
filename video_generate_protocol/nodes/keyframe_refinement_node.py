# nodes/keyframe_refinement_node.py
"""
首帧细化节点 (Node 4B in VGP Workflow)

这是视频一致性保障的第二步，负责：
1. 接收分镜块和全局视觉基因
2. 为每个镜头生成结构化的首帧描述（按照7维度）
3. 确保所有镜头共享同一视觉词典

工作流位置：
- 输入：来自 node_4a_visual_dna (视觉基因) 和 node_3_shot_blocks (分镜块)
- 输出：传递给 node_4c_consistency_check (一致性检查节点)

重要说明：
- 按照 [主体+环境+构图+光影+色彩+风格+情绪] 7维度描述
- 字数限定60字以内，简洁高效
- 每个首帧都参考全局视觉词典
"""

from video_generate_protocol import BaseNode
from typing import Dict, List, Any
from datetime import datetime
import asyncio


class KeyframeRefinementNode(BaseNode):
    """
    首帧细化节点

    将简略的分镜描述转化为结构化的、无歧义的视觉指令。
    """

    required_inputs = [
        {
            "name": "shot_blocks_id",
            "label": "分镜块列表",
            "type": list,
            "required": True,
            "desc": "包含分镜描述的结构化列表",
            "field_type": "json"
        },
        {
            "name": "visual_dna_id",
            "label": "全局视觉基因",
            "type": dict,
            "required": True,
            "desc": "全局视觉基因和视觉词典",
            "field_type": "json"
        }
    ]

    output_schema = [
        {
            "name": "refined_keyframes_id",
            "label": "细化后的首帧列表",
            "type": list,
            "required": True,
            "desc": "结构化的首帧描述列表",
            "field_type": "json"
        }
    ]

    system_parameters = {
        "llm_model": "qwen-plus",
        "temperature": 0.7,
        "max_tokens": 3000,
        "max_description_length": 60  # 首帧描述最大字数
    }

    def __init__(self, node_id: str, name: str = "首帧细化"):
        super().__init__(node_id=node_id, node_type="keyframe_analysis", name=name)

    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_context(context)

        # 提取输入
        shot_blocks = context.get("shot_blocks_id", [])
        visual_dna = context.get("visual_dna_id", {})

        if not shot_blocks:
            raise ValueError("缺少分镜块列表")

        print(f"🎨 [Node 4B] 正在细化 {len(shot_blocks)} 个首帧...")

        # 批量细化（为了保持连续性，一次性处理所有镜头）
        refined_keyframes = await self._refine_keyframes_batch(shot_blocks, visual_dna)

        print(f"✅ [Node 4B] 首帧细化完成，共 {len(refined_keyframes)} 个")

        # 打印示例
        if refined_keyframes:
            print(f"\n{'='*80}")
            print(f"📸 细化后的首帧示例:")
            print(f"{'='*80}")
            for idx, kf in enumerate(refined_keyframes[:3], 1):  # 只打印前3个
                print(f"{idx}. {kf.get('refined_prompt', 'N/A')[:80]}...")
            print(f"{'='*80}\n")

        return {
            "refined_keyframes_id": refined_keyframes
        }

    async def _refine_keyframes_batch(self, shot_blocks: List[Dict], visual_dna: Dict) -> List[Dict]:
        """批量细化首帧（保持连续性）"""

        # 构建提示词
        prompt = self._build_refinement_prompt(shot_blocks, visual_dna)

        # 调用LLM
        refined_prompts = await self._call_llm(prompt, len(shot_blocks))

        # 组装结果
        refined_keyframes = []
        for idx, (block, refined_prompt) in enumerate(zip(shot_blocks, refined_prompts)):
            refined_keyframes.append({
                "shot_id": block.get("shot_id", f"shot_{idx+1}"),
                "original_description": block.get("visual_description", ""),
                "refined_prompt": refined_prompt,
                "visual_dna_ref": {
                    "theme": visual_dna.get("core_theme"),
                    "style": visual_dna.get("target_style")
                },
                "metadata": {
                    "shot_index": idx,
                    "duration": block.get("duration", 5.0)
                }
            })

        return refined_keyframes

    def _build_refinement_prompt(self, shot_blocks: List[Dict], visual_dna: Dict) -> str:
        """构建首帧细化提示词"""

        # 提取视觉约束
        color_palette = visual_dna.get("color_palette", {})
        lighting_rules = visual_dna.get("lighting_rules", {})
        material_language = visual_dna.get("material_language", {})
        target_style = visual_dna.get("target_style", "极简主义")
        core_emotion = visual_dna.get("core_emotion", "平静")

        # 提取分镜描述
        shot_descriptions = []
        for idx, block in enumerate(shot_blocks, 1):
            desc = block.get("visual_description", "")
            shot_descriptions.append(f"镜头{idx}: {desc}")

        shots_text = "\n".join(shot_descriptions)

        prompt = f"""# 角色
你是一个专业的首帧细化师，能够从场景布置、角色姿态、光线效果、色彩氛围等多方面进行详细描写。

# 全局视觉约束
请确保所有首帧都遵循以下视觉规则：

**色彩**: {', '.join(color_palette.get('primary_colors', []))} 为主色调
**光影**: {lighting_rules.get('light_source', '定向光')}，质感为 {lighting_rules.get('quality', '柔和')}
**材质**: {', '.join(material_language.get('materials', [])[:3])}
**风格**: {target_style}
**情绪**: {core_emotion}

# 原始分镜描述
{shots_text}

# 任务
请按照以下结构细化每个镜头的首帧描述：

**万能描述结构**: [主体 & 动作] + [环境 & 背景] + [构图 & 视角] + [光影效果] + [色彩色调] + [艺术风格] + [情绪氛围]

# 要求
1. 每个首帧描述限制在60字以内
2. 必须具体、可量化，避免抽象词汇（如"美丽"、"震撼"）
3. 前后镜头要有统一性和连续性
4. 如果是产品特写，明确说明是产品的哪一部分
5. 坚持单一视觉焦点原则

# ⚠️ 重要：动态运动描述（用于图生视频）
**每个镜头都必须包含明确的动态描述**，这些描述将用于生成5秒视频：

- ✅ 好的动态描述：
  * "镜头从左向右快速环绕产品360度"
  * "产品缓慢旋转展示正面到侧面"
  * "镜头由远及近推进，焦点从背景过渡到产品细节"
  * "产品从静止状态突然启动，光效渐亮"
  * "手部动作：拿起、放下、触摸按钮"

- ❌ 避免静态描述：
  * "产品放置在桌面上" - 太静止
  * "产品特写" - 没有动作
  * "缓慢推移" - 动作不明确

**动态描述模板**：
- 镜头运动：环绕/推进/拉远/平移/俯仰
- 产品运动：旋转/展开/启动/变化
- 人物交互：拿起/触摸/操作/展示
- 光影变化：光线渐强/色彩变化/粒子特效

确保每个镜头都能在5秒内呈现清晰的视觉变化。

# 输出格式
请按照以下JSON格式输出（必须是有效的JSON）：

{{
  "refined_prompts": [
    "镜头1的细化描述（60字以内）",
    "镜头2的细化描述（60字以内）",
    ...
  ]
}}

重要：直接输出JSON，不要添加任何解释性文字。"""

        return prompt

    async def _call_llm(self, prompt: str, expected_count: int) -> List[str]:
        """调用LLM生成细化后的首帧描述"""
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
                {"role": "system", "content": "你是一位专业的首帧细化师，擅长将简略描述转化为结构化的视觉指令。"},
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
            refined_prompts = result.get("refined_prompts", [])

            # 确保数量匹配
            if len(refined_prompts) < expected_count:
                print(f"⚠️ [Node 4B] LLM返回的描述数量不足，补充默认描述")
                while len(refined_prompts) < expected_count:
                    refined_prompts.append("产品特写，柔和侧光，极简构图，低饱和色调")

            return refined_prompts[:expected_count]

        except Exception as e:
            print(f"⚠️ [Node 4B] LLM调用失败: {e}")
            # 返回默认描述
            return [f"镜头{i+1}：产品特写，柔和光影，极简风格" for i in range(expected_count)]

    def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """支持用户干预调整首帧"""
        print("⚠️ regenerate 暂不支持，建议使用 async generate")
        return {"refined_keyframes_id": []}
