# nodes/visual_dna_node.py
"""
全局视觉基因提取节点 (Node 4A in VGP Workflow)

这是视频一致性保障的第一步，负责：
1. 从产品描述中提取视觉基因（材质、形态、颜色、功能）
2. 定义全局视觉词典（色彩、光影、材质、运动、符号）
3. 为后续所有镜头提供统一的视觉规则

工作流位置：
- 输入：来自 node_3_shot_blocks (分镜块生成节点)
- 输出：传递给 node_4b_keyframe_refinement (首帧细化节点)

重要说明：
- 此节点是视频一致性的"宪法"
- 所有后续生成的画面都必须遵循这里定义的视觉规则
"""

from video_generate_protocol import BaseNode
from typing import Dict, List, Any
from datetime import datetime
import asyncio


class VisualDNANode(BaseNode):
    """
    全局视觉基因提取节点

    根据产品信息和分镜脚本，提取并定义：
    1. 视觉基因（核心主题、核心情绪、核心对立、目标风格）
    2. 全局视觉词典（色彩、光影、材质、运动、符号）
    """

    required_inputs = [
        {
            "name": "user_description_id",
            "label": "用户原始输入",
            "type": str,
            "required": True,
            "desc": "用户对产品的描述",
            "field_type": "textarea"
        },
        {
            "name": "shot_blocks_id",
            "label": "分镜块列表",
            "type": list,
            "required": True,
            "desc": "包含分镜描述的结构化列表",
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
            "name": "visual_dna_id",
            "label": "视觉基因",
            "type": dict,
            "required": True,
            "desc": "全局视觉基因和视觉词典",
            "field_type": "json"
        }
    ]

    system_parameters = {
        "llm_model": "qwen-plus",  # 使用千问模型
        "temperature": 0.7,
        "max_tokens": 2000
    }

    def __init__(self, node_id: str, name: str = "全局视觉基因提取"):
        super().__init__(node_id=node_id, node_type="visual_analysis", name=name)

    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_context(context)

        # 提取输入
        user_description = context.get("user_description_id", "")
        shot_blocks = context.get("shot_blocks_id", [])
        reference_media = context.get("reference_media", {})

        if not user_description:
            raise ValueError("缺少用户产品描述")

        # === 步骤1: 分析产品图片（如果有）===
        product_visual_analysis = None
        product_image_url = None

        if reference_media:
            product_images = reference_media.get("product_images", [])
            if product_images and len(product_images) > 0:
                product_image_url = product_images[0].get("url")
                if product_image_url:
                    print(f"📸 [Node 4A] 检测到产品图片: {product_image_url}")
                    print(f"🔍 [Node 4A] 正在使用Qwen-VL分析产品图片...")
                    product_visual_analysis = await self._analyze_product_image(product_image_url)

        # === 步骤2: 构建提示词（融合图片分析结果）===
        prompt = self._build_visual_dna_prompt(
            user_description,
            shot_blocks,
            product_visual_analysis
        )

        # === 步骤3: 调用LLM生成视觉基因 ===
        if product_visual_analysis:
            print(f"🧬 [Node 4A] 正在基于产品图片提取全局视觉基因...")
        else:
            print(f"🧬 [Node 4A] 正在基于文本描述提取全局视觉基因...")

        visual_dna = await self._call_llm(prompt)

        print(f"✅ [Node 4A] 视觉基因提取完成")
        print(f"   - 核心主题: {visual_dna.get('core_theme', 'N/A')}")
        print(f"   - 核心情绪: {visual_dna.get('core_emotion', 'N/A')}")
        print(f"   - 目标风格: {visual_dna.get('target_style', 'N/A')}")
        print(f"   - 数据来源: {'产品图片分析' if product_visual_analysis else '文本描述'}")

        return {
            "visual_dna_id": visual_dna
        }

    def _build_visual_dna_prompt(self, user_description: str, shot_blocks: List[Dict], product_visual_analysis: str = None) -> str:
        """构建视觉基因提取提示词"""

        # 提取分镜描述
        shot_descriptions = []
        for idx, block in enumerate(shot_blocks, 1):
            desc = block.get("visual_description", "")
            if desc:
                shot_descriptions.append(f"{idx}. {desc}")

        shots_text = "\n".join(shot_descriptions)

        # 如果有产品图片分析，添加到提示词中
        product_section = ""
        if product_visual_analysis:
            product_section = f"""
## 产品真实视觉特征（基于Qwen-VL图片分析）
{product_visual_analysis}

⚠️ 重要：必须严格遵循上述产品真实视觉特征，不要凭想象添加不存在的风格元素。
"""

        prompt = f"""# 角色
你是一位视觉总监。你的核心职责是构建一个视觉高度统一、叙事连贯完整的视觉总纲。你不是在生成一堆漂亮的图片，而是在执导一部每一帧都互相关联、共同推进品牌故事的电影。你不是在描述场景，而是在设计一个世界的规则。

# 任务
基于以下产品信息和分镜脚本，提取并定义全局视觉基因和视觉词典。
{product_section}
## 产品描述
{user_description}

## 分镜脚本
{shots_text}

# 输出要求
请按照以下JSON格式输出（必须是有效的JSON，不要有多余的文本）：

{{
  "core_theme": "核心主题，如：重生、内省、连接、未来科技",
  "core_emotion": "核心情绪，如：宁静的忧伤、炽热的期待、神秘的敬畏",
  "core_conflict": "核心对立/张力，如：有机 vs 机械、秩序 vs 混沌、记忆 vs 遗忘",
  "target_style": "目标风格，选择1-2种：生物机械风/赛博朋克/唯美自然/抽象表现主义/史诗电影/极简主义",

  "color_palette": {{
    "primary_colors": ["主色调1", "主色调2"],
    "secondary_colors": ["辅助色1", "辅助色2"],
    "accent_colors": ["点缀色"],
    "constraint": "色彩约束规则"
  }},

  "lighting_rules": {{
    "light_source": "光源类型，如：强烈的定向侧光、弥漫的雾光",
    "quality": "光的质感，如：湿润的反射、粗糙的肌理",
    "constraint": "光影约束规则"
  }},

  "material_language": {{
    "materials": ["材质1", "材质2", "材质3"],
    "constraint": "材质约束规则"
  }},

  "motion_grammar": {{
    "motion_types": ["运动类型1", "运动类型2"],
    "rhythm": "节奏描述",
    "constraint": "运动约束规则"
  }},

  "core_symbols": {{
    "symbols": ["核心符号1", "核心符号2", "核心符号3"],
    "constraint": "符号使用规则"
  }}
}}

重要：请直接输出JSON，不要添加任何解释性文字。"""

        return prompt

    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """调用LLM生成视觉基因"""
        import os
        import json
        import aiohttp

        # 使用千问API
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
                {"role": "system", "content": "你是一位专业的视觉总监，擅长提取视觉基因和定义视觉规则。"},
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

            # 尝试解析JSON
            # 移除可能的markdown代码块标记
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            visual_dna = json.loads(content)
            return visual_dna

        except json.JSONDecodeError as e:
            print(f"⚠️ [Node 4A] JSON解析失败，返回默认视觉基因")
            print(f"   原始内容: {content[:200]}...")
            # 返回默认视觉基因
            return self._get_default_visual_dna()
        except Exception as e:
            print(f"⚠️ [Node 4A] LLM调用失败: {e}")
            return self._get_default_visual_dna()

    def _get_default_visual_dna(self) -> Dict[str, Any]:
        """返回默认的视觉基因"""
        return {
            "core_theme": "产品美学",
            "core_emotion": "平静的期待",
            "core_conflict": "自然 vs 科技",
            "target_style": "极简主义",
            "color_palette": {
                "primary_colors": ["深空灰", "纯白"],
                "secondary_colors": ["暖灰"],
                "accent_colors": ["一瞬的高光白"],
                "constraint": "整体低饱和度，高光区域明亮"
            },
            "lighting_rules": {
                "light_source": "定向侧光",
                "quality": "光滑的陶瓷感反射",
                "constraint": "避免平均光，追求高对比度"
            },
            "material_language": {
                "materials": ["磨砂玻璃", "阳极氧化铝", "柔软织物"],
                "constraint": "材质对比融合"
            },
            "motion_grammar": {
                "motion_types": ["缓慢推轨", "优雅粒子"],
                "rhythm": "慢速，有节奏停顿",
                "constraint": "运动应充满意图"
            },
            "core_symbols": {
                "symbols": ["光晕", "流动线条", "悬浮物体"],
                "constraint": "反复出现形成视觉母题"
            }
        }

    async def _analyze_product_image(self, image_url: str) -> str:
        """使用Qwen-VL API分析产品图片，提取真实的视觉特征"""
        import os
        import aiohttp

        api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('AI__DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("缺少千问API密钥")

        endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 使用Qwen-VL-Max进行图片分析
        request_body = {
            "model": "qwen-vl-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        },
                        {
                            "type": "text",
                            "text": """请详细分析这张产品图片的视觉特征，包括：

1. **主体特征**：产品的形态、材质、颜色
2. **色彩方案**：主色调、辅助色、点缀色（使用具体的颜色名称）
3. **光影效果**：光源类型、光的质感、明暗对比
4. **材质语言**：表面材质、质感描述
5. **艺术风格**：整体视觉风格（如极简主义、工业风、自然风、科技感等）
6. **情绪氛围**：画面传达的情绪感受

请用专业的视觉语言描述，避免主观评价词汇。限制在200字以内。"""
                        }
                    ]
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=request_body, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"⚠️ [Node 4A] Qwen-VL API调用失败 {response.status}: {error_text}")
                        return None

                    result = await response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    print(f"🖼️ [Node 4A] 产品图片分析结果:\n{content}")
                    return content

        except Exception as e:
            print(f"⚠️ [Node 4A] 产品图片分析失败: {e}")
            return None

    def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """支持用户干预调整视觉基因"""
        print("⚠️ regenerate 暂不支持，建议使用 async generate")
        return {"visual_dna_id": self._get_default_visual_dna()}
