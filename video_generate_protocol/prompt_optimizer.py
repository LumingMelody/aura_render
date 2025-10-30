"""
视频生成提示词优化器
实现视频生成.md中的12步提示词优化流程
"""
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisualStyle:
    """视觉风格定义"""
    core_theme: str  # 核心主题
    core_emotion: str  # 核心情绪
    core_tension: str  # 核心对立/张力
    target_style: str  # 目标风格

    # 全局视觉词典
    color_palette: Dict[str, List[str]]  # 主色调、辅助色、点缀色
    lighting_rules: Dict[str, str]  # 光源、质感、约束
    material_language: List[str]  # 材质列表
    motion_grammar: Dict[str, str]  # 运动类型、节奏、约束
    core_symbols: List[str]  # 核心符号


@dataclass
class StoryboardShot:
    """分镜镜头"""
    shot_index: int  # 镜头索引
    description: str  # 镜头描述
    reason: str  # 设计理由
    duration: float  # 时长
    is_continuous: bool  # 是否连续镜头

    # 细化后的内容
    first_frame: Optional[str] = None  # 首帧描述
    first_frame_refined: Optional[str] = None  # 首帧细化（60字）
    first_frame_clean: Optional[str] = None  # 去括号后的首帧

    middle_process: Optional[str] = None  # 中间过程描述
    middle_process_refined: Optional[str] = None  # 中间过程细化（运镜）
    middle_process_clean: Optional[str] = None  # 去括号后的中间过程

    # 一致性策略
    generation_strategy: str = "text_to_image"  # text_to_image 或 image_to_image
    reference_source: str = "none"  # none, previous_frame, product_image


@dataclass
class OptimizedPromptResult:
    """优化后的提示词结果"""
    product_description: str  # 产品描述
    marketing_analysis: Dict[str, Any]  # 宣传偏好分析
    era_preference: str  # 时代偏好
    visual_style: VisualStyle  # 视觉风格
    storyboard: List[StoryboardShot]  # 分镜列表
    total_duration: float  # 总时长


class VideoPromptOptimizer:
    """视频生成提示词优化器"""

    def __init__(self, qwen_llm=None):
        """
        初始化优化器

        参数:
            qwen_llm: Qwen大模型实例（用于调用LLM）
        """
        from llm.qwen import QwenLLM
        self.qwen = qwen_llm or QwenLLM()

    async def optimize(
        self,
        product_name: str,
        user_input: Optional[str] = None,
        target_duration: int = 60
    ) -> OptimizedPromptResult:
        """
        执行完整的12步优化流程

        参数:
            product_name: 产品名称
            user_input: 用户额外输入（可选）
            target_duration: 目标视频时长（秒），默认60秒

        返回:
            优化后的提示词结果
        """
        logger.info(f"🎬 开始视频提示词优化流程: {product_name}")

        # 步骤1：全局产品描述
        product_desc = await self._step1_product_description(product_name)
        logger.info(f"✅ 步骤1完成 - 产品描述: {product_desc[:50]}...")

        # 步骤2：宣传偏好分析
        marketing_analysis = await self._step2_marketing_preference(product_name, product_desc)
        logger.info(f"✅ 步骤2完成 - 宣传偏好分析")

        # 步骤3：产品时代偏好
        era_preference = await self._step3_era_preference(product_name, product_desc, user_input)
        logger.info(f"✅ 步骤3完成 - 时代偏好: {era_preference}")

        # 步骤4：故事线分镜设计
        raw_storyboard = await self._step4_storyboard_design(
            product_name, product_desc, marketing_analysis, era_preference, target_duration
        )
        logger.info(f"✅ 步骤4完成 - 生成{len(raw_storyboard)}个分镜")

        # 步骤5：全局要素统一（视觉基因）
        visual_style = await self._step5_visual_unification(
            product_name, product_desc, raw_storyboard
        )
        logger.info(f"✅ 步骤5完成 - 视觉风格: {visual_style.target_style}")

        # 步骤6：片段分割（判断连续性）
        storyboard = await self._step6_segment_division(raw_storyboard)
        logger.info(f"✅ 步骤6完成 - 连续性分析")

        # 步骤7：首帧和中间过程描述
        storyboard = await self._step7_frame_process_description(storyboard)
        logger.info(f"✅ 步骤7完成 - 首帧和中间过程描述")

        # 步骤8-9：首帧细化 + 去括号（循环处理每个镜头）
        for i, shot in enumerate(storyboard):
            # 步骤8：首帧细化
            shot.first_frame_refined = await self._step8_first_frame_refinement(
                shot, visual_style, product_name, i
            )
            # 步骤9：去括号
            shot.first_frame_clean = self._step9_remove_brackets(shot.first_frame_refined)
            logger.info(f"✅ 步骤8-9完成 - 镜头{i+1}首帧细化")

        # 步骤10：一致性检查（图生图判断）
        storyboard = await self._step10_consistency_check(storyboard, product_name)
        logger.info(f"✅ 步骤10完成 - 一致性策略")

        # 步骤11-12：中间过程细化 + 去括号（循环处理每个镜头）
        for i, shot in enumerate(storyboard):
            # 步骤11：中间过程细化（运镜）
            shot.middle_process_refined = await self._step11_middle_process_refinement(
                shot, visual_style, storyboard
            )
            # 步骤12：去括号
            shot.middle_process_clean = self._step9_remove_brackets(shot.middle_process_refined)
            logger.info(f"✅ 步骤11-12完成 - 镜头{i+1}中间过程细化")

        # 计算总时长
        total_duration = sum(shot.duration for shot in storyboard)

        result = OptimizedPromptResult(
            product_description=product_desc,
            marketing_analysis=marketing_analysis,
            era_preference=era_preference,
            visual_style=visual_style,
            storyboard=storyboard,
            total_duration=total_duration
        )

        logger.info(f"🎉 提示词优化完成！共{len(storyboard)}个镜头，总时长{total_duration}秒")
        return result

    async def _step1_product_description(self, product_name: str) -> str:
        """步骤1：全局产品描述"""
        prompt = f"""你是一个产品描述专家。请为以下产品生成简洁精准的描述，突出其关键特性和主要用途。

产品名称：{product_name}

要求：
1. 描述简洁明了，50字以内
2. 突出产品核心特性
3. 包含主要用途和目标人群

只输出产品描述，不要额外解释。"""

        response = await self._call_llm(prompt)
        return response.strip()

    async def _step2_marketing_preference(self, product_name: str, product_desc: str) -> Dict[str, Any]:
        """步骤2：宣传偏好分析"""
        prompt = f"""你是一位资深的市场营销与品牌策略专家。请分析以下产品的宣传策略。

产品：{product_name}
描述：{product_desc}

请完成以下分析（以JSON格式输出）：
1. product_category: 产品类别（如：食品、美妆、电子产品等）
2. marketing_pitfalls: 3-5个宣传雷点（近1-2年的翻车案例）
3. preference_trends: 3-5个目标受众偏好的宣传方式

输出JSON格式：
{{
  "product_category": "类别",
  "marketing_pitfalls": ["雷点1", "雷点2", "雷点3"],
  "preference_trends": ["偏好1", "偏好2", "偏好3"]
}}"""

        response = await self._call_llm(prompt)
        try:
            # 提取JSON
            json_str = self._extract_json(response)
            return self._parse_json_robust(json_str)
        except:
            return {
                "product_category": "通用产品",
                "marketing_pitfalls": ["避免过度夸张", "避免虚假宣传"],
                "preference_trends": ["真实场景展示", "用户口碑"]
            }

    async def _step3_era_preference(self, product_name: str, product_desc: str, user_input: Optional[str]) -> str:
        """步骤3：产品时代偏好"""
        prompt = f"""你是一位前沿的产品背景洞察官。请判断以下产品适合的时代背景。

产品：{product_name}
描述：{product_desc}
用户需求：{user_input or "无特殊要求"}

请从以下选项中选择最合适的时代背景，并只输出一个词：
- modern（现代化，当下，科技感）
- traditional（传统，古法，经典）
- retro（复古，怀旧，年代感）
- futuristic（未来，前沿，超前）

只输出一个词，不要解释。"""

        response = await self._call_llm(prompt)
        era = response.strip().lower()
        if era not in ["modern", "traditional", "retro", "futuristic"]:
            era = "modern"
        return era

    async def _step4_storyboard_design(
        self,
        product_name: str,
        product_desc: str,
        marketing_analysis: Dict,
        era_preference: str,
        target_duration: int = 60
    ) -> List[StoryboardShot]:
        """步骤4：故事线分镜设计"""

        # ✅ 根据目标时长动态计算分镜数量
        # 规则：每个镜头2-3秒，计算需要多少个镜头
        min_shot_duration = 2.0
        max_shot_duration = 3.0
        avg_shot_duration = 2.5

        # 计算建议的镜头数量（向上取整确保不超时）
        shots_count = max(3, min(10, int(target_duration / avg_shot_duration)))
        avg_duration = target_duration / shots_count

        logger.info(f"📊 [步骤4] 目标时长: {target_duration}秒, 计划生成: {shots_count}个镜头, 平均时长: {avg_duration:.1f}秒")

        prompt = f"""你是一位专业的广告导演。请为以下产品设计一个{target_duration}秒的高端宣传片分镜脚本。

产品：{product_name}
描述：{product_desc}
时代背景：{era_preference}
宣传偏好：{marketing_analysis.get('preference_trends', [])}
避免雷点：{marketing_analysis.get('marketing_pitfalls', [])}

遵循「惊鸿一瞥」高端品牌短片设计规范：
1. 「克制即高级」：用最精准的镜头传递最明确的信息
2. 「静态即力量」：以固定镜头、微动镜头替代复杂运镜
3. 「片段即整体」：2-3秒的短镜头快速组接
4. 「主体原则」：每个镜头必须有明确的主体
5. 画面不要出现完整的人（但可以出现人的部分，如手、眼睛等）
6. 每个分镜只动一个东西，强调慢动作质感
7. 基于静物拍摄，避免复杂动态

⚠️ 重要：严格控制总时长
- 请设计**恰好{shots_count}个**分镜
- 每个分镜时长在{min_shot_duration}-{max_shot_duration}秒之间
- 所有分镜总时长必须接近{target_duration}秒（误差不超过1秒）
- 建议每个分镜平均时长：{avg_duration:.1f}秒

每个分镜包含：
- 画面描述（30字以内，描述镜头内容）
- 时长（{min_shot_duration}-{max_shot_duration}秒之间的小数，确保总和={target_duration}秒）
- 设计理由（说明为什么这样设计）

以JSON数组格式输出：
[
  {{
    "shot_index": 1,
    "description": "画面描述",
    "duration": 2.5,
    "reason": "设计理由"
  }},
  ...
]"""

        response = await self._call_llm(prompt)
        try:
            json_str = self._extract_json(response)
            shots_data = self._parse_json_robust(json_str)

            storyboard = []
            for shot in shots_data:
                storyboard.append(StoryboardShot(
                    shot_index=shot.get("shot_index", len(storyboard) + 1),
                    description=shot.get("description", ""),
                    reason=shot.get("reason", ""),
                    duration=shot.get("duration", 2.5),
                    is_continuous=False  # 将在步骤6判断
                ))

            # ✅ 校验总时长，如果超过target_duration则按比例缩放
            total_duration = sum(shot.duration for shot in storyboard)
            if total_duration > target_duration + 1:  # 允许1秒误差
                logger.warning(f"⚠️ 生成的分镜总时长{total_duration}秒超过目标{target_duration}秒，按比例缩放")
                scale_factor = target_duration / total_duration
                for shot in storyboard:
                    shot.duration = round(shot.duration * scale_factor, 1)
                logger.info(f"✅ 缩放后总时长: {sum(shot.duration for shot in storyboard):.1f}秒")

            # ✅ 给每个镜头增加0.5秒缓冲，防止音频被截断
            # TTS生成的音频长度可能比预期稍长，增加缓冲确保音频完整播放
            for shot in storyboard:
                shot.duration += 0.5
            logger.info(f"✅ 增加缓冲后总时长: {sum(shot.duration for shot in storyboard):.1f}秒")

            logger.info(f"📊 [步骤4] 实际生成: {len(storyboard)}个镜头, 总时长: {sum(shot.duration for shot in storyboard):.1f}秒")

            return storyboard
        except Exception as e:
            logger.warning(f"解析分镜失败: {e}，使用默认分镜")
            return self._create_default_storyboard(product_name)

    async def _step5_visual_unification(
        self,
        product_name: str,
        product_desc: str,
        storyboard: List[StoryboardShot]
    ) -> VisualStyle:
        """步骤5：全局要素统一（视觉基因）"""
        storyboard_summary = "\n".join([f"{i+1}. {shot.description}" for i, shot in enumerate(storyboard[:5])])

        prompt = f"""你是一位视觉总监。请为以下产品和分镜设计统一的视觉风格体系。

产品：{product_name}
描述：{product_desc}

分镜概要：
{storyboard_summary}

请设计全局视觉风格（JSON格式）：
{{
  "core_theme": "核心主题（如：科技与人性的连接）",
  "core_emotion": "核心情绪（如：静谧的期待）",
  "core_tension": "核心对立（如：冰冷的金属 vs 温暖的人性）",
  "target_style": "目标风格（如：极简主义、电影感、赛博朋克等）",
  "color_palette": {{
    "main": ["主色调1", "主色调2"],
    "auxiliary": ["辅助色1", "辅助色2"],
    "accent": ["点缀色"]
  }},
  "lighting_rules": {{
    "source": "光源类型（如：强烈定向侧光、柔和顶光）",
    "texture": "质感（如：湿润的反射、粗糙的肌理）",
    "constraint": "约束（如：高对比度、避免平均光）"
  }},
  "material_language": ["材质1", "材质2", "材质3"],
  "motion_grammar": {{
    "type": "运动类型（如：缓慢推轨、优雅粒子汇聚）",
    "rhythm": "节奏（如：慢速、有节奏停顿）",
    "constraint": "约束（如：充满意图、避免随机晃动）"
  }},
  "core_symbols": ["核心符号1", "核心符号2"]
}}"""

        response = await self._call_llm(prompt)
        try:
            json_str = self._extract_json(response)
            data = self._parse_json_robust(json_str)

            return VisualStyle(
                core_theme=data.get("core_theme", "产品展示"),
                core_emotion=data.get("core_emotion", "专业自信"),
                core_tension=data.get("core_tension", "静止与动态"),
                target_style=data.get("target_style", "写实风格"),
                color_palette=data.get("color_palette", {
                    "main": ["自然色"], "auxiliary": ["白色"], "accent": ["高光"]
                }),
                lighting_rules=data.get("lighting_rules", {
                    "source": "自然光", "texture": "柔和", "constraint": "真实"
                }),
                material_language=data.get("material_language", ["金属", "玻璃"]),
                motion_grammar=data.get("motion_grammar", {
                    "type": "缓慢推进", "rhythm": "平稳", "constraint": "流畅"
                }),
                core_symbols=data.get("core_symbols", ["产品"])
            )
        except Exception as e:
            logger.warning(f"解析视觉风格失败: {e}，使用默认风格")
            return self._create_default_visual_style()

    async def _step6_segment_division(self, storyboard: List[StoryboardShot]) -> List[StoryboardShot]:
        """步骤6：片段分割（判断连续性）"""
        if len(storyboard) <= 1:
            return storyboard

        shots_desc = "\n".join([
            f"镜头{i+1}: {shot.description} (理由: {shot.reason})"
            for i, shot in enumerate(storyboard)
        ])

        prompt = f"""你是连续性执行设计师。请判断以下分镜中哪些镜头是连续的（需要一次拍摄完成）。

{shots_desc}

连续镜头的标志：
- 出现"化为、扩散为、幻化为、浮现、划出、收束为"等平滑转变动词
- 共享同一主体、同一构图
- 属于同一情绪单元
- 导演注明"长镜头"或"呼吸感"

独立镜头的标志：
- 景别、主体、场景跳跃
- 需要强调独立信息点
- 注明"静态即力量"、"克制"的特写

请输出JSON数组，标记每个镜头的连续性（true=连续，false=独立）：
[true, false, true, false, ...]"""

        response = await self._call_llm(prompt)
        try:
            json_str = self._extract_json(response)
            continuity = self._parse_json_robust(json_str)

            for i, is_cont in enumerate(continuity):
                if i < len(storyboard):
                    storyboard[i].is_continuous = bool(is_cont)
        except Exception as e:
            logger.warning(f"解析连续性失败: {e}，默认所有镜头独立")
            for shot in storyboard:
                shot.is_continuous = False

        return storyboard

    async def _step7_frame_process_description(self, storyboard: List[StoryboardShot]) -> List[StoryboardShot]:
        """步骤7：首帧和中间过程描述"""
        for i, shot in enumerate(storyboard):
            prompt = f"""你是分镜处理专家。请为以下镜头确定首帧和中间过程。

镜头{shot.shot_index}：
画面：{shot.description}
理由：{shot.reason}
时长：{shot.duration}秒

请输出JSON格式：
{{
  "first_frame": "首帧描述（起始瞬间的画面元素、场景布局）",
  "middle_process": "中间过程描述（场景变化、动作、与前后镜头的关联）"
}}

要求：
- 首帧描述清晰具体，描述静态画面
- 中间过程描述动态变化，但不要太复杂
- 保持与整体分镜的统一性"""

            response = await self._call_llm(prompt)
            try:
                json_str = self._extract_json(response)
                data = self._parse_json_robust(json_str)
                shot.first_frame = data.get("first_frame", shot.description)
                shot.middle_process = data.get("middle_process", "画面平滑过渡")
            except Exception as e:
                logger.warning(f"镜头{i+1}首帧描述失败: {e}")
                shot.first_frame = shot.description
                shot.middle_process = "画面平滑过渡"

        return storyboard

    async def _step8_first_frame_refinement(
        self,
        shot: StoryboardShot,
        visual_style: VisualStyle,
        product_name: str,
        shot_index: int
    ) -> str:
        """步骤8：首帧细化（60字限定，结构化描述）"""
        # 获取前一个镜头的首帧（用于保持连续性）
        previous_context = ""

        prompt = f"""你是专业的首帧细化师。请将以下首帧描述细化为结构化的视觉指令。

产品：{product_name}
镜头{shot.shot_index}首帧：{shot.first_frame}
设计理由：{shot.reason}

全局视觉风格：
- 主题：{visual_style.core_theme}
- 情绪：{visual_style.core_emotion}
- 风格：{visual_style.target_style}
- 主色调：{visual_style.color_palette.get('main', [])}
- 光源：{visual_style.lighting_rules.get('source', '')}
- 材质：{visual_style.material_language}

请按照以下结构细化首帧（总共不超过60字）：
[主体与动作] + [环境与背景] + [构图与视角] + [光影效果] + [色彩色调] + [艺术风格]

要求：
1. 字数限定60字以内
2. 使用具体的专业术语（特写/中景/全景、俯角/仰角、硬光/柔光等）
3. 避免抽象词汇（美丽、震撼、高级等）
4. 清晰描述，减少歧义
5. 如果是产品特写，明确说明是产品的哪一部分

只输出细化后的首帧描述，不要其他内容。"""

        response = await self._call_llm(prompt)
        refined = response.strip()

        # 确保不超过60字
        if len(refined) > 60:
            refined = refined[:60]

        return refined

    def _step9_remove_brackets(self, text: str) -> str:
        """步骤9：去括号优化"""
        if not text:
            return text

        import re
        # 移除所有中英文括号及其内容
        cleaned = re.sub(r'[（(].*?[）)]', '', text)
        # 移除多余空格
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    async def _step10_consistency_check(
        self,
        storyboard: List[StoryboardShot],
        product_name: str
    ) -> List[StoryboardShot]:
        """步骤10：一致性检查（图生图判断）"""
        shots_desc = "\n".join([
            f"镜头{shot.shot_index}: {shot.first_frame or shot.description}"
            for shot in storyboard
        ])

        prompt = f"""你是场景一致性判断助手。请判断以下分镜中哪些镜头需要使用图生图保持一致性。

产品：{product_name}

{shots_desc}

图生图判断规则：
1. 优先判断是否为同一物体（若是则使用图生图）
2. 若不是同一物体但画面中有产品，则使用产品图进行图生图
3. 若场景高度相似（相同背景、物体或人物，仅视角、动作略有变化），则需要图生图
4. 若场景完全不同（切换背景、物体），则不需要图生图

请输出JSON数组，为每个镜头标记生成策略：
[
  {{
    "shot_index": 1,
    "generation_strategy": "text_to_image",  // 或 "image_to_image"
    "reference_source": "none"  // none, previous_frame, product_image
  }},
  ...
]"""

        response = await self._call_llm(prompt)
        try:
            json_str = self._extract_json(response)
            strategies = self._parse_json_robust(json_str)

            for strategy in strategies:
                idx = strategy.get("shot_index", 0) - 1
                if 0 <= idx < len(storyboard):
                    storyboard[idx].generation_strategy = strategy.get("generation_strategy", "text_to_image")
                    storyboard[idx].reference_source = strategy.get("reference_source", "none")
        except Exception as e:
            logger.warning(f"解析一致性策略失败: {e}，使用默认策略")
            # 第一个镜头使用产品图，后续镜头参考前一帧
            for i, shot in enumerate(storyboard):
                if i == 0:
                    shot.generation_strategy = "image_to_image"
                    shot.reference_source = "product_image"
                else:
                    shot.generation_strategy = "image_to_image"
                    shot.reference_source = "previous_frame"

        return storyboard

    async def _step11_middle_process_refinement(
        self,
        shot: StoryboardShot,
        visual_style: VisualStyle,
        all_shots: List[StoryboardShot]
    ) -> str:
        """步骤11：中间过程细化（运镜描述）"""
        prompt = f"""你是运镜设计专家。请为以下镜头设计详细的中间过程和运镜方式。

镜头{shot.shot_index}：
首帧：{shot.first_frame_clean or shot.first_frame}
中间过程：{shot.middle_process}
时长：{shot.duration}秒

全局运动规则：
- 运动类型：{visual_style.motion_grammar.get('type', '')}
- 节奏：{visual_style.motion_grammar.get('rhythm', '')}
- 约束：{visual_style.motion_grammar.get('constraint', '')}

请使用专业运镜术语优化中间过程：
- 运镜方式：推、拉、摇、移、跟、升、降
- 速度节奏：匀速、先快后慢、先慢后快、突然
- 焦点变化：景深变化、焦点转移

要求：
1. 描述简单流畅，不要过于复杂
2. 每个镜头只动一个东西
3. 实拍画面和特效要区分开
4. 变化描述简单，不要复杂夸张

只输出细化后的中间过程描述，不要其他内容。"""

        response = await self._call_llm(prompt)
        return response.strip()

    def _create_default_storyboard(self, product_name: str) -> List[StoryboardShot]:
        """创建默认分镜（当解析失败时）"""
        return [
            StoryboardShot(1, f"{product_name}产品特写", "展示产品细节", 2.5, False),
            StoryboardShot(2, f"{product_name}使用场景", "展示使用价值", 3.0, False),
            StoryboardShot(3, f"{product_name}核心功能展示", "突出核心卖点", 2.5, False),
            StoryboardShot(4, f"{product_name}全貌展示", "整体呈现", 3.0, False),
        ]

    def _create_default_visual_style(self) -> VisualStyle:
        """创建默认视觉风格"""
        return VisualStyle(
            core_theme="产品展示",
            core_emotion="专业可信",
            core_tension="静止与动态",
            target_style="写实风格",
            color_palette={"main": ["自然色"], "auxiliary": ["白色"], "accent": ["高光"]},
            lighting_rules={"source": "自然光", "texture": "柔和", "constraint": "真实"},
            material_language=["金属", "玻璃", "塑料"],
            motion_grammar={"type": "缓慢推进", "rhythm": "平稳", "constraint": "流畅"},
            core_symbols=["产品主体"]
        )

    async def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """调用LLM（支持异步）"""
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        response = await loop.run_in_executor(
            executor,
            lambda: self.qwen.generate(prompt=prompt, max_retries=max_retries)
        )

        return str(response) if response else ""

    def _extract_json(self, text: str) -> str:
        """从文本中提取JSON"""
        import re

        # 尝试提取JSON代码块
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # 尝试提取普通代码块
        code_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # 尝试提取大括号或中括号包围的内容
        brace_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if brace_match:
            return brace_match.group(1)

        return text

    def _parse_json_robust(self, json_str: str) -> Any:
        """鲁棒的JSON解析，支持多种格式"""
        import re

        # 清理JSON字符串
        cleaned = json_str.strip()

        # 1. 先尝试标准json.loads
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 2. 尝试修复常见问题
        try:
            # 移除注释（// 和 /* */）
            cleaned = re.sub(r'//.*?$', '', cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)

            # 移除尾部逗号
            cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)

            # 替换单引号为双引号（但要小心字符串中的单引号）
            # 这个正则会匹配键名和字符串值
            cleaned = re.sub(r"'([^']*)'", r'"\1"', cleaned)

            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 3. 最后尝试使用json5库（如果可用）
        try:
            import json5
            return json5.loads(json_str)
        except ImportError:
            pass
        except Exception:
            pass

        # 4. 如果都失败，抛出原始错误
        raise json.JSONDecodeError(f"无法解析JSON: {json_str[:100]}...", json_str, 0)
