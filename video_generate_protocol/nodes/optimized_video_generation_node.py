"""
优化的视频生成节点 - 集成12步提示词优化流程
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_generate_protocol.prompt_optimizer import VideoPromptOptimizer, OptimizedPromptResult
from video_generate_protocol.nodes.qwen_integration import StoryboardToVideoProcessor

logger = logging.getLogger(__name__)


class OptimizedVideoGenerationNode:
    """
    优化的视频生成节点

    完整流程：
    1. 使用提示词优化器生成详细的分镜和提示词（12步流程）
    2. 将优化后的提示词转换为视频生成参数
    3. 调用Qwen视频生成API生成视频
    4. 返回完整的视频序列
    """

    required_inputs = [
        {
            "name": "product_name",
            "label": "产品名称",
            "type": str,
            "required": True,
            "desc": "待生成宣传视频的产品名称"
        },
        {
            "name": "product_image_url",
            "label": "产品图片URL",
            "type": str,
            "required": False,
            "desc": "产品参考图片URL（用于一致性保障）"
        },
        {
            "name": "user_requirements",
            "label": "用户需求",
            "type": str,
            "required": False,
            "desc": "用户额外的需求描述"
        }
    ]

    def __init__(self, qwen_api_key: str):
        """
        初始化节点

        参数:
            qwen_api_key: 千问API密钥
        """
        self.optimizer = VideoPromptOptimizer()
        self.video_processor = StoryboardToVideoProcessor(qwen_api_key)
        self.qwen_api_key = qwen_api_key

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行优化的视频生成流程

        参数:
            context: 上下文数据，包含：
                - product_name: 产品名称
                - product_image_url: 产品图片URL（可选）
                - user_requirements: 用户需求（可选）
                - output_dir: 输出目录（可选）

        返回:
            包含视频片段和优化信息的结果
        """
        product_name = context.get("product_name")
        if not product_name:
            raise ValueError("缺少必需参数: product_name")

        product_image_url = context.get("product_image_url")
        user_requirements = context.get("user_requirements")
        output_dir = context.get("output_dir", "/tmp/optimized_video_output")

        logger.info(f"\n{'='*80}")
        logger.info(f"🎬 开始优化的视频生成流程")
        logger.info(f"📦 产品: {product_name}")
        if product_image_url:
            logger.info(f"🖼️  产品图: {product_image_url[:80]}...")
        logger.info(f"{'='*80}\n")

        # ==================== 阶段1: 提示词优化 ====================
        logger.info(f"🔧 阶段1: 执行12步提示词优化流程...")
        optimized_result = await self.optimizer.optimize(
            product_name=product_name,
            user_input=user_requirements
        )

        logger.info(f"✅ 提示词优化完成")
        logger.info(f"   📊 生成{len(optimized_result.storyboard)}个分镜")
        logger.info(f"   🎨 视觉风格: {optimized_result.visual_style.target_style}")
        logger.info(f"   ⏱️  总时长: {optimized_result.total_duration}秒")

        # ==================== 阶段2: 转换为视频生成参数 ====================
        logger.info(f"\n🔄 阶段2: 转换为视频生成参数...")
        keyframes_with_strategy = self._convert_to_keyframes(optimized_result)

        logger.info(f"✅ 参数转换完成")
        logger.info(f"   📸 生成{len(keyframes_with_strategy)}个关键帧参数")

        # ==================== 阶段3: 生成视频片段 ====================
        logger.info(f"\n🎥 阶段3: 生成视频片段...")
        video_clips = await self.video_processor.process_keyframes_with_consistency(
            keyframes_with_strategy=keyframes_with_strategy,
            output_dir=output_dir,
            product_image_url=product_image_url
        )

        logger.info(f"✅ 视频片段生成完成")
        logger.info(f"   🎬 成功生成{len(video_clips)}个视频片段")

        # ==================== 阶段4: 合并视频（可选） ====================
        final_video_url = None
        if video_clips and context.get("merge_clips", True):
            logger.info(f"\n🔗 阶段4: 合并视频片段...")
            try:
                merge_result = await self.video_processor.merge_clips(
                    clip_data=video_clips,
                    output_path=f"{output_dir}/final_video.mp4",
                    subtitle_sequence=context.get("subtitle_sequence"),  # 可选字幕
                    vgp_context=context.get("vgp_context")  # 可选VGP特效
                )

                if merge_result.get("success"):
                    final_video_url = merge_result.get("video_url")
                    logger.info(f"✅ 视频合并完成")
                    logger.info(f"   🎬 最终视频: {final_video_url[:80]}...")
            except Exception as e:
                logger.warning(f"⚠️ 视频合并失败: {e}")
                logger.info(f"   ℹ️  将返回独立的视频片段")

        # ==================== 返回结果 ====================
        logger.info(f"\n{'='*80}")
        logger.info(f"🎉 优化的视频生成流程完成！")
        logger.info(f"{'='*80}\n")

        return {
            "success": True,
            "product_name": product_name,

            # 优化结果
            "optimization": {
                "product_description": optimized_result.product_description,
                "marketing_analysis": optimized_result.marketing_analysis,
                "era_preference": optimized_result.era_preference,
                "visual_style": {
                    "core_theme": optimized_result.visual_style.core_theme,
                    "core_emotion": optimized_result.visual_style.core_emotion,
                    "core_tension": optimized_result.visual_style.core_tension,
                    "target_style": optimized_result.visual_style.target_style,
                    "color_palette": optimized_result.visual_style.color_palette,
                    "lighting_rules": optimized_result.visual_style.lighting_rules,
                },
                "storyboard_count": len(optimized_result.storyboard),
                "total_duration": optimized_result.total_duration
            },

            # 分镜详情
            "storyboard": [
                {
                    "shot_index": shot.shot_index,
                    "description": shot.description,
                    "reason": shot.reason,
                    "duration": shot.duration,
                    "first_frame_refined": shot.first_frame_refined,
                    "first_frame_clean": shot.first_frame_clean,
                    "middle_process_refined": shot.middle_process_refined,
                    "middle_process_clean": shot.middle_process_clean,
                    "generation_strategy": shot.generation_strategy,
                    "reference_source": shot.reference_source
                }
                for shot in optimized_result.storyboard
            ],

            # 视频结果
            "video_clips": video_clips,
            "final_video_url": final_video_url,
            "clips_count": len(video_clips)
        }

    def _convert_to_keyframes(self, optimized_result: OptimizedPromptResult) -> List[Dict]:
        """
        将优化后的分镜转换为视频生成所需的关键帧参数

        参数:
            optimized_result: 优化后的提示词结果

        返回:
            关键帧参数列表
        """
        keyframes = []

        for shot in optimized_result.storyboard:
            # 使用清理后的首帧描述作为图片生成提示词
            image_prompt = shot.first_frame_clean or shot.first_frame_refined or shot.first_frame or shot.description

            # 使用清理后的中间过程作为视频运动提示词
            video_prompt = shot.middle_process_clean or shot.middle_process_refined or shot.middle_process or "画面平滑过渡"

            # 组合完整的提示词（包含视觉风格）
            visual_style = optimized_result.visual_style

            # 为图片生成添加风格约束
            full_image_prompt = self._add_visual_style_to_prompt(
                image_prompt,
                visual_style,
                is_image=True
            )

            # 为视频生成添加运动描述
            full_video_prompt = self._add_visual_style_to_prompt(
                video_prompt,
                visual_style,
                is_image=False
            )

            keyframe = {
                "shot_index": shot.shot_index,
                "refined_prompt": full_image_prompt,  # 用于图片生成
                "video_prompt": full_video_prompt,  # 用于视频生成的运动描述
                "duration": shot.duration,
                "generation_strategy": shot.generation_strategy,
                "reference_source": shot.reference_source,

                # 调试信息
                "original_description": shot.description,
                "design_reason": shot.reason
            }

            keyframes.append(keyframe)

        return keyframes

    def _add_visual_style_to_prompt(
        self,
        base_prompt: str,
        visual_style,
        is_image: bool = True
    ) -> str:
        """
        为基础提示词添加视觉风格约束

        参数:
            base_prompt: 基础提示词
            visual_style: 视觉风格对象
            is_image: 是否为图片生成（True）还是视频运动描述（False）

        返回:
            增强后的提示词
        """
        # 提取关键风格元素
        style = visual_style.target_style
        main_colors = ", ".join(visual_style.color_palette.get("main", []))
        lighting = visual_style.lighting_rules.get("source", "")

        if is_image:
            # 图片生成：强调静态画面、构图、光影、色彩
            style_suffix = f"风格: {style}"
            if main_colors:
                style_suffix += f", 色调: {main_colors}"
            if lighting:
                style_suffix += f", 光源: {lighting}"

            return f"{base_prompt}, {style_suffix}"
        else:
            # 视频运动：强调运动方式、节奏、流畅性
            motion_type = visual_style.motion_grammar.get("type", "")
            rhythm = visual_style.motion_grammar.get("rhythm", "")

            style_suffix = f"风格: {style}, 运动流畅自然"
            if motion_type:
                style_suffix += f", {motion_type}"
            if rhythm:
                style_suffix += f", 节奏{rhythm}"

            return f"{base_prompt}, {style_suffix}"


# ==================== 使用示例 ====================
async def demo():
    """演示完整的优化视频生成流程"""
    import os

    # 初始化节点
    node = OptimizedVideoGenerationNode(
        qwen_api_key=os.getenv("DASHSCOPE_API_KEY", "your_api_key_here")
    )

    # 准备上下文
    context = {
        "product_name": "智能手表",
        "product_image_url": "https://example.com/product.jpg",  # 可选
        "user_requirements": "强调科技感和运动场景",  # 可选
        "output_dir": "/tmp/demo_video_output",
        "merge_clips": True  # 是否合并视频片段
    }

    # 执行生成
    result = await node.execute(context)

    # 输出结果
    print(f"\n{'='*80}")
    print(f"✅ 生成完成！")
    print(f"{'='*80}\n")
    print(f"产品描述: {result['optimization']['product_description']}")
    print(f"视觉风格: {result['optimization']['visual_style']['target_style']}")
    print(f"核心主题: {result['optimization']['visual_style']['core_theme']}")
    print(f"核心情绪: {result['optimization']['visual_style']['core_emotion']}")
    print(f"\n分镜数量: {result['optimization']['storyboard_count']}")
    print(f"视频片段: {result['clips_count']}个")

    if result.get("final_video_url"):
        print(f"\n最终视频: {result['final_video_url']}")

    print(f"\n分镜详情:")
    for i, shot in enumerate(result['storyboard'][:3]):  # 只显示前3个
        print(f"\n镜头{i+1}:")
        print(f"  描述: {shot['description']}")
        print(f"  首帧: {shot['first_frame_clean'][:60]}...")
        print(f"  运动: {shot['middle_process_clean'][:60]}...")
        print(f"  策略: {shot['generation_strategy']} ({shot['reference_source']})")

    return result


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 运行演示
    asyncio.run(demo())
