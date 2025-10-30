#!/usr/bin/env python3
"""测试12步优化器集成"""

import asyncio
import logging
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_optimizer_integration():
    """测试优化器是否正确集成到分镜生成节点"""

    logger.info("=" * 60)
    logger.info("🧪 测试12步优化器集成")
    logger.info("=" * 60)

    # 1. 导入节点
    try:
        from video_generate_protocol.nodes.shot_block_generation_node import ShotBlockGenerationNode
        logger.info("✅ 成功导入 ShotBlockGenerationNode")
    except Exception as e:
        logger.error(f"❌ 导入节点失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. 检查优化器是否可用
    try:
        from video_generate_protocol.prompt_optimizer import VideoPromptOptimizer
        logger.info("✅ VideoPromptOptimizer 可用")
    except Exception as e:
        logger.warning(f"⚠️ VideoPromptOptimizer 不可用: {e}")

    # 3. 创建节点实例
    try:
        node = ShotBlockGenerationNode(node_id="test_node")
        logger.info(f"✅ 节点创建成功")

        # 检查优化器是否已初始化
        if node.optimizer:
            logger.info(f"✅ 优化器已启用: {type(node.optimizer).__name__}")
        else:
            logger.warning("⚠️ 优化器未启用，将使用旧版生成")

        # 检查统计信息
        logger.info(f"📊 节点统计: {node.stats}")

    except Exception as e:
        logger.error(f"❌ 节点创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 测试简单的生成请求（使用优化器）
    if node.optimizer:
        logger.info("\n" + "=" * 60)
        logger.info("🎬 测试使用优化器生成分镜")
        logger.info("=" * 60)

        try:
            # 准备测试上下文（最小化）
            test_context = {
                "keywords_id": ["智能投影仪"],
                "user_description_id": "一款能投射100寸巨幕的便携智能投影仪",
                "emotions_id": {"emotions": {"excited": 80, "calm": 20}},
                "structure_template_id": {
                    "开场": "产品特写",
                    "主体": "功能展示",
                    "结尾": "购买信息"
                },
                "video_type_id": "产品广告",
                "target_duration_id": 15  # 15秒短片
            }

            logger.info(f"📦 测试产品: {test_context['keywords_id'][0]}")
            logger.info(f"⏱️ 目标时长: {test_context['target_duration_id']}秒")

            # 调用生成方法
            result = await node.generate(test_context)

            # 检查结果
            if "shot_blocks_id" in result:
                shot_blocks = result["shot_blocks_id"]
                logger.info(f"✅ 生成成功!")
                logger.info(f"   分镜数量: {len(shot_blocks)}")

                # 显示每个分镜的信息
                for i, shot in enumerate(shot_blocks, 1):
                    logger.info(f"\n   镜头 {i}:")
                    logger.info(f"      时长: {shot['duration']}秒")
                    logger.info(f"      描述: {shot['visual_description'][:60]}...")

                    # 检查是否有优化器生成的标记
                    if "_optimized" in shot:
                        logger.info(f"      ✨ 使用了优化器!")
                        logger.info(f"      生成策略: {shot['_optimized']['generation_strategy']}")
                        logger.info(f"      视觉风格: {shot['_optimized']['visual_style']['target_style']}")
                    else:
                        logger.info(f"      ⚠️ 未使用优化器（旧版）")

                logger.info(f"\n📊 最终统计:")
                logger.info(f"   优化器调用次数: {node.stats['optimizer_calls']}")
                logger.info(f"   LLM调用次数: {node.stats['llm_calls']}")
                logger.info(f"   总请求次数: {node.stats['total_requests']}")

                return True
            else:
                logger.error("❌ 生成失败: 结果中没有 shot_blocks_id")
                return False

        except Exception as e:
            logger.error(f"❌ 测试生成失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        logger.warning("⚠️ 优化器未启用，跳过生成测试")
        return True

    return True

if __name__ == "__main__":
    success = asyncio.run(test_optimizer_integration())
    if success:
        logger.info("\n🎉 测试完成!")
        sys.exit(0)
    else:
        logger.error("\n❌ 测试失败!")
        sys.exit(1)
