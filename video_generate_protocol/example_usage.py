"""
新VGP工作流使用示例 - 产品展示视频生成
"""
import asyncio
import json
from workflow.video_generation_workflow import VideoGenerationWorkflow


async def generate_product_video():
    """生成产品展示视频的完整示例"""

    # 1. 初始化工作流系统
    print("🚀 初始化视频生成系统...")
    workflow_system = VideoGenerationWorkflow()
    await workflow_system.initialize()

    try:
        # 2. 构建请求数据
        request = {
            "template": "vgp_new_pipeline",  # 使用新的工作流模板
            "input": {
                # VGP协议标准输入字段
                "theme_id": "产品展示",
                "keywords_id": [
                    "智能投影仪",
                    "4K高清",
                    "便携",
                    "语音控制"
                ],
                "target_duration_id": 20,
                "user_description_id": (
                    "展示智能投影仪的完整功能演示。"
                    "首先展示投影仪的外观设计，黑色磨砂质感的机身。"
                    "然后展示开机投影，自动对焦。"
                    "接着演示在白墙上投射4K画面。"
                    "展示多种使用场景：客厅观影、办公演示。"
                    "最后展示智能功能：语音控制、无线投屏。"
                ),
                "reference_media": {
                    "product_images": [
                        {
                            "url": "https://ai-movie-cloud-v2.oss-cn-shanghai.aliyuncs.com/%E6%B5%8B%E8%AF%95%E5%95%86%E5%93%81.jpg",
                            "type": "product",
                            "weight": 1.0
                        }
                    ]
                }
            },
            "params": {
                "max_parallel_nodes": 5,      # 允许5个节点并行执行
                "total_timeout": 3600.0,       # 1小时超时
                "auto_retry": True,            # 失败自动重试
                "enable_monitoring": True,     # 启用监控
                "save_intermediate_results": True  # 保存中间结果
            },
            "session_id": "session_20251009_001",
            "user_id": "product_demo_user"
        }

        print(f"📋 请求数据:")
        print(json.dumps(request, indent=2, ensure_ascii=False))

        # 3. 创建视频生成任务
        print("\n📝 创建视频生成任务...")
        instance_id = await workflow_system.create_video_generation_task(request)
        print(f"✅ 任务创建成功！")
        print(f"   任务ID: {instance_id}")

        # 4. 执行任务（异步模式）
        print(f"\n🎬 开始生成视频...")
        execution_result = await workflow_system.execute_video_generation(
            instance_id,
            async_mode=True  # 异步执行，不阻塞
        )

        print(f"✅ 任务已提交:")
        print(f"   执行ID: {execution_result['task_id']}")
        print(f"   实例ID: {execution_result['instance_id']}")

        # 5. 监控任务进度
        print(f"\n📊 监控任务进度...")
        last_status = None

        while True:
            status = await workflow_system.get_generation_status(instance_id)

            # 只在状态改变时打印
            if status.get('status') != last_status:
                last_status = status.get('status')
                print(f"   状态: {last_status}")

                if status.get('current_node'):
                    print(f"   当前节点: {status['current_node']}")

                if status.get('progress'):
                    print(f"   进度: {status['progress']:.1f}%")

            # 检查是否完成
            if status['status'] in ['completed', 'failed', 'cancelled']:
                break

            await asyncio.sleep(2)  # 每2秒查询一次

        # 6. 处理结果
        print(f"\n" + "="*60)
        if status['status'] == 'completed':
            print(f"✅ 视频生成成功！")
            print(f"\n📹 视频信息:")

            result_data = status.get('result', {})

            # 打印各个轨道信息
            if 'preliminary_sequence_id' in result_data:
                clips = result_data['preliminary_sequence_id']
                print(f"   视频片段数: {len(clips)}")
                print(f"   总时长: {result_data.get('total_main_duration_id', 0):.1f}秒")

            if 'bgm_track' in result_data:
                bgm = result_data['bgm_track']
                print(f"   BGM片段: {len(bgm.get('clips', []))}")

            if 'sfx_track' in result_data:
                sfx = result_data['sfx_track']
                print(f"   音效片段: {len(sfx.get('clips', []))}")

            if 'subtitle_sequence_id' in result_data:
                subtitle = result_data['subtitle_sequence_id']
                print(f"   字幕片段: {len(subtitle.get('clips', []))}")

            if 'tts_track_id' in result_data and result_data['tts_track_id']:
                tts = result_data['tts_track_id']
                print(f"   TTS片段: {len(tts.get('clips', []))}")

            # 输出文件路径
            if 'output_path' in status:
                print(f"\n💾 输出路径: {status['output_path']}")

            # 性能统计
            if 'execution_time' in status:
                print(f"\n⏱️ 执行时间: {status['execution_time']:.2f}秒")

            if 'node_execution_times' in status:
                print(f"\n📊 节点执行时间:")
                for node_id, time in status['node_execution_times'].items():
                    print(f"   {node_id}: {time:.2f}秒")

        elif status['status'] == 'failed':
            print(f"❌ 视频生成失败！")
            print(f"\n错误信息:")
            print(f"   {status.get('error_message', '未知错误')}")

            if 'error_log' in status:
                print(f"\n详细日志:")
                for log in status['error_log']:
                    print(f"   {log}")

        else:
            print(f"⚠️ 任务被取消")

        print("="*60)

        return status

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # 7. 清理资源
        print(f"\n🧹 清理资源...")
        await workflow_system.shutdown()
        print(f"✅ 系统已关闭")


async def check_available_templates():
    """查看可用的工作流模板"""
    workflow_system = VideoGenerationWorkflow()

    templates = workflow_system.get_available_templates()

    print("\n📋 可用的工作流模板:")
    for template in templates:
        print(f"  - {template['template_id']}: {template.get('description', '无描述')}")
        if template['template_id'] == 'vgp_new_pipeline':
            print(f"    ✨ (新版本，优化的节点流程)")


async def main():
    """主函数"""
    print("="*60)
    print("🎬 新VGP工作流 - 产品展示视频生成示例")
    print("="*60)

    # 可选：先查看可用模板
    # await check_available_templates()

    # 生成视频
    result = await generate_product_video()

    if result and result['status'] == 'completed':
        print("\n🎉 完成！您的产品展示视频已生成。")
    else:
        print("\n😞 视频生成未能完成，请查看上方错误信息。")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())
