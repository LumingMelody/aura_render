"""
视频生成完整工作流实现 - 集成所有组件的端到端流程
"""
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

from .workflow_orchestrator import WorkflowOrchestrator, WorkflowConfig
from .workflow_manager import WorkflowManager
from .workflow_templates import WorkflowTemplateManager
from nodes.base_node import ProcessingContext, ProcessingPriority

# 导入智能素材供给系统组件
from materials_supplies.style_anchor_manager import StyleAnchorManager
from materials_supplies.three_level_supply_strategy import ThreeLevelSupplyStrategy
from materials_supplies.api_clients.material_client_manager import MaterialClientManager

# 导入各种服务
from tts_services.tts_service_manager import TTSServiceManager
from image_generation.image_generation_manager import ImageGenerationManager
from database.database_manager import DatabaseManager
from cache.cache_manager import CacheManager
from task_queue.task_queue_manager import TaskQueueManager


class VideoGenerationWorkflow:
    """视频生成完整工作流"""

    def __init__(self):
        # 初始化核心组件
        self.workflow_manager = WorkflowManager()
        self.template_manager = WorkflowTemplateManager()

        # 初始化服务管理器
        self.style_anchor_manager = StyleAnchorManager()
        self.supply_strategy = ThreeLevelSupplyStrategy(self.style_anchor_manager)
        self.material_client_manager = MaterialClientManager()
        self.tts_service_manager = TTSServiceManager()
        self.image_generation_manager = ImageGenerationManager()
        self.database_manager = DatabaseManager()
        self.cache_manager = CacheManager()
        self.task_queue_manager = TaskQueueManager()

        # 注册工作流模板
        self._register_workflow_templates()

        # 系统状态
        self.initialized = False

    def _register_workflow_templates(self):
        """注册工作流模板到管理器"""
        # 基础视频生成
        self.workflow_manager.register_workflow_template(
            'basic_video_generation',
            self.template_manager.create_basic_video_workflow
        )

        # 高级视频生成
        self.workflow_manager.register_workflow_template(
            'advanced_video_generation',
            self.template_manager.create_advanced_video_workflow
        )

        # 音频优先生成
        self.workflow_manager.register_workflow_template(
            'audio_focused_generation',
            self.template_manager.create_audio_focused_workflow
        )

        # 快速原型
        self.workflow_manager.register_workflow_template(
            'quick_prototype',
            self.template_manager.create_quick_prototype_workflow
        )

        # ✨ 新增：注册新的VGP工作流（按照新流程图）
        try:
            from video_generate_protocol.vgp_workflow_template_new import VGPWorkflowTemplateNew
            self.workflow_manager.register_workflow_template(
                'vgp_new_pipeline',
                VGPWorkflowTemplateNew.create_new_pipeline
            )
            print("✅ 新VGP工作流模板 (vgp_new_pipeline) 已注册")
        except ImportError as e:
            print(f"⚠️ 无法导入新VGP工作流模板: {e}")

        # 可选：同时注册旧的VGP模板作为备用
        try:
            from video_generate_protocol.vgp_workflow_template import VGPWorkflowTemplate
            self.workflow_manager.register_workflow_template(
                'vgp_full_pipeline',
                VGPWorkflowTemplate.create_full_pipeline
            )
            print("✅ 旧VGP工作流模板 (vgp_full_pipeline) 已注册（备用）")
        except ImportError as e:
            print(f"⚠️ 无法导入旧VGP工作流模板: {e}")

    async def initialize(self) -> bool:
        """初始化系统"""
        print("🚀 Initializing Video Generation Workflow System...")

        initialization_success = True

        # 初始化数据库（可选）
        try:
            if hasattr(self.database_manager, 'initialize'):
                db_config = {
                    'sqlite': {
                        'database': 'video_generation.db'
                    }
                }
                await self.database_manager.initialize(db_config)
                print("  ✅ Database manager initialized")
            else:
                print("  ⚠️  Database manager doesn't require initialization")
        except Exception as e:
            print(f"  ⚠️  Database initialization skipped: {e}")

        # 初始化缓存（可选）
        try:
            if hasattr(self.cache_manager, 'initialize'):
                cache_configs = {
                    'memory': {
                        'type': 'memory',
                        'max_size': 1000
                    }
                }
                await self.cache_manager.initialize(cache_configs)
                print("  ✅ Cache manager initialized")
            else:
                print("  ⚠️  Cache manager doesn't require initialization")
        except Exception as e:
            print(f"  ⚠️  Cache initialization skipped: {e}")

        # 初始化任务队列（可选）
        try:
            if hasattr(self.task_queue_manager, 'initialize'):
                queue_configs = {
                    'video_generation': {
                        'type': 'memory',
                        'max_workers': 5
                    }
                }
                await self.task_queue_manager.initialize(queue_configs)
                print("  ✅ Task queue manager initialized")
            else:
                print("  ⚠️  Task queue manager doesn't require initialization")
        except Exception as e:
            print(f"  ⚠️  Task queue initialization skipped: {e}")

        # 初始化素材客户端（可选）
        try:
            if hasattr(self.material_client_manager, 'initialize'):
                await self.material_client_manager.initialize()
                print("  ✅ Material client manager initialized")
            else:
                print("  ⚠️  Material client manager doesn't require initialization")
        except Exception as e:
            print(f"  ⚠️  Material client initialization skipped: {e}")

        # 初始化TTS服务（可选）
        try:
            if hasattr(self.tts_service_manager, 'initialize'):
                await self.tts_service_manager.initialize()
                print("  ✅ TTS service manager initialized")
            else:
                print("  ⚠️  TTS service manager doesn't require initialization")
        except Exception as e:
            print(f"  ⚠️  TTS service initialization skipped: {e}")

        # 初始化图像生成服务（可选）
        try:
            if hasattr(self.image_generation_manager, 'initialize'):
                await self.image_generation_manager.initialize()
                print("  ✅ Image generation manager initialized")
            else:
                print("  ⚠️  Image generation manager doesn't require initialization")
        except Exception as e:
            print(f"  ⚠️  Image generation initialization skipped: {e}")

        self.initialized = True
        print("✅ Video Generation Workflow System initialized successfully")
        print(f"   Core workflow system is ready (optional services may be skipped)")
        return True

    async def create_video_generation_task(self, request: Dict[str, Any]) -> str:
        """创建视频生成任务"""
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")

        try:
            # 解析请求参数
            template_id = request.get('template', 'basic_video_generation')
            user_input = request.get('input', {})
            project_params = request.get('params', {})

            # 验证模板是否存在（使用 workflow_manager 而不是 template_manager）
            if not self.workflow_manager.has_template(template_id):
                raise ValueError(f"Unknown template: {template_id}")

            # 使用请求中的参数（不需要通过 template_manager 验证）
            validated_params = project_params

            # 创建工作流配置
            workflow_config = WorkflowConfig(
                name=f"video_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description=f"Video generation using template: {template_id}",
                max_parallel_nodes=validated_params.get('max_parallel_nodes', 3),
                total_timeout=validated_params.get('total_timeout', 3600.0),
                auto_retry=validated_params.get('auto_retry', True),
                save_intermediate_results=validated_params.get('save_intermediate_results', True),
                enable_monitoring=validated_params.get('enable_monitoring', True)
            )

            # 创建处理上下文
            context = ProcessingContext(
                task_id=f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                session_id=request.get('session_id', 'default'),
                user_id=request.get('user_id'),
                project_data={
                    'user_input': user_input,
                    'template_id': template_id,
                    'params': validated_params
                },
                intermediate_results={
                    'initial_data': user_input
                },
                metadata={
                    'request_timestamp': datetime.now().isoformat(),
                    'template_used': template_id
                }
            )

            # 创建工作流实例
            instance_id = await self.workflow_manager.create_workflow(
                template_id, workflow_config, context
            )

            print(f"✅ Video generation task created: {instance_id}")
            return instance_id

        except Exception as e:
            print(f"❌ Failed to create video generation task: {e}")
            raise

    async def execute_video_generation(self, instance_id: str, async_mode: bool = True) -> Any:
        """执行视频生成"""
        try:
            if async_mode:
                # 异步执行，返回任务ID（现在task_id就是instance_id）
                task_id = await self.workflow_manager.execute_workflow_async(instance_id)
                print(f"🚀 Video generation started asynchronously: {task_id}")
                # task_id 和 instance_id 现在是相同的（修复后的行为）
                return {'task_id': task_id, 'instance_id': instance_id}
            else:
                # 同步执行，返回结果
                result = await self.workflow_manager.execute_workflow(instance_id)
                print(f"✅ Video generation completed: {instance_id}")
                return result

        except Exception as e:
            print(f"❌ Video generation execution failed: {e}")
            raise

    async def get_generation_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """获取视频生成状态"""
        return self.workflow_manager.get_workflow_status(instance_id)

    async def cancel_generation(self, instance_id: str) -> bool:
        """取消视频生成"""
        return await self.workflow_manager.cancel_workflow(instance_id)

    async def pause_generation(self, instance_id: str) -> bool:
        """暂停视频生成"""
        return await self.workflow_manager.pause_workflow(instance_id)

    async def resume_generation(self, instance_id: str) -> bool:
        """恢复视频生成"""
        return await self.workflow_manager.resume_workflow(instance_id)

    def list_active_generations(self) -> List[Dict[str, Any]]:
        """列出活跃的视频生成任务"""
        return self.workflow_manager.list_workflows()

    def get_available_templates(self) -> List[Dict[str, Any]]:
        """获取可用的工作流模板"""
        return self.template_manager.get_template_list()

    def search_templates(self, query: str) -> List[Dict[str, Any]]:
        """搜索工作流模板"""
        return self.template_manager.search_templates(query)

    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'initialized': self.initialized,
            'timestamp': datetime.now().isoformat(),
            'workflow_manager': self.workflow_manager.get_global_metrics(),
            'services': {}
        }

        if self.initialized:
            # 检查各个服务状态
            try:
                status['services']['database'] = await self.database_manager.health_check()
            except Exception as e:
                status['services']['database'] = {'status': 'error', 'message': str(e)}

            try:
                status['services']['cache'] = await self.cache_manager.health_check()
            except Exception as e:
                status['services']['cache'] = {'status': 'error', 'message': str(e)}

            try:
                status['services']['task_queue'] = await self.task_queue_manager.health_check()
            except Exception as e:
                status['services']['task_queue'] = {'status': 'error', 'message': str(e)}

            try:
                status['services']['material_clients'] = await self.material_client_manager.get_health_status()
            except Exception as e:
                status['services']['material_clients'] = {'status': 'error', 'message': str(e)}

        return status

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = {
            'workflow_metrics': self.workflow_manager.get_global_metrics(),
            'service_metrics': {},
            'system_metrics': {
                'timestamp': datetime.now().isoformat()
            }
        }

        if self.initialized:
            # 收集各服务的性能指标
            try:
                metrics['service_metrics']['cache'] = await self.cache_manager.get_statistics()
            except Exception as e:
                metrics['service_metrics']['cache'] = {'error': str(e)}

            try:
                metrics['service_metrics']['task_queue'] = await self.task_queue_manager.get_metrics()
            except Exception as e:
                metrics['service_metrics']['task_queue'] = {'error': str(e)}

        return metrics

    async def cleanup_completed_tasks(self, max_age_hours: int = 24) -> Dict[str, int]:
        """清理已完成的任务"""
        cleanup_results = {}

        # 清理工作流
        workflow_cleaned = self.workflow_manager.cleanup_completed_workflows(max_age_hours)
        cleanup_results['workflows'] = workflow_cleaned

        # 清理任务队列
        if self.initialized:
            try:
                queue_cleaned = await self.task_queue_manager.cleanup_completed_tasks(max_age_hours)
                cleanup_results['task_queue'] = queue_cleaned
            except Exception as e:
                cleanup_results['task_queue_error'] = str(e)

            # 清理缓存
            try:
                cache_cleaned = await self.cache_manager.cleanup_expired()
                cleanup_results['cache'] = cache_cleaned
            except Exception as e:
                cleanup_results['cache_error'] = str(e)

        print(f"🧹 Cleanup completed: {cleanup_results}")
        return cleanup_results

    async def shutdown(self):
        """关闭系统"""
        try:
            print("🛑 Shutting down Video Generation Workflow System...")

            # 停止所有活跃工作流
            active_workflows = self.workflow_manager.list_workflows()
            for workflow in active_workflows:
                if workflow['is_active']:
                    await self.workflow_manager.cancel_workflow(workflow['instance_id'])

            # 关闭各个服务（安全调用，忽略不存在的方法）
            if self.initialized:
                if hasattr(self.task_queue_manager, 'shutdown'):
                    await self.task_queue_manager.shutdown()
                if hasattr(self.cache_manager, 'shutdown'):
                    await self.cache_manager.shutdown()
                if hasattr(self.database_manager, 'shutdown'):
                    await self.database_manager.shutdown()

            self.initialized = False
            print("✅ System shutdown completed")

        except Exception as e:
            print(f"❌ Error during shutdown: {e}")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.shutdown()


# 便捷函数
async def create_video_from_text(text: str, template: str = 'basic_video_generation',
                               **kwargs) -> str:
    """便捷函数：从文本创建视频"""
    workflow = VideoGenerationWorkflow()
    await workflow.initialize()

    try:
        request = {
            'template': template,
            'input': {'text': text},
            'params': kwargs
        }

        instance_id = await workflow.create_video_generation_task(request)
        result = await workflow.execute_video_generation(instance_id, async_mode=False)

        return result
    finally:
        await workflow.shutdown()


async def get_video_generation_status(instance_id: str) -> Optional[Dict[str, Any]]:
    """便捷函数：获取视频生成状态"""
    workflow = VideoGenerationWorkflow()
    return await workflow.get_generation_status(instance_id)