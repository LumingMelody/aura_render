"""
工作流管理器 - 统一管理多个工作流实例
"""
from typing import Dict, List, Any, Optional, Callable
import asyncio
import uuid
from datetime import datetime
from dataclasses import dataclass, field

from .workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowConfig,
    WorkflowResult,
    WorkflowStatus
)
from nodes.base_node import ProcessingContext
from task_queue.task_queue_manager import TaskQueueManager


@dataclass
class WorkflowInstance:
    """工作流实例"""
    instance_id: str
    orchestrator: WorkflowOrchestrator
    context: ProcessingContext
    created_at: datetime
    status: WorkflowStatus = WorkflowStatus.IDLE
    result: Optional[WorkflowResult] = None


class WorkflowManager:
    """工作流管理器"""

    def __init__(self, queue_manager: Optional[TaskQueueManager] = None):
        self.queue_manager = queue_manager

        # 工作流实例管理
        self.active_workflows: Dict[str, WorkflowInstance] = {}
        self.workflow_templates: Dict[str, Callable] = {}
        self.completed_workflows: Dict[str, WorkflowInstance] = {}

        # 后台任务管理（防止被垃圾回收）
        self.background_tasks: Dict[str, asyncio.Task] = {}

        # 监控数据
        self.global_metrics = {
            'total_workflows': 0,
            'active_workflows': 0,
            'completed_workflows': 0,
            'failed_workflows': 0,
            'average_execution_time': 0.0
        }

    def register_workflow_template(self, template_name: str,
                                 builder_func: Callable[[WorkflowConfig], WorkflowOrchestrator]):
        """注册工作流模板"""
        self.workflow_templates[template_name] = builder_func
        print(f"✅ Workflow template registered: {template_name}")

    async def create_workflow(self, template_name: str, config: WorkflowConfig,
                            context: ProcessingContext, instance_id: Optional[str] = None) -> str:
        """创建新的工作流实例

        Args:
            template_name: 工作流模板名称
            config: 工作流配置
            context: 处理上下文
            instance_id: 可选的预定义instance_id，如果不提供则自动生成
        """
        try:
            if template_name not in self.workflow_templates:
                raise ValueError(f"Unknown workflow template: {template_name}")

            # 创建工作流编排器
            builder = self.workflow_templates[template_name]
            orchestrator = builder(config)

            # 使用提供的instance_id或生成新的
            if instance_id is None:
                instance_id = str(uuid.uuid4())

            instance = WorkflowInstance(
                instance_id=instance_id,
                orchestrator=orchestrator,
                context=context,
                created_at=datetime.now()
            )

            self.active_workflows[instance_id] = instance
            self.global_metrics['total_workflows'] += 1
            self.global_metrics['active_workflows'] += 1

            print(f"✅ Workflow instance created: {instance_id}")
            return instance_id

        except Exception as e:
            print(f"❌ Failed to create workflow: {e}")
            raise

    async def execute_workflow(self, instance_id: str) -> WorkflowResult:
        """执行工作流"""
        if instance_id not in self.active_workflows:
            raise ValueError(f"Workflow instance not found: {instance_id}")

        instance = self.active_workflows[instance_id]

        try:
            print(f"🚀 Starting workflow execution: {instance_id}")

            # 执行工作流
            result = await instance.orchestrator.execute(instance.context)

            # 更新实例状态
            instance.status = result.status
            instance.result = result

            # 移动到完成列表
            self.completed_workflows[instance_id] = instance
            del self.active_workflows[instance_id]

            # 更新统计
            self.global_metrics['active_workflows'] -= 1
            if result.status == WorkflowStatus.COMPLETED:
                self.global_metrics['completed_workflows'] += 1
            else:
                self.global_metrics['failed_workflows'] += 1

            # 更新平均执行时间
            self._update_average_execution_time(result.execution_time)

            print(f"✅ Workflow completed: {instance_id} (Status: {result.status.value})")
            return result

        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            print(f"❌ Workflow execution failed: {instance_id} - {e}")
            raise

    async def execute_workflow_async(self, instance_id: str) -> str:
        """异步执行工作流（返回任务ID）"""
        if self.queue_manager:
            # 使用任务队列异步执行
            task_data = {
                'instance_id': instance_id,
                'action': 'execute_workflow'
            }

            task_id = await self.queue_manager.enqueue_task(
                'workflow_execution',
                'execute_workflow_task',
                args=(instance_id,),
                metadata=task_data
            )
            return task_id
        else:
            # 直接使用asyncio（保存task引用防止被垃圾回收）
            task = asyncio.create_task(self.execute_workflow(instance_id))

            # 保存task引用，防止被垃圾回收
            self.background_tasks[instance_id] = task

            # 添加完成回调：清理已完成的task
            def cleanup_task(t):
                try:
                    # 从字典中移除已完成的task
                    if instance_id in self.background_tasks:
                        del self.background_tasks[instance_id]
                        print(f"🧹 Cleaned up background task for workflow: {instance_id}")
                except Exception as e:
                    print(f"⚠️ Error cleaning up task {instance_id}: {e}")

            task.add_done_callback(cleanup_task)

            # 返回 instance_id（而不是无用的内存地址）
            # 这样用户可以通过 /status/{instance_id} 查询状态
            return instance_id

    def get_workflow_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """获取工作流状态"""
        instance = self.active_workflows.get(instance_id) or self.completed_workflows.get(instance_id)

        if not instance:
            return None

        status_info = instance.orchestrator.get_workflow_status()
        status_info.update({
            'instance_id': instance_id,
            'template_config': {
                'name': instance.orchestrator.config.name,
                'description': instance.orchestrator.config.description
            },
            'created_at': instance.created_at.isoformat(),
            'is_active': instance_id in self.active_workflows
        })

        if instance.result:
            status_info['final_result'] = {
                'status': instance.result.status.value,
                'execution_time': instance.result.execution_time,
                'nodes_executed': instance.result.nodes_executed,
                'nodes_failed': instance.result.nodes_failed
            }

        return status_info

    def list_workflows(self, status_filter: Optional[WorkflowStatus] = None) -> List[Dict[str, Any]]:
        """列出工作流实例"""
        all_workflows = {}
        all_workflows.update(self.active_workflows)
        all_workflows.update(self.completed_workflows)

        workflows = []
        for instance_id, instance in all_workflows.items():
            if status_filter and instance.status != status_filter:
                continue

            workflow_info = {
                'instance_id': instance_id,
                'status': instance.status.value,
                'config_name': instance.orchestrator.config.name,
                'created_at': instance.created_at.isoformat(),
                'is_active': instance_id in self.active_workflows
            }

            if instance.result:
                workflow_info.update({
                    'execution_time': instance.result.execution_time,
                    'nodes_executed': instance.result.nodes_executed,
                    'nodes_failed': instance.result.nodes_failed
                })

            workflows.append(workflow_info)

        return workflows

    async def pause_workflow(self, instance_id: str) -> bool:
        """暂停工作流"""
        if instance_id not in self.active_workflows:
            return False

        instance = self.active_workflows[instance_id]
        await instance.orchestrator.pause()
        instance.status = WorkflowStatus.PAUSED
        print(f"⏸️ Workflow paused: {instance_id}")
        return True

    async def resume_workflow(self, instance_id: str) -> bool:
        """恢复工作流"""
        if instance_id not in self.active_workflows:
            return False

        instance = self.active_workflows[instance_id]
        await instance.orchestrator.resume()
        instance.status = WorkflowStatus.RUNNING
        print(f"▶️ Workflow resumed: {instance_id}")
        return True

    async def cancel_workflow(self, instance_id: str) -> bool:
        """取消工作流"""
        if instance_id not in self.active_workflows:
            return False

        instance = self.active_workflows[instance_id]
        await instance.orchestrator.cancel()
        instance.status = WorkflowStatus.CANCELLED

        # 移动到完成列表
        self.completed_workflows[instance_id] = instance
        del self.active_workflows[instance_id]

        self.global_metrics['active_workflows'] -= 1
        self.global_metrics['failed_workflows'] += 1

        print(f"🛑 Workflow cancelled: {instance_id}")
        return True

    def cleanup_completed_workflows(self, max_age_hours: int = 24) -> int:
        """清理过期的已完成工作流"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)

        to_remove = []
        for instance_id, instance in self.completed_workflows.items():
            if instance.created_at.timestamp() < cutoff_time:
                to_remove.append(instance_id)

        for instance_id in to_remove:
            del self.completed_workflows[instance_id]

        print(f"🧹 Cleaned up {len(to_remove)} completed workflows")
        return len(to_remove)

    def get_global_metrics(self) -> Dict[str, Any]:
        """获取全局指标"""
        metrics = self.global_metrics.copy()

        # 添加实时统计
        metrics.update({
            'active_workflow_details': {
                instance_id: {
                    'config_name': instance.orchestrator.config.name,
                    'status': instance.status.value,
                    'runtime_seconds': (datetime.now() - instance.created_at).total_seconds()
                }
                for instance_id, instance in self.active_workflows.items()
            },
            'background_tasks_count': len(self.background_tasks),
            'background_tasks': list(self.background_tasks.keys())
        })

        return metrics

    def _update_average_execution_time(self, execution_time: float):
        """更新平均执行时间"""
        total_completed = self.global_metrics['completed_workflows'] + self.global_metrics['failed_workflows']

        if total_completed == 1:
            self.global_metrics['average_execution_time'] = execution_time
        else:
            current_avg = self.global_metrics['average_execution_time']
            new_avg = ((current_avg * (total_completed - 1)) + execution_time) / total_completed
            self.global_metrics['average_execution_time'] = new_avg

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'workflows': {
                'active': len(self.active_workflows),
                'templates': len(self.workflow_templates)
            },
            'queue_manager': {
                'available': self.queue_manager is not None,
                'status': 'unknown'
            }
        }

        # 检查队列管理器状态
        if self.queue_manager:
            try:
                # 这里可以添加队列管理器的健康检查
                health_status['queue_manager']['status'] = 'healthy'
            except Exception as e:
                health_status['queue_manager']['status'] = f'error: {e}'
                health_status['status'] = 'degraded'

        # 检查活跃工作流
        stuck_workflows = []
        for instance_id, instance in self.active_workflows.items():
            runtime = (datetime.now() - instance.created_at).total_seconds()
            if runtime > 3600:  # 超过1小时
                stuck_workflows.append({
                    'instance_id': instance_id,
                    'runtime_hours': runtime / 3600
                })

        if stuck_workflows:
            health_status['warnings'] = {
                'stuck_workflows': stuck_workflows
            }
            health_status['status'] = 'warning'

        return health_status

    def get_workflow_templates(self) -> List[str]:
        """获取可用的工作流模板"""
        return list(self.workflow_templates.keys())

    def has_template(self, template_name: str) -> bool:
        """检查模板是否存在"""
        return template_name in self.workflow_templates

    async def restart_workflow(self, instance_id: str) -> str:
        """重启工作流"""
        # 获取原始实例
        instance = self.completed_workflows.get(instance_id) or self.active_workflows.get(instance_id)

        if not instance:
            raise ValueError(f"Workflow instance not found: {instance_id}")

        # 如果是活跃工作流，先取消
        if instance_id in self.active_workflows:
            await self.cancel_workflow(instance_id)

        # 重置编排器
        instance.orchestrator.reset()

        # 创建新实例
        new_instance_id = str(uuid.uuid4())
        new_instance = WorkflowInstance(
            instance_id=new_instance_id,
            orchestrator=instance.orchestrator,
            context=instance.context,
            created_at=datetime.now()
        )

        self.active_workflows[new_instance_id] = new_instance
        self.global_metrics['total_workflows'] += 1
        self.global_metrics['active_workflows'] += 1

        print(f"🔄 Workflow restarted: {instance_id} -> {new_instance_id}")
        return new_instance_id