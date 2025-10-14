"""
工作流编排器 - 统一管理视频生成的端到端流程
"""
from typing import Dict, List, Any, Optional, Set, Callable
import asyncio
import json
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict, deque

from nodes.base_node import BaseNode, NodeResult, ProcessingContext, NodeStatus
from task_queue.task_queue_manager import TaskQueueManager, QueueConfig


class WorkflowStatus(Enum):
    """工作流状态"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class NodeDependency:
    """节点依赖关系"""
    node_id: str
    dependencies: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    retry_on_failure: bool = True


@dataclass
class WorkflowConfig:
    """工作流配置"""
    name: str
    description: str
    max_parallel_nodes: int = 5
    total_timeout: float = 3600.0  # 1小时
    auto_retry: bool = True
    save_intermediate_results: bool = True
    enable_monitoring: bool = True
    error_handling_strategy: str = "fail_fast"  # fail_fast, continue, retry


@dataclass
class WorkflowResult:
    """工作流执行结果"""
    workflow_id: str
    status: WorkflowStatus
    execution_time: float = 0.0
    nodes_executed: int = 0
    nodes_failed: int = 0
    final_output: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class WorkflowOrchestrator:
    """工作流编排器"""

    def __init__(self, config: WorkflowConfig, queue_manager: Optional[TaskQueueManager] = None):
        self.config = config
        self.queue_manager = queue_manager

        # 工作流状态
        self.status = WorkflowStatus.IDLE
        self.workflow_id: Optional[str] = None
        self.start_time: Optional[datetime] = None

        # 节点管理
        self.nodes: Dict[str, BaseNode] = {}
        self.dependency_graph = nx.DiGraph()
        self.node_dependencies: Dict[str, NodeDependency] = {}

        # 执行状态跟踪
        self.executed_nodes: Set[str] = set()
        self.failed_nodes: Set[str] = set()
        self.running_nodes: Set[str] = set()
        self.node_results: Dict[str, NodeResult] = {}

        # 事件处理
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        # 监控数据
        self.metrics = {
            'nodes_executed': 0,
            'nodes_failed': 0,
            'total_execution_time': 0.0,
            'average_node_time': 0.0,
            'bottleneck_nodes': [],
            'resource_usage': {}
        }

    def add_node(self, node: BaseNode, dependencies: List[str] = None,
                 conditions: Dict[str, Any] = None, **kwargs) -> bool:
        """添加节点到工作流"""
        try:
            # 确保使用字符串node_id作为键
            if hasattr(node, 'config') and hasattr(node.config, 'node_id'):
                # 优先从 config 中获取
                node_id = node.config.node_id
            elif hasattr(node, 'node_id'):
                # 备选：直接从 node 获取
                node_id = node.node_id
            else:
                raise ValueError(f"Node must have node_id attribute: {node}")

            # 确保 node_id 是字符串，不是对象
            if not isinstance(node_id, str):
                if hasattr(node_id, 'node_id'):
                    # 如果 node_id 是 NodeConfig 对象，提取其 node_id 字符串
                    node_id = node_id.node_id
                else:
                    node_id = str(node_id)

            # 检查是否已存在
            if node_id in self.nodes:
                print(f"⚠️  Node {node_id} already exists in self.nodes, skipping...")
                return True

            # 检查图中是否已存在
            if node_id in self.dependency_graph:
                print(f"⚠️  Node {node_id} already exists in dependency_graph, removing...")
                self.dependency_graph.remove_node(node_id)

            self.nodes[node_id] = node
            print(f"➕ Adding node to workflow: {node_id}")

            # 添加依赖关系
            dependency = NodeDependency(
                node_id=node_id,
                dependencies=dependencies or [],
                conditions=conditions or {},
                **kwargs
            )
            self.node_dependencies[node_id] = dependency

            # 构建依赖图
            self.dependency_graph.add_node(node_id)
            for dep in dependency.dependencies:
                self.dependency_graph.add_edge(dep, node_id)

            # 检查循环依赖
            if not nx.is_directed_acyclic_graph(self.dependency_graph):
                self.dependency_graph.remove_node(node_id)
                del self.nodes[node_id]
                del self.node_dependencies[node_id]
                raise ValueError(f"Adding node {node_id} would create a circular dependency")

            # 调试：显示当前状态
            graph_node_count = len(self.dependency_graph.nodes())
            nodes_dict_count = len(self.nodes)
            print(f"✅ Node added: {node_id} | self.nodes={nodes_dict_count}, graph={graph_node_count}")

            return True

        except Exception as e:
            print(f"❌ Failed to add node: {e}")
            return False

    def remove_node(self, node_id: str) -> bool:
        """从工作流中移除节点"""
        try:
            if node_id in self.nodes:
                del self.nodes[node_id]
                del self.node_dependencies[node_id]
                self.dependency_graph.remove_node(node_id)
                print(f"✅ Node removed from workflow: {node_id}")
                return True
            return False
        except Exception as e:
            print(f"❌ Failed to remove node {node_id}: {e}")
            return False

    def get_execution_order(self) -> List[str]:
        """获取节点执行顺序（拓扑排序）"""
        try:
            return list(nx.topological_sort(self.dependency_graph))
        except nx.NetworkXError as e:
            raise ValueError(f"Cannot determine execution order: {e}")

    def get_ready_nodes(self) -> List[str]:
        """获取准备执行的节点（所有依赖已完成）"""
        ready_nodes = []

        for node_id in self.nodes:
            if node_id in self.executed_nodes or node_id in self.running_nodes:
                continue

            # 检查依赖是否都已完成
            dependency = self.node_dependencies[node_id]
            dependencies_met = True

            for dep_id in dependency.dependencies:
                if dep_id not in self.executed_nodes:
                    dependencies_met = False
                    break

                # 检查条件
                if dependency.conditions:
                    dep_result = self.node_results.get(dep_id)
                    if not self._check_conditions(dep_result, dependency.conditions):
                        dependencies_met = False
                        break

            if dependencies_met:
                ready_nodes.append(node_id)

        return ready_nodes

    def _check_conditions(self, node_result: Optional[NodeResult],
                         conditions: Dict[str, Any]) -> bool:
        """检查节点执行条件"""
        if not node_result or not conditions:
            return True

        for condition, expected_value in conditions.items():
            if condition == "status":
                if node_result.status.value != expected_value:
                    return False
            elif condition == "data_key":
                if expected_value not in node_result.data:
                    return False
            elif condition == "custom":
                # 自定义条件检查逻辑
                if not self._evaluate_custom_condition(node_result, expected_value):
                    return False

        return True

    def _evaluate_custom_condition(self, node_result: NodeResult,
                                  condition: Dict[str, Any]) -> bool:
        """评估自定义条件"""
        # 这里可以实现复杂的条件评估逻辑
        # 例如：检查数据质量、文件大小等
        return True

    async def execute(self, context: ProcessingContext) -> WorkflowResult:
        """执行工作流"""
        self.workflow_id = context.task_id
        self.start_time = datetime.now()
        self.status = WorkflowStatus.RUNNING

        try:
            print(f"🚀 Starting workflow execution: {self.config.name}")

            # 触发开始事件
            await self._trigger_event('workflow_started', context)

            # 验证工作流
            if not self._validate_workflow():
                raise ValueError("Workflow validation failed")

            # 执行节点
            result = await self._execute_nodes(context)

            # 计算最终结果
            final_result = self._compile_final_result(result, context)

            # 触发完成事件
            await self._trigger_event('workflow_completed', final_result)

            return final_result

        except Exception as e:
            print(f"❌ Workflow execution failed: {e}")
            self.status = WorkflowStatus.FAILED

            error_result = WorkflowResult(
                workflow_id=self.workflow_id,
                status=WorkflowStatus.FAILED,
                execution_time=(datetime.now() - self.start_time).total_seconds(),
                nodes_executed=len(self.executed_nodes),
                nodes_failed=len(self.failed_nodes),
                error_log=[str(e)]
            )

            await self._trigger_event('workflow_failed', error_result)
            return error_result

    def _validate_workflow(self) -> bool:
        """验证工作流配置"""
        if not self.nodes:
            print("❌ No nodes in workflow")
            return False

        # 检查是否有孤立节点
        try:
            execution_order = self.get_execution_order()
            if len(execution_order) != len(self.nodes):
                print(f"❌ Workflow contains unreachable nodes")
                print(f"   Total nodes: {len(self.nodes)}")
                print(f"   Reachable nodes: {len(execution_order)}")
                # 确保打印的是键（字符串），不是值（对象）
                all_node_keys = [str(k) for k in self.nodes.keys()]
                print(f"   All node keys: {all_node_keys}")
                execution_order_str = [str(n) for n in execution_order]
                print(f"   Reachable keys: {execution_order_str}")

                # 调试：检查图中的节点
                graph_nodes = list(self.dependency_graph.nodes())
                print(f"   Graph nodes ({len(graph_nodes)}): {[str(n) for n in graph_nodes]}")

                unreachable = set(self.nodes.keys()) - set(execution_order)
                unreachable_str = [str(n) for n in unreachable]
                print(f"   Unreachable nodes: {unreachable_str}")

                # 打印每个不可达节点的依赖关系
                for node_id in unreachable:
                    deps = self.node_dependencies.get(str(node_id), NodeDependency(node_id=str(node_id))).dependencies
                    print(f"     {node_id} depends on: {deps}")
                return False
        except ValueError as e:
            print(f"❌ Workflow validation failed: {e}")
            return False

        # 检查节点输入输出匹配
        # 注意：VGP 节点通过 context.intermediate_results 动态传递数据
        # 静态 schema 验证可能不准确，暂时禁用严格验证
        # 如果真的缺少输入，节点执行时会报错

        # 可选：启用宽松验证（仅警告，不阻止执行）
        enable_strict_validation = False

        if enable_strict_validation:
            global_context_inputs = [
                'initial_data', 'user_input',
                'user_description_id', 'video_type_id', 'theme_id',
                'keywords_id', 'target_duration_id', 'reference_media',
                'project_data', 'session_id', 'task_id', 'user_id'
            ]

            for node_id, node in self.nodes.items():
                required_inputs = node.get_required_inputs()
                dependency = self.node_dependencies[node_id]

                # 如果节点没有依赖（起始节点），跳过输入验证
                if not dependency.dependencies:
                    continue

                # 确保依赖节点能提供所需输入
                for required_input in required_inputs:
                    # 如果是全局上下文输入，跳过验证
                    if required_input in global_context_inputs:
                        continue

                    input_available = False
                    for dep_id in dependency.dependencies:
                        dep_node = self.nodes[dep_id]
                        output_schema = dep_node.get_output_schema()
                        if required_input in output_schema:
                            input_available = True
                            break

                    if not input_available:
                        print(f"⚠️  Warning: Node {node_id} requires input '{required_input}' not in dependency output schemas")
                        print(f"   (This may be provided via context.intermediate_results at runtime)")
                        # 不返回 False，只是警告

        print("✅ Workflow validation passed")
        return True

    async def _execute_nodes(self, context: ProcessingContext) -> WorkflowResult:
        """执行所有节点"""
        semaphore = asyncio.Semaphore(self.config.max_parallel_nodes)

        while len(self.executed_nodes) + len(self.failed_nodes) < len(self.nodes):
            # 获取准备执行的节点
            ready_nodes = self.get_ready_nodes()

            if not ready_nodes:
                if self.running_nodes:
                    # 等待运行中的节点完成
                    await asyncio.sleep(0.1)
                    continue
                else:
                    # 没有可执行的节点且没有运行中的节点，可能是死锁
                    remaining_nodes = set(self.nodes.keys()) - self.executed_nodes - self.failed_nodes
                    raise RuntimeError(f"Workflow deadlock detected. Remaining nodes: {remaining_nodes}")

            # 并发执行准备就绪的节点
            tasks = []
            for node_id in ready_nodes:
                if len(self.running_nodes) < self.config.max_parallel_nodes:
                    task = asyncio.create_task(
                        self._execute_single_node(node_id, context, semaphore)
                    )
                    tasks.append(task)
                    self.running_nodes.add(node_id)

            # 等待至少一个节点完成
            if tasks:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                # 处理完成的任务
                for task in done:
                    try:
                        await task
                    except Exception as e:
                        print(f"Node execution error: {e}")

        # 编译执行结果
        execution_time = (datetime.now() - self.start_time).total_seconds()

        return WorkflowResult(
            workflow_id=self.workflow_id,
            status=WorkflowStatus.COMPLETED if not self.failed_nodes else WorkflowStatus.FAILED,
            execution_time=execution_time,
            nodes_executed=len(self.executed_nodes),
            nodes_failed=len(self.failed_nodes),
            intermediate_results={node_id: result.data for node_id, result in self.node_results.items()}
        )

    async def _execute_single_node(self, node_id: str, context: ProcessingContext,
                                  semaphore: asyncio.Semaphore):
        """执行单个节点"""
        async with semaphore:
            try:
                node = self.nodes[node_id]
                print(f"🔄 Executing node: {node_id}")

                # 准备节点上下文
                node_context = self._prepare_node_context(node_id, context)

                # 触发节点开始事件
                await self._trigger_event('node_started', {'node_id': node_id, 'context': node_context})

                # 执行节点
                start_time = datetime.now()
                result = await node.execute(node_context)
                execution_time = (datetime.now() - start_time).total_seconds()

                # 更新统计
                self.metrics['nodes_executed'] += 1
                self.metrics['total_execution_time'] += execution_time

                # 保存结果
                self.node_results[node_id] = result

                if result.is_success():
                    self.executed_nodes.add(node_id)
                    print(f"✅ Node completed: {node_id} ({execution_time:.2f}s)")

                    # 更新上下文
                    context.intermediate_results.update(result.data)

                    await self._trigger_event('node_completed', {
                        'node_id': node_id,
                        'result': result,
                        'execution_time': execution_time
                    })
                else:
                    self.failed_nodes.add(node_id)
                    print(f"❌ Node failed: {node_id} - {result.error_message}")

                    await self._trigger_event('node_failed', {
                        'node_id': node_id,
                        'error': result.error_message
                    })

                    # 根据错误处理策略决定是否继续
                    if self.config.error_handling_strategy == "fail_fast":
                        raise RuntimeError(f"Node {node_id} failed: {result.error_message}")

            except Exception as e:
                self.failed_nodes.add(node_id)
                print(f"❌ Node execution exception: {node_id} - {e}")

                await self._trigger_event('node_error', {
                    'node_id': node_id,
                    'exception': str(e)
                })

                if self.config.error_handling_strategy == "fail_fast":
                    raise

            finally:
                self.running_nodes.discard(node_id)

    def _prepare_node_context(self, node_id: str, context: ProcessingContext) -> ProcessingContext:
        """为节点准备执行上下文"""
        # 创建节点专用的上下文副本
        node_context = ProcessingContext(
            task_id=context.task_id,
            session_id=context.session_id,
            user_id=context.user_id,
            project_data=context.project_data.copy(),
            intermediate_results=context.intermediate_results.copy(),
            metadata=context.metadata.copy()
        )

        # 添加节点特定的元数据
        node_context.metadata['current_node'] = node_id
        node_context.metadata['execution_order'] = len(self.executed_nodes) + 1

        return node_context

    def _compile_final_result(self, execution_result: WorkflowResult,
                            context: ProcessingContext) -> WorkflowResult:
        """编译最终工作流结果"""
        # 提取最终输出
        final_output = {}

        # 查找输出节点（没有后续依赖的节点）
        output_nodes = [
            node_id for node_id in self.nodes
            if not list(self.dependency_graph.successors(node_id))
        ]

        for node_id in output_nodes:
            if node_id in self.node_results:
                result = self.node_results[node_id]
                if result.is_success():
                    final_output[node_id] = result.data

        # 计算性能指标
        self._calculate_performance_metrics()

        execution_result.final_output = final_output
        execution_result.performance_metrics = self.metrics

        return execution_result

    def _calculate_performance_metrics(self):
        """计算性能指标"""
        if self.metrics['nodes_executed'] > 0:
            self.metrics['average_node_time'] = (
                self.metrics['total_execution_time'] / self.metrics['nodes_executed']
            )

        # 识别瓶颈节点
        node_times = []
        for node_id, result in self.node_results.items():
            if result.is_success():
                node_times.append((node_id, result.execution_time))

        # 按执行时间排序，找出最慢的节点
        node_times.sort(key=lambda x: x[1], reverse=True)
        self.metrics['bottleneck_nodes'] = node_times[:3]  # 前3个最慢的节点

    async def _trigger_event(self, event_type: str, data: Any):
        """触发事件处理器"""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                print(f"Event handler error ({event_type}): {e}")

    def add_event_handler(self, event_type: str, handler: Callable):
        """添加事件处理器"""
        self.event_handlers[event_type].append(handler)

    def get_workflow_status(self) -> Dict[str, Any]:
        """获取工作流状态"""
        return {
            'workflow_id': self.workflow_id,
            'status': self.status.value,
            'config': {
                'name': self.config.name,
                'description': self.config.description
            },
            'execution': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'nodes_total': len(self.nodes),
                'nodes_executed': len(self.executed_nodes),
                'nodes_running': len(self.running_nodes),
                'nodes_failed': len(self.failed_nodes)
            },
            'metrics': self.metrics
        }

    def get_node_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取节点状态"""
        if node_id not in self.nodes:
            return None

        node = self.nodes[node_id]
        result = self.node_results.get(node_id)

        status_info = {
            'node_id': node_id,
            'node_name': node.node_name,
            'dependencies': self.node_dependencies[node_id].dependencies,
            'executed': node_id in self.executed_nodes,
            'running': node_id in self.running_nodes,
            'failed': node_id in self.failed_nodes
        }

        if result:
            status_info.update({
                'status': result.status.value,
                'execution_time': result.execution_time,
                'error_message': result.error_message
            })

        return status_info

    async def pause(self):
        """暂停工作流执行"""
        self.status = WorkflowStatus.PAUSED
        print(f"⏸️ Workflow paused: {self.config.name}")

    async def resume(self):
        """恢复工作流执行"""
        if self.status == WorkflowStatus.PAUSED:
            self.status = WorkflowStatus.RUNNING
            print(f"▶️ Workflow resumed: {self.config.name}")

    async def cancel(self):
        """取消工作流执行"""
        self.status = WorkflowStatus.CANCELLED
        print(f"🛑 Workflow cancelled: {self.config.name}")

    def reset(self):
        """重置工作流状态"""
        self.status = WorkflowStatus.IDLE
        self.workflow_id = None
        self.start_time = None
        self.executed_nodes.clear()
        self.failed_nodes.clear()
        self.running_nodes.clear()
        self.node_results.clear()
        self.metrics = {
            'nodes_executed': 0,
            'nodes_failed': 0,
            'total_execution_time': 0.0,
            'average_node_time': 0.0,
            'bottleneck_nodes': [],
            'resource_usage': {}
        }
        print(f"🔄 Workflow reset: {self.config.name}")