"""
VGP DAG（有向无环图）执行引擎
支持节点并行执行和依赖关系管理
"""
import asyncio
from typing import Dict, List, Set, Any, Callable, Optional
from collections import defaultdict, deque
import time


class VGPDAGExecutor:
    """VGP工作流DAG执行器"""

    def __init__(self):
        # 节点ID到节点名称的映射
        self.node_mapping = {
            1: 'video_type_identification',    # 视频类型识别
            2: 'emotion_analysis',             # 情感基调分析
            3: 'shot_block_generation',        # 分镜块生成
            4: 'bgm_anchor_planning',          # BGM锚点规划
            5: 'asset_request',                # 素材需求解析（核心素材生成）
            6: 'audio_processing',             # 音频处理
            7: 'sfx_integration',              # 音效集成
            8: 'transition_selection',         # 转场选择
            9: 'bgm_composition',              # BGM合成查找
            10: 'filter_application',          # 滤镜应用
            11: 'dynamic_effects',             # 动态特效添加（汇聚点）
            12: 'aux_media_insertion',         # 额外媒体插入
            13: 'aux_text_insertion',          # 装饰文字插入
            14: 'subtitle_generation',         # 字幕生成
            15: 'intro_outro',                 # 片头片尾生成
            16: 'timeline_integration'         # 最终时间线整合
        }

        # 依赖图：每个节点依赖哪些节点（必须等待这些节点完成）
        self.dependencies = {
            1: [],          # node1 无依赖
            2: [1],         # node2 依赖 node1
            3: [2],         # node3 依赖 node2

            # 从 node3 分出的多个分支
            4: [3],         # node4 依赖 node3
            5: [3],         # node5 依赖 node3（核心素材生成）
            10: [3],        # node10 依赖 node3
            12: [3],        # node12 依赖 node3
            13: [3],        # node13 依赖 node3
            14: [3],        # node14 依赖 node3
            15: [3],        # node15 依赖 node3

            # 分支链条
            9: [4],         # node9 依赖 node4
            6: [5],         # node6 依赖 node5
            7: [6],         # node7 依赖 node6
            8: [7],         # node8 依赖 node7

            # 汇聚到 node11
            11: [9, 10, 14],  # node11 依赖 node9, node10, node14

            # 最终汇聚到 node16
            16: [8, 11, 12, 13, 15]  # node16 依赖多个节点
        }

        # 执行状态跟踪
        self.completed_nodes: Set[int] = set()
        self.running_nodes: Set[int] = set()
        self.node_results: Dict[int, Any] = {}
        self.node_locks = {i: asyncio.Lock() for i in range(1, 17)}

    def get_ready_nodes(self) -> List[int]:
        """获取所有依赖已满足且未执行的节点"""
        ready = []
        for node_id in range(1, 17):
            # 跳过已完成或正在运行的节点
            if node_id in self.completed_nodes or node_id in self.running_nodes:
                continue

            # 检查依赖是否都已完成
            deps = self.dependencies.get(node_id, [])
            if all(dep in self.completed_nodes for dep in deps):
                ready.append(node_id)

        return ready

    async def execute_node(
        self,
        node_id: int,
        node_executor: Callable,
        context: Dict[str, Any],
        on_progress: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        执行单个节点

        Args:
            node_id: 节点ID
            node_executor: 节点执行函数
            context: 执行上下文
            on_progress: 进度回调函数

        Returns:
            节点执行结果
        """
        node_name = self.node_mapping[node_id]

        try:
            # 标记为运行中
            self.running_nodes.add(node_id)

            if on_progress:
                await on_progress(node_id, 'started', f'开始执行节点{node_id}: {node_name}')

            print(f"⚡ [DAG] Executing node {node_id}/{len(self.node_mapping)}: {node_name}")
            start_time = time.time()

            # 执行节点
            result = await node_executor(node_name, context)

            elapsed = time.time() - start_time
            print(f"✅ [DAG] Node {node_id} completed: {node_name} ({elapsed:.2f}s)")

            # 保存结果
            self.node_results[node_id] = result

            # 标记为完成
            self.running_nodes.remove(node_id)
            self.completed_nodes.add(node_id)

            if on_progress:
                await on_progress(node_id, 'completed', f'节点{node_id}执行完成: {node_name}')

            return result

        except Exception as e:
            print(f"❌ [DAG] Node {node_id} failed: {node_name} - {e}")
            self.running_nodes.discard(node_id)
            if on_progress:
                await on_progress(node_id, 'failed', f'节点{node_id}执行失败: {str(e)}')
            raise

    async def execute_dag(
        self,
        node_executor: Callable,
        context: Dict[str, Any],
        on_progress: Optional[Callable] = None
    ) -> Dict[int, Any]:
        """
        执行整个DAG工作流

        Args:
            node_executor: 节点执行函数，签名为 async def executor(node_name: str, context: dict) -> dict
            context: 初始上下文
            on_progress: 进度回调函数，签名为 async def callback(node_id: int, status: str, message: str)

        Returns:
            所有节点的执行结果字典 {node_id: result}
        """
        print("🚀 [DAG] Starting DAG execution...")
        print(f"📊 [DAG] Total nodes: {len(self.node_mapping)}")

        # 重置状态
        self.completed_nodes.clear()
        self.running_nodes.clear()
        self.node_results.clear()

        total_start = time.time()

        while len(self.completed_nodes) < len(self.node_mapping):
            # 获取可以执行的节点
            ready_nodes = self.get_ready_nodes()

            if not ready_nodes:
                if self.running_nodes:
                    # 有节点正在运行，等待
                    await asyncio.sleep(0.1)
                    continue
                else:
                    # 没有可执行的节点，也没有运行中的节点 = 死锁或完成
                    break

            # 并行执行所有准备好的节点
            print(f"🔄 [DAG] Launching {len(ready_nodes)} parallel nodes: {ready_nodes}")

            tasks = [
                self.execute_node(node_id, node_executor, context, on_progress)
                for node_id in ready_nodes
            ]

            # 等待这一批节点全部完成
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 检查是否有错误
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    node_id = ready_nodes[i]
                    node_name = self.node_mapping[node_id]
                    raise Exception(f"节点{node_id} ({node_name})执行失败: {result}")

                # 更新上下文（将节点结果合并到上下文中）
                if isinstance(result, dict):
                    context.update(result)

        total_elapsed = time.time() - total_start
        print(f"🎉 [DAG] DAG execution completed in {total_elapsed:.2f}s")
        print(f"📊 [DAG] Completed nodes: {len(self.completed_nodes)}/{len(self.node_mapping)}")

        return self.node_results

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            'total_nodes': len(self.node_mapping),
            'completed_nodes': len(self.completed_nodes),
            'node_mapping': self.node_mapping,
            'execution_order': list(self.completed_nodes),
            'node_results_keys': list(self.node_results.keys())
        }

    def visualize_dag(self) -> str:
        """可视化DAG结构（返回文本表示）"""
        lines = ["VGP Workflow DAG Structure:", "=" * 60]

        for node_id in sorted(self.node_mapping.keys()):
            node_name = self.node_mapping[node_id]
            deps = self.dependencies.get(node_id, [])

            if deps:
                dep_names = [f"{d}({self.node_mapping[d]})" for d in deps]
                lines.append(f"Node {node_id:2d} [{node_name:25s}] ← depends on: {', '.join(dep_names)}")
            else:
                lines.append(f"Node {node_id:2d} [{node_name:25s}] ← (no dependencies)")

        lines.append("=" * 60)
        return "\n".join(lines)
