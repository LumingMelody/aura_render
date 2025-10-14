"""
内存任务队列 - 基于内存的任务队列实现
"""
from typing import Dict, List, Any, Optional, Set
import asyncio
import heapq
from collections import deque
from datetime import datetime
import threading

from .base_queue import (
    BaseTaskQueue,
    Task,
    TaskStatus,
    TaskPriority,
    QueueStats
)


class PriorityQueue:
    """优先级队列"""

    def __init__(self):
        self._queue = []
        self._index = 0
        self._lock = threading.Lock()

    def put(self, task: Task):
        """添加任务"""
        with self._lock:
            # 使用负数优先级，因为heapq是最小堆
            priority = -task.priority.value
            heapq.heappush(self._queue, (priority, self._index, task))
            self._index += 1

    def get(self) -> Optional[Task]:
        """获取最高优先级的任务"""
        with self._lock:
            if self._queue:
                _, _, task = heapq.heappop(self._queue)
                return task
            return None

    def peek(self) -> Optional[Task]:
        """查看下一个任务但不移除"""
        with self._lock:
            if self._queue:
                _, _, task = self._queue[0]
                return task
            return None

    def remove(self, task_id: str) -> bool:
        """移除指定任务"""
        with self._lock:
            for i, (_, _, task) in enumerate(self._queue):
                if task.id == task_id:
                    del self._queue[i]
                    heapq.heapify(self._queue)
                    return True
            return False

    def size(self) -> int:
        """获取队列大小"""
        with self._lock:
            return len(self._queue)

    def clear(self):
        """清空队列"""
        with self._lock:
            self._queue.clear()
            self._index = 0

    def to_list(self) -> List[Task]:
        """转换为列表"""
        with self._lock:
            return [task for _, _, task in self._queue]


class MemoryTaskQueue(BaseTaskQueue):
    """内存任务队列"""

    def __init__(self, name: str = "memory_queue", max_workers: int = 5):
        super().__init__(name, max_workers)

        # 任务存储
        self.tasks: Dict[str, Task] = {}
        self.task_lock = threading.RLock()

        # 各状态的任务队列
        self.pending_queue = PriorityQueue()
        self.scheduled_queue = deque()  # 延迟任务队列
        self.running_tasks: Set[str] = set()

        # 依赖关系
        self.dependency_graph: Dict[str, Set[str]] = {}  # task_id -> dependent_task_ids
        self.reverse_deps: Dict[str, Set[str]] = {}     # task_id -> dependency_task_ids

        # 调度器任务
        self.scheduler_task = None
        self.scheduler_running = False

    async def start_scheduler(self):
        """启动调度器"""
        if not self.scheduler_running:
            self.scheduler_running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop_scheduler(self):
        """停止调度器"""
        self.scheduler_running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

    async def _scheduler_loop(self):
        """调度器循环"""
        while self.scheduler_running:
            try:
                await self._process_scheduled_tasks()
                await asyncio.sleep(1)  # 每秒检查一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Scheduler error: {e}")
                await asyncio.sleep(1)

    async def _process_scheduled_tasks(self):
        """处理延迟任务"""
        current_time = datetime.now()
        ready_tasks = []

        # 检查延迟任务
        while self.scheduled_queue:
            task = self.scheduled_queue[0]
            if task.scheduled_at and task.scheduled_at <= current_time:
                ready_tasks.append(self.scheduled_queue.popleft())
            else:
                break

        # 将准备就绪的任务移动到待处理队列
        for task in ready_tasks:
            if task.status == TaskStatus.SCHEDULED:
                task.status = TaskStatus.PENDING
                await self._add_to_pending_if_ready(task)

    async def enqueue(self, task: Task) -> bool:
        """添加任务到队列"""
        try:
            with self.task_lock:
                self.tasks[task.id] = task

            # 更新统计
            self.stats.total_tasks += 1

            # 建立依赖关系
            if task.depends_on:
                await self._build_dependencies(task)

            # 根据任务状态分配到相应队列
            if task.status == TaskStatus.SCHEDULED:
                self.scheduled_queue.append(task)
            else:
                await self._add_to_pending_if_ready(task)

            return True

        except Exception as e:
            print(f"Enqueue error: {e}")
            return False

    async def _build_dependencies(self, task: Task):
        """建立任务依赖关系"""
        self.reverse_deps[task.id] = set(task.depends_on)

        for dep_id in task.depends_on:
            if dep_id not in self.dependency_graph:
                self.dependency_graph[dep_id] = set()
            self.dependency_graph[dep_id].add(task.id)

    async def _add_to_pending_if_ready(self, task: Task):
        """如果任务准备就绪，添加到待处理队列"""
        if await self._is_task_ready(task):
            task.status = TaskStatus.PENDING
            self.pending_queue.put(task)
            self.stats.pending_tasks += 1

    async def _is_task_ready(self, task: Task) -> bool:
        """检查任务是否准备执行"""
        # 检查基本条件
        if not task.is_ready():
            return False

        # 检查依赖
        if task.id in self.reverse_deps:
            for dep_id in self.reverse_deps[task.id]:
                if dep_id in self.tasks:
                    dep_task = self.tasks[dep_id]
                    if dep_task.status != TaskStatus.COMPLETED:
                        return False

        return True

    async def dequeue(self, worker_id: str) -> Optional[Task]:
        """从队列获取任务"""
        try:
            task = self.pending_queue.get()
            if task:
                # 更新任务状态
                task.status = TaskStatus.RUNNING
                self.running_tasks.add(task.id)

                # 更新统计
                self.stats.pending_tasks = max(0, self.stats.pending_tasks - 1)
                self.stats.running_tasks += 1

                # 更新工作者状态
                await self.update_worker_status(worker_id, "busy", task.id)

                return task

            return None

        except Exception as e:
            print(f"Dequeue error: {e}")
            return None

    async def get_task(self, task_id: str) -> Optional[Task]:
        """根据ID获取任务"""
        with self.task_lock:
            return self.tasks.get(task_id)

    async def update_task(self, task: Task) -> bool:
        """更新任务状态"""
        try:
            with self.task_lock:
                self.tasks[task.id] = task

            # 更新统计
            await self._update_stats_for_task(task)

            # 如果任务完成，检查依赖任务
            if task.status == TaskStatus.COMPLETED:
                await self._check_dependent_tasks(task.id)

            return True

        except Exception as e:
            print(f"Update task error: {e}")
            return False

    async def _update_stats_for_task(self, task: Task):
        """更新任务相关的统计信息"""
        if task.status == TaskStatus.COMPLETED:
            if task.id in self.running_tasks:
                self.running_tasks.remove(task.id)
                self.stats.running_tasks = max(0, self.stats.running_tasks - 1)

            self.stats.completed_tasks += 1

            if task.result:
                self.stats.total_execution_time_ms += task.result.execution_time_ms

        elif task.status == TaskStatus.FAILED:
            if task.id in self.running_tasks:
                self.running_tasks.remove(task.id)
                self.stats.running_tasks = max(0, self.stats.running_tasks - 1)

            self.stats.failed_tasks += 1

        elif task.status == TaskStatus.CANCELLED:
            if task.id in self.running_tasks:
                self.running_tasks.remove(task.id)
                self.stats.running_tasks = max(0, self.stats.running_tasks - 1)

            if task.status == TaskStatus.PENDING:
                self.stats.pending_tasks = max(0, self.stats.pending_tasks - 1)

            self.stats.cancelled_tasks += 1

    async def _check_dependent_tasks(self, completed_task_id: str):
        """检查依赖已完成任务的其他任务"""
        if completed_task_id in self.dependency_graph:
            dependent_task_ids = self.dependency_graph[completed_task_id]

            for dep_task_id in dependent_task_ids:
                if dep_task_id in self.tasks:
                    dep_task = self.tasks[dep_task_id]
                    if await self._is_task_ready(dep_task):
                        await self._add_to_pending_if_ready(dep_task)

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        try:
            with self.task_lock:
                if task_id not in self.tasks:
                    return False

                task = self.tasks[task_id]

                # 只能取消未执行或等待中的任务
                if task.status in [TaskStatus.PENDING, TaskStatus.SCHEDULED]:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()

                    # 从相应队列中移除
                    if task.status == TaskStatus.PENDING:
                        self.pending_queue.remove(task_id)
                    elif task.status == TaskStatus.SCHEDULED:
                        self.scheduled_queue = deque([t for t in self.scheduled_queue if t.id != task_id])

                    await self._update_stats_for_task(task)
                    return True

            return False

        except Exception as e:
            print(f"Cancel task error: {e}")
            return False

    async def get_tasks(self, status: Optional[TaskStatus] = None,
                       limit: int = 100) -> List[Task]:
        """获取任务列表"""
        with self.task_lock:
            tasks = list(self.tasks.values())

        if status:
            tasks = [task for task in tasks if task.status == status]

        # 按创建时间排序
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return tasks[:limit]

    async def clear_queue(self) -> bool:
        """清空队列"""
        try:
            with self.task_lock:
                self.tasks.clear()

            self.pending_queue.clear()
            self.scheduled_queue.clear()
            self.running_tasks.clear()
            self.dependency_graph.clear()
            self.reverse_deps.clear()

            # 重置统计
            self.stats = QueueStats()

            return True

        except Exception as e:
            print(f"Clear queue error: {e}")
            return False

    async def get_queue_info(self) -> Dict[str, Any]:
        """获取队列详细信息"""
        with self.task_lock:
            return {
                "name": self.name,
                "type": "memory",
                "total_tasks": len(self.tasks),
                "pending_tasks": self.pending_queue.size(),
                "scheduled_tasks": len(self.scheduled_queue),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": self.stats.completed_tasks,
                "failed_tasks": self.stats.failed_tasks,
                "cancelled_tasks": self.stats.cancelled_tasks,
                "workers": len(self.workers),
                "dependencies": len(self.dependency_graph),
                "scheduler_running": self.scheduler_running
            }

    async def get_task_dependencies(self, task_id: str) -> Dict[str, Any]:
        """获取任务依赖关系"""
        dependencies = self.reverse_deps.get(task_id, set())
        dependents = self.dependency_graph.get(task_id, set())

        return {
            "task_id": task_id,
            "depends_on": list(dependencies),
            "dependents": list(dependents),
            "dependency_count": len(dependencies),
            "dependent_count": len(dependents)
        }

    async def get_next_tasks(self, limit: int = 10) -> List[Task]:
        """获取即将执行的任务"""
        next_tasks = []

        # 从待处理队列获取
        pending_tasks = self.pending_queue.to_list()
        next_tasks.extend(pending_tasks[:limit])

        # 从延迟队列获取即将到期的任务
        current_time = datetime.now()
        scheduled_tasks = [
            task for task in self.scheduled_queue
            if task.scheduled_at and (task.scheduled_at - current_time).total_seconds() < 300  # 5分钟内
        ]
        next_tasks.extend(scheduled_tasks[:limit - len(next_tasks)])

        return next_tasks[:limit]

    async def retry_failed_tasks(self, max_retries: Optional[int] = None) -> int:
        """重试失败的任务"""
        retried_count = 0

        with self.task_lock:
            failed_tasks = [task for task in self.tasks.values() if task.status == TaskStatus.FAILED]

        for task in failed_tasks:
            if task.should_retry():
                if max_retries is None or task.retry_count < max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    await self._add_to_pending_if_ready(task)
                    retried_count += 1

        return retried_count

    async def pause_queue(self):
        """暂停队列处理"""
        await self.stop_scheduler()

    async def resume_queue(self):
        """恢复队列处理"""
        await self.start_scheduler()

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        stats = await self.get_stats()

        # 计算队列深度
        queue_depth = self.pending_queue.size() + len(self.scheduled_queue)

        # 计算工作者利用率
        worker_utilization = 0.0
        if self.max_workers > 0:
            worker_utilization = len(self.running_tasks) / self.max_workers

        # 计算任务延迟分布
        with self.task_lock:
            current_time = datetime.now()
            waiting_times = []

            for task in self.tasks.values():
                if task.status == TaskStatus.PENDING:
                    wait_time = (current_time - task.created_at).total_seconds()
                    waiting_times.append(wait_time)

        avg_wait_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
        max_wait_time = max(waiting_times) if waiting_times else 0

        return {
            "throughput_per_minute": stats.throughput_per_minute,
            "error_rate": stats.error_rate,
            "average_execution_time_ms": stats.average_execution_time_ms,
            "queue_depth": queue_depth,
            "worker_utilization": worker_utilization,
            "average_wait_time_seconds": avg_wait_time,
            "max_wait_time_seconds": max_wait_time,
            "memory_usage": {
                "total_tasks": len(self.tasks),
                "dependencies": len(self.dependency_graph),
                "reverse_dependencies": len(self.reverse_deps)
            }
        }