"""
任务队列基础接口 - 统一的任务队列抽象层
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import uuid
import json
from datetime import datetime, timedelta


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"          # 等待执行
    RUNNING = "running"          # 正在执行
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"           # 执行失败
    CANCELLED = "cancelled"     # 已取消
    RETRYING = "retrying"       # 重试中
    SCHEDULED = "scheduled"     # 已调度（延迟执行）


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class TaskResult:
    """任务执行结果"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    func: str  # 函数名或模块路径
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING

    # 调度相关
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 重试配置
    max_retries: int = 3
    retry_count: int = 0
    retry_delay: float = 1.0  # 重试延迟（秒）
    exponential_backoff: bool = True

    # 超时配置
    timeout: Optional[float] = None

    # 依赖关系
    depends_on: List[str] = field(default_factory=list)

    # 结果
    result: Optional[TaskResult] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    def is_ready(self) -> bool:
        """检查任务是否准备执行"""
        return (self.status == TaskStatus.PENDING and
                (self.scheduled_at is None or self.scheduled_at <= datetime.now()))

    def should_retry(self) -> bool:
        """检查是否应该重试"""
        return (self.status == TaskStatus.FAILED and
                self.retry_count < self.max_retries)

    def get_retry_delay(self) -> float:
        """获取重试延迟时间"""
        if self.exponential_backoff:
            return self.retry_delay * (2 ** self.retry_count)
        return self.retry_delay


@dataclass
class QueueStats:
    """队列统计信息"""
    total_tasks: int = 0
    pending_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0

    total_execution_time_ms: int = 0
    average_execution_time_ms: float = 0.0

    throughput_per_minute: float = 0.0
    error_rate: float = 0.0

    uptime_seconds: float = 0.0
    workers_count: int = 0
    active_workers: int = 0


@dataclass
class WorkerInfo:
    """工作者信息"""
    id: str
    status: str  # idle, busy, stopped
    current_task: Optional[str] = None
    tasks_processed: int = 0
    total_execution_time_ms: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTaskQueue(ABC):
    """任务队列基类"""

    def __init__(self, name: str = "default", max_workers: int = 5):
        self.name = name
        self.max_workers = max_workers
        self.workers: Dict[str, WorkerInfo] = {}
        self.start_time = datetime.now()

        # 统计信息
        self.stats = QueueStats()

        # 任务处理器注册
        self.task_handlers: Dict[str, Callable] = {}

        # 中间件
        self.middleware: List[Callable] = []

        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {
            "task_started": [],
            "task_completed": [],
            "task_failed": [],
            "task_retried": [],
            "worker_started": [],
            "worker_stopped": []
        }

    @abstractmethod
    async def enqueue(self, task: Task) -> bool:
        """添加任务到队列"""
        pass

    @abstractmethod
    async def dequeue(self, worker_id: str) -> Optional[Task]:
        """从队列获取任务"""
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Task]:
        """根据ID获取任务"""
        pass

    @abstractmethod
    async def update_task(self, task: Task) -> bool:
        """更新任务状态"""
        pass

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        pass

    @abstractmethod
    async def get_tasks(self, status: Optional[TaskStatus] = None,
                       limit: int = 100) -> List[Task]:
        """获取任务列表"""
        pass

    @abstractmethod
    async def clear_queue(self) -> bool:
        """清空队列"""
        pass

    async def enqueue_delayed(self, task: Task, delay: float) -> bool:
        """添加延迟任务"""
        task.scheduled_at = datetime.now() + timedelta(seconds=delay)
        task.status = TaskStatus.SCHEDULED
        return await self.enqueue(task)

    async def enqueue_at(self, task: Task, scheduled_time: datetime) -> bool:
        """在指定时间执行任务"""
        task.scheduled_at = scheduled_time
        task.status = TaskStatus.SCHEDULED
        return await self.enqueue(task)

    def register_handler(self, name: str, handler: Callable):
        """注册任务处理器"""
        self.task_handlers[name] = handler

    def add_middleware(self, middleware: Callable):
        """添加中间件"""
        self.middleware.append(middleware)

    def on(self, event: str, callback: Callable):
        """注册事件回调"""
        if event in self.event_callbacks:
            self.event_callbacks[event].append(callback)

    async def emit_event(self, event: str, *args, **kwargs):
        """触发事件"""
        if event in self.event_callbacks:
            for callback in self.event_callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)
                except Exception as e:
                    print(f"Event callback error: {e}")

    async def execute_task(self, task: Task, worker_id: str) -> TaskResult:
        """执行任务"""
        start_time = time.time()

        try:
            # 更新任务状态
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            await self.update_task(task)

            # 触发任务开始事件
            await self.emit_event("task_started", task, worker_id)

            # 应用中间件
            for middleware in self.middleware:
                if asyncio.iscoroutinefunction(middleware):
                    await middleware(task, "before")
                else:
                    middleware(task, "before")

            # 执行任务
            handler = self.task_handlers.get(task.func)
            if not handler:
                raise ValueError(f"No handler registered for: {task.func}")

            # 执行任务（支持超时）
            if task.timeout:
                result = await asyncio.wait_for(
                    self._call_handler(handler, task.args, task.kwargs),
                    timeout=task.timeout
                )
            else:
                result = await self._call_handler(handler, task.args, task.kwargs)

            # 应用中间件
            for middleware in reversed(self.middleware):
                if asyncio.iscoroutinefunction(middleware):
                    await middleware(task, "after")
                else:
                    middleware(task, "after")

            execution_time_ms = int((time.time() - start_time) * 1000)

            # 创建成功结果
            task_result = TaskResult(
                success=True,
                result=result,
                execution_time_ms=execution_time_ms,
                retry_count=task.retry_count
            )

            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = task_result
            await self.update_task(task)

            # 触发任务完成事件
            await self.emit_event("task_completed", task, task_result, worker_id)

            return task_result

        except asyncio.TimeoutError:
            execution_time_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Task timeout after {task.timeout} seconds"

            task_result = TaskResult(
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms,
                retry_count=task.retry_count
            )

            await self._handle_task_failure(task, task_result, worker_id)
            return task_result

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)

            task_result = TaskResult(
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms,
                retry_count=task.retry_count
            )

            await self._handle_task_failure(task, task_result, worker_id)
            return task_result

    async def _call_handler(self, handler: Callable, args: tuple, kwargs: Dict[str, Any]):
        """调用任务处理器"""
        if asyncio.iscoroutinefunction(handler):
            return await handler(*args, **kwargs)
        else:
            return handler(*args, **kwargs)

    async def _handle_task_failure(self, task: Task, task_result: TaskResult, worker_id: str):
        """处理任务失败"""
        if task.should_retry():
            # 重试任务
            task.retry_count += 1
            task.status = TaskStatus.RETRYING

            # 计算重试延迟
            retry_delay = task.get_retry_delay()
            task.scheduled_at = datetime.now() + timedelta(seconds=retry_delay)

            await self.update_task(task)
            await self.emit_event("task_retried", task, task_result, worker_id)

            # 重新入队
            await self.enqueue(task)
        else:
            # 任务最终失败
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.result = task_result

            await self.update_task(task)
            await self.emit_event("task_failed", task, task_result, worker_id)

    async def get_stats(self) -> QueueStats:
        """获取队列统计信息"""
        # 基础实现，子类可以重写
        uptime = (datetime.now() - self.start_time).total_seconds()

        self.stats.uptime_seconds = uptime
        self.stats.workers_count = len(self.workers)
        self.stats.active_workers = len([w for w in self.workers.values() if w.status == "busy"])

        # 计算平均执行时间
        if self.stats.completed_tasks > 0:
            self.stats.average_execution_time_ms = (
                self.stats.total_execution_time_ms / self.stats.completed_tasks
            )

        # 计算错误率
        total_finished = self.stats.completed_tasks + self.stats.failed_tasks
        if total_finished > 0:
            self.stats.error_rate = self.stats.failed_tasks / total_finished

        # 计算吞吐量（每分钟完成的任务数）
        if uptime > 0:
            self.stats.throughput_per_minute = (self.stats.completed_tasks / uptime) * 60

        return self.stats

    async def get_worker_info(self, worker_id: str) -> Optional[WorkerInfo]:
        """获取工作者信息"""
        return self.workers.get(worker_id)

    async def get_all_workers(self) -> List[WorkerInfo]:
        """获取所有工作者信息"""
        return list(self.workers.values())

    async def register_worker(self, worker_id: str) -> bool:
        """注册工作者"""
        if worker_id not in self.workers:
            self.workers[worker_id] = WorkerInfo(
                id=worker_id,
                status="idle"
            )
            await self.emit_event("worker_started", worker_id)
            return True
        return False

    async def unregister_worker(self, worker_id: str) -> bool:
        """注销工作者"""
        if worker_id in self.workers:
            del self.workers[worker_id]
            await self.emit_event("worker_stopped", worker_id)
            return True
        return False

    async def update_worker_status(self, worker_id: str, status: str,
                                 current_task: Optional[str] = None):
        """更新工作者状态"""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.status = status
            worker.current_task = current_task
            worker.last_heartbeat = datetime.now()

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        stats = await self.get_stats()

        return {
            "status": "healthy",
            "queue_name": self.name,
            "uptime_seconds": stats.uptime_seconds,
            "total_tasks": stats.total_tasks,
            "pending_tasks": stats.pending_tasks,
            "running_tasks": stats.running_tasks,
            "workers": {
                "total": stats.workers_count,
                "active": stats.active_workers,
                "idle": stats.workers_count - stats.active_workers
            },
            "performance": {
                "throughput_per_minute": stats.throughput_per_minute,
                "error_rate": stats.error_rate,
                "average_execution_time_ms": stats.average_execution_time_ms
            }
        }

    def create_task(self, name: str, func: str, *args,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   max_retries: int = 3,
                   timeout: Optional[float] = None,
                   depends_on: List[str] = None,
                   **kwargs) -> Task:
        """创建任务的便捷方法"""
        return Task(
            id=str(uuid.uuid4()),
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout,
            depends_on=depends_on or []
        )