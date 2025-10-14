"""
任务队列管理器 - 统一管理和协调任务队列系统
"""
from typing import Dict, List, Any, Optional, Union, Type
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
import time

from .base_queue import BaseTaskQueue, Task, TaskStatus, TaskPriority, QueueStats
from .memory_queue import MemoryTaskQueue
from .redis_queue import RedisTaskQueue
from .worker import TaskWorker, WorkerPool
from .task_scheduler import TaskScheduler, ScheduleRule


class QueueType(Enum):
    """队列类型"""
    MEMORY = "memory"
    REDIS = "redis"


@dataclass
class QueueConfig:
    """队列配置"""
    name: str
    type: QueueType
    max_workers: int = 5
    worker_concurrency: int = 1
    redis_url: Optional[str] = None
    redis_db: int = 0
    auto_start: bool = True
    scheduler_enabled: bool = False
    scheduler_check_interval: int = 60


@dataclass
class TaskQueueInfo:
    """任务队列信息"""
    name: str
    type: str
    status: str
    workers: int
    scheduler_enabled: bool
    stats: QueueStats


class TaskQueueManager:
    """任务队列管理器"""

    def __init__(self):
        # 队列实例
        self.queues: Dict[str, BaseTaskQueue] = {}
        self.queue_configs: Dict[str, QueueConfig] = {}

        # 工作者池
        self.worker_pools: Dict[str, WorkerPool] = {}

        # 调度器
        self.schedulers: Dict[str, TaskScheduler] = {}

        # 全局任务处理器注册
        self.global_handlers: Dict[str, callable] = {}

        # 管理器状态
        self.running = False

    async def create_queue(self, config: QueueConfig) -> bool:
        """创建任务队列"""
        try:
            # 创建队列实例
            if config.type == QueueType.MEMORY:
                queue = MemoryTaskQueue(config.name, config.max_workers)
            elif config.type == QueueType.REDIS:
                if not config.redis_url:
                    config.redis_url = "redis://localhost:6379"
                queue = RedisTaskQueue(
                    config.name,
                    config.redis_url,
                    config.max_workers,
                    config.redis_db
                )
                # Redis队列需要连接
                if not await queue.connect():
                    return False
            else:
                raise ValueError(f"Unsupported queue type: {config.type}")

            # 注册全局处理器
            for name, handler in self.global_handlers.items():
                queue.register_handler(name, handler)

            # 保存队列和配置
            self.queues[config.name] = queue
            self.queue_configs[config.name] = config

            # 创建工作者池
            if config.auto_start:
                worker_pool = WorkerPool(
                    queue,
                    config.max_workers,
                    config.worker_concurrency
                )
                self.worker_pools[config.name] = worker_pool

                # 启动工作者池
                await worker_pool.start()

            # 创建调度器
            if config.scheduler_enabled:
                scheduler = TaskScheduler(queue, config.scheduler_check_interval)
                self.schedulers[config.name] = scheduler

                # 启动调度器
                if config.auto_start:
                    await scheduler.start()

            # 启动内存队列的调度器（处理延迟任务）
            if config.type == QueueType.MEMORY and hasattr(queue, 'start_scheduler'):
                await queue.start_scheduler()

            print(f"✅ Queue created: {config.name} ({config.type.value})")
            return True

        except Exception as e:
            print(f"❌ Queue creation failed: {config.name} - {e}")
            return False

    async def remove_queue(self, queue_name: str) -> bool:
        """移除任务队列"""
        try:
            # 停止调度器
            if queue_name in self.schedulers:
                await self.schedulers[queue_name].stop()
                del self.schedulers[queue_name]

            # 停止工作者池
            if queue_name in self.worker_pools:
                await self.worker_pools[queue_name].stop()
                del self.worker_pools[queue_name]

            # 断开队列连接
            if queue_name in self.queues:
                queue = self.queues[queue_name]
                if hasattr(queue, 'disconnect'):
                    await queue.disconnect()
                elif hasattr(queue, 'stop_scheduler'):
                    await queue.stop_scheduler()

                del self.queues[queue_name]

            # 移除配置
            if queue_name in self.queue_configs:
                del self.queue_configs[queue_name]

            print(f"✅ Queue removed: {queue_name}")
            return True

        except Exception as e:
            print(f"❌ Queue removal failed: {queue_name} - {e}")
            return False

    def get_queue(self, queue_name: str) -> Optional[BaseTaskQueue]:
        """获取队列实例"""
        return self.queues.get(queue_name)

    def get_scheduler(self, queue_name: str) -> Optional[TaskScheduler]:
        """获取调度器实例"""
        return self.schedulers.get(queue_name)

    def get_worker_pool(self, queue_name: str) -> Optional[WorkerPool]:
        """获取工作者池实例"""
        return self.worker_pools.get(queue_name)

    async def submit_task(self, queue_name: str, task: Task) -> bool:
        """提交任务到指定队列"""
        queue = self.get_queue(queue_name)
        if queue:
            return await queue.enqueue(task)
        return False

    async def submit_task_to_best_queue(self, task: Task) -> Optional[str]:
        """提交任务到最佳队列"""
        if not self.queues:
            return None

        # 选择负载最轻的队列
        best_queue_name = None
        min_pending_tasks = float('inf')

        for queue_name, queue in self.queues.items():
            try:
                stats = await queue.get_stats()
                if stats.pending_tasks < min_pending_tasks:
                    min_pending_tasks = stats.pending_tasks
                    best_queue_name = queue_name
            except Exception:
                continue

        if best_queue_name:
            await self.submit_task(best_queue_name, task)
            return best_queue_name

        return None

    def register_global_handler(self, name: str, handler: callable):
        """注册全局任务处理器"""
        self.global_handlers[name] = handler

        # 为所有现有队列注册处理器
        for queue in self.queues.values():
            queue.register_handler(name, handler)

    async def get_all_queue_stats(self) -> Dict[str, QueueStats]:
        """获取所有队列的统计信息"""
        stats = {}

        for queue_name, queue in self.queues.items():
            try:
                stats[queue_name] = await queue.get_stats()
            except Exception as e:
                print(f"Failed to get stats for queue {queue_name}: {e}")

        return stats

    async def get_queue_info(self, queue_name: str) -> Optional[TaskQueueInfo]:
        """获取队列详细信息"""
        if queue_name not in self.queues:
            return None

        try:
            queue = self.queues[queue_name]
            config = self.queue_configs[queue_name]
            stats = await queue.get_stats()

            # 确定状态
            status = "running"
            if queue_name in self.worker_pools:
                worker_pool = self.worker_pools[queue_name]
                if not worker_pool.running:
                    status = "stopped"
            else:
                status = "created"

            return TaskQueueInfo(
                name=queue_name,
                type=config.type.value,
                status=status,
                workers=stats.workers_count,
                scheduler_enabled=queue_name in self.schedulers,
                stats=stats
            )

        except Exception as e:
            print(f"Failed to get queue info for {queue_name}: {e}")
            return None

    async def get_all_queue_info(self) -> List[TaskQueueInfo]:
        """获取所有队列信息"""
        queue_info = []

        for queue_name in self.queues:
            info = await self.get_queue_info(queue_name)
            if info:
                queue_info.append(info)

        return queue_info

    async def scale_workers(self, queue_name: str, new_size: int) -> bool:
        """动态调整工作者数量"""
        if queue_name not in self.worker_pools:
            return False

        try:
            worker_pool = self.worker_pools[queue_name]
            result = await worker_pool.scale(new_size)

            if result:
                # 更新配置
                self.queue_configs[queue_name].max_workers = new_size
                print(f"📏 Workers scaled for queue {queue_name}: {new_size}")

            return result

        except Exception as e:
            print(f"❌ Worker scaling failed for {queue_name}: {e}")
            return False

    async def pause_queue(self, queue_name: str) -> bool:
        """暂停队列处理"""
        try:
            # 暂停工作者池
            if queue_name in self.worker_pools:
                await self.worker_pools[queue_name].pause_all()

            # 暂停调度器
            if queue_name in self.schedulers:
                await self.schedulers[queue_name].stop()

            print(f"⏸️ Queue paused: {queue_name}")
            return True

        except Exception as e:
            print(f"❌ Queue pause failed: {queue_name} - {e}")
            return False

    async def resume_queue(self, queue_name: str) -> bool:
        """恢复队列处理"""
        try:
            # 恢复工作者池
            if queue_name in self.worker_pools:
                await self.worker_pools[queue_name].resume_all()

            # 恢复调度器
            if queue_name in self.schedulers:
                await self.schedulers[queue_name].start()

            print(f"▶️ Queue resumed: {queue_name}")
            return True

        except Exception as e:
            print(f"❌ Queue resume failed: {queue_name} - {e}")
            return False

    async def clear_queue(self, queue_name: str) -> bool:
        """清空队列"""
        queue = self.get_queue(queue_name)
        if queue:
            return await queue.clear_queue()
        return False

    async def get_task(self, queue_name: str, task_id: str) -> Optional[Task]:
        """获取任务"""
        queue = self.get_queue(queue_name)
        if queue:
            return await queue.get_task(task_id)
        return None

    async def cancel_task(self, queue_name: str, task_id: str) -> bool:
        """取消任务"""
        queue = self.get_queue(queue_name)
        if queue:
            return await queue.cancel_task(task_id)
        return False

    async def get_tasks(self, queue_name: str, status: Optional[TaskStatus] = None,
                       limit: int = 100) -> List[Task]:
        """获取任务列表"""
        queue = self.get_queue(queue_name)
        if queue:
            return await queue.get_tasks(status, limit)
        return []

    async def add_schedule(self, queue_name: str, schedule: ScheduleRule) -> bool:
        """添加调度规则"""
        scheduler = self.get_scheduler(queue_name)
        if scheduler:
            return scheduler.add_schedule(schedule)
        return False

    async def remove_schedule(self, queue_name: str, schedule_name: str) -> bool:
        """移除调度规则"""
        scheduler = self.get_scheduler(queue_name)
        if scheduler:
            return scheduler.remove_schedule(schedule_name)
        return False

    async def trigger_schedule(self, queue_name: str, schedule_name: str) -> bool:
        """手动触发调度"""
        scheduler = self.get_scheduler(queue_name)
        if scheduler:
            return await scheduler.trigger_schedule(schedule_name)
        return False

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "manager_running": self.running,
            "total_queues": len(self.queues),
            "queues": {}
        }

        for queue_name, queue in self.queues.items():
            try:
                queue_health = await queue.health_check()
                worker_pool_status = "not_running"

                if queue_name in self.worker_pools:
                    worker_pool = self.worker_pools[queue_name]
                    worker_pool_status = "running" if worker_pool.running else "stopped"

                scheduler_status = "not_enabled"
                if queue_name in self.schedulers:
                    scheduler = self.schedulers[queue_name]
                    scheduler_status = "running" if scheduler.running else "stopped"

                health_status["queues"][queue_name] = {
                    "queue_healthy": queue_health,
                    "worker_pool_status": worker_pool_status,
                    "scheduler_status": scheduler_status
                }

            except Exception as e:
                health_status["queues"][queue_name] = {
                    "error": str(e)
                }

        return health_status

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = {
            "global": {
                "total_queues": len(self.queues),
                "total_workers": 0,
                "total_pending_tasks": 0,
                "total_running_tasks": 0,
                "total_completed_tasks": 0,
                "total_failed_tasks": 0,
                "overall_throughput": 0.0,
                "overall_error_rate": 0.0
            },
            "queues": {}
        }

        total_completed = 0
        total_failed = 0
        total_uptime = 0

        for queue_name, queue in self.queues.items():
            try:
                stats = await queue.get_stats()

                metrics["queues"][queue_name] = {
                    "pending_tasks": stats.pending_tasks,
                    "running_tasks": stats.running_tasks,
                    "completed_tasks": stats.completed_tasks,
                    "failed_tasks": stats.failed_tasks,
                    "workers": stats.workers_count,
                    "active_workers": stats.active_workers,
                    "throughput_per_minute": stats.throughput_per_minute,
                    "error_rate": stats.error_rate,
                    "average_execution_time_ms": stats.average_execution_time_ms
                }

                # 累计全局指标
                metrics["global"]["total_workers"] += stats.workers_count
                metrics["global"]["total_pending_tasks"] += stats.pending_tasks
                metrics["global"]["total_running_tasks"] += stats.running_tasks
                metrics["global"]["total_completed_tasks"] += stats.completed_tasks
                metrics["global"]["total_failed_tasks"] += stats.failed_tasks

                total_completed += stats.completed_tasks
                total_failed += stats.failed_tasks
                total_uptime += stats.uptime_seconds

            except Exception as e:
                metrics["queues"][queue_name] = {"error": str(e)}

        # 计算全局指标
        total_tasks = total_completed + total_failed
        if total_tasks > 0:
            metrics["global"]["overall_error_rate"] = total_failed / total_tasks

        if total_uptime > 0:
            avg_uptime = total_uptime / len(self.queues)
            metrics["global"]["overall_throughput"] = (total_completed / avg_uptime) * 60

        return metrics

    async def start_all(self) -> bool:
        """启动所有队列、工作者和调度器"""
        try:
            self.running = True

            # 启动所有工作者池
            for queue_name, worker_pool in self.worker_pools.items():
                if not worker_pool.running:
                    await worker_pool.start()

            # 启动所有调度器
            for queue_name, scheduler in self.schedulers.items():
                if not scheduler.running:
                    await scheduler.start()

            print("✅ All task queue components started")
            return True

        except Exception as e:
            print(f"❌ Start all failed: {e}")
            return False

    async def stop_all(self) -> bool:
        """停止所有队列、工作者和调度器"""
        try:
            self.running = False

            # 停止所有调度器
            for scheduler in self.schedulers.values():
                if scheduler.running:
                    await scheduler.stop()

            # 停止所有工作者池
            for worker_pool in self.worker_pools.values():
                if worker_pool.running:
                    await worker_pool.stop()

            print("✅ All task queue components stopped")
            return True

        except Exception as e:
            print(f"❌ Stop all failed: {e}")
            return False

    async def cleanup(self) -> bool:
        """清理所有资源"""
        try:
            # 停止所有组件
            await self.stop_all()

            # 断开队列连接
            for queue in self.queues.values():
                if hasattr(queue, 'disconnect'):
                    await queue.disconnect()

            # 清空所有集合
            self.queues.clear()
            self.queue_configs.clear()
            self.worker_pools.clear()
            self.schedulers.clear()
            self.global_handlers.clear()

            print("✅ Task queue manager cleaned up")
            return True

        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            return False

    def get_queue_names(self) -> List[str]:
        """获取所有队列名称"""
        return list(self.queues.keys())

    def create_task(self, name: str, func: str, *args,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   **kwargs) -> Task:
        """创建任务的便捷方法"""
        return Task(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority
        )

    async def get_manager_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        return {
            "running": self.running,
            "queues_count": len(self.queues),
            "worker_pools_count": len(self.worker_pools),
            "schedulers_count": len(self.schedulers),
            "global_handlers_count": len(self.global_handlers),
            "queue_names": self.get_queue_names()
        }


# 预设配置函数
def create_default_queue_manager() -> TaskQueueManager:
    """创建默认的任务队列管理器"""
    manager = TaskQueueManager()

    # 注册一些常用的任务处理器示例
    def example_handler(*args, **kwargs):
        """示例任务处理器"""
        return f"Task executed with args: {args}, kwargs: {kwargs}"

    manager.register_global_handler("example_task", example_handler)

    return manager