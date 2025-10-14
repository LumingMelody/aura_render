"""
Redis任务队列 - 基于Redis的分布式任务队列实现
"""
from typing import Dict, List, Any, Optional, Union
import asyncio
import json
import time
from datetime import datetime, timedelta
import redis.asyncio as redis

from .base_queue import (
    BaseTaskQueue,
    Task,
    TaskStatus,
    TaskPriority,
    TaskResult,
    QueueStats
)


class RedisTaskQueue(BaseTaskQueue):
    """Redis任务队列"""

    def __init__(self, name: str = "redis_queue", redis_url: str = "redis://localhost:6379",
                 max_workers: int = 5, db: int = 0):
        super().__init__(name, max_workers)

        self.redis_url = redis_url
        self.db = db
        self.redis_client = None

        # Redis键前缀
        self.key_prefix = f"taskqueue:{name}"

        # 队列键名
        self.pending_key = f"{self.key_prefix}:pending"
        self.running_key = f"{self.key_prefix}:running"
        self.scheduled_key = f"{self.key_prefix}:scheduled"
        self.completed_key = f"{self.key_prefix}:completed"
        self.failed_key = f"{self.key_prefix}:failed"
        self.tasks_key = f"{self.key_prefix}:tasks"
        self.stats_key = f"{self.key_prefix}:stats"
        self.workers_key = f"{self.key_prefix}:workers"

        # 调度器
        self.scheduler_task = None
        self.scheduler_running = False

    async def connect(self) -> bool:
        """连接Redis"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                db=self.db,
                decode_responses=True,
                socket_timeout=30,
                socket_connect_timeout=30,
                retry_on_timeout=True
            )

            # 测试连接
            await self.redis_client.ping()
            print(f"✅ Redis task queue connected: {self.name}")
            return True

        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            return False

    async def disconnect(self) -> bool:
        """断开Redis连接"""
        try:
            if self.scheduler_running:
                await self.stop_scheduler()

            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None

            print(f"✅ Redis task queue disconnected: {self.name}")
            return True

        except Exception as e:
            print(f"❌ Redis disconnect failed: {e}")
            return False

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
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Redis scheduler error: {e}")
                await asyncio.sleep(1)

    async def _process_scheduled_tasks(self):
        """处理延迟任务"""
        current_timestamp = time.time()

        # 获取到期的延迟任务
        results = await self.redis_client.zrangebyscore(
            self.scheduled_key,
            0,
            current_timestamp,
            withscores=True
        )

        for task_id, score in results:
            # 移动到待处理队列
            async with self.redis_client.pipeline() as pipe:
                # 从延迟队列移除
                pipe.zrem(self.scheduled_key, task_id)

                # 获取任务数据
                task_data = await self.redis_client.hget(self.tasks_key, task_id)
                if task_data:
                    task_dict = json.loads(task_data)
                    task = self._dict_to_task(task_dict)

                    # 检查依赖关系
                    if await self._check_dependencies(task):
                        # 添加到待处理队列（按优先级）
                        priority_score = -task.priority.value  # 负数用于高优先级在前
                        pipe.zadd(self.pending_key, {task_id: priority_score})

                        # 更新任务状态
                        task.status = TaskStatus.PENDING
                        pipe.hset(self.tasks_key, task_id, json.dumps(self._task_to_dict(task)))

                await pipe.execute()

    async def enqueue(self, task: Task) -> bool:
        """添加任务到队列"""
        try:
            task_dict = self._task_to_dict(task)
            task_json = json.dumps(task_dict)

            async with self.redis_client.pipeline() as pipe:
                # 存储任务数据
                pipe.hset(self.tasks_key, task.id, task_json)

                # 根据任务状态和调度时间决定放入哪个队列
                if task.status == TaskStatus.SCHEDULED and task.scheduled_at:
                    # 延迟任务，放入调度队列
                    timestamp = task.scheduled_at.timestamp()
                    pipe.zadd(self.scheduled_key, {task.id: timestamp})
                else:
                    # 检查依赖关系
                    if await self._check_dependencies(task):
                        # 立即可执行，放入待处理队列
                        priority_score = -task.priority.value
                        pipe.zadd(self.pending_key, {task.id: priority_score})
                        task.status = TaskStatus.PENDING
                        pipe.hset(self.tasks_key, task.id, json.dumps(self._task_to_dict(task)))

                # 更新统计
                pipe.hincrby(self.stats_key, "total_tasks", 1)

                await pipe.execute()

            return True

        except Exception as e:
            print(f"Redis enqueue error: {e}")
            return False

    async def dequeue(self, worker_id: str) -> Optional[Task]:
        """从队列获取任务"""
        try:
            # 使用BZPOPMIN获取最高优先级的任务（阻塞1秒）
            result = await self.redis_client.bzpopmin(self.pending_key, timeout=1)

            if result:
                queue_name, task_id, score = result

                # 获取任务数据
                task_data = await self.redis_client.hget(self.tasks_key, task_id)
                if not task_data:
                    return None

                task_dict = json.loads(task_data)
                task = self._dict_to_task(task_dict)

                # 更新任务状态
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()

                async with self.redis_client.pipeline() as pipe:
                    # 移动到运行队列
                    pipe.sadd(self.running_key, task_id)

                    # 更新任务数据
                    pipe.hset(self.tasks_key, task_id, json.dumps(self._task_to_dict(task)))

                    # 更新统计
                    pipe.hincrby(self.stats_key, "running_tasks", 1)

                    # 更新工作者状态
                    worker_data = {
                        "status": "busy",
                        "current_task": task_id,
                        "last_heartbeat": datetime.now().isoformat()
                    }
                    pipe.hset(self.workers_key, worker_id, json.dumps(worker_data))

                    await pipe.execute()

                return task

            return None

        except Exception as e:
            print(f"Redis dequeue error: {e}")
            return None

    async def get_task(self, task_id: str) -> Optional[Task]:
        """根据ID获取任务"""
        try:
            task_data = await self.redis_client.hget(self.tasks_key, task_id)
            if task_data:
                task_dict = json.loads(task_data)
                return self._dict_to_task(task_dict)
            return None

        except Exception as e:
            print(f"Redis get task error: {e}")
            return None

    async def update_task(self, task: Task) -> bool:
        """更新任务状态"""
        try:
            async with self.redis_client.pipeline() as pipe:
                # 更新任务数据
                task_dict = self._task_to_dict(task)
                pipe.hset(self.tasks_key, task.id, json.dumps(task_dict))

                # 根据状态更新队列
                if task.status == TaskStatus.COMPLETED:
                    pipe.srem(self.running_key, task.id)
                    pipe.sadd(self.completed_key, task.id)
                    pipe.hincrby(self.stats_key, "running_tasks", -1)
                    pipe.hincrby(self.stats_key, "completed_tasks", 1)

                    # 更新执行时间统计
                    if task.result:
                        pipe.hincrby(self.stats_key, "total_execution_time_ms",
                                   task.result.execution_time_ms)

                    # 检查依赖任务
                    await self._check_dependent_tasks(task.id)

                elif task.status == TaskStatus.FAILED:
                    pipe.srem(self.running_key, task.id)
                    pipe.sadd(self.failed_key, task.id)
                    pipe.hincrby(self.stats_key, "running_tasks", -1)
                    pipe.hincrby(self.stats_key, "failed_tasks", 1)

                elif task.status == TaskStatus.RETRYING:
                    pipe.srem(self.running_key, task.id)
                    # 重新调度
                    if task.scheduled_at:
                        timestamp = task.scheduled_at.timestamp()
                        pipe.zadd(self.scheduled_key, {task.id: timestamp})

                await pipe.execute()

            return True

        except Exception as e:
            print(f"Redis update task error: {e}")
            return False

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        try:
            task = await self.get_task(task_id)
            if not task:
                return False

            if task.status in [TaskStatus.PENDING, TaskStatus.SCHEDULED]:
                async with self.redis_client.pipeline() as pipe:
                    # 从相应队列移除
                    pipe.zrem(self.pending_key, task_id)
                    pipe.zrem(self.scheduled_key, task_id)

                    # 更新任务状态
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
                    pipe.hset(self.tasks_key, task_id, json.dumps(self._task_to_dict(task)))

                    # 更新统计
                    pipe.hincrby(self.stats_key, "cancelled_tasks", 1)

                    await pipe.execute()

                return True

            return False

        except Exception as e:
            print(f"Redis cancel task error: {e}")
            return False

    async def get_tasks(self, status: Optional[TaskStatus] = None,
                       limit: int = 100) -> List[Task]:
        """获取任务列表"""
        try:
            if status:
                # 根据状态获取任务ID
                if status == TaskStatus.PENDING:
                    task_ids = await self.redis_client.zrange(self.pending_key, 0, limit - 1)
                elif status == TaskStatus.RUNNING:
                    task_ids = await self.redis_client.smembers(self.running_key)
                elif status == TaskStatus.COMPLETED:
                    task_ids = await self.redis_client.smembers(self.completed_key)
                elif status == TaskStatus.FAILED:
                    task_ids = await self.redis_client.smembers(self.failed_key)
                elif status == TaskStatus.SCHEDULED:
                    task_ids = await self.redis_client.zrange(self.scheduled_key, 0, limit - 1)
                else:
                    task_ids = []
            else:
                # 获取所有任务ID
                task_ids = await self.redis_client.hkeys(self.tasks_key)

            # 批量获取任务数据
            if task_ids:
                tasks_data = await self.redis_client.hmget(self.tasks_key, *task_ids)
                tasks = []

                for task_data in tasks_data:
                    if task_data:
                        task_dict = json.loads(task_data)
                        tasks.append(self._dict_to_task(task_dict))

                # 按创建时间排序
                tasks.sort(key=lambda t: t.created_at, reverse=True)
                return tasks[:limit]

            return []

        except Exception as e:
            print(f"Redis get tasks error: {e}")
            return []

    async def clear_queue(self) -> bool:
        """清空队列"""
        try:
            keys_to_delete = [
                self.pending_key,
                self.running_key,
                self.scheduled_key,
                self.completed_key,
                self.failed_key,
                self.tasks_key,
                self.stats_key
            ]

            await self.redis_client.delete(*keys_to_delete)
            return True

        except Exception as e:
            print(f"Redis clear queue error: {e}")
            return False

    async def get_stats(self) -> QueueStats:
        """获取队列统计信息"""
        try:
            # 从Redis获取统计数据
            stats_data = await self.redis_client.hgetall(self.stats_key)

            # 获取当前队列长度
            pending_count = await self.redis_client.zcard(self.pending_key)
            running_count = await self.redis_client.scard(self.running_key)
            completed_count = await self.redis_client.scard(self.completed_key)
            failed_count = await self.redis_client.scard(self.failed_key)
            scheduled_count = await self.redis_client.zcard(self.scheduled_key)

            stats = QueueStats(
                total_tasks=int(stats_data.get("total_tasks", 0)),
                pending_tasks=pending_count,
                running_tasks=running_count,
                completed_tasks=completed_count,
                failed_tasks=failed_count,
                cancelled_tasks=int(stats_data.get("cancelled_tasks", 0)),
                total_execution_time_ms=int(stats_data.get("total_execution_time_ms", 0)),
                uptime_seconds=(datetime.now() - self.start_time).total_seconds(),
                workers_count=len(self.workers),
                active_workers=len([w for w in self.workers.values() if w.status == "busy"])
            )

            # 计算衍生统计
            if stats.completed_tasks > 0:
                stats.average_execution_time_ms = (
                    stats.total_execution_time_ms / stats.completed_tasks
                )

            total_finished = stats.completed_tasks + stats.failed_tasks
            if total_finished > 0:
                stats.error_rate = stats.failed_tasks / total_finished

            if stats.uptime_seconds > 0:
                stats.throughput_per_minute = (stats.completed_tasks / stats.uptime_seconds) * 60

            return stats

        except Exception as e:
            print(f"Redis get stats error: {e}")
            return QueueStats()

    async def _check_dependencies(self, task: Task) -> bool:
        """检查任务依赖是否满足"""
        if not task.depends_on:
            return True

        for dep_id in task.depends_on:
            # 检查依赖任务是否已完成
            is_completed = await self.redis_client.sismember(self.completed_key, dep_id)
            if not is_completed:
                return False

        return True

    async def _check_dependent_tasks(self, completed_task_id: str):
        """检查依赖已完成任务的其他任务"""
        # 使用模式匹配找到依赖此任务的其他任务
        dependency_key = f"{self.key_prefix}:deps:{completed_task_id}"
        dependent_task_ids = await self.redis_client.smembers(dependency_key)

        for dep_task_id in dependent_task_ids:
            task = await self.get_task(dep_task_id)
            if task and await self._check_dependencies(task):
                # 将任务移动到待处理队列
                priority_score = -task.priority.value
                await self.redis_client.zadd(self.pending_key, {dep_task_id: priority_score})

                # 更新任务状态
                task.status = TaskStatus.PENDING
                await self.redis_client.hset(
                    self.tasks_key,
                    dep_task_id,
                    json.dumps(self._task_to_dict(task))
                )

    def _task_to_dict(self, task: Task) -> Dict[str, Any]:
        """将Task对象转换为字典"""
        return {
            "id": task.id,
            "name": task.name,
            "func": task.func,
            "args": task.args,
            "kwargs": task.kwargs,
            "priority": task.priority.value,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "scheduled_at": task.scheduled_at.isoformat() if task.scheduled_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "max_retries": task.max_retries,
            "retry_count": task.retry_count,
            "retry_delay": task.retry_delay,
            "exponential_backoff": task.exponential_backoff,
            "timeout": task.timeout,
            "depends_on": task.depends_on,
            "result": {
                "success": task.result.success,
                "result": task.result.result,
                "error": task.result.error,
                "execution_time_ms": task.result.execution_time_ms,
                "retry_count": task.result.retry_count,
                "metadata": task.result.metadata
            } if task.result else None,
            "metadata": task.metadata
        }

    def _dict_to_task(self, task_dict: Dict[str, Any]) -> Task:
        """将字典转换为Task对象"""
        task = Task(
            id=task_dict["id"],
            name=task_dict["name"],
            func=task_dict["func"],
            args=tuple(task_dict["args"]),
            kwargs=task_dict["kwargs"],
            priority=TaskPriority(task_dict["priority"]),
            status=TaskStatus(task_dict["status"]),
            created_at=datetime.fromisoformat(task_dict["created_at"]),
            max_retries=task_dict["max_retries"],
            retry_count=task_dict["retry_count"],
            retry_delay=task_dict["retry_delay"],
            exponential_backoff=task_dict["exponential_backoff"],
            timeout=task_dict["timeout"],
            depends_on=task_dict["depends_on"],
            metadata=task_dict["metadata"]
        )

        if task_dict["scheduled_at"]:
            task.scheduled_at = datetime.fromisoformat(task_dict["scheduled_at"])

        if task_dict["started_at"]:
            task.started_at = datetime.fromisoformat(task_dict["started_at"])

        if task_dict["completed_at"]:
            task.completed_at = datetime.fromisoformat(task_dict["completed_at"])

        if task_dict["result"]:
            result_data = task_dict["result"]
            task.result = TaskResult(
                success=result_data["success"],
                result=result_data["result"],
                error=result_data["error"],
                execution_time_ms=result_data["execution_time_ms"],
                retry_count=result_data["retry_count"],
                metadata=result_data["metadata"]
            )

        return task

    async def register_worker(self, worker_id: str) -> bool:
        """注册工作者"""
        try:
            worker_data = {
                "id": worker_id,
                "status": "idle",
                "current_task": None,
                "tasks_processed": 0,
                "total_execution_time_ms": 0,
                "last_heartbeat": datetime.now().isoformat(),
                "metadata": {}
            }

            await self.redis_client.hset(
                self.workers_key,
                worker_id,
                json.dumps(worker_data)
            )

            await self.emit_event("worker_started", worker_id)
            return True

        except Exception as e:
            print(f"Redis register worker error: {e}")
            return False

    async def unregister_worker(self, worker_id: str) -> bool:
        """注销工作者"""
        try:
            await self.redis_client.hdel(self.workers_key, worker_id)
            await self.emit_event("worker_stopped", worker_id)
            return True

        except Exception as e:
            print(f"Redis unregister worker error: {e}")
            return False

    async def get_all_workers(self) -> List[Dict[str, Any]]:
        """获取所有工作者信息"""
        try:
            workers_data = await self.redis_client.hgetall(self.workers_key)
            workers = []

            for worker_id, worker_data in workers_data.items():
                worker_info = json.loads(worker_data)
                workers.append(worker_info)

            return workers

        except Exception as e:
            print(f"Redis get workers error: {e}")
            return []