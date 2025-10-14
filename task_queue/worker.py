"""
任务工作者 - 负责执行队列中的任务
"""
from typing import Dict, List, Any, Optional, Callable
import asyncio
import signal
import time
import uuid
from datetime import datetime
import traceback

from .base_queue import BaseTaskQueue, Task, TaskStatus, WorkerInfo


class TaskWorker:
    """任务工作者"""

    def __init__(self, queue: BaseTaskQueue, worker_id: Optional[str] = None,
                 concurrency: int = 1, heartbeat_interval: int = 30):
        self.queue = queue
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.concurrency = concurrency
        self.heartbeat_interval = heartbeat_interval

        # 工作者状态
        self.running = False
        self.tasks_processed = 0
        self.total_execution_time_ms = 0
        self.start_time = None

        # 当前执行的任务
        self.current_tasks: Dict[str, Task] = {}

        # 工作者任务
        self.worker_tasks: List[asyncio.Task] = []
        self.heartbeat_task: Optional[asyncio.Task] = None

        # 信号处理
        self.shutdown_event = asyncio.Event()

    async def start(self) -> bool:
        """启动工作者"""
        if self.running:
            return False

        try:
            # 注册工作者
            await self.queue.register_worker(self.worker_id)

            self.running = True
            self.start_time = datetime.now()

            # 启动工作者协程
            for i in range(self.concurrency):
                task = asyncio.create_task(
                    self._worker_loop(f"{self.worker_id}_{i}")
                )
                self.worker_tasks.append(task)

            # 启动心跳任务
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # 注册信号处理
            self._setup_signal_handlers()

            print(f"✅ Worker started: {self.worker_id} (concurrency: {self.concurrency})")
            return True

        except Exception as e:
            print(f"❌ Worker start failed: {e}")
            self.running = False
            return False

    async def stop(self, graceful: bool = True) -> bool:
        """停止工作者"""
        if not self.running:
            return False

        try:
            print(f"🛑 Stopping worker: {self.worker_id}")
            self.running = False

            if graceful:
                # 优雅停止：等待当前任务完成
                print("⏳ Waiting for current tasks to complete...")
                await self._wait_for_current_tasks()

            # 取消所有工作者任务
            for task in self.worker_tasks:
                task.cancel()

            if self.heartbeat_task:
                self.heartbeat_task.cancel()

            # 等待任务取消完成
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)

            if self.heartbeat_task:
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass

            # 注销工作者
            await self.queue.unregister_worker(self.worker_id)

            print(f"✅ Worker stopped: {self.worker_id}")
            return True

        except Exception as e:
            print(f"❌ Worker stop failed: {e}")
            return False

    async def _worker_loop(self, worker_instance_id: str):
        """工作者主循环"""
        while self.running:
            try:
                # 从队列获取任务
                task = await self.queue.dequeue(worker_instance_id)

                if task:
                    # 执行任务
                    await self._execute_task(task, worker_instance_id)
                else:
                    # 没有任务，短暂休眠
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker loop error: {e}")
                await asyncio.sleep(1)

    async def _execute_task(self, task: Task, worker_instance_id: str):
        """执行单个任务"""
        start_time = time.time()

        try:
            # 记录当前任务
            self.current_tasks[worker_instance_id] = task

            # 更新工作者状态
            await self.queue.update_worker_status(
                worker_instance_id,
                "busy",
                task.id
            )

            # 执行任务
            result = await self.queue.execute_task(task, worker_instance_id)

            # 更新统计
            self.tasks_processed += 1
            execution_time = int((time.time() - start_time) * 1000)
            self.total_execution_time_ms += execution_time

            print(f"✅ Task completed: {task.name} ({execution_time}ms)")

        except Exception as e:
            print(f"❌ Task execution error: {task.name} - {e}")
            traceback.print_exc()

        finally:
            # 清理当前任务记录
            if worker_instance_id in self.current_tasks:
                del self.current_tasks[worker_instance_id]

            # 更新工作者状态为空闲
            await self.queue.update_worker_status(
                worker_instance_id,
                "idle",
                None
            )

    async def _heartbeat_loop(self):
        """心跳循环"""
        while self.running:
            try:
                # 发送心跳
                await self._send_heartbeat()
                await asyncio.sleep(self.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Heartbeat error: {e}")
                await asyncio.sleep(self.heartbeat_interval)

    async def _send_heartbeat(self):
        """发送心跳"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        worker_info = WorkerInfo(
            id=self.worker_id,
            status="busy" if self.current_tasks else "idle",
            current_task=next(iter(self.current_tasks.values())).id if self.current_tasks else None,
            tasks_processed=self.tasks_processed,
            total_execution_time_ms=self.total_execution_time_ms,
            last_heartbeat=datetime.now(),
            metadata={
                "uptime_seconds": uptime,
                "concurrency": self.concurrency,
                "current_task_count": len(self.current_tasks)
            }
        )

        # 这里可以实现具体的心跳发送逻辑
        # 比如更新Redis中的工作者状态

    async def _wait_for_current_tasks(self, timeout: int = 300):
        """等待当前任务完成"""
        start_time = time.time()

        while self.current_tasks and (time.time() - start_time) < timeout:
            await asyncio.sleep(1)

        if self.current_tasks:
            print(f"⚠️ Timeout waiting for tasks, {len(self.current_tasks)} tasks still running")

    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            print(f"📡 Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop(graceful=True))

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def get_status(self) -> Dict[str, Any]:
        """获取工作者状态"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        avg_execution_time = (
            self.total_execution_time_ms / self.tasks_processed
            if self.tasks_processed > 0 else 0
        )

        return {
            "worker_id": self.worker_id,
            "running": self.running,
            "concurrency": self.concurrency,
            "tasks_processed": self.tasks_processed,
            "current_task_count": len(self.current_tasks),
            "current_tasks": [task.name for task in self.current_tasks.values()],
            "uptime_seconds": uptime,
            "total_execution_time_ms": self.total_execution_time_ms,
            "average_execution_time_ms": avg_execution_time,
            "start_time": self.start_time.isoformat() if self.start_time else None
        }

    async def pause(self):
        """暂停工作者（停止接收新任务，但完成当前任务）"""
        self.running = False
        print(f"⏸️ Worker paused: {self.worker_id}")

    async def resume(self):
        """恢复工作者"""
        if not self.running:
            self.running = True
            print(f"▶️ Worker resumed: {self.worker_id}")

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        throughput = self.tasks_processed / uptime if uptime > 0 else 0

        return {
            "throughput_per_second": throughput,
            "throughput_per_minute": throughput * 60,
            "average_execution_time_ms": (
                self.total_execution_time_ms / self.tasks_processed
                if self.tasks_processed > 0 else 0
            ),
            "utilization": len(self.current_tasks) / self.concurrency,
            "uptime_seconds": uptime,
            "tasks_per_hour": throughput * 3600 if throughput > 0 else 0
        }


class WorkerPool:
    """工作者池"""

    def __init__(self, queue: BaseTaskQueue, pool_size: int = 5,
                 worker_concurrency: int = 1):
        self.queue = queue
        self.pool_size = pool_size
        self.worker_concurrency = worker_concurrency

        self.workers: Dict[str, TaskWorker] = {}
        self.running = False

    async def start(self) -> bool:
        """启动工作者池"""
        if self.running:
            return False

        try:
            self.running = True

            # 创建并启动工作者
            for i in range(self.pool_size):
                worker_id = f"pool_worker_{i}"
                worker = TaskWorker(
                    self.queue,
                    worker_id,
                    self.worker_concurrency
                )

                if await worker.start():
                    self.workers[worker_id] = worker
                else:
                    print(f"❌ Failed to start worker: {worker_id}")

            print(f"✅ Worker pool started: {len(self.workers)} workers")
            return True

        except Exception as e:
            print(f"❌ Worker pool start failed: {e}")
            return False

    async def stop(self, graceful: bool = True) -> bool:
        """停止工作者池"""
        if not self.running:
            return False

        try:
            print(f"🛑 Stopping worker pool ({len(self.workers)} workers)")
            self.running = False

            # 停止所有工作者
            stop_tasks = []
            for worker in self.workers.values():
                stop_tasks.append(worker.stop(graceful))

            results = await asyncio.gather(*stop_tasks, return_exceptions=True)

            # 检查结果
            success_count = sum(1 for result in results if result is True)
            print(f"✅ Worker pool stopped: {success_count}/{len(self.workers)} workers stopped successfully")

            self.workers.clear()
            return True

        except Exception as e:
            print(f"❌ Worker pool stop failed: {e}")
            return False

    async def scale(self, new_size: int) -> bool:
        """动态调整工作者数量"""
        if not self.running:
            return False

        try:
            current_size = len(self.workers)

            if new_size > current_size:
                # 增加工作者
                for i in range(current_size, new_size):
                    worker_id = f"pool_worker_{i}"
                    worker = TaskWorker(
                        self.queue,
                        worker_id,
                        self.worker_concurrency
                    )

                    if await worker.start():
                        self.workers[worker_id] = worker

            elif new_size < current_size:
                # 减少工作者
                workers_to_remove = list(self.workers.keys())[new_size:]
                for worker_id in workers_to_remove:
                    worker = self.workers[worker_id]
                    await worker.stop(graceful=True)
                    del self.workers[worker_id]

            self.pool_size = new_size
            print(f"📏 Worker pool scaled to {len(self.workers)} workers")
            return True

        except Exception as e:
            print(f"❌ Worker pool scaling failed: {e}")
            return False

    async def get_pool_status(self) -> Dict[str, Any]:
        """获取工作者池状态"""
        worker_statuses = {}
        total_tasks_processed = 0
        total_execution_time = 0

        for worker_id, worker in self.workers.items():
            status = await worker.get_status()
            worker_statuses[worker_id] = status
            total_tasks_processed += status["tasks_processed"]
            total_execution_time += status["total_execution_time_ms"]

        return {
            "pool_size": len(self.workers),
            "target_size": self.pool_size,
            "running": self.running,
            "total_tasks_processed": total_tasks_processed,
            "total_execution_time_ms": total_execution_time,
            "average_execution_time_ms": (
                total_execution_time / total_tasks_processed
                if total_tasks_processed > 0 else 0
            ),
            "workers": worker_statuses
        }

    async def get_pool_metrics(self) -> Dict[str, Any]:
        """获取工作者池性能指标"""
        metrics = []
        for worker in self.workers.values():
            worker_metrics = await worker.get_performance_metrics()
            metrics.append(worker_metrics)

        if not metrics:
            return {
                "pool_throughput_per_second": 0,
                "pool_utilization": 0,
                "average_worker_throughput": 0
            }

        # 聚合指标
        total_throughput = sum(m["throughput_per_second"] for m in metrics)
        avg_utilization = sum(m["utilization"] for m in metrics) / len(metrics)
        avg_worker_throughput = total_throughput / len(metrics)

        return {
            "pool_throughput_per_second": total_throughput,
            "pool_throughput_per_minute": total_throughput * 60,
            "pool_utilization": avg_utilization,
            "average_worker_throughput": avg_worker_throughput,
            "worker_count": len(self.workers),
            "individual_metrics": metrics
        }

    async def restart_worker(self, worker_id: str) -> bool:
        """重启指定工作者"""
        if worker_id not in self.workers:
            return False

        try:
            # 停止旧工作者
            old_worker = self.workers[worker_id]
            await old_worker.stop(graceful=True)

            # 创建新工作者
            new_worker = TaskWorker(
                self.queue,
                worker_id,
                self.worker_concurrency
            )

            if await new_worker.start():
                self.workers[worker_id] = new_worker
                print(f"🔄 Worker restarted: {worker_id}")
                return True
            else:
                print(f"❌ Failed to restart worker: {worker_id}")
                return False

        except Exception as e:
            print(f"❌ Worker restart failed: {e}")
            return False

    def get_worker_ids(self) -> List[str]:
        """获取所有工作者ID"""
        return list(self.workers.keys())

    async def pause_all(self):
        """暂停所有工作者"""
        for worker in self.workers.values():
            await worker.pause()

    async def resume_all(self):
        """恢复所有工作者"""
        for worker in self.workers.values():
            await worker.resume()