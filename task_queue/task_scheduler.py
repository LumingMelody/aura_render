"""
任务调度器 - 高级任务调度和管理功能
"""
from typing import Dict, List, Any, Optional, Callable, Union
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import cron_descriptor
from croniter import croniter

from .base_queue import BaseTaskQueue, Task, TaskStatus, TaskPriority


@dataclass
class ScheduleRule:
    """调度规则"""
    name: str
    cron_expression: str
    task_template: Dict[str, Any]
    enabled: bool = True
    max_instances: int = 1  # 最大并发实例数
    timeout: Optional[float] = None
    retry_policy: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get_description(self) -> str:
        """获取调度规则的描述"""
        try:
            return cron_descriptor.get_description(self.cron_expression)
        except:
            return self.cron_expression

    def get_next_run_time(self, base_time: Optional[datetime] = None) -> datetime:
        """获取下次执行时间"""
        base = base_time or datetime.now()
        cron = croniter(self.cron_expression, base)
        return cron.get_next(datetime)

    def get_prev_run_time(self, base_time: Optional[datetime] = None) -> datetime:
        """获取上次执行时间"""
        base = base_time or datetime.now()
        cron = croniter(self.cron_expression, base)
        return cron.get_prev(datetime)


@dataclass
class ScheduledTask:
    """调度任务记录"""
    schedule_name: str
    task_id: str
    scheduled_time: datetime
    created_time: datetime
    status: TaskStatus
    next_scheduled_time: Optional[datetime] = None


class TaskScheduler:
    """任务调度器"""

    def __init__(self, queue: BaseTaskQueue, check_interval: int = 60):
        self.queue = queue
        self.check_interval = check_interval

        # 调度规则
        self.schedules: Dict[str, ScheduleRule] = {}

        # 调度任务记录
        self.scheduled_tasks: Dict[str, List[ScheduledTask]] = {}

        # 调度器状态
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None

        # 钩子函数
        self.before_schedule_hooks: List[Callable] = []
        self.after_schedule_hooks: List[Callable] = []
        self.schedule_error_hooks: List[Callable] = []

    async def start(self) -> bool:
        """启动调度器"""
        if self.running:
            return False

        try:
            self.running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            print("✅ Task scheduler started")
            return True

        except Exception as e:
            print(f"❌ Task scheduler start failed: {e}")
            self.running = False
            return False

    async def stop(self) -> bool:
        """停止调度器"""
        if not self.running:
            return False

        try:
            self.running = False
            if self.scheduler_task:
                self.scheduler_task.cancel()
                try:
                    await self.scheduler_task
                except asyncio.CancelledError:
                    pass

            print("✅ Task scheduler stopped")
            return True

        except Exception as e:
            print(f"❌ Task scheduler stop failed: {e}")
            return False

    async def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                await self._check_schedules()
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Scheduler loop error: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_schedules(self):
        """检查所有调度规则"""
        current_time = datetime.now()

        for schedule_name, schedule in self.schedules.items():
            if not schedule.enabled:
                continue

            try:
                await self._check_single_schedule(schedule_name, schedule, current_time)

            except Exception as e:
                print(f"Schedule check error for {schedule_name}: {e}")
                # 调用错误钩子
                for hook in self.schedule_error_hooks:
                    try:
                        if asyncio.iscoroutinefunction(hook):
                            await hook(schedule_name, schedule, e)
                        else:
                            hook(schedule_name, schedule, e)
                    except Exception as hook_error:
                        print(f"Schedule error hook failed: {hook_error}")

    async def _check_single_schedule(self, schedule_name: str, schedule: ScheduleRule, current_time: datetime):
        """检查单个调度规则"""
        # 获取下次执行时间
        next_run_time = schedule.get_next_run_time(current_time - timedelta(minutes=1))

        # 检查是否应该执行
        if next_run_time <= current_time:
            # 检查最大实例数限制
            if await self._check_max_instances(schedule_name, schedule):
                await self._create_scheduled_task(schedule_name, schedule, next_run_time)

    async def _check_max_instances(self, schedule_name: str, schedule: ScheduleRule) -> bool:
        """检查最大实例数限制"""
        if schedule.max_instances <= 0:
            return True

        # 统计当前运行的实例数
        current_instances = 0
        scheduled_tasks = self.scheduled_tasks.get(schedule_name, [])

        for scheduled_task in scheduled_tasks:
            task = await self.queue.get_task(scheduled_task.task_id)
            if task and task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                current_instances += 1

        return current_instances < schedule.max_instances

    async def _create_scheduled_task(self, schedule_name: str, schedule: ScheduleRule, scheduled_time: datetime):
        """创建调度任务"""
        try:
            # 调用前置钩子
            for hook in self.before_schedule_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook(schedule_name, schedule, scheduled_time)
                    else:
                        hook(schedule_name, schedule, scheduled_time)
                except Exception as e:
                    print(f"Before schedule hook failed: {e}")

            # 创建任务
            task_template = schedule.task_template.copy()

            # 添加调度元数据
            if 'metadata' not in task_template:
                task_template['metadata'] = {}

            task_template['metadata'].update({
                'schedule_name': schedule_name,
                'scheduled_time': scheduled_time.isoformat(),
                'created_by_scheduler': True
            })

            # 应用重试策略
            if schedule.retry_policy:
                task_template.update(schedule.retry_policy)

            # 应用超时设置
            if schedule.timeout:
                task_template['timeout'] = schedule.timeout

            # 创建任务对象
            task = Task(
                name=task_template.get('name', f"{schedule_name}_{scheduled_time.strftime('%Y%m%d_%H%M%S')}"),
                func=task_template['func'],
                args=task_template.get('args', ()),
                kwargs=task_template.get('kwargs', {}),
                priority=TaskPriority(task_template.get('priority', TaskPriority.NORMAL.value)),
                max_retries=task_template.get('max_retries', 3),
                timeout=task_template.get('timeout'),
                metadata=task_template.get('metadata', {})
            )

            # 入队任务
            if await self.queue.enqueue(task):
                # 记录调度任务
                scheduled_task = ScheduledTask(
                    schedule_name=schedule_name,
                    task_id=task.id,
                    scheduled_time=scheduled_time,
                    created_time=datetime.now(),
                    status=TaskStatus.PENDING,
                    next_scheduled_time=schedule.get_next_run_time(scheduled_time)
                )

                if schedule_name not in self.scheduled_tasks:
                    self.scheduled_tasks[schedule_name] = []

                self.scheduled_tasks[schedule_name].append(scheduled_task)

                print(f"📅 Scheduled task created: {task.name} (schedule: {schedule_name})")

                # 调用后置钩子
                for hook in self.after_schedule_hooks:
                    try:
                        if asyncio.iscoroutinefunction(hook):
                            await hook(schedule_name, schedule, task, scheduled_task)
                        else:
                            hook(schedule_name, schedule, task, scheduled_task)
                    except Exception as e:
                        print(f"After schedule hook failed: {e}")

            else:
                print(f"❌ Failed to enqueue scheduled task: {schedule_name}")

        except Exception as e:
            print(f"Create scheduled task failed: {schedule_name} - {e}")
            raise

    def add_schedule(self, schedule: ScheduleRule) -> bool:
        """添加调度规则"""
        try:
            # 验证cron表达式
            croniter(schedule.cron_expression)

            self.schedules[schedule.name] = schedule
            print(f"✅ Schedule added: {schedule.name} - {schedule.get_description()}")
            return True

        except Exception as e:
            print(f"❌ Add schedule failed: {schedule.name} - {e}")
            return False

    def remove_schedule(self, schedule_name: str) -> bool:
        """移除调度规则"""
        if schedule_name in self.schedules:
            del self.schedules[schedule_name]

            # 清理调度任务记录
            if schedule_name in self.scheduled_tasks:
                del self.scheduled_tasks[schedule_name]

            print(f"✅ Schedule removed: {schedule_name}")
            return True

        return False

    def get_schedule(self, schedule_name: str) -> Optional[ScheduleRule]:
        """获取调度规则"""
        return self.schedules.get(schedule_name)

    def get_all_schedules(self) -> Dict[str, ScheduleRule]:
        """获取所有调度规则"""
        return self.schedules.copy()

    def enable_schedule(self, schedule_name: str) -> bool:
        """启用调度规则"""
        if schedule_name in self.schedules:
            self.schedules[schedule_name].enabled = True
            print(f"✅ Schedule enabled: {schedule_name}")
            return True
        return False

    def disable_schedule(self, schedule_name: str) -> bool:
        """禁用调度规则"""
        if schedule_name in self.schedules:
            self.schedules[schedule_name].enabled = False
            print(f"⏸️ Schedule disabled: {schedule_name}")
            return True
        return False

    async def trigger_schedule(self, schedule_name: str) -> bool:
        """手动触发调度"""
        if schedule_name not in self.schedules:
            return False

        try:
            schedule = self.schedules[schedule_name]
            current_time = datetime.now()

            await self._create_scheduled_task(schedule_name, schedule, current_time)
            print(f"🔥 Schedule triggered manually: {schedule_name}")
            return True

        except Exception as e:
            print(f"❌ Manual trigger failed: {schedule_name} - {e}")
            return False

    def get_schedule_status(self, schedule_name: str) -> Dict[str, Any]:
        """获取调度状态"""
        if schedule_name not in self.schedules:
            return {"error": "Schedule not found"}

        schedule = self.schedules[schedule_name]
        scheduled_tasks = self.scheduled_tasks.get(schedule_name, [])

        # 计算统计信息
        total_tasks = len(scheduled_tasks)
        completed_tasks = 0
        failed_tasks = 0
        running_tasks = 0

        for scheduled_task in scheduled_tasks:
            if scheduled_task.status == TaskStatus.COMPLETED:
                completed_tasks += 1
            elif scheduled_task.status == TaskStatus.FAILED:
                failed_tasks += 1
            elif scheduled_task.status == TaskStatus.RUNNING:
                running_tasks += 1

        # 获取下次执行时间
        next_run_time = schedule.get_next_run_time()

        return {
            "schedule_name": schedule_name,
            "enabled": schedule.enabled,
            "cron_expression": schedule.cron_expression,
            "description": schedule.get_description(),
            "max_instances": schedule.max_instances,
            "next_run_time": next_run_time.isoformat(),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "running_tasks": running_tasks,
            "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0
        }

    def get_all_schedule_status(self) -> Dict[str, Any]:
        """获取所有调度状态"""
        status = {}
        for schedule_name in self.schedules:
            status[schedule_name] = self.get_schedule_status(schedule_name)
        return status

    async def get_scheduled_tasks(self, schedule_name: Optional[str] = None,
                                 limit: int = 100) -> List[ScheduledTask]:
        """获取调度任务记录"""
        if schedule_name:
            tasks = self.scheduled_tasks.get(schedule_name, [])
        else:
            tasks = []
            for schedule_tasks in self.scheduled_tasks.values():
                tasks.extend(schedule_tasks)

        # 按创建时间倒序排序
        tasks.sort(key=lambda t: t.created_time, reverse=True)

        return tasks[:limit]

    def add_before_schedule_hook(self, hook: Callable):
        """添加调度前钩子"""
        self.before_schedule_hooks.append(hook)

    def add_after_schedule_hook(self, hook: Callable):
        """添加调度后钩子"""
        self.after_schedule_hooks.append(hook)

    def add_schedule_error_hook(self, hook: Callable):
        """添加调度错误钩子"""
        self.schedule_error_hooks.append(hook)

    async def cleanup_old_records(self, days: int = 30):
        """清理旧的调度记录"""
        cutoff_time = datetime.now() - timedelta(days=days)
        cleaned_count = 0

        for schedule_name, tasks in self.scheduled_tasks.items():
            # 过滤掉旧记录
            old_count = len(tasks)
            self.scheduled_tasks[schedule_name] = [
                task for task in tasks
                if task.created_time > cutoff_time
            ]
            cleaned_count += old_count - len(self.scheduled_tasks[schedule_name])

        print(f"🧹 Cleaned up {cleaned_count} old schedule records")
        return cleaned_count

    def create_schedule_from_template(self, name: str, cron_expression: str,
                                    func: str, **kwargs) -> ScheduleRule:
        """从模板创建调度规则"""
        template = {
            'func': func,
            'args': kwargs.get('args', ()),
            'kwargs': kwargs.get('kwargs', {}),
            'priority': kwargs.get('priority', TaskPriority.NORMAL.value),
            'max_retries': kwargs.get('max_retries', 3),
            'timeout': kwargs.get('timeout')
        }

        return ScheduleRule(
            name=name,
            cron_expression=cron_expression,
            task_template=template,
            enabled=kwargs.get('enabled', True),
            max_instances=kwargs.get('max_instances', 1),
            timeout=kwargs.get('timeout'),
            retry_policy=kwargs.get('retry_policy'),
            metadata=kwargs.get('metadata', {})
        )

    def validate_cron_expression(self, cron_expression: str) -> Dict[str, Any]:
        """验证cron表达式"""
        try:
            cron = croniter(cron_expression)
            next_runs = []

            # 获取接下来5次执行时间作为示例
            for _ in range(5):
                next_runs.append(cron.get_next(datetime).isoformat())

            return {
                "valid": True,
                "description": cron_descriptor.get_description(cron_expression),
                "next_runs": next_runs
            }

        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }

    async def get_scheduler_metrics(self) -> Dict[str, Any]:
        """获取调度器性能指标"""
        total_schedules = len(self.schedules)
        enabled_schedules = sum(1 for s in self.schedules.values() if s.enabled)

        # 计算任务统计
        total_scheduled_tasks = 0
        total_completed = 0
        total_failed = 0

        for tasks in self.scheduled_tasks.values():
            total_scheduled_tasks += len(tasks)
            for task in tasks:
                if task.status == TaskStatus.COMPLETED:
                    total_completed += 1
                elif task.status == TaskStatus.FAILED:
                    total_failed += 1

        success_rate = total_completed / total_scheduled_tasks if total_scheduled_tasks > 0 else 0

        return {
            "scheduler_running": self.running,
            "check_interval_seconds": self.check_interval,
            "total_schedules": total_schedules,
            "enabled_schedules": enabled_schedules,
            "disabled_schedules": total_schedules - enabled_schedules,
            "total_scheduled_tasks": total_scheduled_tasks,
            "completed_tasks": total_completed,
            "failed_tasks": total_failed,
            "success_rate": success_rate,
            "hook_counts": {
                "before_schedule": len(self.before_schedule_hooks),
                "after_schedule": len(self.after_schedule_hooks),
                "error_hooks": len(self.schedule_error_hooks)
            }
        }