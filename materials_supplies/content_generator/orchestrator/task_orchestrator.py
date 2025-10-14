# orchestrator/task_orchestrator.py
from .retry_policy import RetryPolicy

class TaskOrchestrator:
    def __init__(self, generator_pool):
        self.generator_pool = generator_pool

    async def run_task(self, task: Task, params: dict):
        generator = self.generator_pool.get(task.generator_key)
        if not generator:
            return {"status": "failed", "error": f"Generator not found: {task.generator_key}"}

        # 获取任务级重试策略
        policy_config = params.pop("retry_policy", None)
        retry_policy = RetryPolicy(policy_config)

        async def _generate():
            result = await generator.generate(**params)
            if result.get("status") != "success":
                raise RuntimeError(result.get("error"))
            return result

        try:
            result = await retry_policy.execute(_generate)
            return result
        except Exception as e:
            return {"status": "failed", "error": f"Task failed after retries: {str(e)}"}