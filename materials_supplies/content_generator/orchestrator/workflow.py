# ai_content_pipeline/orchestrator/workflow.py
class Workflow:
    def __init__(self, name: str):
        self.name = name
        self.tasks = []
        self.results = {}

    def add_task(self, task: Task):
        self.tasks.append(task)

    async def execute(self, executor):
        for task in self.tasks:
            # 替换模板变量
            resolved_params = self._resolve_params(task.params)
            task.result = await executor.run_task(task, resolved_params)
            if task.output_key:
                self.results[task.output_key] = task.result.get("video_url") or task.result.get("audio_url")
            if task.result["status"] != "success":
                return {"status": "failed", "error": task.result.get("error")}
        return {"status": "success", "output": self.results}

    def _resolve_params(self, params):
        # 替换 {{audio_url}} -> 实际值
        if isinstance(params, dict):
            return {k: self._resolve_value(v) for k, v in params.items()}
        return params

    def _resolve_value(self, value):
        if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
            key = value[2:-2]
            return self.results.get(key, value)
        return value