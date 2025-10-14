# ai_content_pipeline/orchestrator/task.py
class Task:
    def __init__(self, generator_key: str, params: dict, output_key: str = None):
        self.generator_key = generator_key
        self.params = params  # 支持模板变量 {{var}}
        self.output_key = output_key
        self.result = None
        self.status = "pending"