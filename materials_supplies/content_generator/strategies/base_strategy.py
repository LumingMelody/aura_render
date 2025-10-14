# strategies/base_strategy.py
from abc import ABC, abstractmethod
from ai_content_pipeline.orchestrator.workflow import Workflow

class Strategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def build_workflow(self, context: Dict) -> Workflow:
        """
        根据上下文构建任务流
        """
        pass