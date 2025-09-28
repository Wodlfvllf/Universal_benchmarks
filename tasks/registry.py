from typing import Dict, Type, List
from .base import BaseTask

class TaskRegistry:
    """Central registry for all task implementations"""

    _tasks: Dict[str, Type[BaseTask]] = {}

    @classmethod
    def register(cls, name: str, task_class: Type[BaseTask]):
        """Register a task implementation"""
        if name in cls._tasks:
            print(f"Warning: Task '{name}' is already registered. Overwriting.")
        cls._tasks[name] = task_class

    @classmethod
    def get_task(cls, name: str, config: Dict = None) -> BaseTask:
        """Get task instance by name"""
        if name not in cls._tasks:
            raise ValueError(f"Task '{name}' not registered. Available tasks: {list(cls._tasks.keys())}")
        return cls._tasks[name](config)

    @classmethod
    def list_tasks(cls) -> List[str]:
        """List all registered tasks"""
        return list(cls._tasks.keys())

def register_all_tasks():
    """Register all built-in tasks"""
    from .classification.task import ClassificationTask
    from .multiple_choice.task import MultipleChoiceQATask
    from .text_generation.task import TextGenerationTask
    from .code_generation.task import CodeGenerationTask
    from .summarization.task import SummarizationTask
    from .translation.task import TranslationTask
    from .visual_qa.task import VisualQATask
    from .image_captioning.task import ImageCaptioningTask
    from .extractive_qa.task import ExtractiveQATask

    TaskRegistry.register('classification', ClassificationTask)
    TaskRegistry.register('multiple_choice', MultipleChoiceQATask)
    TaskRegistry.register('text_generation', TextGenerationTask)
    TaskRegistry.register('code_generation', CodeGenerationTask)
    TaskRegistry.register('summarization', SummarizationTask)
    TaskRegistry.register('translation', TranslationTask)
    TaskRegistry.register('visual_qa', VisualQATask)
    TaskRegistry.register('image_captioning', ImageCaptioningTask)
    TaskRegistry.register('extractive_qa', ExtractiveQATask)
    # Add more registrations for other tasks here

# Auto-register tasks when this module is imported
register_all_tasks()
