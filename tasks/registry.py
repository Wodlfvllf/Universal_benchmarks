from typing import Dict, Type, List
from tasks.base import BaseTask

class TaskRegistry:
    """Central registry for all task implementations"""
    
    _tasks: Dict[str, Type[BaseTask]] = {}
    
    @classmethod
    def register(cls, name: str, task_class: Type[BaseTask]):
        """Register a task implementation"""
        cls._tasks[name] = task_class
        
    @classmethod
    def get_task(cls, name: str, config: Dict = None) -> BaseTask:
        """Get task instance by name"""
        if name not in cls._tasks:
            raise ValueError(f"Task {name} not registered")
        return cls._tasks[name](config)
        
    @classmethod
    def list_tasks(cls) -> List[str]:
        """List all registered tasks"""
        return list(cls._tasks.keys())

# Auto-registration
def register_all_tasks():
    """Register all built-in tasks"""
    from tasks.classification.task import ClassificationTask
    from tasks.multiple_choice.task import MultipleChoiceQATask
    from tasks.text_generation.task import TextGenerationTask
    from tasks.code_generation.task import CodeGenerationTask
    from tasks.extractive_qa.task import ExtractiveQATask
    from tasks.generative_qa.task import GenerativeQATask
    from tasks.image_captioning.task import ImageCaptioningTask
    from tasks.summarization.task import SummarizationTask
    from tasks.translation.task import TranslationTask
    from tasks.visual_qa.task import VisualQATask
    from tasks.token_classification.task import TokenClassificationTask
    from tasks.commonsense_reasoning.task import CommonsenseReasoningTask
    from tasks.document_qa.task import DocumentQATask
    from tasks.image_classification.task import ImageClassificationTask
    from tasks.math_reasoning.task import MathReasoningTask
    from tasks.sequence_pair_classification.task import SequencePairClassificationTask
    from tasks.regression.task import RegressionTask
    from tasks.nli.task import NLITask
    from tasks.sentence_similarity.task import SentenceSimilarityTask
    
    TaskRegistry.register('classification', ClassificationTask)
    TaskRegistry.register('multiple_choice', MultipleChoiceQATask)
    TaskRegistry.register('text_generation', TextGenerationTask)
    TaskRegistry.register('code_generation', CodeGenerationTask)
    TaskRegistry.register('extractive_qa', ExtractiveQATask)
    TaskRegistry.register('generative_qa', GenerativeQATask)
    TaskRegistry.register('image_captioning', ImageCaptioningTask)
    TaskRegistry.register('summarization', SummarizationTask)
    TaskRegistry.register('translation', TranslationTask)
    TaskRegistry.register('visual_qa', VisualQATask)
    TaskRegistry.register('token_classification', TokenClassificationTask)
    TaskRegistry.register('commonsense_reasoning', CommonsenseReasoningTask)
    TaskRegistry.register('document_qa', DocumentQATask)
    TaskRegistry.register('image_classification', ImageClassificationTask)
    TaskRegistry.register('math_reasoning', MathReasoningTask)
    TaskRegistry.register('sequence_pair_classification', SequencePairClassificationTask)
    TaskRegistry.register('regression', RegressionTask)
    TaskRegistry.register('nli', NLITask)
    TaskRegistry.register('sentence_similarity', SentenceSimilarityTask)
    
register_all_tasks()