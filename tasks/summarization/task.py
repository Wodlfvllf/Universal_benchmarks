from typing import List, Dict, Any
from ..text_generation.task import TextGenerationTask

class SummarizationTask(TextGenerationTask):
    """
    Implementation for summarization tasks.
    Inherits from TextGenerationTask and uses ROUGE scores for evaluation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    def prepare_inputs(self, dataset: Any, **kwargs) -> List[Dict[str, Any]]:
        """
        Prepares inputs for summarization.
        Assumes dataset has 'document' and 'summary' columns.
        """
        # The TextGenerationTask expects 'input' and 'output' keys.
        # We adapt the summarization dataset format here.
        for example in dataset:
            example['input'] = example.pop('document', example.get('input'))
            example['output'] = example.pop('summary', example.get('output'))
        
        return super().prepare_inputs(dataset, **kwargs)
