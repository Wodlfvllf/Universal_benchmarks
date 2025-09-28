from typing import List, Dict, Any
from ..text_generation.task import TextGenerationTask

class TranslationTask(TextGenerationTask):
    """
    Implementation for translation tasks.
    Inherits from TextGenerationTask and uses BLEU/ROUGE scores for evaluation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    def prepare_inputs(self, dataset: Any, **kwargs) -> List[Dict[str, Any]]:
        """
        Prepares inputs for translation.
        Assumes dataset has columns for source and target languages.
        Example: {'en': 'Hello', 'fr': 'Bonjour'}
        """
        source_lang = self.config.get('source_lang', 'en')
        target_lang = self.config.get('target_lang', 'fr')

        for example in dataset:
            example['input'] = example.pop(source_lang, example.get('input'))
            example['output'] = example.pop(target_lang, example.get('output'))

        return super().prepare_inputs(dataset, **kwargs)

    def format_prompt(self, input_data: Dict[str, Any]) -> str:
        """Formats the prompt for translation, optionally adding a prefix."""
        prompt = super().format_prompt(input_data)
        target_lang = self.config.get('target_lang', 'fr')
        return f"Translate to {target_lang}: {prompt}"
