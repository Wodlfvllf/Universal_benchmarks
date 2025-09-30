
import numpy as np
from typing import List, Dict, Any
from ..base import BaseTask, TaskInput, TaskOutput
from metrics.registry import MetricRegistry

class TranslationTask(BaseTask):
    """Implementation for translation tasks"""
    
    def setup(self):
        self.max_length = self.config.get('max_length', 512)
        self.temperature = self.config.get('temperature', 0.7)
        self.top_p = self.config.get('top_p', 0.95)
        self.bleu_metric = MetricRegistry.get_metric('bleu')
        self.rouge_metric = MetricRegistry.get_metric('rouge')
        self.source_lang = self.config.get('source_lang', 'en')
        self.target_lang = self.config.get('target_lang', 'de')
        
    def prepare_inputs(self, dataset: Any, input_column: str, label_column: str) -> List[TaskInput]:
        """Prepare translation inputs"""
        inputs = []
        
        texts = dataset[input_column]
        translations = dataset[label_column]
        
        for i, text in enumerate(texts):
            inputs.append(TaskInput(
                data=text,
                labels=translations[i],
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for translation prompt"""
        if self.config.get('prompt_template'):
            return self.config['prompt_template'].format(
                text=input_data.data, 
                source_lang=self.source_lang, 
                target_lang=self.target_lang
            )
        return f"Translate the following text from {self.source_lang} to {self.target_lang}:\n\n{input_data.data}"
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 1) -> List[TaskOutput]:
        """Generate translation predictions"""
        outputs = []
        
        prompts = [self.format_prompt(inp) for inp in inputs]
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            generations = model.generate(
                batch_prompts,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            for j, generation in enumerate(generations):
                outputs.append(TaskOutput(
                    predictions=generation,
                    metadata={'prompt_length': len(batch_prompts[j])}
                ))
            
        return outputs

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse translation output"""
        return TaskOutput(predictions=raw_output.strip())
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute translation metrics"""
        
        preds = [p.predictions for p in predictions]
        
        bleu_result = self.bleu_metric.compute(preds, references)
        rouge_result = self.rouge_metric.compute(preds, references)
        
        return {
            'bleu': bleu_result.score,
            'rougeL': rouge_result.score,
        }
