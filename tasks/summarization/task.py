
import numpy as np
from typing import List, Dict, Any
from ..base import BaseTask, TaskInput, TaskOutput
from metrics.registry import MetricRegistry

class SummarizationTask(BaseTask):
    """Implementation for summarization tasks"""
    
    def setup(self):
        self.max_length = self.config.get('max_length', 512)
        self.temperature = self.config.get('temperature', 0.7)
        self.top_p = self.config.get('top_p', 0.95)
        self.rouge_metric = MetricRegistry.get_metric('rouge')
        
    def prepare_inputs(self, dataset: Any, input_column: str, label_column: str) -> List[TaskInput]:
        """Prepare summarization inputs"""
        inputs = []
        
        texts = dataset[input_column]
        summaries = dataset[label_column]
        
        for i, text in enumerate(texts):
            inputs.append(TaskInput(
                data=text,
                labels=summaries[i],
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for summarization prompt"""
        if self.config.get('prompt_template'):
            return self.config['prompt_template'].format(text=input_data.data)
        return f"Summarize the following text:\n\n{input_data.data}"
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 1) -> List[TaskOutput]:
        """Generate summarization predictions"""
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
        """Parse summarization output"""
        return TaskOutput(predictions=raw_output.strip())
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute summarization metrics"""
        
        preds = [p.predictions for p in predictions]
        
        rouge_result = self.rouge_metric.compute(preds, references)
        
        return {
            'rouge1': rouge_result.additional_metrics['rouge1'],
            'rouge2': rouge_result.additional_metrics['rouge2'],
            'rougeL': rouge_result.score,
        }
