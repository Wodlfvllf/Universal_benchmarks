from typing import List, Dict, Any
from ..base import BaseTask, TaskInput, TaskOutput
from ...metrics.registry import MetricRegistry

class GenerativeQATask(BaseTask):
    """Implementation for generative question answering tasks"""
    
    def setup(self):
        self.max_length = self.config.get('max_length', 512)
        self.temperature = self.config.get('temperature', 0.7)
        self.top_p = self.config.get('top_p', 0.95)
        self.rouge_metric = MetricRegistry.get_metric('rouge')
        self.bleu_metric = MetricRegistry.get_metric('bleu')
        
    def prepare_inputs(self, dataset: Any, question_column: str, answer_column: str) -> List[TaskInput]:
        """Prepare generative QA inputs"""
        inputs = []
        
        questions = dataset[question_column]
        answers = dataset[answer_column]
        
        for i, question in enumerate(questions):
            inputs.append(TaskInput(
                data=question,
                labels=answers[i],
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for generative QA prompt"""
        if self.config.get('prompt_template'):
            return self.config['prompt_template'].format(question=input_data.data)
        return f"Question: {input_data.data}\n\nAnswer:"
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 1) -> List[TaskOutput]:
        """Generate generative QA predictions"""
        outputs = []
        
        prompts = [self.format_prompt(inp) for inp in inputs]
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            generations = model.generate(
                batch_prompts,
                max_new_tokens=self.config.get('max_new_tokens', 128)
            )
            
            for j, generation in enumerate(generations):
                outputs.append(self.parse_output(generation))
            
        return outputs

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse generative QA output"""
        return TaskOutput(predictions=raw_output.strip())
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute generative QA metrics"""
        
        preds = [p.predictions for p in predictions]
        
        rouge_result = self.rouge_metric.compute(preds, references)
        bleu_result = self.bleu_metric.compute(preds, references)
        
        return {
            'rougeL': rouge_result.score,
            'bleu': bleu_result.score,
        }