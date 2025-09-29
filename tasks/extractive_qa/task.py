from typing import List, Dict, Any
from ..base import BaseTask, TaskInput, TaskOutput
from ...metrics.registry import MetricRegistry

class ExtractiveQATask(BaseTask):
    """Implementation for extractive question answering tasks"""
    
    def setup(self):
        self.max_length = self.config.get('max_length', 512)
        self.exact_match_metric = MetricRegistry.get_metric('exact_match')
        self.f1_qa_metric = MetricRegistry.get_metric('f1_qa')
        
    def prepare_inputs(self, dataset: Any, question_column: str, context_column: str, answer_column: str) -> List[TaskInput]:
        """Prepare extractive QA inputs"""
        inputs = []
        
        questions = dataset[question_column]
        contexts = dataset[context_column]
        answers = dataset[answer_column]
        
        for i, question in enumerate(questions):
            inputs.append(TaskInput(
                data={
                    'question': question,
                    'context': contexts[i]
                },
                labels=answers[i]['text'][0],
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for extractive QA prompt"""
        if self.config.get('prompt_template'):
            return self.config['prompt_template'].format(
                question=input_data.data['question'], 
                context=input_data.data['context']
            )
        return f"Answer the following question based on the context provided.\n\nContext: {input_data.data['context']}\n\nQuestion: {input_data.data['question']}\n\nAnswer:"
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 1) -> List[TaskOutput]:
        """Generate extractive QA predictions"""
        outputs = []
        
        prompts = [self.format_prompt(inp) for inp in inputs]
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            generations = model.generate(
                batch_prompts,
                max_new_tokens=self.config.get('max_new_tokens', 64)
            )
            
            for j, generation in enumerate(generations):
                outputs.append(self.parse_output(generation))
            
        return outputs

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse extractive QA output"""
        return TaskOutput(predictions=raw_output.strip())
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute extractive QA metrics"""
        
        preds = [p.predictions for p in predictions]
        
        em_result = self.exact_match_metric.compute(preds, references)
        f1_result = self.f1_qa_metric.compute(preds, references)
        
        return {
            'exact_match': em_result.score,
            'f1': f1_result.score,
        }
