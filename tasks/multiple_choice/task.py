from typing import List, Dict, Any
from ..base import BaseTask, TaskInput, TaskOutput
from ...metrics.registry import MetricRegistry

class MultipleChoiceQATask(BaseTask):
    """Implementation for multiple choice question answering"""
    
    def setup(self):
        self.choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.accuracy_metric = MetricRegistry.get_metric('accuracy')
        
    def prepare_inputs(self, dataset: Any, question_column: str, choices_column: str, answer_column: str) -> List[TaskInput]:
        """Prepare multiple choice inputs"""
        inputs = []
        
        questions = dataset[question_column]
        choices_list = dataset[choices_column]
        answers = dataset[answer_column]
        
        for i, question in enumerate(questions):
            inputs.append(TaskInput(
                data={
                    'question': question,
                    'choices': choices_list[i]
                },
                labels=answers[i],
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format multiple choice prompt"""
        data = input_data.data
        
        prompt_parts = []
        prompt_parts.append(f"Question: {data['question']}\n")
        
        prompt_parts.append("Choices:\n")
        for i, choice in enumerate(data['choices']):
            prompt_parts.append(f"{self.choice_labels[i]}) {choice}\n")
            
        prompt_parts.append("\nAnswer (A/B/C/D/...): ")
        
        return "".join(prompt_parts)
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 1) -> List[TaskOutput]:
        """Generate multiple choice predictions"""
        outputs = []
        
        prompts = [self.format_prompt(inp) for inp in inputs]
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            generations = model.generate(
                batch_prompts,
                max_new_tokens=self.config.get('max_new_tokens', 5)
            )
            
            for j, generation in enumerate(generations):
                outputs.append(self.parse_output(generation))
            
        return outputs

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Extract choice from output"""
        output = raw_output.strip().upper()
        
        for i, label in enumerate(self.choice_labels):
            if label in output:
                return TaskOutput(predictions=i)
                
        return TaskOutput(predictions=0) # Default to first choice
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute multiple choice metrics"""
        
        preds = [p.predictions for p in predictions]
        
        accuracy_result = self.accuracy_metric.compute(preds, references)
        
        return {
            'accuracy': accuracy_result.score,
        }
