from typing import List, Dict, Any
import re
from ..base import BaseTask, TaskInput, TaskOutput
from ...metrics.registry import MetricRegistry

class MathReasoningTask(BaseTask):
    """Implementation for math reasoning tasks"""
    
    def setup(self):
        self.max_length = self.config.get('max_length', 1024)
        self.exact_match_metric = MetricRegistry.get_metric('exact_match', normalize=False) # Normalization is handled here
        
    def prepare_inputs(self, dataset: Any, question_column: str, answer_column: str) -> List[TaskInput]:
        """Prepare math reasoning inputs"""
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
        """Format for math reasoning prompt"""
        if self.config.get('prompt_template'):
            return self.config['prompt_template'].format(question=input_data.data)
        return f"Solve the following math problem:\n\n{input_data.data}\n\nGive the final answer after \"####\"."
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 1) -> List[TaskOutput]:
        """Generate math reasoning predictions"""
        outputs = []
        
        prompts = [self.format_prompt(inp) for inp in inputs]
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            generations = model.generate(
                batch_prompts,
                max_new_tokens=self.max_length
            )
            
            for j, generation in enumerate(generations):
                outputs.append(self.parse_output(generation))
            
        return outputs

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse math reasoning output to extract the final answer."""
        match = re.search(r"#### (.*)", raw_output)
        if match:
            answer = match.group(1).strip()
        else:
            answer = raw_output.strip()
            
        return TaskOutput(predictions=answer)
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize the answer for comparison."""
        # Remove commas, dollar signs, etc.
        answer = re.sub(r"[$,]", "", answer)
        # Convert to float if possible
        try:
            answer = str(float(answer))
        except ValueError:
            pass
        return answer

    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute math reasoning metrics"""
        
        preds = [self.normalize_answer(p.predictions) for p in predictions]
        refs = [self.normalize_answer(r) for r in references]
        
        em_result = self.exact_match_metric.compute(preds, refs)
        
        return {
            'exact_match': em_result.score,
        }