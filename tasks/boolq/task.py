from typing import List, Dict, Any
from tasks.base import BaseTask, TaskInput, TaskOutput

class BoolQTask(BaseTask):
    """Implementation for the BoolQ task"""
    
    def setup(self):
        self.choice_labels = ['False', 'True']
        
    def prepare_inputs(self, dataset: Any, 
                      input_columns: List[str], 
                      label_column: str = None) -> List[TaskInput]:
        """Prepare BoolQ inputs"""
        inputs = []
        
        if len(input_columns) != 2:
            raise ValueError("BoolQ requires two input columns (question and passage)")

        questions = dataset[input_columns[0]]
        passages = dataset[input_columns[1]]
        
        labels = None
        if label_column and label_column in dataset.column_names:
            labels = dataset[label_column]
        
        for i in range(len(questions)):
            inputs.append(TaskInput(
                data={
                    'question': questions[i],
                    'passage': passages[i]
                },
                labels=labels[i] if labels else None,
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for zero/few-shot BoolQ"""
        data = input_data.data
        return f"{data['passage']}\nquestion: {data['question']}\ntrue or false?"
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 32) -> List[TaskOutput]:
        """Generate BoolQ predictions"""
        outputs = []
        
        if hasattr(model, 'classify'):
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i+batch_size]
                batch_outputs = model.classify(
                    [self.format_prompt(inp) for inp in batch],
                    labels=self.choice_labels
                )
                outputs.extend(batch_outputs)
        else:
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
        """Parse BoolQ output"""
        output = raw_output.strip().lower()
        
        if 'true' in output:
            return TaskOutput(predictions=1)
        elif 'false' in output:
            return TaskOutput(predictions=0)
        else:
            return TaskOutput(predictions=0) # Default to false
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute BoolQ metrics"""
        from sklearn.metrics import accuracy_score
        
        preds = [p.predictions for p in predictions]
        
        metrics = {
            'accuracy': accuracy_score(references, preds)
        }
        
        return metrics
