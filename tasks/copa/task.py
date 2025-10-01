from typing import List, Dict, Any
from tasks.base import BaseTask, TaskInput, TaskOutput

class CopaTask(BaseTask):
    """Implementation for the COPA task"""
    
    def setup(self):
        pass
        
    def prepare_inputs(self, dataset: Any, 
                      input_columns: List[str], 
                      label_column: str = None) -> List[TaskInput]:
        """Prepare COPA inputs"""
        inputs = []
        
        if len(input_columns) != 3:
            raise ValueError("COPA requires three input columns (premise, choice1, choice2)")

        premises = dataset[input_columns[0]]
        choices1 = dataset[input_columns[1]]
        choices2 = dataset[input_columns[2]]
        
        labels = None
        if label_column and label_column in dataset.column_names:
            labels = dataset[label_column]
        
        for i in range(len(premises)):
            inputs.append(TaskInput(
                data={
                    'premise': premises[i],
                    'choices': [choices1[i], choices2[i]]
                },
                labels=labels[i] if labels else None,
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for zero/few-shot COPA"""
        data = input_data.data
        return f"{data['premise']}\nquestion: what is the cause or effect?\nchoice1: {data['choices'][0]}\nchoice2: {data['choices'][1]}"
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 32) -> List[TaskOutput]:
        """Generate COPA predictions"""
        outputs = []
        
        prompts = [self.format_prompt(inp) for inp in inputs]
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            generations = model.generate(
                batch_prompts,
                max_new_tokens=self.config.get('max_new_tokens', 20)
            )
            
            for j, generation in enumerate(generations):
                outputs.append(self.parse_output(generation, inputs[i+j].data['choices']))
            
        return outputs

    def parse_output(self, raw_output: str, choices: List[str]) -> TaskOutput:
        """Parse COPA output"""
        output = raw_output.strip().lower()
        
        # Simple text matching
        if choices[0].lower() in output:
            return TaskOutput(predictions=0)
        elif choices[1].lower() in output:
            return TaskOutput(predictions=1)
        else:
            # Check for choice index
            if '1' in output or 'one' in output:
                return TaskOutput(predictions=0)
            elif '2' in output or 'two' in output:
                return TaskOutput(predictions=1)
            else:
                return TaskOutput(predictions=0) # Default to first choice
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute COPA metrics"""
        from sklearn.metrics import accuracy_score
        
        preds = [p.predictions for p in predictions]
        
        metrics = {
            'accuracy': accuracy_score(references, preds)
        }
        
        return metrics
