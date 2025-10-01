from typing import List, Dict, Any
from tasks.base import BaseTask, TaskInput, TaskOutput

class RegressionTask(BaseTask):
    """Implementation for regression tasks"""
    
    def setup(self):
        pass
        
    def prepare_inputs(self, dataset: Any, 
                      input_columns: List[str], 
                      label_column: str = None) -> List[TaskInput]:
        """Prepare regression inputs"""
        inputs = []
        
        # Handle single or multiple input columns
        if len(input_columns) == 1:
            texts = dataset[input_columns[0]]
        else:
            # Concatenate multiple columns
            texts = [" ".join([row[col] for col in input_columns]) 
                    for row in dataset]
        
        labels = None
        if label_column and label_column in dataset.column_names:
            labels = dataset[label_column]
        
        # Create TaskInput objects
        for i, text in enumerate(texts):
            inputs.append(TaskInput(
                data=text,
                labels=labels[i] if labels else None,
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for zero/few-shot regression"""
        if self.config.get('prompt_template'):
            return self.config['prompt_template'].format(text=input_data.data)
        return input_data.data
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 32) -> List[TaskOutput]:
        """Generate regression predictions"""
        outputs = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            
            if hasattr(model, 'classify'):
                batch_outputs = model.classify(
                    [inp.data for inp in batch]
                )
            else:
                prompts = [self.format_prompt(inp) for inp in batch]
                raw_outputs = model.generate(prompts)
                batch_outputs = [self.parse_output(out) for out in raw_outputs]
            
            outputs.extend(batch_outputs)
            
        return outputs
    
    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse regression output"""
        try:
            prediction = float(raw_output.strip())
        except ValueError:
            prediction = 0.0
            
        return TaskOutput(predictions=prediction)
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute regression metrics"""
        from scipy.stats import pearsonr, spearmanr
        
        preds = [p.predictions for p in predictions]
        
        pearson_corr, _ = pearsonr(references, preds)
        spearman_corr, _ = spearmanr(references, preds)
        
        metrics = {
            'pearson': pearson_corr,
            'spearman': spearman_corr
        }
        
        return metrics