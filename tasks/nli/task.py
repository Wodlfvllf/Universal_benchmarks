from typing import List, Dict, Any
from tasks.base import BaseTask, TaskInput, TaskOutput
from sklearn.preprocessing import LabelEncoder

class NLITask(BaseTask):
    """Implementation for Natural Language Inference tasks"""
    
    def setup(self):
        self.label_encoder = LabelEncoder()
        self.label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        self.num_labels = 3
        
    def prepare_inputs(self, dataset: Any, 
                      input_columns: List[str], 
                      label_column: str = None) -> List[TaskInput]:
        """Prepare NLI inputs"""
        inputs = []
        
        if len(input_columns) != 2:
            raise ValueError("NLI requires two input columns (premise and hypothesis)")

        premise = dataset[input_columns[0]]
        hypothesis = dataset[input_columns[1]]
        
        labels = None
        if label_column and label_column in dataset.column_names:
            labels = dataset[label_column]
            labels = [self.label_map[label] for label in labels]
        
        for i in range(len(premise)):
            inputs.append(TaskInput(
                data=[premise[i], hypothesis[i]],
                labels=labels[i] if labels else None,
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for zero/few-shot NLI"""
        if self.config.get('prompt_template'):
            return self.config['prompt_template'].format(premise=input_data.data[0], hypothesis=input_data.data[1])
        return f"premise: {input_data.data[0]}\nhypothesis: {input_data.data[1]}"
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 32) -> List[TaskOutput]:
        """Generate NLI predictions"""
        outputs = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            
            if hasattr(model, 'classify'):
                batch_outputs = model.classify(
                    [inp.data for inp in batch],
                    labels=list(self.label_map.keys())
                )
            else:
                prompts = [self.format_prompt(inp) for inp in batch]
                raw_outputs = model.generate(prompts)
                batch_outputs = [self.parse_output(out) for out in raw_outputs]
            
            outputs.extend(batch_outputs)
            
        return outputs
    
    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse NLI output"""
        predicted_label = raw_output.strip().lower()
        
        best_match = None
        for label in self.label_map.keys():
            if str(label).lower() in predicted_label:
                best_match = label
                break
        
        if best_match:
            prediction = self.label_map[best_match]
        else:
            prediction = 1  # Default to neutral
            
        return TaskOutput(predictions=prediction)
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute NLI metrics"""
        from sklearn.metrics import accuracy_score, f1_score
        
        preds = [p.predictions for p in predictions]
        
        metrics = {
            'accuracy': accuracy_score(references, preds),
            'f1_macro': f1_score(references, preds, average='macro')
        }
        
        return metrics