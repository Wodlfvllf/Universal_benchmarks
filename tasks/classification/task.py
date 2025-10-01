import numpy as np
from typing import List, Dict, Any, Optional
from tasks.base import BaseTask, TaskInput, TaskOutput
from sklearn.preprocessing import LabelEncoder

class ClassificationTask(BaseTask):
    """Implementation for text classification tasks"""
    
    def setup(self):
        self.label_encoder = LabelEncoder()
        self.num_labels = None
        self.label_map = {}
        
    def prepare_inputs(self, dataset: Any, 
                      input_columns: List[str], 
                      label_column: str = None) -> List[TaskInput]:
        """Prepare classification inputs"""
        inputs = []
        
        # Handle single or multiple input columns
        if len(input_columns) == 1:
            texts = dataset[input_columns[0]]
        else:
            # Concatenate multiple columns
            texts = [" ".join([row[col] for col in input_columns]) 
                    for row in dataset]
        
        # Encode labels if present
        labels = None
        if label_column and label_column in dataset.column_names:
            labels = dataset[label_column]
            if not self.label_map:
                unique_labels = list(set(labels))
                self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
                self.num_labels = len(unique_labels)
            labels = [self.label_map[label] for label in labels]
        
        # Create TaskInput objects
        for i, text in enumerate(texts):
            inputs.append(TaskInput(
                data=text,
                labels=labels[i] if labels else None,
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for zero/few-shot classification"""
        if self.config.get('prompt_template'):
            return self.config['prompt_template'].format(text=input_data.data)
        return input_data.data
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 32) -> List[TaskOutput]:
        """Generate classification predictions"""
        outputs = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            
            if hasattr(model, 'classify'):
                # Direct classification method
                batch_outputs = model.classify(
                    [inp.data for inp in batch],
                    labels=list(self.label_map.keys())
                )
            else:
                # Use generation and parse
                prompts = [self.format_prompt(inp) for inp in batch]
                raw_outputs = model.generate(prompts)
                batch_outputs = [self.parse_output(out) for out in raw_outputs]
            
            outputs.extend(batch_outputs)
            
        return outputs
    
    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse classification output"""
        # Try to extract label from generated text
        predicted_label = raw_output.strip().lower()
        
        # Map to known labels
        best_match = None
        for label in self.label_map.keys():
            if str(label).lower() in predicted_label:
                best_match = label
                break
        
        if best_match:
            prediction = self.label_map[best_match]
        else:
            # Default to first label if no match
            prediction = 0
            
        return TaskOutput(predictions=prediction)
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute classification metrics"""
        from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
        
        preds = [p.predictions for p in predictions]
        
        metrics = {
            'accuracy': accuracy_score(references, preds)
        }
        
        if self.num_labels == 2:
            # Binary classification
            metrics['f1'] = f1_score(references, preds, average='binary')
            precision, recall, f1, _ = precision_recall_fscore_support(
                references, preds, average='binary'
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
        else:
            # Multi-class
            metrics['f1_macro'] = f1_score(references, preds, average='macro')
            metrics['f1_weighted'] = f1_score(references, preds, average='weighted')
            
        return metrics