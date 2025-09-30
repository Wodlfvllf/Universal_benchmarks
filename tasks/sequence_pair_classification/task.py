from typing import List, Dict, Any
from tasks.base import BaseTask, TaskInput, TaskOutput
from sklearn.preprocessing import LabelEncoder

class SequencePairClassificationTask(BaseTask):
    """Implementation for sequence pair classification tasks"""
    
    def setup(self):
        self.label_encoder = LabelEncoder()
        self.num_labels = None
        self.label_map = {}
        
    def prepare_inputs(self, dataset: Any, 
                      input_columns: List[str], 
                      label_column: str = None) -> List[TaskInput]:
        """Prepare sequence pair classification inputs"""
        inputs = []
        
        sentence1 = dataset[input_columns[0]]
        sentence2 = dataset[input_columns[1]]
        
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
        for i in range(len(sentence1)):
            inputs.append(TaskInput(
                data={'sentence1': sentence1[i], 'sentence2': sentence2[i]},
                labels=labels[i] if labels else None,
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for zero/few-shot sequence pair classification"""
        if self.config.get('prompt_template'):
            return self.config['prompt_template'].format(
                sentence1=input_data.data['sentence1'], 
                sentence2=input_data.data['sentence2']
            )
        return f"Sentence 1: {input_data.data['sentence1']}\nSentence 2: {input_data.data['sentence2']}"
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 32) -> List[TaskOutput]:
        """Generate classification predictions"""
        outputs = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            
            batch_outputs = model.classify(
                [(inp.data['sentence1'], inp.data['sentence2']) for inp in batch],
                candidate_labels=list(self.label_map.keys())
            )
            
            outputs.extend(batch_outputs)
            
        return outputs
    
    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse classification output"""
        predicted_label = raw_output.strip().lower()
        
        best_match = None
        for label in self.label_map.keys():
            if str(label).lower() in predicted_label:
                best_match = label
                break
        
        if best_match:
            prediction = self.label_map[best_match]
        else:
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
            metrics['f1'] = f1_score(references, preds, average='binary')
            precision, recall, f1, _ = precision_recall_fscore_support(
                references, preds, average='binary'
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
        else:
            metrics['f1_macro'] = f1_score(references, preds, average='macro')
            metrics['f1_weighted'] = f1_score(references, preds, average='weighted')
            
        return metrics
