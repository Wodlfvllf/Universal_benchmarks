from typing import List, Dict, Any
from tasks.base import BaseTask, TaskInput, TaskOutput
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

class SentenceSimilarityTask(BaseTask):
    """Implementation for sentence similarity tasks"""
    
    def setup(self):
        self.model = None
        
    def prepare_inputs(self, dataset: Any, 
                      input_columns: List[str], 
                      label_column: str = None) -> List[TaskInput]:
        """Prepare sentence similarity inputs"""
        inputs = []
        
        if len(input_columns) != 2:
            raise ValueError("Sentence similarity requires two input columns")

        sentence1 = dataset[input_columns[0]]
        sentence2 = dataset[input_columns[1]]
        
        labels = None
        if label_column and label_column in dataset.column_names:
            labels = dataset[label_column]
        
        for i in range(len(sentence1)):
            inputs.append(TaskInput(
                data=[sentence1[i], sentence2[i]],
                labels=labels[i] if labels else None,
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Sentence similarity does not use prompts"""
        return ""
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 32) -> List[TaskOutput]:
        """Generate sentence similarity predictions"""
        if not hasattr(model, 'model') or not isinstance(model.model, SentenceTransformer):
            raise ValueError("SentenceSimilarityTask requires a SentenceTransformer model")

        sentence_model = model.model
        outputs = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            
            sentences1 = [inp.data[0] for inp in batch]
            sentences2 = [inp.data[1] for inp in batch]
            
            embeddings1 = sentence_model.encode(sentences1, convert_to_tensor=True)
            embeddings2 = sentence_model.encode(sentences2, convert_to_tensor=True)
            
            similarities = cos_sim(embeddings1, embeddings2)
            
            for j in range(len(batch)):
                similarity = similarities[j][j].item()
                # Scale similarity from [-1, 1] to [0, 5] for stsb
                prediction = (similarity + 1) * 2.5
                outputs.append(TaskOutput(predictions=prediction))
            
        return outputs
    
    def parse_output(self, raw_output: str) -> TaskOutput:
        """Sentence similarity does not use parsing"""
        return TaskOutput(predictions=0.0)
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute sentence similarity metrics"""
        from scipy.stats import pearsonr, spearmanr
        
        preds = [p.predictions for p in predictions]
        
        pearson_corr, _ = pearsonr(references, preds)
        spearman_corr, _ = spearmanr(references, preds)
        
        metrics = {
            'pearson': pearson_corr,
            'spearman': spearman_corr
        }
        
        return metrics
