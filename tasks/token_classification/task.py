from typing import List, Dict, Any
from ..base import BaseTask, TaskInput, TaskOutput
from ...metrics.registry import MetricRegistry
from seqeval.metrics import f1_score, precision_score, recall_score

class TokenClassificationTask(BaseTask):
    """Implementation for token classification tasks (e.g., NER, POS tagging)"""
    
    def setup(self):
        self.max_length = self.config.get('max_length', 512)
        
    def prepare_inputs(self, dataset: Any, tokens_column: str, tags_column: str) -> List[TaskInput]:
        """Prepare token classification inputs"""
        inputs = []
        
        tokens_list = dataset[tokens_column]
        tags_list = dataset[tags_column]
        
        for i, tokens in enumerate(tokens_list):
            inputs.append(TaskInput(
                data=tokens,
                labels=tags_list[i],
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for token classification prompt"""
        if self.config.get('prompt_template'):
            return self.config['prompt_template'].format(tokens=' '.join(input_data.data))
        return f"Perform named entity recognition on the following text: {' '.join(input_data.data)}"
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 16) -> List[TaskOutput]:
        """Generate token classification predictions"""
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
        """Parse token classification output"""
        # This is a simplified parsing. A real implementation would need a more robust parser.
        # This assumes the model outputs a list of (token, label) pairs.
        try:
            # Example format: "[('John', 'B-PER'), ('Doe', 'I-PER')]"
            parsed = eval(raw_output)
            if isinstance(parsed, list):
                predictions = [tag for token, tag in parsed]
                return TaskOutput(predictions=predictions)
        except:
            pass
        return TaskOutput(predictions=[])
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute token classification metrics"""
        
        preds = [p.predictions for p in predictions]
        
        # Ensure predictions and references have the same length
        for i in range(len(preds)):
            if len(preds[i]) != len(references[i]):
                preds[i] = (preds[i] + ['O'] * len(references[i]))[:len(references[i])]

        precision = precision_score(references, preds)
        recall = recall_score(references, preds)
        f1 = f1_score(references, preds)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }