from typing import List, Dict, Any
from tasks.base import BaseTask, TaskInput, TaskOutput

class ExtractiveQATask(BaseTask):
    """Implementation for extractive question answering tasks"""
    
    def setup(self):
        pass
        
    def prepare_inputs(self, dataset: Any, 
                      input_columns: List[str], 
                      label_column: str = None) -> List[TaskInput]:
        """Prepare extractive QA inputs"""
        inputs = []
        
        if len(input_columns) != 2:
            raise ValueError("Extractive QA requires two input columns (context and question)")

        contexts = dataset[input_columns[0]]
        questions = dataset[input_columns[1]]
        
        labels = None
        if label_column and label_column in dataset.column_names:
            labels = dataset[label_column]
        
        for i in range(len(contexts)):
            inputs.append(TaskInput(
                data=[contexts[i], questions[i]],
                labels=labels[i] if labels else None,
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for zero/few-shot extractive QA"""
        if self.config.get('prompt_template'):
            return self.config['prompt_template'].format(context=input_data.data[0], question=input_data.data[1])
        return f"Answer the question based on the context.\nContext: {input_data.data[0]}\nQuestion: {input_data.data[1]}\nAnswer:"
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 32) -> List[TaskOutput]:
        """Generate extractive QA predictions"""
        outputs = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            
            prompts = [self.format_prompt(inp) for inp in batch]
            raw_outputs = model.generate(prompts)
            batch_outputs = [self.parse_output(out) for out in raw_outputs]
            
            outputs.extend(batch_outputs)
            
        return outputs
    
    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse extractive QA output"""
        return TaskOutput(predictions=raw_output.strip())
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute extractive QA metrics"""
        from metrics.question_answering import ExactMatchMetric, F1ScoreQAMetric
        
        preds = [p.predictions for p in predictions]
        refs = [ref['text'] for ref in references]

        exact_match_metric = ExactMatchMetric()
        f1_qa_metric = F1ScoreQAMetric()

        exact_match = exact_match_metric.compute(preds, refs).score
        f1 = f1_qa_metric.compute(preds, refs).score
        
        metrics = {
            'exact_match': exact_match,
            'f1': f1
        }
        
        return metrics