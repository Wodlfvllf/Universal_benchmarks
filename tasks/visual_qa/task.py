
from typing import List, Dict, Any
from ..base import BaseTask, TaskInput, TaskOutput
from ...metrics.registry import MetricRegistry
from PIL import Image

class VisualQATask(BaseTask):
    """Implementation for visual question answering tasks"""
    
    def setup(self):
        self.max_length = self.config.get('max_length', 128)
        self.exact_match_metric = MetricRegistry.get_metric('exact_match')
        
    def prepare_inputs(self, dataset: Any, image_column: str, question_column: str, answer_column: str) -> List[TaskInput]:
        """Prepare visual QA inputs"""
        inputs = []
        
        images = dataset[image_column]
        questions = dataset[question_column]
        answers = dataset[answer_column]
        
        for i, image_path in enumerate(images):
            inputs.append(TaskInput(
                data={
                    'image': image_path,
                    'question': questions[i]
                },
                labels=answers[i],
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for visual QA prompt"""
        return input_data.data['question']
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 8) -> List[TaskOutput]:
        """Generate visual QA predictions"""
        outputs = []
        
        images = [Image.open(inp.data['image']).convert("RGB") for inp in inputs]
        questions = [self.format_prompt(inp) for inp in inputs]
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_questions = questions[i:i+batch_size]
            
            # This assumes a multimodal model with a `visual_question_answering` method.
            generations = model.visual_question_answering(
                images=batch_images,
                questions=batch_questions,
                max_new_tokens=self.max_length
            )
            
            for generation in generations:
                outputs.append(self.parse_output(generation))
            
        return outputs

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse visual QA output"""
        return TaskOutput(predictions=raw_output.strip())
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute visual QA metrics"""
        
        preds = [p.predictions for p in predictions]
        
        # In VQA, often there are multiple correct answers.
        # Here we take the first one for simplicity.
        refs = [ref[0] if isinstance(ref, list) else ref for ref in references]

        em_result = self.exact_match_metric.compute(preds, refs)
        
        return {
            'exact_match': em_result.score,
        }
