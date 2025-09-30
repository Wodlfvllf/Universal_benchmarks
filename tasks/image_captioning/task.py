
from typing import List, Dict, Any
from ..base import BaseTask, TaskInput, TaskOutput
from metrics.registry import MetricRegistry
from PIL import Image

class ImageCaptioningTask(BaseTask):
    """Implementation for image captioning tasks"""
    
    def setup(self):
        self.max_length = self.config.get('max_length', 128)
        self.bleu_metric = MetricRegistry.get_metric('bleu')
        self.rouge_metric = MetricRegistry.get_metric('rouge')
        
    def prepare_inputs(self, dataset: Any, image_column: str, caption_column: str) -> List[TaskInput]:
        """Prepare image captioning inputs"""
        inputs = []
        
        images = dataset[image_column]
        captions = dataset[caption_column]
        
        for i, image_path in enumerate(images):
            inputs.append(TaskInput(
                data=image_path,
                labels=captions[i],
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for image captioning prompt"""
        # For many multimodal models, the prompt is implicit in the task.
        # However, some models might take a text prompt along with the image.
        if self.config.get('prompt_template'):
            return self.config['prompt_template']
        return "Generate a caption for this image."
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 1) -> List[TaskOutput]:
        """Generate image captioning predictions"""
        outputs = []
        
        images = [Image.open(inp.data).convert("RGB") for inp in inputs]
        prompts = [self.format_prompt(inp) for inp in inputs]
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            
            # This assumes a multimodal model with a `generate_from_image` method.
            generations = model.generate_from_image(
                images=batch_images,
                prompts=batch_prompts,
                max_new_tokens=self.max_length
            )
            
            for generation in generations:
                outputs.append(self.parse_output(generation))
            
        return outputs

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse image captioning output"""
        return TaskOutput(predictions=raw_output.strip())
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute image captioning metrics"""
        
        preds = [p.predictions for p in predictions]
        
        bleu_result = self.bleu_metric.compute(preds, references)
        rouge_result = self.rouge_metric.compute(preds, references)
        
        return {
            'bleu': bleu_result.score,
            'rougeL': rouge_result.score,
        }
