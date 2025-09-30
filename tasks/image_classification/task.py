from typing import List, Dict, Any
from ..base import BaseTask, TaskInput, TaskOutput
from metrics.registry import MetricRegistry
from PIL import Image
from sklearn.preprocessing import LabelEncoder

class ImageClassificationTask(BaseTask):
    """Implementation for image classification tasks"""
    
    def setup(self):
        self.accuracy_metric = MetricRegistry.get_metric('accuracy')
        self.label_encoder = LabelEncoder()
        
    def prepare_inputs(self, dataset: Any, image_column: str, label_column: str) -> List[TaskInput]:
        """Prepare image classification inputs"""
        inputs = []
        
        images = dataset[image_column]
        labels = dataset[label_column]
        
        # Fit label encoder
        self.label_encoder.fit(labels)
        
        for i, image_path in enumerate(images):
            inputs.append(TaskInput(
                data=image_path,
                labels=self.label_encoder.transform([labels[i]])[0],
                metadata={'index': i}
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format for image classification prompt"""
        # Usually not needed for image classification models that have a dedicated classifier head.
        return ""
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 32) -> List[TaskOutput]:
        """Generate image classification predictions"""
        outputs = []
        
        images = [Image.open(inp.data).convert("RGB") for inp in inputs]
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            # This assumes a vision model with a `classify` method.
            # The classify method should return predictions as integer labels.
            predictions = model.classify(images=batch_images)
            
            for pred in predictions:
                outputs.append(TaskOutput(predictions=pred))
            
        return outputs

    def parse_output(self, raw_output: Any) -> TaskOutput:
        """Parse image classification output"""
        # The model.classify method should ideally return structured output.
        # If it returns raw logits, this method would need to perform argmax.
        return TaskOutput(predictions=raw_output)
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute image classification metrics"""
        
        preds = [p.predictions for p in predictions]
        
        accuracy_result = self.accuracy_metric.compute(preds, references)
        
        return {
            'accuracy': accuracy_result.score,
        }