from PIL import Image
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
from ..interfaces.base import BaseModel, ModelConfig, ModelOutput

class MultimodalModelAdapter(BaseModel):
    """Adapter for vision-language models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.image_processor = None
        
    def setup(self):
        """Initialize multimodal model"""
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            torch_dtype=self._get_torch_dtype(),
            device_map="auto" if self.config.device == "cuda" else None
        )
        
        self.model.eval()

    def _get_torch_dtype(self):
        """Get torch dtype from config"""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8
        }
        return dtype_map.get(self.config.precision, torch.float32)
        
    def process_image(self, image_path: Union[str, Image.Image]) -> Image.Image:
        """Process image input"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        return image
        
    def generate_from_image(self,
                           images: Union[str, Image.Image, List],
                           prompts: Union[str, List[str]],
                           max_new_tokens: int = 512,
                           **kwargs) -> Union[str, List[str]]:
        """Generate text from image and text inputs"""
        
        # Handle single inputs
        single_input = isinstance(images, (str, Image.Image))
        if single_input:
            images = [images]
            prompts = [prompts] if isinstance(prompts, str) else prompts
            
        # Process images
        processed_images = [self.process_image(img) for img in images]
        
        # Prepare inputs
        inputs = self.processor(
            text=prompts,
            images=processed_images,
            return_tensors="pt",
            padding=True
        )
        
        if self.config.device == "cuda":
            inputs = {k: v.cuda() if torch.is_tensor(v) else v 
                     for k, v in inputs.items()}
            
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            
        # Decode
        generated_texts = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        return generated_texts[0] if single_input else generated_texts
        
    def visual_question_answering(self,
                                  images: Union[str, Image.Image, List],
                                  questions: Union[str, List[str]],
                                  **kwargs) -> Union[str, List[str]]:
        """Perform VQA"""
        # Format as VQA prompt
        if isinstance(questions, str):
            vqa_prompt = f"Question: {questions}\nAnswer:"
        else:
            vqa_prompt = [f"Question: {q}\nAnswer:" for q in questions]
            
        return self.generate_from_image(images, vqa_prompt, **kwargs)
        
    def caption_image(self,
                     images: Union[str, Image.Image, List],
                     **kwargs) -> Union[str, List[str]]:
        """Generate image captions"""
        prompt = "Generate a detailed caption for this image:"
        
        single_input = isinstance(images, (str, Image.Image))
        if not single_input:
            prompt = [prompt] * len(images)
            
        return self.generate_from_image(images, prompt, **kwargs)

    def generate(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        raise NotImplementedError("Use generate_from_image for multimodal models")

    def classify(self, inputs: Union[str, List[str]], labels: Optional[List[str]] = None, **kwargs) -> ModelOutput:
        raise NotImplementedError("Classification is not directly supported for this model type, use VQA.")

    def embed(self, inputs: Union[str, List[str]], **kwargs) -> np.ndarray:
        raise NotImplementedError("Embedding is not directly supported for this model type.")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.config.model_name,
            "type": "multimodal",
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.model.device),
            "precision": self.config.precision,
            "max_length": self.config.max_length
        }