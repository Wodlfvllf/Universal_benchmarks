from PIL import Image
import torch
from torchvision import transforms
from typing import List, Tuple, Optional, Union
import numpy as np

class ImageProcessor:
    """Image preprocessing utilities"""

    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 augment: bool = False):
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment

        # Build transform pipeline
        transform_list = []

        if augment:
            transform_list.extend([
                transforms.RandomResizedCrop(target_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ])
        else:
            transform_list.extend([
                transforms.Resize(target_size),
                transforms.CenterCrop(target_size)
            ])

        transform_list.append(transforms.ToTensor())

        if normalize:
            # ImageNet normalization
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            )

        self.transform = transforms.Compose(transform_list)

    def process(self, image_path: Union[str, Image.Image]) -> torch.Tensor:
        """Process single image"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path

        return self.transform(image)

    def batch_process(self, image_paths: List[Union[str, Image.Image]]) -> torch.Tensor:
        """Process batch of images"""
        processed = [self.process(img) for img in image_paths]
        return torch.stack(processed)

    def load_image(self, path: str) -> Image.Image:
        """Load image from path"""
        return Image.open(path).convert('RGB')

    def save_image(self, tensor: torch.Tensor, path: str):
        """Save tensor as image"""
        # Denormalize if needed
        if self.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = tensor * std + mean

        # Convert to PIL Image
        tensor = torch.clamp(tensor * 255, 0, 255).byte()
        array = tensor.permute(1, 2, 0).numpy()
        image = Image.fromarray(array)
        image.save(path)
