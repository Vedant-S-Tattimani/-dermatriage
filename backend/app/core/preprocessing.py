"""
Image preprocessing utilities for EfficientNet-B0 inference.
"""
import io
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.config import settings


def get_transform(augment: bool = False) -> transforms.Compose:
    """Return the preprocessing pipeline."""
    ops = []
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    ops += [
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=settings.IMAGE_MEAN, std=settings.IMAGE_STD),
    ]
    return transforms.Compose(ops)


_inference_transform = get_transform(augment=False)


def load_image(source: Union[str, Path, bytes]) -> Image.Image:
    """Load a PIL Image from a file path or raw bytes."""
    if isinstance(source, (str, Path)):
        img = Image.open(source).convert("RGB")
    else:
        img = Image.open(io.BytesIO(source)).convert("RGB")
    return img


def apply_clahe(img: Image.Image) -> Image.Image:
    """Apply Contrast Limited Adaptive Histogram Equalization to the image."""
    # Convert PIL to OpenCV (numpy)
    img_np = np.array(img.convert("RGB"))
    # Convert to LAB color space
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(final_img)

def resize_and_pad(img: Image.Image, size: int) -> Image.Image:
    """Resize image to fit within 'size' while maintaining aspect ratio, then pad to square."""
    w, h = img.size
    ratio = float(size) / max(h, w)
    new_size = tuple([int(x * ratio) for x in (w, h)])
    img = img.resize(new_size, Image.LANCZOS)
    
    # Create new square image with black padding
    new_img = Image.new("RGB", (size, size), (0, 0, 0))
    # Paste resized image in the center
    new_img.paste(img, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return new_img

def preprocess(source: Union[str, Path, bytes, Image.Image]) -> torch.Tensor:
    """
    Preprocess an image for model inference.

    Returns:
        Tensor of shape (1, 3, IMAGE_SIZE, IMAGE_SIZE).
    """
    if isinstance(source, Image.Image):
        img = source.convert("RGB")
    else:
        img = load_image(source)
    
    # 1. Apply CLAHE for robust illumination normalization
    img = apply_clahe(img)
    
    # 2. Adaptive resizing with padding to prevent distortion
    img = resize_and_pad(img, settings.IMAGE_SIZE)
    
    # 3. Final transforms (ToTensor, Normalize) - bypassing the internal resize
    # We define a specific transform for this to avoid redundant resizing
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=settings.IMAGE_MEAN, std=settings.IMAGE_STD),
    ])
    
    tensor = final_transform(img)
    return tensor.unsqueeze(0)           # (1, 3, H, W)

import cv2 # Ensure cv2 is imported for CLAHE


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW tensor to HWC uint8 numpy array (for display)."""
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Denormalise
    mean = np.array(settings.IMAGE_MEAN)
    std  = np.array(settings.IMAGE_STD)
    arr  = arr * std + mean
    arr  = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return arr
