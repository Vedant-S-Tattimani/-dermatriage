"""
HAM10000 dataset loader — updated for structured train/val/test directories.

Usage:
    from ml.data_loader import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
"""
import torch
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# ── Transforms ────────────────────────────────────────────────────────────────
# TODO: Centralize these in app/config.py
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class AddGaussianNoise(object):
    """Custom transform to add Gaussian noise for robustness."""
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class JPEGCompression(object):
    """Simulates JPEG compression artifacts."""
    def __init__(self, quality_range=(10, 50)):
        self.quality_range = quality_range

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            return img
        quality = np.random.randint(self.quality_range[0], self.quality_range[1])
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality)
        output.seek(0)
        return Image.open(output)

import io
import numpy as np
import cv2
from PIL import Image

class ApplyCLAHE(object):
    """Custom transform to apply CLAHE for robust contrast normalization."""
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            return img
        # Convert PIL to OpenCV (numpy)
        img_np = np.array(img.convert("RGB"))
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final_img)

train_transform = transforms.Compose([
    # Standard resizing
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    
    # Geometric invariance (rotations are highly effective for skin lesions)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),
    
    # Mild color variations to simulate smartphone/camera sensors
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),
    
    # Conversion and Normalization
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

val_transform = transforms.Compose([
    # Resizing without CLAHE to perfectly match production inference TRANSFORM
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def get_dataloaders(batch_size: int = 32, num_workers: int = 0, use_segmented: bool = True):
    """
    Build DataLoaders from organized directories with Weighted Sampling.
    """
    segmented_dir = BASE_DIR / "data" / "processed_segmented"
    
    if use_segmented and segmented_dir.exists():
        print(f"[data_loader] Using ROBUST SEGMENTED dataset: {segmented_dir}")
        data_root = segmented_dir
    else:
        print(f"[data_loader] Using standard processed dataset: {PROCESSED_DIR}")
        data_root = PROCESSED_DIR

    train_dir = data_root / "train"
    val_dir   = data_root / "val"
    test_dir  = data_root / "test"

    # 1. Datasets
    train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_ds   = datasets.ImageFolder(root=str(val_dir),   transform=val_transform)
    test_ds  = datasets.ImageFolder(root=str(test_dir),  transform=val_transform)

    print(f"[data_loader] Detected class mapping: {train_ds.class_to_idx}")
    
    # ── 2. WEIGHTED SAMPLING (Handle Imbalance) ──────────────────────────────
    from collections import Counter
    targets = train_ds.targets
    counts = Counter(targets)
    num_classes = len(train_ds.classes)
    
    # Weight per class: 1.0 / frequency
    class_weights = [1.0 / counts[i] for i in range(num_classes)]
    
    # Weight per sample
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # ── 3. LOSS WEIGHTS ──────────────────────────────────────────────────────
    # Normalize class weights so they sum to num_classes
    weights_sum = sum(class_weights)
    normalized_weights = [w / weights_sum * num_classes for w in class_weights]
    loss_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32)

    # ── 4. LOADERS ───────────────────────────────────────────────────────────
    loader_kwargs = dict(
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    
    # Use sampler for train, shuffle must be False when sampler is used
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, loss_weights_tensor

if __name__ == "__main__":
    tl, vl, tsl, weights = get_dataloaders()
    print(f"Train batches: {len(tl)}")
    print(f"Class weights: {weights}")
