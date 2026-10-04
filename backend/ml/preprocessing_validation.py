import sys
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

def validate_preprocessing():
    print("="*60)
    print("PHASE 2: PREPROCESSING VALIDATION")
    print("="*60)
    
    BASE_DIR = Path("C:/Users/Lenovo/medical-assistant-working/backend")
    sys.path.append(str(BASE_DIR))
    
    from app.core.preprocessing import preprocess, apply_clahe, resize_and_pad
    from ml.data_loader import train_transform, val_transform
    
    # Create a dummy image (e.g., a non-square image with some features)
    dummy_img = np.zeros((300, 450, 3), dtype=np.uint8)
    cv2.circle(dummy_img, (225, 150), 50, (100, 50, 200), -1) # "Lesion"
    dummy_img_pil = Image.fromarray(dummy_img)
    
    # 1. Test Inference Preprocessing
    print("\n[1/3] Testing Inference Preprocessing Pipeline...")
    tensor_inf = preprocess(dummy_img_pil)
    print(f"  - Inference Tensor Shape: {tensor_inf.shape}")
    
    # 2. Test Consistency with Training Preprocessing
    print("\n[2/3] Checking Normalization Consistency...")
    # Get mean/std from settings
    from app.config import settings
    inf_mean = settings.IMAGE_MEAN
    inf_std = settings.IMAGE_STD
    
    # Check data_loader transforms
    # Note: data_loader.py uses global MEAN/STD variables
    from ml.data_loader import MEAN, STD
    
    print(f"  - Inference Mean: {inf_mean}")
    print(f"  - Training Mean:  {MEAN}")
    
    if inf_mean == MEAN and inf_std == STD:
        print("  - Normalization: MATCHED")
    else:
        print("  - WARNING: Normalization mismatch detected!")

    # 3. Visual Verification of Adaptive Resize
    print("\n[3/3] Generating Visual Comparison (saved to ml/preprocessing_validation_combined.jpg)...")
    
    # Inference flow steps
    clahe_img = apply_clahe(dummy_img_pil)
    padded_img = resize_and_pad(clahe_img, 224)
    
    # Create a contact sheet using PIL
    combined = Image.new('RGB', (dummy_img_pil.width + clahe_img.width + padded_img.width + 40, max(dummy_img_pil.height, padded_img.height) + 20))
    combined.paste(dummy_img_pil, (10, 10))
    combined.paste(clahe_img, (dummy_img_pil.width + 20, 10))
    combined.paste(padded_img, (dummy_img_pil.width + clahe_img.width + 30, 10))
    
    combined.save(str(BASE_DIR / "ml" / "preprocessing_validation_combined.jpg"))
    print("  - Visual report generated (Side-by-side: Original | CLAHE | Padded).")

    print("\n" + "="*60)
    print("PHASE 2 COMPLETE")
    print("="*60)

if __name__ == "__main__":
    validate_preprocessing()
