import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path

def validate_segmentation():
    print("="*60)
    print("PHASE 3: SEGMENTATION VALIDATION")
    print("="*60)
    
    BASE_DIR = Path("C:/Users/Lenovo/medical-assistant-working/backend")
    sys.path.append(str(BASE_DIR))
    
    from app.core.segmenter import segmenter
    
    # Paths for testing (Phase 3.1: Run on various images)
    test_images = []
    
    # 1. HAM10000 image (pick one from processed/test)
    ham_img = list((BASE_DIR / "data/processed/test").rglob("*.jpg"))[0]
    test_images.append(("HAM10000", ham_img))
    
    # 2. External image (pick one from uploads)
    ext_img = list((BASE_DIR / "uploads").glob("*.jpg"))[0]
    test_images.append(("External", ext_img))
    
    print(f"\n[1/3] Running Segmentation on {len(test_images)} samples...")
    
    for label, img_path in test_images:
        print(f"  - Testing {label}: {img_path.name}")
        try:
            cropped_path, mask_path = segmenter.segment_and_crop(str(img_path))
            
            if cropped_path and Path(cropped_path).exists():
                print(f"    - Success: Cropped image saved at {Path(cropped_path).name}")
            else:
                print(f"    - FAILED: No cropped image generated.")
                
            if mask_path and Path(mask_path).exists():
                print(f"    - Success: Mask saved at {Path(mask_path).name}")
            else:
                print(f"    - WARNING: No mask generated.")
                
        except Exception as e:
            print(f"    - ERROR: {e}")

    # 3. Detection of failures (Phase 3.4)
    print("\n[2/3] Validating Automatic Failure Detection...")
    # Test on a "random" image (blank)
    blank_img = Image.new('RGB', (224, 224), (255, 255, 255))
    blank_path = BASE_DIR / "ml" / "blank_test.jpg"
    blank_img.save(blank_path)
    
    cropped_path, mask_path = segmenter.segment_and_crop(str(blank_path))
    if cropped_path == str(blank_path):
        print("  - Success: Segmenter correctly identified no lesion and returned original path.")
    else:
        print("  - WARNING: Segmenter returned a crop for a blank image.")

    print("\n" + "="*60)
    print("PHASE 3 COMPLETE")
    print("="*60)

if __name__ == "__main__":
    validate_segmentation()
