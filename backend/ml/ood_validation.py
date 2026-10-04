import sys
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

def validate_ood():
    print("="*60)
    print("PHASE 6: OOD & CONFIDENCE VALIDATION")
    print("="*60)
    
    BASE_DIR = Path("C:/Users/Lenovo/medical-assistant-working/backend")
    sys.path.append(str(BASE_DIR))
    
    from app.core.classifier import classifier
    
    # 1. Test unrelated images (OOD)
    print("\n[1/2] Testing Out-of-Distribution (OOD) Detection...")
    
    # Create a random noise image
    noise_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    noise_path = BASE_DIR / "ml" / "noise_test.jpg"
    Image.fromarray(noise_img).save(noise_path)
    
    # Create a simple geometric image
    geo_img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.rectangle(geo_img, (50, 50), (170, 170), (255, 255, 255), -1)
    geo_path = BASE_DIR / "ml" / "geo_test.jpg"
    Image.fromarray(geo_img).save(geo_path)
    
    test_cases = [
        ("Random Noise", noise_path),
        ("Geometric Shape", geo_path)
    ]
    
    for label, path in test_cases:
        print(f"  - Testing {label}...")
        res = classifier.predict_dict(str(path))
        print(f"    - Pred: {res['label']} | Conf: {res['confidence']:.2f} | Entropy: {res['entropy']:.2f}")
        print(f"    - Flags: {res['uncertainty_flags']}")
        print(f"    - Is OOD: {res['is_ood']}")
        
        if res['is_ood']:
            print("    - Success: OOD flag triggered correctly.")
        else:
            print("    - WARNING: OOD detection failed to flag this unrelated image.")

    # 2. Test Low Quality (Blurry)
    print("\n[2/2] Testing Low-Quality (Blurry) Confidence Reduction...")
    # Take an existing lesion and blur it heavily
    sample_img_path = list((BASE_DIR / "data/processed/test").rglob("*.jpg"))[0]
    sample_img = cv2.imread(str(sample_img_path))
    blurry_img = cv2.GaussianBlur(sample_img, (31, 31), 0)
    blurry_path = BASE_DIR / "ml" / "blurry_test.jpg"
    cv2.imwrite(str(blurry_path), blurry_img)
    
    print("  - Testing Blurry Lesion...")
    res_orig = classifier.predict_dict(str(sample_img_path))
    res_blur = classifier.predict_dict(str(blurry_path))
    
    print(f"    - Original Conf: {res_orig['confidence']:.2f}")
    print(f"    - Blurry Conf:   {res_blur['confidence']:.2f}")
    
    if res_blur['confidence'] < res_orig['confidence']:
        print("    - Success: Confidence reduced for lower quality image.")
    else:
        print("    - WARNING: Confidence did not decrease for blurry image.")

    print("\n" + "="*60)
    print("PHASE 6 COMPLETE")
    print("="*60)

if __name__ == "__main__":
    validate_ood()
