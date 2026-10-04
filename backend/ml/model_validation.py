import sys
import torch
import json
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

def validate_model():
    print("="*60)
    print("PHASE 4: MODEL VALIDATION")
    print("="*60)
    
    BASE_DIR = Path("C:/Users/Lenovo/medical-assistant-working/backend")
    sys.path.append(str(BASE_DIR))
    
    from app.core.classifier import classifier
    from ml.data_loader import get_dataloaders
    
    # 1. Full Evaluation on Test Set (Phase 4)
    print("\n[1/3] Running Full Evaluation on Test Set...")
    # Use num_workers=0 to avoid multiprocessing issues in the check script
    _, _, test_loader, _ = get_dataloaders(batch_size=32, num_workers=0)
    
    all_preds = []
    all_labels = []
    
    # We sample a subset for quick validation in the audit phase
    # (Full evaluation is usually done in ml/evaluate.py)
    max_samples = 200 
    count = 0
    
    start_time = time.time()
    
    # We manually iterate classes since we want to test the full Classifier.predict (which includes segmenter)
    data_path = BASE_DIR / "data/processed/test"
    classes = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    for actual_cls in classes:
        cls_path = data_path / actual_cls
        images = list(cls_path.glob("*.jpg"))[:30] # Sample per class
        
        print(f"  - Evaluating {actual_cls}...")
        for img_p in images:
            try:
                res = classifier.predict_dict(str(img_p))
                all_preds.append(res["label"])
                all_labels.append(actual_cls)
                count += 1
            except Exception as e:
                print(f"    - Error on {img_p}: {e}")
            
            if count >= max_samples:
                break
        if count >= max_samples:
            break

    duration = time.time() - start_time
    print(f"\n[2/3] Generating Metrics (Sample size: {count})...")
    
    report = classification_report(all_labels, all_preds, output_dict=True)
    print(f"  - Accuracy: {report['accuracy']:.2%}")
    print(f"  - Macro F1: {report['macro avg']['f1-score']:.2%}")
    print(f"  - Avg Inference Time: {(duration/count)*1000:.2f}ms (including segmentation)")

    # 3. Detect prediction collapse (Phase 4.Overfitting check)
    unique_preds = set(all_preds)
    print(f"\n[3/3] Prediction Analysis...")
    print(f"  - Unique classes predicted: {len(unique_preds)} / {len(classes)}")
    if len(unique_preds) < 3:
        print("  - WARNING: Prediction collapse detected! Model is only predicting a few classes.")
    else:
        print("  - Prediction diversity: OK")

    print("\n" + "="*60)
    print("PHASE 4 COMPLETE")
    print("="*60)

if __name__ == "__main__":
    validate_model()
