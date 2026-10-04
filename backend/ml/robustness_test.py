"""
Robustness and Stress Testing Suite for Skin Lesion Classification.
Applies real-world distortions to the test set and measures performance degradation.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import io
from pathlib import Path
from PIL import Image
from torchvision import transforms
from ml.data_loader import get_dataloaders
from app.core.classifier import Classifier
from app.config import settings

# ── Distortion Functions ──────────────────────────────────────────────────────

def apply_blur(img):
    return transforms.GaussianBlur(kernel_size=(7, 7), sigma=(2.0, 2.0))(img)

def apply_low_light(img):
    # Reduce brightness to 30% of original
    return transforms.functional.adjust_brightness(img, 0.3)

def apply_rotation(img):
    return transforms.functional.rotate(img, 45)

def apply_zoom(img):
    # Zoom in by taking a 50% center crop and resizing back
    w, h = img.size
    crop_size = min(w, h) // 2
    img_cropped = transforms.functional.center_crop(img, [crop_size, crop_size])
    return transforms.functional.resize(img_cropped, [w, h])

def apply_noise(img):
    # Add Gaussian noise
    img_np = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 30, img_np.shape)
    img_noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_noisy)

def apply_compression(img):
    # Simulate heavy JPEG compression
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=15)
    buffer.seek(0)
    return Image.open(buffer)

# ── Test Runner ───────────────────────────────────────────────────────────────

def run_stress_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing stress tests on: {device}")

    # 1. Setup
    _, _, test_loader, _ = get_dataloaders(batch_size=32, num_workers=0)
    clf = Classifier()
    model = clf.model
    if model is None:
        print("ERROR: Model not found.")
        return
    model.to(device).eval()
    
    class_names = [clf.classes[i] for i in range(clf.num_classes)]
    
    distortions = {
        "Baseline": lambda x: x,
        "Blurry": apply_blur,
        "Low Light": apply_low_light,
        "Rotated (45 deg)": apply_rotation,
        "Zoomed (2x)": apply_zoom,
        "Noisy (Gaussian)": apply_noise,
        "Compressed (LQ)": apply_compression
    }

    results = {}

    # 2. Iterate through distortions
    for name, transform_fn in distortions.items():
        print(f"Testing distortion: {name}...")
        correct = 0
        total = 0
        total_conf = 0.0
        
        # We test on a subset to keep it fast but statistically significant
        max_batches = 10 
        
        with torch.no_grad():
            for i, (inputs_raw, labels) in enumerate(test_loader):
                if i >= max_batches: break
                
                # We need to distorts before normalization
                # So we take the raw dataset items (which aren't normalized yet if we get them from ImageFolder)
                # But get_dataloaders returns normalized tensors.
                # Let's rebuild the input manually from the paths for this test.
                
                # Re-loading images manually to apply custom PIL transforms
                distorted_tensors = []
                for idx_in_batch in range(inputs_raw.size(0)):
                    # Get path from dataset (test_loader.dataset is ImageFolder)
                    path, _ = test_loader.dataset.samples[i * 32 + idx_in_batch]
                    img = Image.open(path).convert("RGB")
                    
                    # Apply distortion
                    distorted_img = transform_fn(img)
                    
                    # Apply standard inference preprocessing
                    from app.core.classifier import TRANSFORM
                    tensor = TRANSFORM(distorted_img)
                    distorted_tensors.append(tensor)
                
                inputs = torch.stack(distorted_tensors).to(device)
                labels = labels.to(device)
                
                logits = model(inputs)
                probs = torch.softmax(logits / settings.CALIBRATION_TEMPERATURE, dim=1)
                confs, preds = torch.max(probs, 1)
                
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                total_conf += confs.sum().item()

        results[name] = {
            "accuracy": correct / total,
            "avg_confidence": total_conf / total
        }

    # 3. Generate Report
    print("\n" + "="*60)
    print("STRESS TEST & ROBUSTNESS REPORT")
    print("="*60)
    print(f"{'Condition':<20} | {'Accuracy':<10} | {'Confidence':<10}")
    print("-" * 60)
    
    baseline_acc = results["Baseline"]["accuracy"]
    
    for name, metrics in results.items():
        acc = metrics["accuracy"]
        conf = metrics["avg_confidence"]
        diff = acc - baseline_acc
        print(f"{name:<20} | {acc:<10.2%} | {conf:<10.2%}")

    # 4. Identification of Fragility
    worst_distortion = min(results, key=lambda k: results[k]["accuracy"])
    degradation = baseline_acc - results[worst_distortion]["accuracy"]
    
    report = {
        "metrics": results,
        "worst_mode": worst_distortion,
        "max_degradation": degradation,
        "recommendations": []
    }
    
    if degradation > 0.1:
        report["recommendations"].append(f"Heavy accuracy drop in {worst_distortion} mode. Increase targeted augmentations.")
    if results["Low Light"]["accuracy"] < baseline_acc - 0.05:
        report["recommendations"].append("Model is sensitive to lighting. Add RandomBrightness to training.")
    if results["Blurry"]["accuracy"] < baseline_acc - 0.05:
        report["recommendations"].append("Model is sensitive to blur. Add GaussianBlur to training.")

    output_path = Path("ml/robustness_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
    
    print(f"\nReport saved to: {output_path}")

if __name__ == "__main__":
    run_stress_test()
