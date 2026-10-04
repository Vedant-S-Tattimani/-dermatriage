"""
Calibration script for medical skin triage.
- Optimizes Temperature (T) using the validation set.
- Calculates Expected Calibration Error (ECE).
- Generates reliability data for visualization.
"""
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from ml.data_loader import get_dataloaders
from app.core.classifier import Classifier
from app.config import settings

def calculate_ece(probs, labels, n_bins=10):
    """Calculates Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated confidence and accuracy in each bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calibrating on device: {device}")

    # 1. Load Data
    _, val_loader, _, _ = get_dataloaders(batch_size=32, num_workers=0)
    
    # 2. Load Model via refined Classifier class
    clf = Classifier()
    model = clf.model
    if model is None:
        print("ERROR: Model not found. Run training first.")
        return
    model.to(device).eval()

    all_logits = []
    all_labels = []

    print("Extracting validation logits...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()

    # ── 3. Optimize Temperature ──────────────────────────────────────────────
    print("\nOptimizing Temperature Scaling...")
    
    def nll_criterion(t, logits, labels):
        t_scaled = logits / t
        # CrossEntropyLoss expects (N, C) and (N)
        return nn.functional.cross_entropy(t_scaled, torch.from_numpy(labels)).item()

    # Simple Grid Search for optimal T (clinically reliable range)
    temperatures = np.linspace(0.1, 5.0, 50)
    losses = [nll_criterion(t, logits, labels) for t in temperatures]
    optimal_t = temperatures[np.argmin(losses)]
    
    print(f"Optimal Temperature: {optimal_t:.4f}")

    # ── 4. Compare Metrics ───────────────────────────────────────────────────
    # Raw Softmax (T=1.0)
    raw_probs = torch.softmax(logits, dim=1).numpy()
    raw_ece = calculate_ece(raw_probs, labels)

    # Calibrated Softmax (T=optimal_t)
    cal_probs = torch.softmax(logits / optimal_t, dim=1).numpy()
    cal_ece = calculate_ece(cal_probs, labels)

    print("\n" + "="*50)
    print("CALIBRATION RESULTS")
    print("="*50)
    print(f"Uncalibrated ECE (T=1.0): {raw_ece:.4f}")
    print(f"Calibrated ECE   (T={optimal_t:.2f}): {cal_ece:.4f}")
    print(f"ECE Improvement: {((raw_ece - cal_ece) / raw_ece * 100):.2f}%")
    print("-" * 50)

    # ── 5. Prevent Overconfidence ────────────────────────────────────────────
    max_cal_conf = np.max(cal_probs, axis=1)
    overconfident_count = np.sum(max_cal_conf > 0.99)
    print(f"Samples with >99% confidence (calibrated): {overconfident_count}")

    # ── 6. Save Metrics ──────────────────────────────────────────────────────
    output_dir = Path("ml/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "optimal_temperature": optimal_t,
        "raw_ece": raw_ece,
        "calibrated_ece": cal_ece,
        "improvement_pct": (raw_ece - cal_ece) / raw_ece * 100,
        "overconfident_samples": int(overconfident_count)
    }
    
    with open(output_dir / "calibration_metrics.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\nCalibration report saved to: {output_dir / 'calibration_metrics.json'}")
    print("\n[Action] Update app/config.py with optimal_t for production use.")

if __name__ == "__main__":
    main()
