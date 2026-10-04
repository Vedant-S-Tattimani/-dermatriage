"""
Advanced evaluation suite for the skin lesion classification model.
Generates precision/recall/F1, confusion matrix, ROC curves, and clinical error analysis.
Exports results to CSV and PNG.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    roc_curve, auc, precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize
from ml.data_loader import get_dataloaders
from app.core.classifier import Classifier
from app.config import settings

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing advanced evaluation on: {device}")

    # 1. Setup Output Directories
    out_dir = Path("ml/evaluation_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Data and Model
    _, _, test_loader, _ = get_dataloaders(batch_size=32, num_workers=0)
    clf = Classifier()
    model = clf.model
    if model is None:
        print("ERROR: Model not found.")
        return
    model.to(device).eval()
    
    class_names = [clf.classes[i] for i in range(clf.num_classes)]
    num_classes = len(class_names)

    # 3. Collect Predictions
    all_preds = []
    all_labels = []
    all_probs = []
    image_paths = [] # To track specific errors if we had the paths

    print("Running batch inference...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            # Use production calibration
            probs = torch.softmax(logits / settings.CALIBRATION_TEMPERATURE, dim=1)
            _, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 4. Core Metrics (Precision, Recall, F1)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds)
    
    metrics_df = pd.DataFrame({
        'class': class_names,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    })
    metrics_df.to_csv(out_dir / "class_metrics.csv", index=False)

    # 5. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    # 6. Clinical Error Analysis (Isolate critical failures)
    # Target: High risk classes (Melanoma, BCC) misclassified as benign
    critical_errors = []
    malignant_indices = [i for i, name in enumerate(class_names) if name.lower() in ['melanoma', 'bcc']]
    benign_indices = [i for i, name in enumerate(class_names) if name.lower() in ['nevus', 'bkl', 'df', 'vascular']]

    for i in range(len(all_labels)):
        actual = all_labels[i]
        pred = all_preds[i]
        if actual in malignant_indices and pred in benign_indices:
            critical_errors.append({
                "index": i,
                "actual": class_names[actual],
                "predicted": class_names[pred],
                "confidence": float(all_probs[i][pred])
            })

    with open(out_dir / "critical_failures.json", "w") as f:
        json.dump(critical_errors, f, indent=4)

    # 7. ROC Curve Data (Binarize for multi-class)
    y_test_bin = label_binarize(all_labels, classes=range(num_classes))
    roc_data = {}
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_data[class_names[i]] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": auc(fpr, tpr)
        }
    
    with open(out_dir / "roc_data.json", "w") as f:
        json.dump(roc_data, f, indent=4)

    # 8. Visualization (Optional Matplotlib/Seaborn)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Plot 1: Confusion Matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Clinical Confusion Matrix')
        plt.savefig(plots_dir / "confusion_matrix.png")
        
        # Plot 2: ROC Curves
        plt.figure(figsize=(10, 8))
        for name, data in roc_data.items():
            plt.plot(data["fpr"], data["tpr"], label=f'{name} (AUC = {data["auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.savefig(plots_dir / "roc_curves.png")
        
        print(f"Charts saved to: {plots_dir}")
    except ImportError:
        print("[Note] Visualization libraries not found, skipping PNG generation.")

    # 9. Dashboard Summary
    dashboard = {
        "overall_accuracy": accuracy_score(all_labels, all_preds),
        "total_test_samples": len(all_labels),
        "critical_failure_rate": len(critical_errors) / len(all_labels),
        "best_performing_class": class_names[np.argmax(f1)],
        "weakest_class": class_names[np.argmin(f1)],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(out_dir / "evaluation_dashboard.json", "w") as f:
        json.dump(dashboard, f, indent=4)

    print("\n" + "="*60)
    print("ADVANCED EVALUATION COMPLETE")
    print("="*60)
    print(f"Overall Accuracy: {dashboard['overall_accuracy']:.2%}")
    print(f"Critical Errors (Malignant -> Benign): {len(critical_errors)}")
    print(f"Weakest Class: {dashboard['weakest_class']}")
    print(f"Reports saved in: {out_dir}")

if __name__ == "__main__":
    evaluate()
