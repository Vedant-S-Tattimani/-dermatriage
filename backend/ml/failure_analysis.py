import json
import torch
from pathlib import Path
from app.core.classifier import classifier
from PIL import Image
import numpy as np

def run_failure_analysis(data_dir, output_file):
    """Analyzes incorrect predictions to find patterns (Phase 7)."""
    data_path = Path(data_dir)
    results = []
    
    classes = [d.name for d in data_path.iterdir() if d.is_dir()]
    print(f"Analyzing failures in {data_dir}...")
    
    for actual_cls in classes:
        cls_path = data_path / actual_cls
        images = list(cls_path.glob("*.jpg"))
        
        print(f"  Processing {actual_cls} ({len(images)} images)...")
        for img_p in images:
            try:
                # Use predict_dict which now includes segmentation, uncertainty, and OOD
                pred = classifier.predict_dict(str(img_p))
                
                is_correct = (pred["label"] == actual_cls)
                
                results.append({
                    "filename": img_p.name,
                    "actual": actual_cls,
                    "predicted": pred["label"],
                    "confidence": pred["confidence"],
                    "is_correct": is_correct,
                    "uncertainty_flags": pred["uncertainty_flags"],
                    "entropy": pred["entropy"],
                    "is_ood": pred["is_ood"]
                })
            except Exception as e:
                print(f"Error analyzing {img_p}: {e}")

    # Cluster failures
    failures = [r for r in results if not r["is_correct"]]
    
    analysis = {
        "summary": {
            "total": len(results),
            "correct": len(results) - len(failures),
            "failed": len(failures),
            "accuracy": (len(results) - len(failures)) / len(results) if results else 0
        },
        "failure_patterns": {}
    }
    
    # Simple clustering by actual vs predicted
    for f in failures:
        key = f"{f['actual']} -> {f['predicted']}"
        if key not in analysis["failure_patterns"]:
            analysis["failure_patterns"][key] = []
        analysis["failure_patterns"][key].append(f)

    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=4)
        
    print(f"\nFailure analysis complete. Report saved to {output_file}")
    print(f"Total Accuracy on this set: {analysis['summary']['accuracy']:.2%}")

if __name__ == "__main__":
    BASE = Path("C:/Users/Lenovo/medical-assistant-working/backend")
    # First analyze on standard test set
    run_failure_analysis(
        BASE / "data/processed/test",
        BASE / "ml/failure_analysis_report.json"
    )
