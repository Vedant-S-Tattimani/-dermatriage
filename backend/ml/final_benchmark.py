import json
import torch
import time
from pathlib import Path
from app.core.classifier import classifier
from ml.data_loader import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix

def benchmark_dataset(data_dir, name):
    """Evaluates the current model on a specific dataset."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Dataset {name} not found at {data_dir}. Skipping.")
        return None

    results = []
    actual_labels = []
    pred_labels = []
    
    classes = [d.name for d in data_path.iterdir() if d.is_dir()]
    print(f"\nBenchmarking {name}...")
    
    start_time = time.time()
    for actual_cls in classes:
        cls_path = data_path / actual_cls
        images = list(cls_path.glob("*.jpg"))
        
        for img_p in images:
            try:
                # Use current improved prediction logic
                pred = classifier.predict_dict(str(img_p))
                
                actual_labels.append(actual_cls)
                pred_labels.append(pred["label"])
                
                results.append({
                    "is_correct": (pred["label"] == actual_cls),
                    "confidence": pred["confidence"],
                    "is_ood": pred["is_ood"]
                })
            except Exception as e:
                pass

    duration = time.time() - start_time
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / total if total else 0
    avg_conf = sum(r["confidence"] for r in results) / total if total else 0
    ood_rate = sum(1 for r in results if r["is_ood"]) / total if total else 0
    
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Avg Confidence: {avg_conf:.2f}")
    print(f"  OOD Flag Rate: {ood_rate:.2%}")
    print(f"  Total Images: {total}")
    print(f"  Inference Speed: {total/duration:.2f} img/sec")

    return {
        "dataset": name,
        "accuracy": accuracy,
        "avg_confidence": avg_conf,
        "ood_rate": ood_rate,
        "total": total,
        "report": classification_report(actual_labels, pred_labels, output_dict=True)
    }

def run_final_benchmark():
    BASE = Path("C:/Users/Lenovo/medical-assistant-working/backend")
    
    # 1. HAM10000 Benchmark
    ham_results = benchmark_dataset(BASE / "data/processed/test", "HAM10000 (Internal)")
    
    # 2. External Benchmark (Google/Smartphone)
    ext_results = benchmark_dataset(BASE / "data/external_val", "Real-World (External)")
    
    summary = {
        "ham10000": ham_results,
        "external": ext_results,
        "timestamp": time.ctime()
    }
    
    with open(BASE / "ml/final_benchmark_report.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nPhase 9: Final Benchmark Complete. Report saved to {BASE}/ml/final_benchmark_report.json")

if __name__ == "__main__":
    run_final_benchmark()
