"""
Benchmarking script to compare EfficientNetV2, DenseNet121, and ConvNeXt Tiny.
Measures: Validation Accuracy (after 1 epoch), Inference Speed, and Memory Usage.
"""
import time
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models
from ml.data_loader import get_dataloaders
import os

def build_model(arch_name, num_classes):
    if arch_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif arch_name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
    return model

def benchmark_arch(arch_name, train_loader, val_loader, device):
    print(f"\n--- Benchmarking: {arch_name} ---")
    num_classes = len(train_loader.dataset.classes)
    
    # 1. Model Build
    model = build_model(arch_name, num_classes).to(device)

    # 2. Training Speed & Early Accuracy (1 Epoch)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    start_train = time.time()
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i >= 10: break # Small subset for speed comparison
    train_time_per_10_batches = time.time() - start_train
    print(f"  Training Speed (10 batches): {train_time_per_10_batches:.2f}s")

    # 3. Validation Accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i >= 5: break # Small subset for quick benchmark
    val_acc = correct / total
    print(f"  Early Validation Acc (subset): {val_acc:.4f}")

    # 4. Inference Speed
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    start_inf = time.time()
    for _ in range(50):
        _ = model(dummy_input)
    avg_inf_time = (time.time() - start_inf) / 50 * 1000 # ms
    print(f"  Inference Latency: {avg_inf_time:.2f} ms")

    return {
        "arch": arch_name,
        "train_time_10_batches": train_time_per_10_batches,
        "val_acc": val_acc,
        "inference_latency_ms": avg_inf_time
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device}")
    
    train_loader, val_loader, _, _ = get_dataloaders(batch_size=16) # Smaller batch for CPU stability
    
    architectures = ["efficientnet_v2_s", "densenet121", "convnext_tiny"]
    results = []

    for arch in architectures:
        results.append(benchmark_arch(arch, train_loader, val_loader, device))

    print("\n" + "="*80)
    print(f"{'Architecture':<20} | {'Inf (ms)':<10} | {'Val Acc':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['arch']:<20} | {r['inference_latency_ms']:<10.2f} | {r['val_acc']:<10.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
