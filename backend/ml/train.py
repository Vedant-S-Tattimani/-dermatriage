"""
Upgraded training script using ConvNeXt Tiny.
Features: Dropout, Early Stopping, and Learning Rate Scheduler.
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from torchvision import models
from ml.data_loader import get_dataloaders
import time

def main():
    base_dir = Path(__file__).resolve().parent.parent
    weights_dir = base_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 1. Load data
    batch_size = 32
    train_loader, val_loader, test_loader, loss_weights_tensor = get_dataloaders(
        batch_size=batch_size, num_workers=4
    )
    num_classes = len(train_loader.dataset.classes)

    # 2. Build ConvNeXt Tiny Model (Best from benchmark)
    print("Initializing ConvNeXt Tiny...")
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    
    # 3. Add Dropout and update head
    # ConvNeXt-Tiny classifier is a sequence: (LayerNorm2d, Flatten, Linear)
    # We'll add Dropout before the final Linear layer.
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)

    # 4. Optimizer & Scheduler
    loss_weights_tensor = loss_weights_tensor.to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # 5. Training Settings
    epochs = 3 
    best_val_acc = 0.0
    patience = 2
    epochs_no_improve = 0
    best_model_path = weights_dir / "best_model.pth"
    
    metrics = {
        "history": [],
        "best_val_acc": 0.0,
        "arch": "convnext_tiny"
    }

    print("=" * 60)
    print(f"Starting upgraded training (ConvNeXt Tiny)...")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step(val_loss)
        
        metrics["history"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(f"Epoch {epoch:02d}/{epochs} | Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Early Stopping & Best Model Save
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            metrics["best_val_acc"] = best_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  >> New best model! (Acc: {val_acc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    # Save metrics
    with open(weights_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nUpgrade complete. Best Val Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
