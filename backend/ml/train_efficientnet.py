"""
Training script for the 7-class HAM10000 skin lesion classification model.
Supports:
- Pre-trained EfficientNet-B0 fine-tuning (all layers).
- Weighted Cross-Entropy Loss with Label Smoothing.
- Focal Loss.
- ReduceLROnPlateau learning rate scheduler.
- Early Stopping based on validation loss.
- Multi-threaded CPU execution.
"""
import argparse
import json
import time
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

# Ensure backend directory is in path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.data_loader import get_dataloaders

# Optimize CPU execution
torch.set_num_threads(8)

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights tensor (num_classes,)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def build_model(num_classes=7):
    """Loads pre-trained EfficientNet-B0 and replaces the classifier head."""
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    # EfficientNet-B0 classifier has dropout and a linear layer:
    # classifier = Sequential(Dropout(p=0.2, inplace=True), Linear(in_features=1280, out_features=1000, bias=True))
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="Train HAM10000 Skin Lesion Model")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--loss", type=str, choices=["ce", "focal"], default="ce", help="Loss function (ce or focal)")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon for CE loss")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    weights_dir = BASE_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loss Configuration: {args.loss.upper()}")
    if args.loss == "ce":
        print(f"Label Smoothing: {args.label_smoothing}")

    # 1. Load Datasets via Loader (handles class weighting and WeightedRandomSampler)
    train_loader, val_loader, test_loader, loss_weights_tensor = get_dataloaders(
        batch_size=args.batch_size, num_workers=0, use_segmented=False
    )
    num_classes = len(train_loader.dataset.classes)
    
    # 2. Build EfficientNet-B0 Model (Fine-tune all layers)
    print("Building EfficientNet-B0 model...")
    model = build_model(num_classes=num_classes)
    
    # Ensure all layers are trainable
    for param in model.parameters():
        param.requires_grad = True
        
    model = model.to(device)

    # 3. Setup Loss Criterion
    loss_weights_tensor = loss_weights_tensor.to(device)
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=loss_weights_tensor, label_smoothing=args.label_smoothing)
    else:
        criterion = FocalLoss(alpha=loss_weights_tensor, gamma=2.0)

    # 4. Optimizer & ReduceLROnPlateau Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # 5. Training State
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    checkpoint_path = weights_dir / "model_robust_v2.pth"
    
    # Track metrics
    history = []

    print("=" * 60)
    print(f"Starting Training on HAM10000 (7 classes)...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
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
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch:02d} | Batch {batch_idx+1:03d}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validation ────────────────────────────────────────────────────────
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

        val_loss /= val_total
        val_acc = val_correct / val_total

        # ── Learning Rate Step ───────────────────────────────────────────────
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch info
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr
        })

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{args.epochs} - {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {current_lr:.1e}")

        # ── Early Stopping & Best Model Saving ────────────────────────────────
        # We base best checkpoint on val_loss as requested by early stopping policy
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            
            # Save state dict
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc
            }, checkpoint_path)
            
            print(f"  >> Saved new best model checkpoint! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered: no validation loss improvement for {args.patience} epochs.")
                break

    # Save metrics history
    metrics_path = weights_dir / "metrics_v2.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "history": history,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "loss_configured": args.loss,
            "label_smoothing": args.label_smoothing if args.loss == "ce" else 0.0,
            "arch": "efficientnet_b0"
        }, f, indent=4)

    print(f"\nTraining completed. Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Model Checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()
