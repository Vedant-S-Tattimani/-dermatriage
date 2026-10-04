"""
Model evaluation script — computes per-class metrics on the held-out test set.

Usage:
    python ml/evaluate.py [--weights weights/model.pth]

Outputs a classification report and saves a confusion matrix PNG.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.data_loader import get_dataloaders, LABEL_MAP, NUM_CLASSES
from app.utils.logger import get_logger

logger = get_logger("evaluate")

CLASS_NAMES = list(LABEL_MAP.keys())   # short codes
LONG_NAMES  = [
    "Melanocytic nevi",
    "Melanoma",
    "Benign keratosis",
    "Basal cell carcinoma",
    "Actinic keratoses",
    "Vascular lesions",
    "Dermatofibroma",
]


def load_model(weights_path: Path):
    from torchvision import models
    import torch.nn as nn

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    state = torch.load(str(weights_path), map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict(model, loader, device):
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


def save_confusion_matrix(cm: np.ndarray, output_path: Path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LONG_NAMES,
            yticklabels=LONG_NAMES,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix — HAM10000 Test Set")
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150)
        logger.info("Confusion matrix saved: %s", output_path)
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — skipping confusion matrix plot.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/model.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers",    type=int, default=4)
    args = parser.parse_args()

    weights_path = BASE_DIR / args.weights
    if not weights_path.exists():
        logger.error("Weights file not found: %s", weights_path)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    _, _, test_loader, _ = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.workers
    )

    model = load_model(weights_path).to(device)
    preds, labels = predict(model, test_loader, device)

    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(
        classification_report(
            labels, preds,
            target_names=LONG_NAMES,
            digits=4,
        )
    )

    cm = confusion_matrix(labels, preds)
    save_confusion_matrix(
        cm, output_path=BASE_DIR / "weights" / "confusion_matrix.png"
    )


if __name__ == "__main__":
    main()
