"""
Classifier — wraps the trained EfficientNet-B0 multi-class model.

Inference contract:
    label, confidence, all_probs, top_3 = classifier.predict(image_path)

    label      : str        — top predicted class name
    confidence : float      — calibrated softmax probability [0, 1]
    all_probs  : list[float] — full distribution across all 7 classes
    top_3      : list[dict]  — sorted top 3 predictions
"""

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from app.config import settings

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODE = "hackathon"
# NOTE: Temperature is now exclusively read from settings.CALIBRATION_TEMPERATURE
# (loaded from ml/calibration/calibration_metrics.json, default 1.0).
# Do NOT set a local override here — use the calibration pipeline.

TRANSFORM = transforms.Compose([
    transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=settings.IMAGE_MEAN, std=settings.IMAGE_STD),
])

# ── Model Builder ──────────────────────────────────────────────────────────────

def _build_convnext_tiny(num_classes: int) -> nn.Module:
    """Build ConvNeXt Tiny with the dropout-enhanced head."""
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

# ── Classifier class ───────────────────────────────────────────────────────────

from app.core.segmenter import segmenter

class Classifier:
    """
    Singleton wrapper for the dermatology classification model.
    Implements Temperature Scaling and Top-K results.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = settings.MEDIUM_RISK_THRESHOLD
        self.use_segmentation = True # New flag to control segmentation

        base_dir = Path(__file__).resolve().parent.parent.parent
        self._model_path   = settings.MODEL_PATH
        self._classes_path = base_dir / "weights" / "classes.json"

        # Load class mapping
        if self._classes_path.exists():
            with open(self._classes_path, "r") as f:
                idx_to_class_json = json.load(f)
            self.classes = {int(k): v for k, v in idx_to_class_json.items()}
        else:
            logger.warning("classes.json not found! Inference mapping will be empty.")
            self.classes = {}

        self.num_classes = len(self.classes)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Eagerly loads weights from settings.MODEL_PATH."""
        if not self._model_path.exists():
            logger.error("Model weights NOT FOUND at %s", self._model_path)
            return

        try:
            net = _build_convnext_tiny(num_classes=self.num_classes)
            net.load_state_dict(torch.load(self._model_path, map_location=self.device))
            net.to(self.device).eval()
            self.model = net
            logger.info("Successfully loaded model: %s", self._model_path.name)
        except Exception as exc:
            logger.exception("Failed to load model: %s", exc)
            self.model = None

    def predict(self, image_path: str, use_segmentation: bool = True):
        """
        Runs inference with Temperature Scaling.

        Returns:
            (top_label, confidence, all_probabilities, top_3_list)
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Check weights directory.")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            start_time = time.time()
            
            # ── 0. Segmentation & Cropping ────────────────────────────────────
            # Automatically isolate the lesion to remove background bias
            target_path = image_path
            if use_segmentation:
                try:
                    cropped_path, _ = segmenter.segment_and_crop(image_path)
                    if cropped_path and Path(cropped_path).exists():
                        target_path = cropped_path
                        logger.info("Using segmented and cropped image for classification: %s", target_path)
                except Exception as seg_exc:
                    logger.error("Segmentation failed, falling back to original: %s", seg_exc)

            image = Image.open(target_path).convert("RGB")
            input_tensor = TRANSFORM(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(input_tensor)
                
                # ── 1. Temperature Scaling (Calibration) ──────────────────────
                # Uses setting from config (optimised via ml/calibrate_model.py)
                scaled_logits = logits / settings.CALIBRATION_TEMPERATURE
                probs = torch.softmax(scaled_logits, dim=1)

            all_probs_list = probs[0].tolist()
            
            # ── 2. Get Top-1 with Overconfidence Cap ──────────────────────────
            conf_tensor, idx_tensor = torch.max(probs, 1)
            confidence = conf_tensor.item()
            
            # Prevent unrealistic 99-100% scores in a medical context
            if confidence > settings.PREVENT_OVERCONFIDENCE_CAP:
                confidence = settings.PREVENT_OVERCONFIDENCE_CAP

            pred_idx = idx_tensor.item()
            label    = self.classes.get(pred_idx, "UNKNOWN")

            # ── Get Top-3 ──────────────────────────────────────────────────────
            # Ensure topk does not exceed logits size (vital for tests with mocked models)
            num_logits = probs.size(1)
            top_k = min(3, self.num_classes, num_logits)
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=1)
            
            top_3 = []
            for i in range(top_k):
                idx = top_k_indices[0][i].item()
                top_3.append({
                    "class_name": self.classes.get(idx, "UNKNOWN"),
                    "probability": top_k_probs[0][i].item()
                })

            # ── 3. OOD Detection & Uncertainty Flags ──────────────────────────
            is_ood = False
            uncertainty_flags = []
            
            # Entropy check: High entropy means the model is "confused" across classes
            # HAM10000 images at T=1.0 have typical entropy ~1.4–1.7; threshold set to
            # 1.85 to avoid false-positive OOD flags on valid dermatoscopy images.
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).item()
            
            if confidence < settings.UNCERTAINTY_THRESHOLD:
                uncertainty_flags.append("LOW_CONFIDENCE")
            
            if entropy > 1.85:  # Raised from 1.5 — avoids OOD-flagging normal images
                uncertainty_flags.append("HIGH_UNCERTAINTY")
                is_ood = True
                
            if confidence < 0.18:  # Extremely low confidence = likely OOD
                uncertainty_flags.append("POSSIBLE_OUT_OF_DISTRIBUTION")
                is_ood = True

            inference_ms = (time.time() - start_time) * 1000
            logger.info("[%s] Inference done in %.2fms | Pred: %s (%.2f) | Flags: %s", 
                        path.name, inference_ms, label, confidence, uncertainty_flags)

            return {
                "label": label,
                "confidence": confidence,
                "all_probs": all_probs_list,
                "top_3": top_3,
                "uncertainty_flags": uncertainty_flags,
                "is_ood": is_ood,
                "entropy": entropy
            }

        except Exception as exc:
            logger.error("Inference failure: %s", exc)
            raise

    def predict_dict(self, image_path: str, use_segmentation: bool = True) -> dict:
        """Updated dict wrapper."""
        return self.predict(image_path, use_segmentation=use_segmentation)

# ── Module Singleton ───────────────────────────────────────────────────────────
classifier = Classifier()

def predict(image_path: str, use_segmentation: bool = True) -> dict:
    return classifier.predict_dict(image_path, use_segmentation=use_segmentation)
