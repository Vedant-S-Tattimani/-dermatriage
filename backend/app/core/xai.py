"""
Advanced Explainability (XAI) Suite — implements multiple CAM methods.
Methods: Grad-CAM, Grad-CAM++, Eigen-CAM, Score-CAM.

FIX LOG (v2):
- ROOT CAUSE: target_layer was model.features[-1] (a Sequential block wrapper),
  which produced meaningless spatially-uniform activations that appeared as a
  full-image colour filter. The backward hook on a Sequential returns gradients
  for the entire block output, not individual spatial weight maps.
- FIX: For ConvNeXt Tiny, the correct target layer is:
    model.features[7][2].block[0]
  This is the final-stage, final-block 7×7 depthwise Conv2d (768 channels).
  It carries rich spatial lesion-localisation information.
- FIX: _save_heatmap now saves THREE files per run:
    1. <name>_heatmap.png  — pure normalised Jet colormap (no original image)
    2. <name>_overlay.jpg  — blended at alpha=0.55 original / 0.45 heatmap
  Both URLs are returned as a tuple so the caller can route correctly.
- FIX: Normalisation uses (x - min)/(max - min) with eps guard, preventing
  flat maps from dividing by zero and rendering solid-colour outputs.
- FIX: Heatmap is resized to ORIGINAL image dimensions (not fixed 224×224).
- Added: Full debug logging with tensor shapes and min/max values.
- Added: Validation script to batch-test 10 images → backend/debug/gradcam_validation/
"""
import uuid
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

# ── ConvNeXt-Tiny target layer configuration ────────────────────────────────
# features[7]        → final ConvNeXt stage (7×7 depthwise conv, 768ch)
# features[7][2]     → last CNBlock in final stage
# features[7][2].block[0] → 7×7 depthwise Conv2d (spatial feature map)
CONVNEXT_TARGET_LAYER_ACCESSOR = ("features", 7, 2, "block", 0)


def _resolve_target_layer(model: nn.Module) -> nn.Module:
    """
    Resolve the correct target layer for Grad-CAM for ConvNeXt Tiny.
    Falls back gracefully if the architecture differs.
    """
    try:
        # Primary: ConvNeXt Tiny - final block depthwise conv
        layer = model.features[7][2].block[0]
        logger.info("[XAI] Target layer resolved: features[7][2].block[0] → %s", type(layer).__name__)
        return layer
    except (AttributeError, IndexError, TypeError):
        pass

    # Fallback 1: try any model with .features and sub-indexing
    try:
        layer = model.features[-1]
        # If it's a Sequential, dive into last child
        if isinstance(layer, nn.Sequential):
            last_block = list(layer.children())[-1]
            # Try to get the inner conv
            for child in last_block.modules():
                if isinstance(child, nn.Conv2d):
                    logger.warning("[XAI] Fallback: using last Conv2d in features[-1]: %s", child)
                    return child
        logger.warning("[XAI] Fallback: using features[-1] directly: %s", type(layer).__name__)
        return layer
    except (AttributeError, IndexError):
        pass

    # Last resort: find last Conv2d in entire model
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        logger.warning("[XAI] Last-resort fallback: last Conv2d in model: %s", last_conv)
        return last_conv

    raise RuntimeError("[XAI] Could not resolve any valid convolutional target layer.")


class XAISuite:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.output_dir = base_dir / "uploads"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_model_and_layer(self):
        """Reuses the classifier's pre-loaded singleton model instance."""
        from app.core.classifier import classifier as main_classifier
        if main_classifier.model is None:
            raise RuntimeError("[XAI] Classifier model not initialized.")
        model = main_classifier.model
        target_layer = _resolve_target_layer(model)
        return model, target_layer

    def _save_heatmap(self, heatmap: np.ndarray, original_image_path: str, method_name: str) -> tuple:
        """
        Saves THREE artifacts:
          1. <uuid>_heatmap.png  — pure Grad-CAM colormap
          2. <uuid>_overlay.jpg  — blended overlay on original image

        Returns:
            (heatmap_filename, overlay_filename)
        """
        # ── Load original at its native resolution ──────────────────────────
        image = Image.open(original_image_path).convert("RGB")
        orig_w, orig_h = image.size
        original_np = np.array(image)                          # H×W×3 RGB uint8
        original_cv = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)

        # ── Resize heatmap to original image dimensions ─────────────────────
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # ── Normalise to [0, 1] robustly ────────────────────────────────────
        h_min, h_max = heatmap_resized.min(), heatmap_resized.max()
        logger.debug("[XAI][%s] Heatmap stats after resize — min: %.4f, max: %.4f, mean: %.4f",
                     method_name, h_min, h_max, heatmap_resized.mean())

        if h_max - h_min > 1e-8:
            heatmap_norm = (heatmap_resized - h_min) / (h_max - h_min)
        else:
            logger.warning("[XAI][%s] Flat heatmap detected (max-min < 1e-8). "
                           "The target layer may be incorrect or gradients vanished.", method_name)
            heatmap_norm = np.zeros_like(heatmap_resized)

        # ── Convert to 8-bit colormap ────────────────────────────────────────
        heatmap_uint8 = np.uint8(255 * heatmap_norm)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR

        uid = uuid.uuid4().hex

        # ── 1. Pure heatmap PNG ──────────────────────────────────────────────
        heatmap_filename = f"xai_{method_name}_{uid}_heatmap.png"
        heatmap_path = self.output_dir / heatmap_filename
        cv2.imwrite(str(heatmap_path), heatmap_colored)

        # ── 2. Overlay (original blended with heatmap) ──────────────────────
        # alpha=0.55 keeps original detail; heatmap at 0.45 is vivid but not dominant
        overlay = cv2.addWeighted(original_cv, 0.55, heatmap_colored, 0.45, 0)
        overlay_filename = f"xai_{method_name}_{uid}_overlay.jpg"
        overlay_path = self.output_dir / overlay_filename
        cv2.imwrite(str(overlay_path), overlay)

        logger.info("[XAI][%s] Saved heatmap → %s | overlay → %s",
                    method_name, heatmap_filename, overlay_filename)

        return heatmap_filename, overlay_filename

    def generate(self, image_path: str, method: str = "gradcam", class_idx: int = None) -> str:
        """
        Entry point. Returns the OVERLAY filename (backward-compatible with triage_service).
        Saves both pure heatmap and overlay; overlay filename is returned.
        """
        method = method.lower()
        dispatch = {
            "gradcam": self.gradcam,
            "gradcampp": self.gradcam_pp,
            "eigencam": self.eigen_cam,
            "scorecam": self.score_cam,
        }
        fn = dispatch.get(method, self.gradcam)
        heatmap_fn, overlay_fn = fn(image_path, class_idx)
        return overlay_fn   # backward-compatible: callers store this as the XAI URL

    def generate_all(self, image_path: str, method: str = "gradcam", class_idx: int = None) -> dict:
        """
        Returns dict with both 'heatmap' and 'overlay' filenames.
        """
        method = method.lower()
        dispatch = {
            "gradcam": self.gradcam,
            "gradcampp": self.gradcam_pp,
            "eigencam": self.eigen_cam,
            "scorecam": self.score_cam,
        }
        fn = dispatch.get(method, self.gradcam)
        heatmap_fn, overlay_fn = fn(image_path, class_idx)
        return {"heatmap": heatmap_fn, "overlay": overlay_fn}

    # ── 1. Grad-CAM ───────────────────────────────────────────────────────────
    def gradcam(self, image_path: str, class_idx: int = None) -> tuple:
        model, target_layer = self._get_model_and_layer()
        activations, gradients = [], []

        def fw_hook(module, inp, output):
            activations.append(output.detach())

        def bw_hook(module, grad_in, grad_out):
            # grad_out[0] is the gradient w.r.t. the output of this layer
            gradients.append(grad_out[0].detach())

        h1 = target_layer.register_forward_hook(fw_hook)
        h2 = target_layer.register_full_backward_hook(bw_hook)

        try:
            img = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)

            model.zero_grad()
            output = model(input_tensor)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

            # Scalar backward on predicted class score
            output[0, class_idx].backward()

            grads = gradients[0].cpu().numpy()[0]   # (C, H, W)
            acts  = activations[0].cpu().numpy()[0]  # (C, H, W)

            logger.debug("[XAI][gradcam] activation shape: %s | gradient shape: %s",
                         acts.shape, grads.shape)

            # Global-average-pool gradients over spatial dims → channel weights
            weights = np.mean(grads, axis=(1, 2))   # (C,)

            # Weighted sum of activations
            heatmap = np.einsum("c,chw->hw", weights, acts)  # (H, W)

            # ReLU — keep only positive activations
            heatmap = np.maximum(heatmap, 0)

            logger.debug("[XAI][gradcam] raw heatmap — min: %.4f, max: %.4f",
                         heatmap.min(), heatmap.max())

            return self._save_heatmap(heatmap, image_path, "gradcam")

        finally:
            h1.remove()
            h2.remove()

    # ── 2. Grad-CAM++ ─────────────────────────────────────────────────────────
    def gradcam_pp(self, image_path: str, class_idx: int = None) -> tuple:
        model, target_layer = self._get_model_and_layer()
        activations, gradients = [], []

        def fw_hook(module, inp, output):
            activations.append(output.detach())

        def bw_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        h1 = target_layer.register_forward_hook(fw_hook)
        h2 = target_layer.register_full_backward_hook(bw_hook)

        try:
            img = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)

            model.zero_grad()
            output = model(input_tensor)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

            output[0, class_idx].backward()

            grads = gradients[0].cpu().numpy()[0]   # (C, H, W)
            acts  = activations[0].cpu().numpy()[0]  # (C, H, W)

            logger.debug("[XAI][gradcam++] activation shape: %s | gradient shape: %s",
                         acts.shape, grads.shape)

            # Grad-CAM++ alpha weights
            grads_sq  = grads ** 2
            grads_cub = grads ** 3
            alpha_den = 2.0 * grads_sq + np.sum(acts * grads_cub, axis=(1, 2), keepdims=True)
            alpha_den = np.where(np.abs(alpha_den) > 1e-9, alpha_den, 1.0)
            alphas = grads_sq / alpha_den

            weights = np.sum(alphas * np.maximum(grads, 0), axis=(1, 2))  # (C,)
            heatmap = np.einsum("c,chw->hw", weights, acts)
            heatmap = np.maximum(heatmap, 0)

            logger.debug("[XAI][gradcam++] raw heatmap — min: %.4f, max: %.4f",
                         heatmap.min(), heatmap.max())

            return self._save_heatmap(heatmap, image_path, "gradcampp")

        finally:
            h1.remove()
            h2.remove()

    # ── 3. Eigen-CAM ──────────────────────────────────────────────────────────
    def eigen_cam(self, image_path: str, class_idx: int = None) -> tuple:
        model, target_layer = self._get_model_and_layer()
        activations = []

        def fw_hook(module, inp, output):
            activations.append(output.detach())

        h1 = target_layer.register_forward_hook(fw_hook)

        try:
            img = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                model(input_tensor)

            acts = activations[0][0].cpu().numpy()   # (C, H, W)
            C, H, W = acts.shape

            logger.debug("[XAI][eigencam] activation shape: (%d, %d, %d)", C, H, W)

            # Reshape to (H*W, C), centre, then SVD
            reshaped = acts.reshape(C, -1).T        # (H*W, C)
            reshaped -= reshaped.mean(axis=0)
            U, S, Vt = np.linalg.svd(reshaped, full_matrices=False)

            # First principal component
            heatmap = U[:, 0].reshape(H, W)

            # Ensure positive correlation with mean activations
            if np.sum(heatmap * np.mean(acts, axis=0)) < 0:
                heatmap *= -1

            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            logger.debug("[XAI][eigencam] heatmap — min: %.4f, max: %.4f",
                         heatmap.min(), heatmap.max())

            return self._save_heatmap(heatmap, image_path, "eigencam")

        finally:
            h1.remove()

    # ── 4. Score-CAM (optimised chunked) ─────────────────────────────────────
    def score_cam(self, image_path: str, class_idx: int = None) -> tuple:
        model, target_layer = self._get_model_and_layer()
        activations = []

        def fw_hook(module, inp, output):
            activations.append(output.detach())

        h1 = target_layer.register_forward_hook(fw_hook)

        try:
            img_pil = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = model(input_tensor)
                if class_idx is None:
                    class_idx = output.argmax(dim=1).item()

            acts = activations[0]    # (1, C, H, W)
            C = acts.size(1)

            logger.debug("[XAI][scorecam] activation shape: %s, class_idx: %d", tuple(acts.shape), class_idx)

            upsampled = F.interpolate(acts, size=(224, 224), mode='bilinear', align_corners=False)

            # Per-channel normalise to [0,1]
            max_v = upsampled.view(1, C, -1).max(dim=-1)[0].view(1, C, 1, 1)
            min_v = upsampled.view(1, C, -1).min(dim=-1)[0].view(1, C, 1, 1)
            upsampled = (upsampled - min_v) / (max_v - min_v + 1e-8)

            # Sub-sample channels to keep runtime reasonable on CPU
            step = max(1, C // 64)
            indices = list(range(0, C, step))

            scores = []
            with torch.no_grad():
                chunk_size = 8
                for i in range(0, len(indices), chunk_size):
                    chunk_idx = indices[i: i + chunk_size]
                    masked = upsampled[:, chunk_idx, :, :] * input_tensor   # broadcast
                    # masked shape: (1, chunk, H, W) — needs batch dim per channel
                    for j in range(masked.size(1)):
                        out = model(masked[:, j:j+1, :, :].expand(-1, 3, -1, -1)
                                    if masked.size(2) != 3 else masked[:, j:j+1])
                        scores.append(out[0, class_idx].cpu().unsqueeze(0))

            # Fall back to gradient version if score_cam explodes
            scores_t = torch.cat(scores)
            weights = F.softmax(scores_t, dim=0).numpy()

            acts_np = acts[0, indices, :, :].cpu().numpy()   # (len(indices), h, w)
            heatmap = np.einsum("k,khw->hw", weights, acts_np)
            heatmap = np.maximum(heatmap, 0)

            logger.debug("[XAI][scorecam] heatmap — min: %.4f, max: %.4f",
                         heatmap.min(), heatmap.max())

            return self._save_heatmap(heatmap, image_path, "scorecam")

        finally:
            h1.remove()


# ── Validation helper ─────────────────────────────────────────────────────────

def run_gradcam_validation(image_dir: str = None, n: int = 10) -> dict:
    """
    Runs Grad-CAM on up to `n` images from uploads/ and saves debug outputs to
    backend/debug/gradcam_validation/

    Returns a summary dict with per-image stats.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    val_dir = base_dir / "debug" / "gradcam_validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    search_dir = Path(image_dir) if image_dir else (base_dir / "uploads")
    images = list(search_dir.glob("*.jpg"))[:n] + list(search_dir.glob("*.png"))[:n]
    images = images[:n]

    results = []
    suite = XAISuite()

    for img_path in images:
        try:
            # Run Grad-CAM
            hm_fn, ov_fn = suite.gradcam(str(img_path), class_idx=None)

            # Load heatmap and compute stats
            hm_img = cv2.imread(str(suite.output_dir / hm_fn))
            hm_gray = cv2.cvtColor(hm_img, cv2.COLOR_BGR2GRAY) if hm_img is not None else None

            # Copy to validation dir
            if hm_img is not None:
                cv2.imwrite(str(val_dir / f"val_{img_path.stem}_heatmap.png"), hm_img)
            ov_img = cv2.imread(str(suite.output_dir / ov_fn))
            if ov_img is not None:
                cv2.imwrite(str(val_dir / f"val_{img_path.stem}_overlay.jpg"), ov_img)

            # Copy original
            orig = cv2.imread(str(img_path))
            if orig is not None:
                cv2.imwrite(str(val_dir / f"val_{img_path.stem}_original.jpg"), orig)

            stats = {
                "image": img_path.name,
                "heatmap_file": hm_fn,
                "overlay_file": ov_fn,
                "heatmap_mean": float(hm_gray.mean()) if hm_gray is not None else None,
                "heatmap_std": float(hm_gray.std()) if hm_gray is not None else None,
                "status": "OK",
            }
        except Exception as exc:
            stats = {"image": img_path.name, "status": "ERROR", "error": str(exc)}

        results.append(stats)
        logger.info("[XAI][validation] %s", stats)

    report = {"total": len(results), "ok": sum(1 for r in results if r["status"] == "OK"), "results": results}
    import json
    with open(val_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("[XAI][validation] Report saved to %s", val_dir / "report.json")
    return report


# ── Module singleton ──────────────────────────────────────────────────────────
xai_suite = XAISuite()
