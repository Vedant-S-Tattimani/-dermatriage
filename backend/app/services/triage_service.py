"""
Triage service — orchestrates: segment → classify → XAI → decide.
Multimodal integration with patient metadata.
"""
import time
from pathlib import Path
from typing import Optional
from app.config import settings
from app.core.classifier import classifier
from app.core.xai import xai_suite
from app.core.decision_layer import classify_risk
from app.core.segmenter import segmenter
from app.models.schemas import TriageResponse, ClassProbability, TriageMetadata
from app.utils.disclaimers import STANDARD_DISCLAIMER
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TriageService:
    def run(self, image_path: str, request_id: str, metadata: Optional[TriageMetadata] = None) -> TriageResponse:
        """
        Full multimodal triage pipeline with performance tracking.
        """
        pipeline_start = time.time()
        
        # ── Step 0: Segmentation ─────────────────────────────────────────────
        logger.info("[%s] Step 0/3 — Segmentation", request_id)
        try:
            cropped_path, mask_path = segmenter.segment_and_crop(image_path)
            cropped_url = f"/uploads/{Path(cropped_path).name}" if cropped_path != image_path else None
            mask_url = f"/uploads/{Path(mask_path).name}" if mask_path else None
            # Use cropped image for the rest of the pipeline
            analysis_path = cropped_path
        except Exception as exc:
            logger.warning("[%s] Segmentation failed (non-fatal): %s", request_id, exc)
            analysis_path = image_path
            cropped_url = None
            mask_url = None

        # ── Step 1: Classification ───────────────────────────────────────────
        logger.info("[%s] Step 1/3 — Classification", request_id)
        prediction = classifier.predict(analysis_path, use_segmentation=False)
        class_name = prediction["label"]
        confidence = prediction["confidence"]
        all_probs  = prediction["all_probs"]
        top_3      = prediction["top_3"]
        uncertainty_flags = prediction.get("uncertainty_flags", [])
        is_ood = prediction.get("is_ood", False)

        # ── Step 2: XAI Heatmaps ──────────────────────────────────────────────
        logger.info("[%s] Step 2/3 — XAI Heatmaps", request_id)
        xai_reports = {}         # overlay URLs
        xai_heatmap_reports = {} # pure heatmap URLs
        try:
            class_idx = settings.CLASS_NAMES.index(class_name)
            for method in ["gradcam", "gradcampp", "eigencam"]:
                both = xai_suite.generate_all(analysis_path, method=method, class_idx=class_idx)
                xai_reports[method] = f"/uploads/{both['overlay']}"
                xai_heatmap_reports[method] = f"/uploads/{both['heatmap']}"
            gradcam_url = xai_reports.get("gradcam")
        except Exception as exc:
            logger.warning("[%s] XAI Suite failed (non-fatal): %s", request_id, exc)
            gradcam_url = None

        # ── Step 3: Decision Layer (Multimodal) ──────────────────────────────
        logger.info("[%s] Step 3/3 — Decision layer", request_id)
        decision = classify_risk(class_name, confidence, metadata=metadata, all_probabilities=all_probs)

        # Map to Pydantic models
        all_probabilities = [
            ClassProbability(class_name=settings.CLASS_NAMES[i], probability=p)
            for i, p in enumerate(all_probs)
        ]
        
        top_3_probabilities = [
            ClassProbability(class_name=item["class_name"], probability=item["probability"])
            for item in top_3
        ]

        total_time_ms = (time.time() - pipeline_start) * 1000

        return TriageResponse(
            request_id=request_id,
            class_name=class_name,
            full_name=decision["full_name"],
            condition_type=decision["condition_type"],
            confidence=confidence,
            risk_level=decision["risk_level"],
            recommendation=decision["recommendation"],
            clinical_description=decision["clinical_description"],
            clinical_features=decision["clinical_features"],
            urgency=decision["urgency"],
            all_probabilities=all_probabilities,
            top_3=top_3_probabilities,
            gradcam_url=gradcam_url,
            xai_reports=xai_reports,
            xai_heatmap_reports=xai_heatmap_reports,
            cropped_image_url=cropped_url,
            segmentation_mask_url=mask_url,
            uncertainty_flags=uncertainty_flags,
            is_ood=is_ood,
            inference_time_ms=total_time_ms,
            disclaimer=STANDARD_DISCLAIMER,
        )


triage_service = TriageService()
