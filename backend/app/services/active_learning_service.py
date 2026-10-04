"""
Active Learning Service — handles prediction monitoring and user feedback collection.
Logs uncertain cases and build retraining queues.
"""
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiofiles
from app.config import settings
from app.models.schemas import TriageResponse, FeedbackRequest, AnalyticsResponse, ClassStat
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ActiveLearningService:
    def __init__(self):
        # Ensure directories exist
        settings.ACTIVE_LEARNING_DIR.mkdir(parents=True, exist_ok=True)
        (settings.ACTIVE_LEARNING_DIR / "difficult_cases").mkdir(parents=True, exist_ok=True)

    async def log_prediction(self, response: TriageResponse, image_path: str):
        """Logs every prediction for drift monitoring and saves difficult cases."""
        log_entry = {
            "timestamp": time.time(),
            "request_id": response.request_id,
            "prediction": response.class_name,
            "confidence": response.confidence,
            "risk_level": response.risk_level,
            "is_uncertain": response.confidence < settings.UNCERTAINTY_THRESHOLD
        }

        # 1. Append to monitoring log
        async with aiofiles.open(settings.MONITORING_LOG_PATH, mode="a") as f:
            await f.write(json.dumps(log_entry) + "\n")

        # 2. Archive 'difficult' cases for expert re-labeling
        # Condition: Low confidence OR high-risk prediction
        if log_entry["is_uncertain"] or response.risk_level == "HIGH":
            dest = settings.ACTIVE_LEARNING_DIR / "difficult_cases" / f"{response.request_id}.jpg"
            try:
                shutil.copy2(image_path, dest)
                logger.info("[%s] Archived as difficult case for active learning.", response.request_id)
            except Exception as e:
                logger.error("Failed to archive difficult case: %s", e)

    async def save_feedback(self, feedback: FeedbackRequest):
        """Saves user/clinician feedback to the active learning queue."""
        log_entry = {
            "timestamp": time.time(),
            "request_id": feedback.request_id,
            "correct_class": feedback.correct_class,
            "comments": feedback.user_comments
        }
        
        async with aiofiles.open(settings.FEEDBACK_LOG_PATH, mode="a") as f:
            await f.write(json.dumps(log_entry) + "\n")
            
        logger.info("[%s] Feedback received: %s", feedback.request_id, feedback.correct_class)

    async def get_analytics(self) -> AnalyticsResponse:
        """Computes model performance and drift metrics from logs."""
        predictions = []
        if settings.MONITORING_LOG_PATH.exists():
            async with aiofiles.open(settings.MONITORING_LOG_PATH, mode="r") as f:
                async for line in f:
                    if not line.strip():
                        continue
                    predictions.append(json.loads(line))

        feedbacks = []
        if settings.FEEDBACK_LOG_PATH.exists():
            async with aiofiles.open(settings.FEEDBACK_LOG_PATH, mode="r") as f:
                async for line in f:
                    if not line.strip():
                        continue
                    feedbacks.append(json.loads(line))

        if not predictions:
            return AnalyticsResponse(
                total_predictions=0, total_feedback=len(feedbacks),
                avg_system_confidence=0.0, uncertain_count=0,
                class_distribution=[], recent_feedback=feedbacks[-10:]
            )

        total_conf = sum(p["confidence"] for p in predictions)
        uncertain = sum(1 for p in predictions if p["is_uncertain"])
        
        # Class distribution
        dist_map = {}
        for p in predictions:
            cls = p["prediction"]
            if cls not in dist_map:
                dist_map[cls] = {"count": 0, "conf_sum": 0.0}
            dist_map[cls]["count"] += 1
            dist_map[cls]["conf_sum"] += p["confidence"]

        class_stats = [
            ClassStat(
                class_name=cls,
                count=data["count"],
                avg_confidence=data["conf_sum"] / data["count"]
            )
            for cls, data in dist_map.items()
        ]

        return AnalyticsResponse(
            total_predictions=len(predictions),
            total_feedback=len(feedbacks),
            avg_system_confidence=total_conf / len(predictions),
            uncertain_count=uncertain,
            class_distribution=class_stats,
            recent_feedback=feedbacks[-10:]
        )

active_learning_service = ActiveLearningService()
