"""
GET /health  — liveness + readiness probe.
"""
from fastapi import APIRouter
from datetime import datetime, timezone

from app.config import settings
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    """Returns service status and basic configuration."""
    from app.core.classifier import classifier
    
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_ready=(classifier.model is not None),
    )
