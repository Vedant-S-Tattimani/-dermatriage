"""
POST /api/v1/triage — Primary endpoint for uploading an image and receiving a triage result.
"""
from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Form
from app.api.dependencies import get_optional_api_key
from app.models.schemas import TriageResponse, TriageMetadata
from app.services.image_service import image_service
from app.services.triage_service import triage_service
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/triage",
    response_model=TriageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit image for triage",
    description=(
        "Upload a skin lesion image (JPEG/PNG/WebP) to be classified. "
        "Returns the top prediction, risk level, clinical recommendation, and Grad-CAM saliency map."
    ),
)
async def process_triage(
    file: UploadFile = File(...),
    age: Optional[int] = Form(None),
    location: Optional[str] = Form(None),
    duration: Optional[str] = Form(None),
    itching: bool = Form(False),
    bleeding: bool = Form(False),
    rapid_change: bool = Form(False),
    _: None = Depends(get_optional_api_key),
):
    """
    Triage endpoint:
    1. Validate file and metadata.
    2. Save to /uploads.
    3. Run multimodal triage pipeline (Service layer).
    """
    request_id = image_service.generate_request_id()
    logger.info("[%s] Received triage request: %s", request_id, file.filename)

    # ── 1. Validation ────────────────────────────────────────────────────────
    image_service.validate_file(file)
    
    metadata = TriageMetadata(
        age=age,
        location=location,
        duration=duration,
        itching=itching,
        bleeding=bleeding,
        rapid_change=rapid_change
    )

    # ── 2. Save file ─────────────────────────────────────────────────────────
    try:
        temp_path = await image_service.save_upload(file, request_id)
    except Exception as exc:
        logger.exception("[%s] Failed to save upload", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error saving the uploaded image.",
        ) from exc

    # ── 3. Triage Pipeline ───────────────────────────────────────────────────
    try:
        # Offload CPU-heavy ML/XAI tasks to a thread pool (Async Safety)
        from fastapi.concurrency import run_in_threadpool
        result = await run_in_threadpool(triage_service.run, temp_path, request_id, metadata)
        
        # ── 4. Active Learning & Monitoring ──────────────────────────────────
        from app.services.active_learning_service import active_learning_service
        # Background task for logging (non-blocking for the user)
        from fastapi import BackgroundTasks
        # Note: We need BackgroundTasks in the function signature, or just await it here
        # For simplicity in this script, we'll await it since it's already in a threadpool-heavy flow
        await active_learning_service.log_prediction(result, temp_path)
        
    except Exception as exc:
        logger.exception("[%s] Triage pipeline failed", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Triage processing failed: {exc}",
        ) from exc

    return result
