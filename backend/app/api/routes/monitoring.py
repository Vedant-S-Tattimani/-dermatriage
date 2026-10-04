"""
Active Learning & Monitoring Routes.
- Feedback collection
- Performance analytics
"""
from fastapi import APIRouter, HTTPException, status, Depends
from app.models.schemas import FeedbackRequest, AnalyticsResponse
from app.services.active_learning_service import active_learning_service
from app.api.dependencies import get_optional_api_key

router = APIRouter()

@router.post(
    "/feedback",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit diagnostic feedback",
    description="Submit correct labels for a triage request to improve future model versions."
)
async def submit_feedback(
    feedback: FeedbackRequest,
    _: None = Depends(get_optional_api_key)
):
    """Enqueues feedback for the active learning pipeline."""
    try:
        await active_learning_service.save_feedback(feedback)
        return {"status": "success", "message": "Feedback recorded for analysis."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save feedback: {e}"
        )

@router.get(
    "/analytics",
    response_model=AnalyticsResponse,
    summary="Get model monitoring analytics",
    description="Returns metrics on prediction distribution, confidence trends, and feedback count."
)
async def get_analytics(_: None = Depends(get_optional_api_key)):
    """Calculates production performance metrics from monitoring logs."""
    try:
        return await active_learning_service.get_analytics()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics engine error: {e}"
        )
