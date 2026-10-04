"""
POST /api/v1/explain — Generate an LLM explanation for a triage result.

This is a separate endpoint so that explanation can be called independently
(e.g., after retrieving a cached triage result) without re-running the model.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_optional_api_key
from app.models.schemas import ExplainRequest, ExplainResponse
from app.llm.groq_client import groq_client
from app.llm.guardrails import guardrails
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="LLM Explanation",
    description=(
        "Send a triage result and receive a plain-language explanation. "
        "Guardrails block requests for prescriptions or definitive diagnoses."
    ),
)
async def explain_result(
    payload: ExplainRequest,
    _: None = Depends(get_optional_api_key),
):
    # ── Guardrail check ───────────────────────────────────────────────────────
    if payload.user_question:
        blocked, reason = guardrails.check(payload.user_question)
        if blocked:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Request blocked by safety guardrail: {reason}",
            )

    try:
        explanation = await groq_client.explain(payload)
    except Exception as exc:
        logger.exception("LLM explanation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM service error: {exc}",
        ) from exc

    return explanation
