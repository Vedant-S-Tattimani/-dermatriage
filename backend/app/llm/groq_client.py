"""
Async Groq client wrapper for generating patient-facing explanations.
"""
from groq import AsyncGroq

from app.config import settings
from app.llm.prompts import (
    SYSTEM_PROMPT,
    EXPLAIN_USER_TEMPLATE,
    USER_QUESTION_SECTION_TEMPLATE,
)
from app.models.schemas import ExplainRequest, ExplainResponse
from app.utils.disclaimers import STANDARD_DISCLAIMER
from app.utils.logger import get_logger
from app.llm.rag import retrieve_context

logger = get_logger(__name__)


class GroqClient:
    """Thin async wrapper around the Groq chat-completions API."""

    def __init__(self):
        self._client: AsyncGroq | None = None

    @property
    def client(self) -> AsyncGroq:
        if self._client is None:
            if not settings.GROQ_API_KEY:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. "
                    "Add it to your .env file to enable LLM explanations."
                )
            self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        return self._client

    def _build_user_message(self, payload: ExplainRequest) -> str:
        uq_section = ""
        if payload.user_question:
            uq_section = USER_QUESTION_SECTION_TEMPLATE.format(
                user_question=payload.user_question
            )
        
        # Retrieve grounding medical context
        medical_context = retrieve_context(payload.class_name)

        return EXPLAIN_USER_TEMPLATE.format(
            class_name=payload.class_name,
            severity_level=payload.condition_type.value,
            confidence_pct=f"{payload.confidence * 100:.1f}",
            risk_level=payload.risk_level.value,
            recommendation=payload.recommendation,
            medical_context_section=medical_context,
            user_question_section=uq_section,
            language=payload.language,
        )

    async def explain(self, payload: ExplainRequest) -> ExplainResponse:
        """
        Call the Groq chat-completions API and return an ExplainResponse.
        Falls back to a canned explanation if the API key is missing.
        """
        try:
            user_message = self._build_user_message(payload)
            logger.info(
                "LLM explain call — model=%s class=%s risk=%s",
                settings.GROQ_MODEL,
                payload.class_name,
                payload.risk_level.value,
            )
            response = await self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                max_tokens=settings.GROQ_MAX_TOKENS,
                temperature=settings.GROQ_TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            explanation_text = response.choices[0].message.content.strip()
        except RuntimeError as exc:
            # API key not configured — return fallback
            logger.warning("Groq not configured: %s", exc)
            explanation_text = self._fallback_explanation(payload)

        return ExplainResponse(
            explanation=explanation_text,
            disclaimer=STANDARD_DISCLAIMER,
            model_used=settings.GROQ_MODEL,
        )

    @staticmethod
    def _fallback_explanation(payload: ExplainRequest) -> str:
        return (
            f"The AI screening tool identified the lesion as a possible "
            f"**{payload.class_name}** ({payload.condition_type.value}) with "
            f"{payload.confidence * 100:.1f}% confidence. "
            f"Risk level: **{payload.risk_level.value}**.\n\n"
            f"{payload.recommendation}\n\n"
            "*(LLM explanations are unavailable — GROQ_API_KEY not configured.)*"
        )


groq_client = GroqClient()
