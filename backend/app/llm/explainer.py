"""
LLM-powered explanation generator using the project's Groq client.
"""
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

DISCLAIMER = "This is not a medical diagnosis. Please consult a licensed healthcare professional."

SYSTEM_PROMPT = (
    "You are a medical AI assistant for a skin-lesion triage system. Your role is to provide a calm, "
    "professional, and easy-to-understand clinical explanation based on the model's prediction. "
    "CRITICAL RULES: "
    "1. Explain what the condition means in simple language. "
    "2. Mention common visual characteristics (color, shape, irregularity) associated with the condition. "
    "3. Keep the explanation under 120 words. "
    "4. DO NOT claim a diagnosis or provide prescriptions. "
    "5. You MUST include the disclaimer: 'This is not a medical diagnosis.' "
    "6. Tone must be professional and empathetic."
)


def _build_user_prompt(prediction: str, confidence: float, risk_level: str) -> str:
    return (
        f"The AI model predicted the lesion as '{prediction}' "
        f"with a confidence of {confidence:.2f} ({confidence * 100:.1f}%). "
        f"The determined risk level is '{risk_level}'. "
        "Please provide a clear clinical explanation of what this means, its typical visual features, "
        "and appropriate safe next steps."
    )


def _fallback_explanation(prediction: str, confidence: float, risk_level: str) -> str:
    return (
        f"{DISCLAIMER} The system indicates a result of "
        f"'{prediction}' with {confidence * 100:.1f}% confidence (Risk: {risk_level}). "
        "Please consult a healthcare professional or dermatologist for a proper evaluation."
    )


async def generate_explanation_async(prediction: str, confidence: float, risk_level: str) -> dict:
    """
    Async version — generates a safe, plain-language explanation via Groq.
    """
    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — returning fallback explanation.")
        return {"explanation": _fallback_explanation(prediction, confidence, risk_level)}

    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        user_prompt = _build_user_prompt(prediction, confidence, risk_level)

        response = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.2,
        )

        explanation = response.choices[0].message.content.strip()

        if "This is not a medical diagnosis" not in explanation:
            explanation = DISCLAIMER + " " + explanation

        return {"explanation": explanation}

    except Exception as e:
        logger.error("LLM explanation failed: %s", e)
        return {"explanation": _fallback_explanation(prediction, confidence, risk_level)}


def generate_explanation(prediction: str, confidence: float, risk_level: str) -> dict:
    """
    Sync wrapper — for use outside of async contexts (e.g. test scripts).
    """
    return {"explanation": _fallback_explanation(prediction, confidence, risk_level)}
