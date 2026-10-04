"""
AI Chat endpoint — allows follow-up questions about triage results via Groq.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

DISCLAIMER = "This is not a medical diagnosis. Please consult a licensed healthcare professional."

CHAT_SYSTEM_PROMPT = (
    "You are a helpful, empathetic medical AI assistant for a skin-lesion triage platform. "
    "You help users understand their triage results and answer follow-up questions. "
    "CRITICAL RULES: "
    "1. Be professional, calm, and empathetic. "
    "2. NEVER diagnose or prescribe medication. "
    "3. Always recommend consulting a dermatologist for definitive diagnosis. "
    "4. Keep responses concise (under 150 words). "
    "5. If asked about something outside dermatology or skin health, politely redirect. "
    "6. End every response with a reminder that this is not a medical diagnosis."
)


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context: Optional[str] = None  # triage context (prediction, confidence, risk)
    language: str = "en"


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """
    Send a chat message and receive an AI response grounded in the triage context.
    """
    # ── Guardrail check ───────────────────────────────────────────────────────
    from app.llm.guardrails import guardrails
    if request.messages:
        user_messages = [m for m in request.messages if m.role == "user"]
        if user_messages:
            latest_content = user_messages[-1].content
            blocked, reason = guardrails.check(latest_content)
            if blocked:
                raise HTTPException(
                    status_code=400,
                    detail=f"Request blocked by safety guardrail: {reason}\n\n⚠️ {DISCLAIMER}"
                )

    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — returning fallback response.")
        return ChatResponse(
            reply="AI chat is currently unavailable. Please consult a dermatologist for further questions. " + DISCLAIMER
        )

    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=settings.GROQ_API_KEY)

        # Build system message with triage context
        system_content = CHAT_SYSTEM_PROMPT
        if request.context:
            system_content += (
                f"\n\nCURRENT TRIAGE CONTEXT:\n{request.context}\n"
                "Use this context to give relevant, grounded answers about the user's results."
            )
        
        system_content += f"\n\nCRITICAL: You MUST respond entirely in the language corresponding to the code '{request.language}'."

        messages = [{"role": "system", "content": system_content}]

        # Add conversation history
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        response = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0.3,
        )

        reply = response.choices[0].message.content.strip()

        # Ensure disclaimer is present
        if "not a medical diagnosis" not in reply.lower():
            reply += f"\n\n⚠️ {DISCLAIMER}"

        return ChatResponse(reply=reply)

    except Exception as e:
        logger.error("Chat error: %s", e)
        return ChatResponse(
            reply="I'm sorry, I encountered an error processing your question. Please try again or consult a healthcare professional. " + DISCLAIMER
        )
