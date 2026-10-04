import re

# Blocked keyword categories — easy to extend by adding new entries
BLOCKED_KEYWORDS = {
    "prescription": [
        "medicine", "medication", "prescription", "prescribe", "prescribed",
    ],
    "treatment": [
        "treatment", "treat", "therapy", "procedure", "surgery",
    ],
    "drugs": [
        "drug", "dose", "dosage", "tablet", "capsule", "injection",
        "antibiotic", "steroid", "ointment", "cream",
    ],
    "diagnosis_claim": [
        "diagnose me", "what disease do i have", "tell me my condition",
        "confirm diagnosis", "definitely", "certain", "guarantee",
    ],
    "emergency": [
        "911", "emergency", "ER", "urgent care",
    ],
    "medical_advice": [
        "should i get a biopsy", "do i need surgery",
    ],
}

BLOCK_MESSAGE = (
    "This system cannot provide medical prescriptions, drug recommendations, "
    "or diagnostic claims. Please consult a licensed healthcare professional."
)


def _build_pattern():
    """Compile a single regex from all blocked keywords for efficient matching."""
    all_keywords = []
    for keywords in BLOCKED_KEYWORDS.values():
        all_keywords.extend(keywords)
    escaped = [re.escape(kw) for kw in all_keywords]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


_BLOCKED_PATTERN = _build_pattern()


def validate_prompt(user_query: str) -> dict:
    """
    Validates a user query against the blocked keyword rules.

    Returns:
        {"allowed": True} if the query is safe.
        {"allowed": False, "message": "...", "matched": "..."} if blocked.
    """
    if not user_query or not user_query.strip():
        return {"allowed": True}

    match = _BLOCKED_PATTERN.search(user_query)
    if match:
        return {
            "allowed": False,
            "message": BLOCK_MESSAGE,
            "matched": match.group(0),
        }

    return {"allowed": True}


class Guardrails:
    """Object-oriented interface used by the existing explain route."""

    def check(self, user_query: str) -> tuple[bool, str]:
        """
        Returns (blocked: bool, reason: str).
        blocked=True means the query should be rejected.
        """
        result = validate_prompt(user_query)
        if not result.get("allowed"):
            return True, result.get("message", BLOCK_MESSAGE)
        return False, ""

    def safe_question(self, user_query: str) -> bool:
        """Returns True if the query is safe, False otherwise."""
        return not self.check(user_query)[0]


guardrails = Guardrails()
