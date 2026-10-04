"""
Integration tests for the FastAPI routes (using TestClient).

These tests mock the triage_service so they run without a GPU or model weights.
"""
import io
import sys
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app
from app.models.schemas import TriageResponse, ClassProbability
from app.models.enums import RiskLevel, ConditionType


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _fake_image_bytes() -> bytes:
    """Return a minimal JPEG image as bytes."""
    buf = io.BytesIO()
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


_MOCK_TRIAGE = TriageResponse(
    request_id="test-uuid",
    class_name="nevus",
    full_name="Melanocytic nevi",
    condition_type=ConditionType.BENIGN,
    confidence=0.82,
    risk_level=RiskLevel.LOW,
    recommendation="Test recommendation",
    clinical_description="Test description",
    clinical_features=["Feature 1", "Feature 2"],
    urgency="Non-urgent",
    all_probabilities=[
        ClassProbability(class_name="nevus", probability=0.82),
    ],
    top_3=[
        ClassProbability(class_name="nevus", probability=0.82),
    ],
    gradcam_url="/uploads/gradcam_test.png",
    inference_time_ms=250.0,
    disclaimer="Test disclaimer",
)



# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_ok(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "timestamp" in data
        assert "model_ready" in data


# ── Triage endpoint ───────────────────────────────────────────────────────────

class TestTriageEndpoint:
    def test_valid_upload(self, client):
        with patch("app.services.triage_service.triage_service.run", return_value=_MOCK_TRIAGE):
            response = client.post(
                "/api/v1/triage",
                files={"file": ("test.jpg", _fake_image_bytes(), "image/jpeg")},
            )
        assert response.status_code == 201
        data = response.json()
        assert data["class_name"] == "nevus"
        assert data["risk_level"] == "LOW"
        assert "disclaimer" in data

    def test_unsupported_content_type(self, client):
        response = client.post(
            "/api/v1/triage",
            files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert response.status_code == 415

    def test_missing_file(self, client):
        response = client.post("/api/v1/triage")
        assert response.status_code == 422   # Unprocessable Entity


# ── Explain endpoint ──────────────────────────────────────────────────────────

class TestExplainEndpoint:
    _PAYLOAD = {
        "class_name": "Melanocytic nevi",
        "condition_type": "BENIGN",
        "confidence": 0.82,
        "risk_level": "HIGH",
        "recommendation": "Seek evaluation",
        "user_question": "What does this mean?",
    }

    def test_valid_explain(self, client):
        mock_response = MagicMock()
        mock_response.explanation = "This is a test explanation."
        mock_response.disclaimer  = "Test disclaimer."
        mock_response.model_used  = "llama3-70b-8192"

        with patch(
            "app.llm.groq_client.groq_client.explain",
            return_value=mock_response,
        ):
            response = client.post("/api/v1/explain", json=self._PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        assert "explanation" in data
        assert "disclaimer" in data

    def test_blocked_user_question(self, client):
        payload = {**self._PAYLOAD, "user_question": "Prescribe me a cream"}
        response = client.post("/api/v1/explain", json=payload)
        assert response.status_code == 400
        assert "guardrail" in response.json()["detail"].lower()

    def test_missing_required_fields(self, client):
        response = client.post("/api/v1/explain", json={"class_name": "Melanoma"})
        assert response.status_code == 422


# ── Chat endpoint ─────────────────────────────────────────────────────────────

class TestChatEndpoint:
    def test_valid_chat(self, client):
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "This is a benign mole."
        mock_completion.choices = [mock_choice]

        with patch("groq.AsyncGroq") as mock_groq_class:
            mock_client = MagicMock()
            async def mock_create(*args, **kwargs):
                return mock_completion
            mock_client.chat.completions.create = mock_create
            mock_groq_class.return_value = mock_client

            with patch("app.api.routes.chat.settings") as mock_settings:
                mock_settings.GROQ_API_KEY = "fake-key"
                mock_settings.GROQ_MODEL = "llama-3.3-70b-versatile"
                
                payload = {
                    "messages": [{"role": "user", "content": "Tell me about my mole."}],
                    "context": "triage context"
                }
                response = client.post("/api/v1/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert "This is a benign mole." in data["reply"]

    def test_blocked_chat(self, client):
        with patch("app.api.routes.chat.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = "fake-key"
            payload = {
                "messages": [{"role": "user", "content": "Can you prescribe me a cream?"}],
                "context": "triage context"
            }
            response = client.post("/api/v1/chat", json=payload)
        
        assert response.status_code == 400
        assert "blocked by safety guardrail" in response.json()["detail"].lower()
