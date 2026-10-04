"""
Tests for the decision layer (multimodal risk engine).
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.decision_layer import classify_risk, get_condition_data
from app.models.enums import RiskLevel, ConditionType
from app.models.schemas import TriageMetadata


def _flat_probs(n: int = 7, high_idx: int = 0, confidence: float = 0.9):
    """Helper: build a probability list with one dominant class."""
    probs = [(1 - confidence) / (n - 1)] * n
    probs[high_idx] = confidence
    return probs


class TestClassifyRisk:
    def test_high_confidence_low_risk(self):
        # Nevi is benign, so high confidence -> LOW risk
        decision = classify_risk("nevus", 0.95, all_probabilities=_flat_probs(confidence=0.95))
        assert decision["risk_level"] == RiskLevel.LOW
        assert "Melanocytic Nevus" in decision["full_name"]

    def test_melanoma_medium_confidence_is_still_high(self):
        """Melanoma at 50% confidence should still be HIGH risk."""
        decision = classify_risk("melanoma", 0.50, all_probabilities=_flat_probs(confidence=0.50))
        assert decision["risk_level"] == RiskLevel.HIGH
        assert "HIGH RISK" in decision["recommendation"]

    def test_bcc_medium_confidence_is_high(self):
        decision = classify_risk("bcc", 0.45, all_probabilities=_flat_probs(confidence=0.45))
        assert decision["risk_level"] == RiskLevel.HIGH

    def test_medium_confidence_benign(self):
        decision = classify_risk("df", 0.55, all_probabilities=_flat_probs(confidence=0.55))
        assert decision["risk_level"] == RiskLevel.LOW

    def test_low_confidence(self):
        decision = classify_risk("df", 0.25, all_probabilities=_flat_probs(confidence=0.25))
        assert decision["risk_level"] == RiskLevel.UNCERTAIN

    def test_multimodal_escalation_bleeding(self):
        """Metadata red flags should escalate risk."""
        # Nevus is normally LOW risk, but bleeding should make it HIGH or at least escalate
        metadata = TriageMetadata(bleeding=True)
        decision = classify_risk("nevus", 0.80, metadata=metadata)
        assert decision["risk_level"] == RiskLevel.HIGH
        assert "RED FLAG" in decision["recommendation"]

    def test_multimodal_age_escalation(self):
        """Age should be noted for malignant-like lesions."""
        metadata = TriageMetadata(age=70)
        decision = classify_risk("melanoma", 0.80, metadata=metadata)
        assert "patient age (>50)" in decision["recommendation"]

    def test_cumulative_malignant_risk_escalation(self):
        """
        Test that a top-1 benign class (e.g. nevus) with high/medium confidence
        escalates to HIGH risk if the cumulative malignant risk (Melanoma + BCC) exceeds 15%.
        """
        # settings.CLASS_NAMES: ["actinic", "bcc", "bkl", "df", "melanoma", "nevus", "vascular"]
        # Index of bcc is 1, melanoma is 4, nevus is 5.
        probs = [0.05, 0.05, 0.05, 0.05, 0.35, 0.36, 0.09]
        decision = classify_risk("nevus", 0.36, all_probabilities=probs)
        assert decision["risk_level"] == RiskLevel.HIGH
        assert "Cumulative malignant risk (Melanoma + BCC) exceeds 15%" in decision["recommendation"]


class TestConditionData:
    def test_melanoma_is_malignant(self):
        data = get_condition_data("melanoma")
        assert data["condition_type"] == ConditionType.MALIGNANT
        assert "Melanoma" == data["full_name"]

    def test_nevi_is_benign(self):
        data = get_condition_data("nevus")
        assert data["condition_type"] == ConditionType.BENIGN

    def test_actinic_is_pre_malignant(self):
        data = get_condition_data("actinic")
        assert data["condition_type"] == ConditionType.PRE_MALIGNANT
