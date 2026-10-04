"""
Contextual Risk Engine — combines AI predictions with patient metadata.
Provides multimodal triage logic.
"""
from typing import List, Optional, Dict, Any
from app.models.enums import RiskLevel, ConditionType
from app.core.medical_profiles import MEDICAL_PROFILES
from app.config import settings
from app.models.schemas import TriageMetadata

def get_condition_data(class_name: str) -> Dict[str, Any]:
    """Returns the detailed profile for a class code."""
    code = class_name.lower().strip()
    return MEDICAL_PROFILES.get(code, {
        "full_name": "Unknown Condition",
        "description": "The condition could not be specifically identified.",
        "clinical_features": [],
        "urgency": "Consult a professional",
        "risk_level": RiskLevel.UNCERTAIN,
        "condition_type": ConditionType.UNKNOWN
    })

def classify_risk(
    class_name: str, 
    confidence: float, 
    metadata: Optional[TriageMetadata] = None,
    all_probabilities: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Multimodal Risk Engine:
    1. Visual Prediction (AI)
    2. Clinical Context (Metadata)
    3. Uncertainty Safeguards
    """
    profile = get_condition_data(class_name)
    
    base_risk = profile["risk_level"]
    condition_type = profile["condition_type"]
    recommendation = ""
    clinical_notes = []

    # ── 1. Metadata Risk Escalation ──────────────────────────────────────────
    elevated_risk = base_risk
    
    if metadata:
        # Age Factor (Higher concern for older patients with malignant-like lesions)
        if metadata.age and metadata.age > 50 and condition_type == ConditionType.MALIGNANT:
            clinical_notes.append("Elevated risk due to patient age (>50) for suspected malignancy.")

        # Symptom Red Flags
        if metadata.bleeding:
            clinical_notes.append("⚠️ RED FLAG: Lesion is reported to be bleeding.")
            if base_risk != RiskLevel.HIGH:
                elevated_risk = RiskLevel.HIGH
        
        if metadata.rapid_change:
            clinical_notes.append("⚠️ RED FLAG: Lesion reported to be changing rapidly.")
            elevated_risk = RiskLevel.HIGH

        if metadata.itching:
            clinical_notes.append("Symptom: Patient reports itching (Pruritus).")

    # ── 1.5. Cumulative Malignant Risk Escalation ───────────────────────────
    if all_probabilities:
        try:
            bcc_idx = settings.CLASS_NAMES.index("bcc")
            mel_idx = settings.CLASS_NAMES.index("melanoma")
            bcc_prob = all_probabilities[bcc_idx]
            mel_prob = all_probabilities[mel_idx]
            cumulative_malignant_prob = bcc_prob + mel_prob
            if cumulative_malignant_prob > 0.15:
                elevated_risk = RiskLevel.HIGH
                clinical_notes.append("⚠️ CLINICAL WARNING: Cumulative malignant risk (Melanoma + BCC) exceeds 15%.")
        except Exception:
            pass

    # ── 2. Uncertainty Detection (Clinical Safeguard) ─────────────────────────
    if confidence < settings.UNCERTAINTY_THRESHOLD:
        risk_level = RiskLevel.UNCERTAIN
        recommendation = (
            f"❓ LOW CONFIDENCE ({confidence*100:.1f}%) — The AI visual analysis is below the certainty threshold. "
            "Because automated pattern matching is inconclusive, a physical examination is required."
        )
    else:
        risk_level = elevated_risk
        
        # Build tailored recommendation
        if risk_level == RiskLevel.HIGH:
            recommendation = (
                f"⚠️ HIGH RISK — {profile['full_name']} is a serious clinical concern. "
                f"{profile['urgency']}. "
            )
        elif risk_level == RiskLevel.MEDIUM:
            recommendation = (
                f"🔔 MEDIUM RISK — {profile['full_name']} detected. "
                f"{profile['urgency']}. "
            )
        else:
            recommendation = (
                f"✅ LOW RISK — {profile['full_name']} appears likely. "
                f"{profile['urgency']}. "
            )

        # Add context from metadata to recommendation
        if clinical_notes:
            recommendation += " " + " ".join(clinical_notes)

    return {
        "risk_level": risk_level,
        "condition_type": condition_type,
        "recommendation": recommendation,
        "full_name": profile["full_name"],
        "clinical_description": profile["description"],
        "clinical_features": profile["clinical_features"],
        "urgency": profile["urgency"]
    }
