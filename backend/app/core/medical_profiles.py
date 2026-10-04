"""
Tailored medical insights for each of the 7 skin condition classes.
"""
from typing import Dict, TypedDict
from app.models.enums import RiskLevel, ConditionType

class ConditionProfile(TypedDict):
    full_name: str
    description: str
    clinical_features: list[str]
    urgency: str
    risk_level: RiskLevel
    condition_type: ConditionType

MEDICAL_PROFILES: Dict[str, ConditionProfile] = {
    "actinic": {
        "full_name": "Actinic Keratosis",
        "description": "A pre-cancerous skin growth usually caused by sun damage.",
        "clinical_features": [
            "Rough, scaly patch on sun-exposed areas",
            "May feel like sandpaper",
            "Color varies from brown to red or yellowish"
        ],
        "urgency": "Non-urgent, but requires monitoring",
        "risk_level": RiskLevel.MEDIUM,
        "condition_type": ConditionType.PRE_MALIGNANT
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "description": "A common, slow-growing form of skin cancer.",
        "clinical_features": [
            "Pearly or waxy bump",
            "Flat, flesh-colored or brown scar-like lesion",
            "Bleeding or scabbing sore that heals and returns"
        ],
        "urgency": "Consult a dermatologist within 1-2 weeks",
        "risk_level": RiskLevel.HIGH,
        "condition_type": ConditionType.MALIGNANT
    },
    "bkl": {
        "full_name": "Benign Keratosis-like Lesions",
        "description": "Non-cancerous skin growths (e.g., seborrheic keratosis).",
        "clinical_features": [
            "Waxy, 'stuck-on' appearance",
            "Round or oval shape",
            "Usually brown, black, or tan"
        ],
        "urgency": "No urgent action needed",
        "risk_level": RiskLevel.LOW,
        "condition_type": ConditionType.BENIGN
    },
    "df": {
        "full_name": "Dermatofibroma",
        "description": "A common benign fibrous nodule, often on the legs.",
        "clinical_features": [
            "Firm, raised bump",
            "Dimples inward when pinched",
            "Stable in size and color over time"
        ],
        "urgency": "No urgent action needed",
        "risk_level": RiskLevel.LOW,
        "condition_type": ConditionType.BENIGN
    },
    "melanoma": {
        "full_name": "Melanoma",
        "description": "A dangerous form of skin cancer that starts in pigment cells.",
        "clinical_features": [
            "Asymmetrical shape",
            "Irregular or notched borders",
            "Multiple colors or changes in an existing mole"
        ],
        "urgency": "URGENT — Consult a specialist immediately",
        "risk_level": RiskLevel.HIGH,
        "condition_type": ConditionType.MALIGNANT
    },
    "nevus": {
        "full_name": "Melanocytic Nevus",
        "description": "A common benign mole.",
        "clinical_features": [
            "Uniform color (usually brown)",
            "Distinct, smooth borders",
            "Symmetrical shape"
        ],
        "urgency": "Routine monitoring (ABCDE rules)",
        "risk_level": RiskLevel.LOW,
        "condition_type": ConditionType.BENIGN
    },
    "vascular": {
        "full_name": "Vascular Lesions",
        "description": "Benign growths made of blood vessels (e.g., cherry angiomas).",
        "clinical_features": [
            "Bright red, purple, or blue color",
            "Blanches (turns white) when pressed",
            "Smooth or slightly raised texture"
        ],
        "urgency": "No urgent action needed",
        "risk_level": RiskLevel.LOW,
        "condition_type": ConditionType.BENIGN
    }
}
