"""
Pydantic request / response models (schemas).
"""
from __future__ import annotations

from typing import List, Optional, Dict
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.enums import RiskLevel, ConditionType


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    version: str
    timestamp: str
    model_ready: bool


# ── Classification ────────────────────────────────────────────────────────────

class ClassProbability(BaseModel):
    class_name: str
    probability: float = Field(..., ge=0.0, le=1.0)


class XAIResult(BaseModel):
    method: str
    url: str


class TriageMetadata(BaseModel):
    age: Optional[int] = Field(None, ge=0, le=120)
    location: Optional[str] = Field(None, description="Body location (e.g., scalp, back, chest)")
    symptoms: List[str] = Field(default_factory=list, description="List of symptoms like itching, bleeding")
    duration: Optional[str] = Field(None, description="How long the lesion has been present")
    itching: bool = False
    bleeding: bool = False
    rapid_change: bool = False


class TriageResponse(BaseModel):
    request_id: str
    class_name: str = Field(..., description="Top-1 predicted condition category code")
    full_name: str = Field(..., description="Full descriptive name of the condition")
    condition_type: ConditionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    recommendation: str
    clinical_description: str = ""
    clinical_features: List[str] = Field(default_factory=list)
    urgency: str = ""
    all_probabilities: List[ClassProbability]
    top_3: List[ClassProbability] = Field(default_factory=list)
    gradcam_url: Optional[str] = Field(
        None, description="Relative URL to Grad-CAM overlay image"
    )
    xai_reports: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of XAI method names to overlay image URLs"
    )
    xai_heatmap_reports: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of XAI method names to pure heatmap (no original image) URLs"
    )
    cropped_image_url: Optional[str] = Field(
        None, description="Relative URL to the segmented/cropped lesion image"
    )
    segmentation_mask_url: Optional[str] = Field(
        None, description="Relative URL to the binary segmentation mask"
    )
    uncertainty_flags: List[str] = Field(default_factory=list, description="Flags indicating potential issues (OOD, low confidence)")
    is_ood: bool = Field(False, description="Whether the image is likely out-of-distribution")
    inference_time_ms: float = 0.0
    disclaimer: str


# ── LLM Explanation ───────────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    class_name: str
    condition_type: ConditionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    recommendation: str
    user_question: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional follow-up question from the patient (≤ 500 chars)",
    )
    language: str = "en"

    @field_validator("user_question", mode="before")
    @classmethod
    def strip_question(cls, v):
        return v.strip() if isinstance(v, str) else v


class ExplainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    explanation: str
    disclaimer: str
    model_used: str


# ── Active Learning & Analytics ───────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    request_id: str
    correct_class: str
    user_comments: Optional[str] = None


class ClassStat(BaseModel):
    class_name: str
    count: int
    avg_confidence: float


class AnalyticsResponse(BaseModel):
    total_predictions: int
    total_feedback: int
    avg_system_confidence: float
    uncertain_count: int
    class_distribution: List[ClassStat]
    recent_feedback: List[Dict[str, Any]]
