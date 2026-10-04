"""
Application configuration — loads from .env and defines thresholds.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import List
import json


BASE_DIR = Path(__file__).resolve().parent.parent


def load_calibration_temperature() -> float:
    metrics_path = BASE_DIR / "ml" / "calibration" / "calibration_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return float(data.get("optimal_temperature", 0.8))
        except Exception:
            pass
    return 0.8


class Settings(BaseSettings):
    # ── General ──────────────────────────────────────────────────────────────
    APP_NAME: str = "Medical Skin Triage API"
    APP_VERSION: str = "1.1.0"
    APP_DESCRIPTION: str = "Production-grade dermatology AI triage backend with multimodal support."
    ENV: str = "development" # "production" or "development"
    DEBUG: bool = False

    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── Paths ─────────────────────────────────────────────────────────────────
    WEIGHTS_DIR: Path = BASE_DIR / "weights"
    MODEL_PATH: Path = BASE_DIR / "weights" / "model_robust.pth"
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    FRONTEND_DIR: Path = BASE_DIR.parent / "frontend" / "triage-app" / "dist"
    
    # Active Learning & Monitoring
    ACTIVE_LEARNING_DIR: Path = BASE_DIR / "data" / "active_learning"
    FEEDBACK_LOG_PATH: Path = BASE_DIR / "data" / "active_learning" / "feedback.jsonl"
    MONITORING_LOG_PATH: Path = BASE_DIR / "data" / "active_learning" / "predictions.jsonl"

    # ── Classification ────────────────────────────────────────────────────────
    # HAM10000 class labels (7 classes) - Concise names
    CLASS_NAMES: List[str] = [
        "actinic",   # akiec (0)
        "bcc",       # bcc (1)
        "bkl",       # bkl (2)
        "df",        # df (3)
        "melanoma",  # mel (4)
        "nevus",     # nv (5)
        "vascular",  # vasc (6)
    ]

    # Risk thresholds (confidence %)
    HIGH_RISK_THRESHOLD: float = 0.65     # ≥65 % → HIGH (slightly more sensitive)
    MEDIUM_RISK_THRESHOLD: float = 0.35   # 35–65 % → MEDIUM
    # < 35 % → LOW / UNCERTAIN

    # Grad-CAM
    GRADCAM_TARGET_LAYER: str = "features.8"   # EfficientNet-B0 last conv block
    GRADCAM_ALPHA: float = 0.5                 # Slightly more intense heatmap overlay

    # Calibration & Reliability
    CALIBRATION_TEMPERATURE: float = load_calibration_temperature()       # Optimised via ml/calibrate_model.py
    UNCERTAINTY_THRESHOLD: float = 0.35        # Below this, triage switches to LOW_CONFIDENCE mode
    PREVENT_OVERCONFIDENCE_CAP: float = 0.98   # Soft-cap to prevent "fake" 99% scores

    # ── Groq ───────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_MAX_TOKENS: int = 600
    GROQ_TEMPERATURE: float = 0.3

    # ── Image preprocessing ───────────────────────────────────────────────────
    IMAGE_SIZE: int = 224
    IMAGE_MEAN: List[float] = [0.485, 0.456, 0.406]
    IMAGE_STD: List[float] = [0.229, 0.224, 0.225]
    MAX_UPLOAD_SIZE_MB: int = 10

    # ── Security / Auth ───────────────────────────────────────────────────────
    API_KEY_HEADER: str = "X-API-Key"
    API_KEY: str = ""          # Set in .env; empty = auth disabled

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
