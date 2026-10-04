# Medical Skin Lesion Triage API 🏥

An AI-powered dermatology triage backend built with **FastAPI**, **EfficientNet-B0** (HAM10000), **Grad-CAM**, and **GPT-4o-mini**.

> ⚠️ **This system is for educational/research purposes only and is NOT a medical device.**

---

## Architecture

```
POST /api/v1/triage   →  Upload image
                          ├── EfficientNet-B0 (classify)
                          ├── Grad-CAM (saliency map)
                          └── Decision layer (risk level)

POST /api/v1/explain  →  GPT-4o-mini (patient-friendly explanation)
                          └── Guardrails (block Rx / diagnosis queries)

GET  /health          →  Liveness probe
```

---

## Quick Start

### 1. Clone & set up environment

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY
```

### 3. (Optional) Download HAM10000 dataset & train

```bash
# Download dataset (requires Kaggle credentials in .env)
python ml/download_dataset.py

# Fine-tune EfficientNet-B0
python ml/train.py --epochs 30 --batch-size 32

# Evaluate on test set
python ml/evaluate.py
```

### 4. Run the API

```bash
python run.py --reload        # development
python run.py --workers 2     # production
```

API docs available at: `http://localhost:8000/docs`

---

## API Reference

### `POST /api/v1/triage`

| Field | Type | Description |
|-------|------|-------------|
| `file` | `File` | JPEG/PNG/WebP image (≤ 10 MB) |

**Response**
```json
{
  "request_id": "uuid",
  "class_name": "Melanocytic nevi",
  "condition_type": "BENIGN",
  "confidence": 0.87,
  "risk_level": "HIGH",
  "recommendation": "⚠️  HIGH RISK — Seek prompt dermatological evaluation...",
  "all_probabilities": [...],
  "gradcam_url": "/uploads/gradcam_abc12345.png",
  "disclaimer": "..."
}
```

### `POST /api/v1/explain`

```json
{
  "class_name": "Melanocytic nevi",
  "condition_type": "BENIGN",
  "confidence": 0.87,
  "risk_level": "HIGH",
  "recommendation": "...",
  "user_question": "What does this mean for me?"
}
```

---

## HAM10000 Classes

| Code | Condition | Type |
|------|-----------|------|
| `nv`    | Melanocytic nevi | Benign |
| `mel`   | Melanoma | **Malignant** |
| `bkl`   | Benign keratosis-like lesions | Benign |
| `bcc`   | Basal cell carcinoma | **Malignant** |
| `akiec` | Actinic keratoses | Pre-malignant |
| `vasc`  | Vascular lesions | Benign |
| `df`    | Dermatofibroma | Benign |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Docker

```bash
docker build -t medical-triage-api .
docker run -p 8000:8000 --env-file .env medical-triage-api
```

---

## Project Structure

```
backend/
├── app/
│   ├── api/routes/      # FastAPI route handlers
│   ├── core/            # Classifier, Grad-CAM, decision logic
│   ├── llm/             # OpenAI client, prompts, guardrails, RAG
│   ├── models/          # Pydantic schemas + enums
│   ├── services/        # Triage & image orchestration
│   └── utils/           # Logger, disclaimers
├── ml/                  # Training & evaluation scripts
├── weights/             # Model checkpoints (gitignored)
├── data/                # HAM10000 dataset (gitignored)
├── tests/               # pytest test suite
└── run.py               # Uvicorn launcher
```

---

## Safety & Ethics

- All responses include a mandatory **medical disclaimer**
- The LLM is constrained by a **system prompt** that prohibits diagnoses and prescriptions
- **Guardrails** intercept out-of-scope user questions before they reach the LLM
- High-risk conditions (Melanoma, BCC, Actinic keratoses) are flagged at lower confidence thresholds
