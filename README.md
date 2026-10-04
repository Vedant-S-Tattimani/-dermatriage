# DermaTriage AI 🏥

An explainable AI medical assistant for dermatology skin lesion classification, triage, and patient guidance. Powered by **FastAPI**, **EfficientNet-B0** (HAM10000 dataset), **Grad-CAM saliency maps**, and **LLM-driven explanations with medical guardrails**.

> ⚠️ **Disclaimer:** *This system is developed strictly for educational, clinical research, and triage assistance purposes. It does not replace professional medical advice, clinical diagnosis, or treatment.*

---

## 🌟 Key Features

- **Deep Learning Classification:** Fine-tuned EfficientNet-B0 classifier trained on the HAM10000 dermatoscopy dataset (melanoma, nevus, basal cell carcinoma, actinic keratoses, etc.).
- **Explainable AI (XAI):** Real-time Grad-CAM visual heatmaps highlighting suspicious morphological features for clinician interpretability.
- **Risk Triage & Safety Scoring:** Automated risk-level stratification (Low, Moderate, High, Critical) based on class probabilities and confidence thresholds.
- **Guardrailed Patient Explanations:** Natural-language summaries powered by LLM reasoning, strictly constrained by medical safety guardrails (preventing prescription and diagnostic overreach).
- **Modern Responsive UI:** Interactive frontend built with React, Vite, Tailwind CSS, and custom medical design components.
- **Production-Ready Containerization:** Dockerized backend and frontend with Docker Compose and Nginx reverse proxy.

---

## 🏗️ System Architecture

```text
       ┌────────────────────────┐
       │   React SPA (Vite)     │
       └───────────┬────────────┘
                   │
         [Image Upload & Triage]
                   │
                   ▼
       ┌────────────────────────┐
       │     FastAPI Backend    │
       └───────────┬────────────┘
                   ├──► EfficientNet-B0 (Multi-class Lesion Classification)
                   ├──► Grad-CAM (Saliency Map Generation)
                   ├──► Clinical Decision Layer (Risk Stratification)
                   └──► LLM Medical Guardrail (Patient-Friendly Summary)
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ & npm
- Docker & Docker Compose *(optional, for containerized run)*

### 2. Backend Setup
```bash
cd backend
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Update your .env with necessary keys (e.g. OpenAI/Groq API keys)

python run.py
```
Backend will be live at `http://localhost:8000` (API Docs at `http://localhost:8000/docs`).

### 3. Frontend Setup
```bash
cd frontend/triage-app
npm install
npm run dev
```
Frontend will be available at `http://localhost:5173`.

### 4. Running with Docker Compose
```bash
docker-compose up --build
```

---

## 🔬 Dataset & Model

- **Dataset:** HAM10000 ("Human Against Machine with 10000 training images")
- **Classes:**
  - `akiec`: Actinic keratoses and intraepithelial carcinoma
  - `bcc`: Basal cell carcinoma
  - `bkl`: Benign keratosis-like lesions
  - `df`: Dermatofibroma
  - `mel`: Melanoma
  - `nv`: Melanocytic nevi
  - `vasc`: Vascular lesions

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
