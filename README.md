# DermaTriage AI 🏥🔬
### *Explainable AI-Powered Dermatology Screening & Clinical Triage Platform*

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/ML%20Engine-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?logo=react&logoColor=black)](https://vitejs.dev/)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Table of Contents
1. [🌟 The Big Picture (Non-Technical Summary)](#-the-big-picture-non-technical-summary)
   - [Why This Project Exists](#why-this-project-exists)
   - [How It Works for Everyday Users](#how-it-works-for-everyday-users)
   - [What It Does vs. What It Doesn't Do](#what-it-does-vs-what-it-doesnt-do)
   - [Key User Features](#key-user-features)
2. [⚙️ Under the Hood (Technical Architecture)](#️-under-the-hood-technical-architecture)
   - [System Architecture](#system-architecture)
   - [Machine Learning & Computer Vision](#machine-learning--computer-vision)
   - [Explainable AI (XAI) & Grad-CAM](#explainable-ai-xai--grad-cam)
   - [Medical LLM Guardrails & Reasoning](#medical-llm-guardrails--reasoning)
   - [Tech Stack Overview](#tech-stack-overview)
3. [🚀 Getting Started & Installation](#-getting-started--installation)
   - [Option A: Docker Compose (Recommended)](#option-a-docker-compose-recommended)
   - [Option B: Manual Local Setup](#option-b-manual-local-setup)
4. [📡 API Endpoints Reference](#-api-endpoints-reference)
5. [📁 Project Structure](#-project-structure)
6. [⚠️ Medical & Ethical Disclaimer](#️-medical--ethical-disclaimer)

---

## 🌟 The Big Picture (Non-Technical Summary)

### Why This Project Exists
Noticing an unusual mole or skin lesion can be scary. Unfortunately, booking an appointment with a certified dermatologist often takes weeks or months. Searching symptoms online frequently leads to alarmist misinformation, causing unnecessary panic or false reassurance.

**DermaTriage AI** bridges this gap. It acts like a **digital triage nurse for skin concerns**:
- It doesn't replace doctors.
- Instead, it helps individuals quickly understand **how urgent** a skin spot looks, provides **transparent visual proof** of what the AI detected, and offers **clear, jargon-free medical guidance** on what to do next.

---

### How It Works for Everyday Users

```text
  [ 1. Snap & Upload ]            [ 2. Transparent AI Scan ]           [ 3. Clear Action Plan ]
  Take a clear photo of          The system highlights borders        Receive a simple risk level
  a mole, rash, or lesion.       and patterns using a heatmap.        and advice on next steps.
```

1. **📸 Take a Photo:** Upload a well-lit photo of the skin lesion from your phone or computer.
2. **🔍 See What the AI Sees:** Traditional AI acts like a "black box" that gives answers without reasons. DermaTriage uses **visual heatmaps (like a digital highlighter)** to reveal the exact pigment irregularities or borders that triggered its evaluation.
3. **📋 Receive Plain-English Guidance:** You get a clean breakdown:
   - **Risk Level:** *Low*, *Moderate*, *High*, or *Critical*.
   - **What It Might Mean:** Clear, empathetic language (no intimidating medical jargon).
   - **Next Steps:** Actionable advice (e.g., *"Monitor changes over 30 days"* vs. *"Schedule an in-person dermatology visit within 48 hours"*).

---

### What It Does vs. What It Doesn't Do

| ✅ What DermaTriage AI DOES | ❌ What DermaTriage AI DOES NOT DO |
| :--- | :--- |
| **Triages urgency:** Categorizes skin lesions into clear risk tiers. | **Does NOT give official diagnoses:** Only a licensed physician can diagnose skin conditions. |
| **Explains visually:** Shows transparent heatmaps of suspicious features. | **Does NOT prescribe drugs:** Never dispenses prescriptions or recommends dosage. |
| **Speaks your language:** Supports multiple local languages. | **Does NOT operate blindly:** Calibrated to alert you if an image is blurry or out-of-domain. |

---

### Key User Features

- **🎯 Transparent Visual Explanations:** View the original image alongside an AI-generated heatmap showing areas of interest.
- **🌐 Multilingual Accessibility:** Built-in internationalization supporting **English**, **Hindi (हिन्दी)**, **Kannada (ಕನ್ನಡ)**, **Marathi (मराठी)**, **Tamil (தமிழ்)**, and **Telugu (తెలుగు)**.
- **💬 Safe Interactive Assistant:** Ask follow-up questions about skin care, preparation for a doctor's visit, or common symptoms.
- **📱 Clean, Modern Interface:** Designed for quick mobile and desktop scanning with minimal friction.

---

## ⚙️ Under the Hood (Technical Architecture)

For developers, data scientists, and ML researchers, DermaTriage AI combines **state-of-the-art computer vision**, **Explainable AI (XAI)**, and **constrained Large Language Models (LLMs)**.

### System Architecture

```text
 ┌─────────────────────────────────────────────────────────────┐
 │                      React SPA (Vite)                       │
 │    Tailwind CSS • Responsive Layout • i18n Localization     │
 └──────────────────────────────┬──────────────────────────────┘
                                │ HTTPS / REST (Multipart)
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                      FastAPI Gateway                        │
 │        Request Validation • Tracing Middleware • CORS       │
 └───────┬──────────────────────┬───────────────────────┬──────┘
         │                      │                       │
         ▼                      ▼                       ▼
 ┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
 │ Deep Learning │     │  Explainable AI │     │ Clinical LLM    │
 │ (EfficientNet)│     │   (Grad-CAM)    │     │  (Guardrailed)  │
 ├───────────────┤     ├─────────────────┤     ├─────────────────┤
 │ • PyTorch     │     │ • Penultimate   │     │ • Groq / GPT-4o │
 │ • HAM10000    │     │   Conv Layer    │     │ • RAG Protocols │
 │ • 7 Classes   │     │ • Saliency Map  │     │ • Strict Safety │
 │ • Calibrated  │     │ • Jet Heatmap   │     │   Shielding     │
 └───────┬───────┘     └────────┬────────┘     └────────┬────────┘
         │                      │                       │
         └──────────────────────┼───────────────────────┘
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                     Decision Layer                          │
 │     Multi-Threshold Risk Matrix (Low / Mod / High / Crit)   │
 └─────────────────────────────────────────────────────────────┘
```

---

### Machine Learning & Computer Vision

- **Backbone Architecture:** Fine-tuned **EfficientNet-B0** chosen for high top-1 accuracy combined with lightweight parameter overhead suitable for real-time inference.
- **Dataset:** Trained on the **HAM10000** ("Human Against Machine with 10,000 training images") dermatoscopic benchmark dataset.
- **7-Class Taxonomy:**
  1. `mel` — Melanoma
  2. `nv` — Melanocytic Nevi (common moles)
  3. `bcc` — Basal Cell Carcinoma
  4. `akiec` — Actinic Keratoses & Intraepithelial Carcinoma
  5. `bkl` — Benign Keratosis-like Lesions (solar lentigines, seborrheic keratoses)
  6. `df` — Dermatofibroma
  7. `vasc` — Vascular Lesions (angiomas, pyogenic granulomas)
- **Probability Calibration:** Employs temperature scaling to prevent overconfident predictions, ensuring calibrated confidence metrics before risk assignment.

---

### Explainable AI (XAI) & Grad-CAM

Medical AI requires trust and verification. Rather than outputting raw softmax values:
1. DermaTriage hooks into the final convolutional layer of EfficientNet.
2. Computes the gradient of the winning class score with respect to feature maps (**Grad-CAM**).
3. Produces a 2D activation map normalized between `[0, 1]` and overlays a semi-transparent OpenCV color heatmap on the patient image.
4. Clinicians and users can visually confirm whether the model focused on real pigment architecture or peripheral artifacts (hair, ruler marks, glare).

---

### Medical LLM Guardrails & Reasoning

The explanation pipeline feeds the model's top predictions, confidence intervals, and risk flags into an LLM engine (Groq / OpenAI) with **strict constitutional medical guardrails**:
- **Negative Constraint Enforcement:** The system prompt explicitly bans prescribing drug names, dosages, or claiming a definitive medical diagnosis.
- **Tone & Empathy Control:** Formulates responses at an accessible 8th-grade reading level.
- **RAG & Knowledge Grounding:** Injects validated clinical profiles from `app/llm/knowledge_base.json` to eliminate hallucinations.

---

### Tech Stack Overview

| Domain | Technologies Used |
| :--- | :--- |
| **Backend API** | Python 3.10+, FastAPI, Uvicorn, Pydantic v2 |
| **Machine Learning** | PyTorch, Torchvision, Albumentations, scikit-learn, OpenCV |
| **Explainability (XAI)** | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| **LLM & Reasoning** | Groq API / OpenAI API, LangChain / Custom Medical Prompt Guards |
| **Frontend UI** | React 18, Vite, Tailwind CSS, Lucide Icons, i18next |
| **DevOps & Infra** | Docker, Docker Compose, Nginx (Reverse Proxy & Static Cache) |

---

## 🚀 Getting Started & Installation

### Prerequisites
- [Git](https://git-scm.com/)
- [Docker & Docker Compose](https://www.docker.com/) *(for containerized deployment)*
- Or [Python 3.10+](https://www.python.org/) and [Node.js 18+](https://nodejs.org/) *(for manual setup)*

---

### Option A: Docker Compose (Recommended)

Run both the frontend and backend with a single command:

```bash
# 1. Clone repository
git clone https://github.com/Vedant-S-Tattimani/-dermatriage.git
cd -dermatriage

# 2. Configure environment
cp backend/.env.example backend/.env
# Open backend/.env and set GROQ_API_KEY or OPENAI_API_KEY

# 3. Build & start all containers
docker-compose up --build
```
- **Web Application:** `http://localhost` (or `http://localhost:5173`)
- **API Documentation:** `http://localhost:8000/docs`

---

### Option B: Manual Local Setup

#### 1. Backend Setup
```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Launch backend server
python run.py
```
*API will run on `http://localhost:8000`.*

#### 2. Frontend Setup
```bash
# Open a new terminal
cd frontend/triage-app

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```
*Frontend will run on `http://localhost:5173`.*

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/triage` | Uploads skin image; returns predicted classes, risk tier, and Grad-CAM overlay image. |
| `POST` | `/api/v1/explain` | Generates safe, patient-friendly explanation based on triage results. |
| `POST` | `/api/v1/chat` | Safe medical conversational assistant with guardrail enforcement. |
| `GET` | `/api/v1/monitoring` | Telemetry endpoint monitoring model performance, latency, and data drift. |
| `GET` | `/api/health` | Service health check and ML device readiness probe. |

---

## 📁 Project Structure

```text
├── backend/
│   ├── app/
│   │   ├── api/routes/          # FastAPI route controllers (triage, explain, chat)
│   │   ├── core/                # EfficientNet classifier, Grad-CAM XAI, Decision layer
│   │   ├── llm/                 # LLM client, medical prompts, safety guardrails
│   │   └── models/              # Pydantic schemas and enums
│   ├── ml/                      # HAM10000 training, evaluation, and calibration scripts
│   ├── weights/                 # Model checkpoints, classes.json, and metrics
│   ├── Dockerfile               # Backend Docker container definition
│   └── requirements.txt         # Python package dependencies
│
├── frontend/
│   └── triage-app/
│       ├── src/
│       │   ├── components/      # UI components (StarBorder, GooeyNav, Layout)
│       │   ├── locales/         # Multi-language translations (en, hi, kn, mr, ta, te)
│       │   └── pages/           # Pages (Landing, Upload, Results, HowItWorks)
│       ├── Dockerfile           # Optimized Nginx frontend container
│       └── package.json         # Node.js dependencies
│
├── docker-compose.yml           # Unified orchestration configuration
├── .gitignore                   # Security & environment exclusion rules
└── README.md                    # Project documentation
```

---

## ⚠️ Medical & Ethical Disclaimer

> **IMPORTANT MEDICAL NOTICE:**
>
> DermaTriage AI is an experimental software platform developed for **educational, clinical triage research, and proof-of-concept purposes only**.
>
> - It is **NOT** a certified medical diagnostic device under FDA, CE, or CDSCO regulations.
> - It does **NOT** provide medical diagnoses, treatment plans, or prescriptions.
> - An algorithmic prediction should never replace a physical examination, dermoscopy, or biopsy performed by a qualified, board-certified healthcare provider.
> - If you notice a spot that is rapidly evolving, bleeding, itching, or displaying irregular borders, **seek immediate professional medical attention regardless of an algorithm's output.**

---

## 🤝 Contributing & License

Contributions, feedback, and issues are warmly welcomed! Please feel free to open a PR or submit an issue.

Distributed under the **MIT License**. See `LICENSE` for details.
