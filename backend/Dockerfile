# ── Build Stage ──────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime AS builder

WORKDIR /build

# System dependencies for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ── Production Stage (GPU Enabled) ───────────────────────────────────────────
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Medical AI Backend Team"
LABEL version="1.1.0"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"
ENV ENV=production

WORKDIR /app

# Runtime system dependencies (OpenCV + GL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY app/ ./app/
COPY run.py .
COPY .env.example .env

# Create data volumes and mount points
RUN mkdir -p uploads weights data/raw data/processed data/active_learning

# Expose API port
EXPOSE 8000

# Health check using curl (installed in slim images usually, otherwise use python script)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; exit(0 if requests.get('http://localhost:8000/api/health').status_code == 200 else 1)" || exit 1

# Launch production server with Gunicorn-like worker management
# Note: we use uvicorn directly with workers for simplicity in this prototype,
# but gunicorn is recommended for full multi-process management.
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--no-reload"]
