"""
FastAPI application entry point.
Production-grade configuration with middleware and error handling.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
import os
import time

from app.config import settings
from app.api.routes import health, triage, explain, chat, monitoring
from app.utils.logger import get_logger, TracingMiddleware

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle handler."""
    logger.info("Starting %s v%s [ENV: %s]", settings.APP_NAME, settings.APP_VERSION, settings.ENV)
    
    # 1. Ensure upload dir exists
    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Force model load at startup (Performance Optimization)
    try:
        from app.core.classifier import classifier
        if classifier.model is not None:
            logger.info("ML Model pre-loaded and ready on %s", classifier.device)
    except Exception as exc:
        logger.error("Failed to pre-load model: %s", exc)

    yield
    logger.info("Shutting down %s.", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs" if settings.ENV != "production" else None,
    redoc_url="/redoc" if settings.ENV != "production" else None,
)

# ── Middleware ───────────────────────────────────────────────────────────────
app.add_middleware(TracingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Exception Handlers ────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled application error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred. Our team has been notified.",
            "trace_id": request.headers.get("X-Request-ID")
        }
    )

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(triage.router, prefix="/api/v1", tags=["Triage"])
app.include_router(explain.router, prefix="/api/v1", tags=["Explanation"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["Monitoring"])

# ── Static files (serve Grad-CAM images) ─────────────────────────────────────
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOADS_DIR)), name="uploads")

# ── SPA Fallback (Optional) ──────────────────────────────────────────────────
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if full_path.startswith("api/") or full_path.startswith("uploads/"):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    
    file_path = settings.FRONTEND_DIR / full_path
    if file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(settings.FRONTEND_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=True)
