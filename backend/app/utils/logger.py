"""
Production-grade structured logging with JSON support and context tracing.
"""
import logging
import sys
import json
import time
from typing import Any, Dict
from app.config import settings

# Context variable to store request_id for the current task
from contextvars import ContextVar
import uuid

request_id_var: ContextVar[str] = ContextVar("request_id", default="system")

# Log filter to inject request_id into every record automatically
class RequestIDFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True

# Apply filter to root logger
logging.getLogger().addFilter(RequestIDFilter())

class JSONFormatter(logging.Formatter):
    """Formats log records as JSON strings."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
        }
        # Include request_id if available in context
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        # Include extra attributes
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)
    
    if not any(isinstance(f, RequestIDFilter) for f in logger.filters):
        logger.addFilter(RequestIDFilter())

    if not logger.handlers:
        level = logging.DEBUG if settings.DEBUG else logging.INFO
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # In production, use JSON; in dev, use readable format
        if settings.ENV == "production":
            formatter = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z")
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | [%(request_id)s] %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger

# Middleware for request tracing
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class TracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        token = request_id_var.set(request_id)
        
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-MS"] = f"{process_time:.2f}"
        
        request_id_var.reset(token)
        return response

