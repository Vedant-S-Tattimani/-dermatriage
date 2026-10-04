"""
Shared FastAPI dependencies (auth, model loader, etc.).
"""
from fastapi import Header, HTTPException, status
from app.config import settings


async def get_optional_api_key(x_api_key: str = Header(default="")) -> None:
    """
    Optional API key guard. If settings.API_KEY is set, the request must
    supply a matching X-Api-Key header. If the env var is empty, auth is
    disabled (useful for development).
    """
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
