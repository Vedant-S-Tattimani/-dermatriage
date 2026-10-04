"""
Image service — handles saving and loading of uploaded images.
"""
import shutil
from pathlib import Path
from typing import Optional

import aiofiles

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


import uuid
from fastapi import UploadFile, HTTPException, status


class ImageService:
    def __init__(self, upload_dir: Path = settings.UPLOADS_DIR):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())

    def validate_file(self, file: UploadFile) -> None:
        """Robust validation for image uploads."""
        # 1. Check Content-Type
        allowed_types = ["image/jpeg", "image/png", "image/webp"]
        if file.content_type not in allowed_types:
            logger.warning("Rejected upload with type: %s", file.content_type)
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type. Please upload JPEG, PNG, or WebP.",
            )
        
        # 2. Check File Extension (double-check)
        ext = Path(file.filename).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file extension.",
            )

        # 3. Check File Size (via content-length header if present, otherwise on read)
        # Note: In FastAPI, file.size is available for SpooledTemporaryFile
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if file.size and file.size > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image too large. Max size is {settings.MAX_UPLOAD_SIZE_MB}MB.",
            )

    async def save_upload(self, file: UploadFile, request_id: str) -> str:
        """Save an UploadFile to the uploads directory."""
        extension = Path(file.filename).suffix or ".jpg"
        filename = f"{request_id}{extension}"
        dest = self.upload_dir / filename
        
        async with aiofiles.open(dest, "wb") as f:
            while content := await file.read(1024 * 1024):  # Read in 1MB chunks
                await f.write(content)
        
        return str(dest)

    async def save(self, contents: bytes, filename: str) -> Path:
        """Persist raw bytes to the uploads directory asynchronously."""
        dest = self.upload_dir / filename
        async with aiofiles.open(dest, "wb") as f:
            await f.write(contents)
        logger.debug("Saved upload: %s (%d bytes)", dest, len(contents))
        return dest

    def delete(self, filename: str) -> None:
        """Remove a file from the uploads directory (best-effort)."""
        target = self.upload_dir / filename
        if target.exists():
            target.unlink()
            logger.debug("Deleted upload: %s", target)

    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Delete uploads older than *max_age_hours*.
        Returns the number of files removed.
        """
        import time
        cutoff = time.time() - max_age_hours * 3600
        removed = 0
        for f in self.upload_dir.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        logger.info("Cleaned up %d stale uploads", removed)
        return removed


image_service = ImageService()
