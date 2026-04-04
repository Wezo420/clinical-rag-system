"""
POST /api/v1/upload-image — Medical image upload endpoint.
Handles validation, storage, and async CLIP embedding.
"""

import uuid
from typing import Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.auth import get_optional_user
from backend.core.config import settings
from backend.core.database import get_db
from backend.middleware.rate_limit import rate_limit
from backend.models.schemas import ImageModality, ImageUploadResponse
from backend.services.multimodal_service import validate_medical_image

logger = structlog.get_logger(__name__)
router = APIRouter()

ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/tiff",
    "image/bmp", "image/webp", "application/dicom",
}


async def _process_image_embedding(image_id: str, image_bytes: bytes):
    """Background task: compute CLIP embedding and update DB."""
    try:
        from backend.services.multimodal_service import MultimodalFusionService
        from backend.services.storage_service import StorageService
        from backend.core.database import AsyncSessionLocal
        from backend.models.sql_models import MedicalImage
        from sqlalchemy import update

        svc = MultimodalFusionService()
        embedding, description = await svc.encode_image(image_bytes)

        # Save embedding
        storage = StorageService()
        emb_path = await storage.save_embedding(image_id, embedding)

        # Update DB status
        async with AsyncSessionLocal() as db:
            await db.execute(
                update(MedicalImage)
                .where(MedicalImage.id == image_id)
                .values(embedding_status="completed", embedding_path=emb_path)
            )
            await db.commit()

        logger.info("Image embedding completed", image_id=image_id)

    except Exception as e:
        logger.error("Image embedding failed", image_id=image_id, error=str(e))
        try:
            from backend.core.database import AsyncSessionLocal
            from backend.models.sql_models import MedicalImage
            from sqlalchemy import update
            async with AsyncSessionLocal() as db:
                await db.execute(
                    update(MedicalImage)
                    .where(MedicalImage.id == image_id)
                    .values(embedding_status="failed")
                )
                await db.commit()
        except Exception:
            pass


@router.post(
    "/upload-image",
    response_model=ImageUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a medical image for analysis",
)
@rate_limit("30/minute")
async def upload_image(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    modality: ImageModality = Form(default=ImageModality.OTHER),
    description: Optional[str] = Form(default=None, max_length=500),
    patient_id: Optional[str] = Form(default=None, max_length=100),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    image_id = str(uuid.uuid4())
    user_id = current_user.get("sub") if current_user else None

    # Validate content type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Allowed: JPEG, PNG, TIFF, BMP, WEBP, DICOM.",
        )

    # Read file
    image_bytes = await file.read()

    # Size check
    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size exceeds {settings.MAX_IMAGE_SIZE_MB}MB limit.",
        )

    # Medical image validation
    is_valid, error_msg = validate_medical_image(image_bytes, settings.MAX_IMAGE_SIZE_MB)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid medical image: {error_msg}",
        )

    # Store image
    from backend.services.storage_service import StorageService
    storage = StorageService()
    safe_filename = f"{image_id}_{file.filename}"
    storage_path = await storage.save_image(image_id, image_bytes, safe_filename)

    # Persist to DB
    from backend.models.sql_models import MedicalImage
    db_image = MedicalImage(
        id=image_id,
        user_id=user_id,
        filename=safe_filename,
        storage_path=storage_path,
        modality=modality.value,
        size_bytes=len(image_bytes),
        mime_type=file.content_type,
        embedding_status="pending",
        description=description,
    )
    db.add(db_image)
    await db.commit()

    # Trigger async embedding
    background_tasks.add_task(_process_image_embedding, image_id, image_bytes)

    logger.info(
        "Image uploaded",
        image_id=image_id,
        modality=modality.value,
        size_bytes=len(image_bytes),
    )

    return ImageUploadResponse(
        image_id=image_id,
        filename=safe_filename,
        modality=modality,
        size_bytes=len(image_bytes),
        embedding_status="pending",
        message="Image uploaded successfully. Embedding computation in progress.",
    )
