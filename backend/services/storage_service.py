"""
Storage service — local filesystem or S3 backend.
Handles image and embedding persistence.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


class StorageService:
    def __init__(self):
        self.backend = settings.STORAGE_BACKEND
        if self.backend == "local":
            Path(settings.LOCAL_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
            Path(f"{settings.LOCAL_STORAGE_PATH}/embeddings").mkdir(parents=True, exist_ok=True)

    async def save_image(self, image_id: str, data: bytes, filename: str) -> str:
        if self.backend == "s3":
            return await self._s3_upload(f"images/{image_id}/{filename}", data, "image/jpeg")
        path = Path(settings.LOCAL_STORAGE_PATH) / "images" / image_id
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / filename
        file_path.write_bytes(data)
        return str(file_path)

    async def get_image_bytes(self, image_id: str) -> Optional[bytes]:
        if self.backend == "s3":
            return await self._s3_download(f"images/{image_id}/")
        # Local: find first file in image_id directory
        base = Path(settings.LOCAL_STORAGE_PATH) / "images" / image_id
        if not base.exists():
            return None
        files = list(base.iterdir())
        if not files:
            return None
        return files[0].read_bytes()

    async def save_embedding(self, image_id: str, embedding: np.ndarray) -> str:
        path = Path(settings.LOCAL_STORAGE_PATH) / "embeddings" / f"{image_id}.npy"
        np.save(str(path), embedding)
        return str(path)

    async def get_embedding(self, image_id: str) -> Optional[np.ndarray]:
        path = Path(settings.LOCAL_STORAGE_PATH) / "embeddings" / f"{image_id}.npy"
        if not path.exists():
            return None
        return np.load(str(path))

    async def _s3_upload(self, key: str, data: bytes, content_type: str) -> str:
        import boto3
        s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        s3.put_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return f"s3://{settings.S3_BUCKET_NAME}/{key}"

    async def _s3_download(self, prefix: str) -> Optional[bytes]:
        import boto3
        s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        resp = s3.list_objects_v2(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix)
        contents = resp.get("Contents", [])
        if not contents:
            return None
        obj = s3.get_object(Bucket=settings.S3_BUCKET_NAME, Key=contents[0]["Key"])
        return obj["Body"].read()
