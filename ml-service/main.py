"""
ML Service — Standalone FastAPI app for CLIP embeddings and multimodal inference.
Runs on port 8001, called internally by the backend.
"""

import io
import base64
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import structlog
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

# Lazy-loaded models
_clip_encoder = None
_text_encoder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ML Service starting — loading models...")
    # Pre-warm models
    _get_text_encoder()
    logger.info("ML Service ready")
    yield
    logger.info("ML Service shutting down")


def _get_clip_encoder():
    global _clip_encoder
    if _clip_encoder is None:
        from backend.services.multimodal_service import CLIPImageEncoder
        _clip_encoder = CLIPImageEncoder()
    return _clip_encoder


def _get_text_encoder():
    global _text_encoder
    if _text_encoder is None:
        from sentence_transformers import SentenceTransformer
        _text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _text_encoder


app = FastAPI(
    title="Clinical RAG ML Service",
    description="CLIP image encoding and text embedding service",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request/Response models ──────────────────────────────────────────────────

class TextEmbedRequest(BaseModel):
    texts: List[str]
    normalize: bool = True


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dim: int
    model: str


class ImageEmbedResponse(BaseModel):
    embedding: List[float]
    description: str
    dim: int


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ml-service"}


@app.post("/embed/text", response_model=EmbeddingResponse)
async def embed_text(req: TextEmbedRequest):
    """Encode a batch of texts to embeddings."""
    if not req.texts:
        raise HTTPException(400, "texts list is empty")
    if len(req.texts) > 128:
        raise HTTPException(400, "Maximum 128 texts per request")

    try:
        model = _get_text_encoder()
        embeddings = model.encode(
            req.texts,
            normalize_embeddings=req.normalize,
            show_progress_bar=False,
            batch_size=32,
        )
        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            dim=embeddings.shape[1],
            model="all-MiniLM-L6-v2",
        )
    except Exception as e:
        logger.error("Text embedding failed", error=str(e))
        raise HTTPException(500, f"Embedding failed: {str(e)}")


@app.post("/embed/image", response_model=ImageEmbedResponse)
async def embed_image(file: UploadFile = File(...)):
    """Encode a medical image using CLIP and return embedding + description."""
    from backend.services.multimodal_service import validate_medical_image, MultimodalFusionService
    import asyncio

    image_bytes = await file.read()

    # Validate
    is_valid, error_msg = validate_medical_image(image_bytes)
    if not is_valid:
        raise HTTPException(422, error_msg)

    try:
        svc = MultimodalFusionService()
        loop = asyncio.get_event_loop()
        embedding, description = await svc.encode_image(image_bytes)

        return ImageEmbedResponse(
            embedding=embedding.tolist(),
            description=description,
            dim=len(embedding),
        )
    except Exception as e:
        logger.error("Image embedding failed", error=str(e))
        raise HTTPException(500, f"Image embedding failed: {str(e)}")


@app.post("/embed/image/base64", response_model=ImageEmbedResponse)
async def embed_image_base64(payload: dict):
    """Encode a base64-encoded image."""
    b64 = payload.get("image_base64", "")
    if not b64:
        raise HTTPException(400, "image_base64 is required")

    try:
        image_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 encoding")

    from backend.services.multimodal_service import MultimodalFusionService
    import asyncio

    svc = MultimodalFusionService()
    embedding, description = await svc.encode_image(image_bytes)

    return ImageEmbedResponse(
        embedding=embedding.tolist(),
        description=description,
        dim=len(embedding),
    )


@app.post("/similarity")
async def compute_similarity(payload: dict):
    """Compute cosine similarity between image and text."""
    image_b64 = payload.get("image_base64")
    text = payload.get("text", "")

    if not image_b64 or not text:
        raise HTTPException(400, "image_base64 and text are required")

    try:
        image_bytes = base64.b64decode(image_b64)
        encoder = _get_clip_encoder()
        image_emb = encoder.encode_image_from_bytes(image_bytes)
        text_emb = encoder.encode_text_clip(text)
        score = encoder.compute_similarity(image_emb, text_emb)
        return {"similarity": float(score)}
    except Exception as e:
        logger.error("Similarity computation failed", error=str(e))
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ml-service.main:app", host="0.0.0.0", port=8001, reload=False)
