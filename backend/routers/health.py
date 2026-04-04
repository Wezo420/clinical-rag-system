"""Health check endpoints for liveness and readiness probes."""

import structlog
from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = structlog.get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    services: dict


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(request: Request):
    """Kubernetes liveness probe."""
    return HealthResponse(status="ok", services={"api": "ok"})


@router.get("/health/ready", response_model=HealthResponse, tags=["health"])
async def readiness_check(request: Request):
    """Kubernetes readiness probe — checks all dependencies."""
    services = {}

    # Check Redis
    try:
        from backend.core.redis_client import get_redis
        r = get_redis()
        await r.ping()
        services["redis"] = "ok"
    except Exception:
        services["redis"] = "error"

    # Check RAG service
    try:
        rag = request.app.state.rag_service
        services["rag"] = "ok" if rag._initialized else "initializing"
    except Exception:
        services["rag"] = "error"

    all_ok = all(v in ("ok", "initializing") for v in services.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        services=services,
    )
