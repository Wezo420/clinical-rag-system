"""
Clinical Intelligence RAG System - FastAPI Backend
Production-grade multimodal medical diagnostics API
"""
import os
# This forces the app to look at the 'postgres' user with NO password
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres@localhost:5432/postgres"

import logging
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from backend.core.config import settings
from backend.core.database import init_db, close_db
from backend.core.redis_client import init_redis, close_redis
from backend.middleware.rate_limit import setup_rate_limiting
from backend.middleware.security import SecurityMiddleware
from backend.routers import analyze, images, results, auth, health
from backend.core.logging_config import configure_logging

# Configure structured logging
configure_logging()
logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager."""
    logger.info("Starting Clinical Intelligence RAG System", env=settings.APP_ENV)
    
    # Initialize connections
    await init_db()
    await init_redis()
    
    # Load FAISS index (pre-warm)
    from backend.services.rag_service import RAGService
    rag = RAGService()
    await rag.initialize()
    app.state.rag_service = rag
    
    logger.info("All services initialized successfully")
    yield
    
    # Cleanup
    logger.info("Shutting down services...")
    await close_db()
    await close_redis()
    logger.info("Shutdown complete")


def create_application() -> FastAPI:
    app = FastAPI(
        title="Clinical Intelligence RAG API",
        description=(
            "Multimodal medical diagnostics powered by RAG, CLIP, and Groq LLM. "
            "⚠️ FOR RESEARCH PURPOSES ONLY — NOT FOR CLINICAL USE."
        ),
        version="1.0.0",
        docs_url="/api/docs" if settings.APP_DEBUG else None,
        redoc_url="/api/redoc" if settings.APP_DEBUG else None,
        lifespan=lifespan,
    )

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # --- Trusted Hosts ---
    if not settings.APP_DEBUG:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*.yourdomain.com", "localhost"]
        )

    # --- Custom Security Middleware ---
    app.add_middleware(SecurityMiddleware)

    # --- Rate Limiting ---
    setup_rate_limiting(app)

    # --- Request Timing & Metrics Middleware ---
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration = time.perf_counter() - start
        
        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(duration)
        
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        return response

    # --- Routers ---
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
    app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])
    app.include_router(images.router, prefix="/api/v1", tags=["images"])
    app.include_router(results.router, prefix="/api/v1", tags=["results"])

    # --- Prometheus Metrics Endpoint ---
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # --- Global Exception Handler ---
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request.headers.get("X-Request-ID")}
        )

    return app


app = create_application()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )
