"""
Rate limiting using slowapi + Redis sliding window.
Per-IP and per-authenticated-user limits.
"""

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.core.config import settings

logger = structlog.get_logger(__name__)


def get_user_identifier(request: Request) -> str:
    """
    Returns user ID from JWT if present, otherwise falls back to IP.
    This allows higher limits for authenticated users.
    """
    # Try to get user from token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            from backend.core.auth import decode_token
            token = auth_header.split(" ")[1]
            payload = decode_token(token)
            if payload:
                return f"user:{payload.get('sub', get_remote_address(request))}"
        except Exception:
            pass
    return f"ip:{get_remote_address(request)}"


# Initialize the limiter with Redis backend
limiter = Limiter(
    key_func=get_user_identifier,
    default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"],
    storage_uri=settings.REDIS_URL,
)


def setup_rate_limiting(app: FastAPI) -> None:
    """Attach rate limiter to the FastAPI app."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    logger.info(
        "Rate limiting configured",
        limit_per_minute=settings.RATE_LIMIT_PER_MINUTE,
    )


# Convenience decorator factory
def rate_limit(limit: str = None):
    """
    Usage: @rate_limit("5/minute")
    """
    effective = limit or f"{settings.RATE_LIMIT_PER_MINUTE}/minute"
    return limiter.limit(effective)
