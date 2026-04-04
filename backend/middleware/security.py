"""
Security middleware:
- Request ID injection
- Prompt injection detection
- Input length enforcement
- Security headers
"""

import re
import uuid
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core.config import settings

logger = structlog.get_logger(__name__)

# -----------------------------------------------------------------------
# Prompt injection detection patterns
# -----------------------------------------------------------------------

INJECTION_PATTERNS = [
    # Classic jailbreak
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+(a\s+)?(?:dan|jailbreak|evil|unfiltered)",
    r"pretend\s+(you\s+are|to\s+be)\s+(?!a\s+doctor|a\s+medical)",
    r"act\s+as\s+(if\s+you\s+are\s+)?(?:a\s+)?(?:evil|unfiltered|uncensored)",
    # System prompt leaking attempts
    r"(reveal|show|print|output|repeat)\s+(your\s+)?(system|hidden|secret)\s+(prompt|instructions)",
    r"what\s+(are|were)\s+your\s+(original\s+|system\s+)?instructions",
    # Role manipulation
    r"your\s+new\s+(role|persona|identity|task)\s+is",
    r"from\s+now\s+on\s+you\s+(are|will\s+be|must)",
    # Medical misuse
    r"prescribe\s+me",
    r"diagnose\s+me\s+(with|as)",
    r"give\s+me\s+a\s+(definitive\s+)?diagnosis",
    r"tell\s+me\s+(exactly\s+)?what\s+(disease|condition|illness)\s+i\s+have",
    # Prompt boundary escape
    r"[\]\[}{<>]{3,}",  # repeated brackets used to escape prompts
    r"<!--.*?-->",       # HTML comment injection
    r"<script",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in INJECTION_PATTERNS]


def detect_prompt_injection(text: str) -> tuple[bool, str]:
    """
    Returns (is_malicious, matched_pattern_description).
    """
    if not settings.ENABLE_PROMPT_INJECTION_DETECTION:
        return False, ""

    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, pattern.pattern[:60]

    return False, ""


def sanitize_text_input(text: str) -> str:
    """
    Sanitize user text input:
    - Strip HTML tags
    - Normalize whitespace
    - Remove null bytes
    - Enforce length limit
    """
    import bleach

    # Remove null bytes
    text = text.replace("\x00", "")

    # Strip all HTML
    text = bleach.clean(text, tags=[], strip=True)

    # Normalize whitespace (preserve single newlines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Enforce max length
    if len(text) > settings.MAX_TEXT_INPUT_LENGTH:
        text = text[: settings.MAX_TEXT_INPUT_LENGTH]
        logger.warning("Input truncated to max length", max_len=settings.MAX_TEXT_INPUT_LENGTH)

    return text.strip()


# -----------------------------------------------------------------------
# Middleware class
# -----------------------------------------------------------------------

class SecurityMiddleware(BaseHTTPMiddleware):
    """Adds security headers and request ID to every response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Assign unique request ID
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response: Response = await call_next(request)

        # Security headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data: blob:; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline';"
        )

        if settings.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        return response
