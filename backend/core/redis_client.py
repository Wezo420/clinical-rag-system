"""Redis client — rate limiting, caching, session store."""

import json
from typing import Any, Optional

import redis.asyncio as aioredis
import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)

_redis: Optional[aioredis.Redis] = None


async def init_redis():
    global _redis
    _redis = await aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True,
    )
    await _redis.ping()
    logger.info("Redis initialized")


async def close_redis():
    global _redis
    if _redis:
        await _redis.close()
        _redis = None


def get_redis() -> aioredis.Redis:
    if _redis is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _redis


# -----------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------

async def cache_set(key: str, value: Any, ttl: int = 300) -> None:
    """Store JSON-serializable value with TTL (seconds)."""
    r = get_redis()
    await r.setex(key, ttl, json.dumps(value))


async def cache_get(key: str) -> Optional[Any]:
    """Retrieve cached value; returns None on miss."""
    r = get_redis()
    raw = await r.get(key)
    if raw is None:
        return None
    return json.loads(raw)


async def cache_delete(key: str) -> None:
    r = get_redis()
    await r.delete(key)


async def rate_limit_check(identifier: str, limit: int, window: int) -> tuple[bool, int]:
    """
    Sliding window rate limit.
    Returns (is_allowed, remaining_requests).
    """
    r = get_redis()
    key = f"rl:{identifier}"
    pipe = r.pipeline()
    pipe.incr(key)
    pipe.ttl(key)
    results = await pipe.execute()
    count, ttl = results

    if ttl == -1:  # key exists but no TTL
        await r.expire(key, window)

    if count == 1:
        await r.expire(key, window)

    remaining = max(0, limit - count)
    return count <= limit, remaining
