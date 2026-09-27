"""In-memory sliding-window rate limiting (stdlib only, no new dependencies).

Keyed by client IP plus a hashed OpenRouter key when present, plus an
optional per-session identifier forwarded by the frontend. The session
header is the fix for the classic reverse-proxy/Docker topology where
every user's traffic arrives from one IP (Streamlit -> FastAPI over
localhost inside one container): without it, all users would share one
bucket. The header is optional and opaque - the backend only hashes and
bucket-keys it, never parses or trusts its content.

State lives in a single process; this is fine for the current single-worker
deployment and intentionally not distributed.
"""
from __future__ import annotations

import hashlib
import threading
import time

from fastapi import Depends, HTTPException, Request

from src.api.deps import get_api_key
from src.config import settings

_WINDOW_SECONDS = 60.0
_MAX_TRACKED_KEYS = 10_000
SESSION_ID_HEADER = "X-Session-ID"

_lock = threading.Lock()
_hits: dict[str, list[float]] = {}


def _hash_key(api_key: str | None) -> str:
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def _client_key(request: Request, api_key: str | None) -> str:
    client_ip = request.client.host if request.client else "unknown"
    parts = [client_ip, _hash_key(api_key)]
    session_id = request.headers.get(SESSION_ID_HEADER)
    if session_id:
        parts.append(hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16])
    return ":".join(parts)


def _check_and_record(bucket_key: str, limit: int) -> float | None:
    """Record a hit and return seconds to wait if over the limit, else None."""
    now = time.monotonic()
    cutoff = now - _WINDOW_SECONDS

    with _lock:
        if len(_hits) > _MAX_TRACKED_KEYS:
            _hits.clear()

        timestamps = [t for t in _hits.get(bucket_key, []) if t > cutoff]

        if len(timestamps) >= limit:
            retry_after = timestamps[0] + _WINDOW_SECONDS - now
            _hits[bucket_key] = timestamps
            return max(retry_after, 0.0)

        timestamps.append(now)
        _hits[bucket_key] = timestamps
        return None


def rate_limit(scope: str):
    """FastAPI dependency factory enforcing a per-minute limit for `scope`.

    A limit of 0 or an unconfigured scope means unlimited (no-op).
    """

    def _dependency(request: Request, api_key: str | None = Depends(get_api_key)) -> None:
        limit = settings.rate_limits.get(scope, 0)
        if not limit:
            return

        bucket_key = f"{scope}:{_client_key(request, api_key)}"
        retry_after = _check_and_record(bucket_key, limit)
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded for '{scope}'. Try again later.",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

    return _dependency