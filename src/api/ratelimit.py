"""In-memory sliding-window rate limiting (stdlib only, no new dependencies).

Every request is always charged to a HARD bucket keyed on client IP plus a
hashed OpenRouter key when present. In addition, when the frontend forwards
the optional `X-Session-ID` header, the request is also charged to a soft
per-session SUB-bucket so users behind one reverse-proxy IP (Streamlit ->
FastAPI over localhost inside the shipped Docker container) do not collapse
into a single shared bucket.

Security property: the session header is fully client-controlled and may be
rotated/forged at will, so it can only ever SPLIT a hard bucket - it never
replaces the IP accounting that actually bounds abuse. Rotating the header
spawns fresh sub-buckets, but the hard per-IP bucket still trips at the same
`limit` per minute, so the endpoint stays rate limited for a script that
rotates headers just as tightly as for one that doesn't send any.

The header is opaque: the backend hashes it with SHA-256 and never parses or
logs its content. Tracking sub-buckets can be disabled with the environment
variable `CV_MATCHER_TRACK_SESSION_RATELIMIT=0` (e.g. for tests that assert
IP-only accounting).

State lives in a single process; this is fine for the current single-worker
deployment and intentionally not distributed.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time

from fastapi import Depends, HTTPException, Request

from src.api.deps import get_api_key
from src.config import settings

_WINDOW_SECONDS = 60.0
# Hard cap on distinct bucket keys we track. Instead of wiping the whole
# table when the cap is reached (which would reset EVERY user's accounting
# - a global denial-of-accounting attack), we evict expired keys and then
# drop the oldest keys that have no recent hits. A key whose window has
# passed contributes nothing to enforcement and can be forgotten safely.
_MAX_TRACKED_KEYS = 10_000
SESSION_ID_HEADER = "X-Session-ID"

# If disabled, X-Session-ID is ignored entirely (IP-only accounting).
_TRACK_SESSION_SUB_BUCKETS = os.getenv("CV_MATCHER_TRACK_SESSION_RATELIMIT", "1") != "0"

_lock = threading.Lock()
_hits: dict[str, list[float]] = {}


def _hash_key(api_key: str | None) -> str:
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def _client_key(request: Request, api_key: str | None) -> str:
    """HARD bucket key: client IP + a hash of the OpenRouter key.

    Never includes anything the client could rotate to reset accounting.
    """
    client_ip = request.client.host if request.client else "unknown"
    return ":".join([client_ip, _hash_key(api_key)])


def _session_sub_key(request: Request) -> str | None:
    """Optional sub-bucket from the client-supplied session header, if any.

    Returns a hashed key or None. Because the header is attacker-controlled
    it may only split a hard bucket - the hard IP+key bucket is always
    enforced on top (see _client_key), so rotating this header never resets
    the per-IP accounting that actually bounds abuse.
    """
    if not _TRACK_SESSION_SUB_BUCKETS:
        return None
    session_id = request.headers.get(SESSION_ID_HEADER)
    if not session_id:
        return None
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


def _evict_old_and_overflow() -> None:
    """Drop expired buckets, then the oldest buckets if still over cap.

    Runs while holding `_lock`. Expired keys (all hits older than one
    window) are removed unconditionally - they contribute nothing to
    accounting. If the table is still over `_MAX_TRACKED_KEYS`, evict the
    keys with the OLDEST most-recent hit (LRU-ish) instead of wiping
    everyone's accounting with a global `clear()`. The hostile case this
    guards against is a rotating attacker spawning many fresh sub-bucket
    keys that are all inside the window; dropping the `over` oldest-hit
    keys is a far cheaper loss than resetting every user's bucket (F4-08).
    """
    now = time.monotonic()
    cutoff = now - _WINDOW_SECONDS

    expired = [k for k, v in _hits.items() if not v or v[-1] <= cutoff]
    for k in expired:
        del _hits[k]

    over = len(_hits) - _MAX_TRACKED_KEYS
    if over <= 0:
        return

    lru = sorted(
        _hits.items(),
        key=lambda kv: kv[1][-1] if kv[1] else 0.0,
    )
    for k, _v in lru[:over]:
        del _hits[k]


def _check_and_record(bucket_key: str, limit: int) -> float | None:
    """Record a hit and return seconds to wait if over the limit, else None."""
    now = time.monotonic()
    cutoff = now - _WINDOW_SECONDS

    with _lock:
        # Enforce the key cap BEFORE recording (table may already be over
        # from a prior burst) and AFTER (this record may push it over).
        if len(_hits) >= _MAX_TRACKED_KEYS:
            _evict_old_and_overflow()

        timestamps = [t for t in _hits.get(bucket_key, []) if t > cutoff]

        if len(timestamps) >= limit:
            retry_after = timestamps[0] + _WINDOW_SECONDS - now
            _hits[bucket_key] = timestamps
            return max(retry_after, 0.0)

        timestamps.append(now)
        _hits[bucket_key] = timestamps

        if len(_hits) >= _MAX_TRACKED_KEYS:
            _evict_old_and_overflow()
        return None


def _enforce_bucket(bucket_key: str, limit: int) -> None:
    """Record a hit on `bucket_key`; raise 429 if the bucket is over `limit`."""
    retry_after = _check_and_record(bucket_key, limit)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


def rate_limit(scope: str):
    """FastAPI dependency factory enforcing a per-minute limit for `scope`.

    A limit of 0 or an unconfigured scope means unlimited (no-op).
    """

    def _dependency(request: Request, api_key: str | None = Depends(get_api_key)) -> None:
        limit = settings.rate_limits.get(scope, 0)
        if not limit:
            return

        # Hard bucket: always enforced. The client IP (or the trusted
        # reverse-proxy X-Forwarded-For, with --proxy-headers) is the source
        # of truth; a client-supplied session header never removes this
        # accounting.
        _enforce_bucket(f"{scope}:{_client_key(request, api_key)}", limit)

        # Optional session sub-bucket: splits the same limit per session, but
        # the hard bucket above already bounds abuse even when the header is
        # rotated, forged, or omitted.
        session_key = _session_sub_key(request)
        if session_key is not None:
            _enforce_bucket(f"{scope}:session:{session_key}", limit)

    return _dependency