"""Shared FastAPI dependencies."""

import secrets
from typing import Optional

from fastapi import Header, HTTPException

from src.config import settings

API_KEY_HEADER = "X-OpenRouter-Key"
AUTH_TOKEN_HEADER = "X-API-Token"


def get_api_key(
    x_openrouter_key: Optional[str] = Header(default=None, alias=API_KEY_HEADER),
) -> Optional[str]:
    """Per-request OpenRouter API key supplied by the frontend, if any.

    Passed through to call_llm/call_llm_sync, which fall back to
    src.config.settings.openrouter_api_key when this is None - supporting
    both a "bring your own key" public deployment (no server-side key set)
    and a local/dev setup with a key in .env.
    """
    return x_openrouter_key or None


def require_auth(
    x_api_token: str | None = Header(default=None, alias=AUTH_TOKEN_HEADER),
    authorization: str | None = Header(default=None),
) -> None:
    """Require a matching API token on protected routes.

    Accepts either the `X-API-Token` header or a `Bearer` token in
    `Authorization`. When `settings.api_auth_token` is empty (the default),
    this is a no-op so existing tests and local/dev deployments keep
    working without auth. The configured token is never logged.
    """
    expected = settings.api_auth_token
    if not expected:
        return

    supplied = x_api_token
    if not supplied and authorization and authorization[:7].lower() == "bearer ":
        supplied = authorization[7:].strip()

    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API token.")
