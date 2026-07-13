"""Shared FastAPI dependencies."""

from fastapi import Header

API_KEY_HEADER = "X-OpenRouter-Key"


def get_api_key(x_openrouter_key: str | None = Header(default=None, alias=API_KEY_HEADER)) -> str | None:
    """Per-request OpenRouter API key supplied by the frontend, if any.

    Passed through to call_llm/call_llm_sync, which fall back to
    src.config.settings.openrouter_api_key when this is None - supporting
    both a "bring your own key" public deployment (no server-side key set)
    and a local/dev setup with a key in .env.
    """
    return x_openrouter_key or None
