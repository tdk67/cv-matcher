"""OpenRouter API key validation.

Lets the frontend give immediate feedback on whether a user-supplied key
is valid, without spending it on a real completion — OpenRouter's own
key-info endpoint is free and just reports whether the key is accepted.
"""

import logging

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.deps import get_api_key

logger = logging.getLogger(__name__)

router = APIRouter()

OPENROUTER_KEY_INFO_URL = "https://openrouter.ai/api/v1/auth/key"


class KeyValidationResponse(BaseModel):
    valid: bool
    detail: str
    limit: float | None = None
    usage: float | None = None
    is_free_tier: bool | None = None


@router.get("/validate", response_model=KeyValidationResponse)
async def validate_api_key(api_key: str | None = Depends(get_api_key)):
    """Check whether the supplied key (X-OpenRouter-Key header) is valid."""
    if not api_key:
        return KeyValidationResponse(valid=False, detail="No API key provided.")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                OPENROUTER_KEY_INFO_URL,
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.HTTPError as e:
        logger.warning(f"Key validation request failed: {e}")
        return KeyValidationResponse(valid=False, detail=f"Could not reach OpenRouter: {e}")

    if resp.status_code == 200:
        data = resp.json().get("data", {})
        return KeyValidationResponse(
            valid=True,
            detail="Key is valid.",
            limit=data.get("limit"),
            usage=data.get("usage"),
            is_free_tier=data.get("is_free_tier"),
        )
    if resp.status_code == 401:
        return KeyValidationResponse(valid=False, detail="Invalid API key.")

    return KeyValidationResponse(
        valid=False,
        detail=f"OpenRouter returned {resp.status_code}: {resp.text[:200]}",
    )
