"""LLM client wrapper for OpenRouter API.

All model calls go through this module. Model names come from config,
never hardcoded. Only the synchronous `call_llm_sync` is used by agents
(the async twin was dead code and was removed).
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

# HTTP status codes considered transient (worth retrying).
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# Transport-level exceptions considered transient (worth retrying).
_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.TransportError,
    httpx.NetworkError,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# Test seam (F-14): pipelines can install a fake callable here:
#   call_llm_sync(prompt, system_prompt, model, temperature, max_tokens, api_key)
# returning an LLMResponse. When set, real HTTP calls are never made.
CALL_LLM_OVERRIDE = None


def _retry_delay_seconds(attempt: int, retry_after: str | None) -> float:
    """Compute how long to sleep before the next retry attempt.

    Honors an OpenRouter `Retry-After` header if present (seconds), else
    falls back to exponential backoff with jitter:
    base * 2**attempt + random(0, base).

    `Retry-After` is capped at `llm_retry_after_cap_seconds` so a hostile
    or misconfigured upstream can't park a worker thread for an unbounded
    time (a 3600s Retry-After would otherwise hold a thread for an hour).
    """
    if retry_after:
        try:
            return min(
                float(retry_after),
                settings.llm_retry_after_cap_seconds,
            )
        except ValueError:
            pass
    base = settings.llm_retry_backoff_base
    return base * (2 ** attempt) + random.uniform(0, base)


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    error: str | None = None


def call_llm_sync(
    prompt: str,
    system_prompt: str = "",
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    api_key: str | None = None,
) -> LLMResponse:
    """Call an LLM via OpenRouter API (or the installed test override).

    Args:
        prompt: User prompt
        system_prompt: System instruction (optional)
        model: Model override (defaults to settings.rag_model)
        temperature: Sampling temperature
        max_tokens: Max output tokens
        api_key: Per-request OpenRouter key (e.g. supplied by the frontend).
            Falls back to settings.openrouter_api_key for local/dev setups.

    Returns:
        LLMResponse with content and metadata
    """
    if CALL_LLM_OVERRIDE is not None:
        return CALL_LLM_OVERRIDE(prompt, system_prompt, model, temperature, max_tokens, api_key)

    model = model or settings.rag_model
    resolved_key = api_key or settings.openrouter_api_key

    if not resolved_key:
        return LLMResponse(
            content="",
            model=model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            success=False,
            error="No OpenRouter API key configured. Enter your key in the sidebar, "
                  "or set OPENROUTER_API_KEY for a local/server-side deployment.",
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "HTTP-Referer": "https://agentic-rag-cv.local",
        "X-Title": "Agentic RAG CV Matcher",
        "Content-Type": "application/json",
    }

    start = time.time()
    max_retries = settings.llm_max_retries
    last_error = "Unknown error"
    last_latency_ms = 0.0

    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    OPENROUTER_URL,
                    json=payload,
                    headers=headers,
                )
                latency_ms = (time.time() - start) * 1000

                if resp.status_code != 200:
                    last_error = f"API error {resp.status_code}: {resp.text[:500]}"
                    last_latency_ms = latency_ms

                    if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries:
                        logger.warning(
                            f"LLM call attempt {attempt + 1}/{max_retries + 1} failed with "
                            f"transient status {resp.status_code}, retrying: {last_error}"
                        )
                        delay = _retry_delay_seconds(attempt, resp.headers.get("Retry-After"))
                        time.sleep(delay)
                        continue

                    return LLMResponse(
                        content="",
                        model=model,
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency_ms,
                        success=False,
                        error=last_error,
                    )

                data = resp.json()
                choice = data.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                usage = data.get("usage", {})

                return LLMResponse(
                    content=content,
                    model=model,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    latency_ms=latency_ms,
                    success=True,
                )

        except _RETRYABLE_EXCEPTIONS as e:
            latency_ms = (time.time() - start) * 1000
            last_error = f"Request failed: {str(e)}"
            last_latency_ms = latency_ms

            if attempt < max_retries:
                logger.warning(
                    f"LLM call attempt {attempt + 1}/{max_retries + 1} failed with "
                    f"transient error, retrying: {last_error}"
                )
                delay = _retry_delay_seconds(attempt, None)
                time.sleep(delay)
                continue

            return LLMResponse(
                content="",
                model=model,
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                success=False,
                error=last_error,
            )

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return LLMResponse(
                content="",
                model=model,
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                success=False,
                error=f"Request failed: {str(e)}",
            )

    return LLMResponse(
        content="",
        model=model,
        input_tokens=0,
        output_tokens=0,
        latency_ms=last_latency_ms,
        success=False,
        error=last_error,
    )