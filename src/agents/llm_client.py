"""LLM client wrapper for OpenRouter API.

All model calls go through this module. Model names come from config,
never hardcoded.
"""

import json
import time
from dataclasses import dataclass

import httpx

from src.config import settings


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


async def call_llm(
    prompt: str,
    system_prompt: str = "",
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    response_format: str | None = None,
) -> LLMResponse:
    """Call an LLM via OpenRouter API.

    Args:
        prompt: User prompt
        system_prompt: System instruction (optional)
        model: Model override (defaults to settings.rag_model)
        temperature: Sampling temperature
        max_tokens: Max output tokens
        response_format: "json" for JSON mode (optional)

    Returns:
        LLMResponse with content and metadata
    """
    model = model or settings.rag_model

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
    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "HTTP-Referer": "https://agentic-rag-cv.local",
        "X-Title": "Agentic RAG CV Matcher",
        "Content-Type": "application/json",
    }

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            latency_ms = (time.time() - start) * 1000

            if resp.status_code != 200:
                error_text = resp.text[:500]
                return LLMResponse(
                    content="",
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    success=False,
                    error=f"API error {resp.status_code}: {error_text}",
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


def call_llm_sync(
    prompt: str,
    system_prompt: str = "",
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> LLMResponse:
    """Synchronous version of call_llm for use in non-async contexts."""
    model = model or settings.rag_model

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
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "HTTP-Referer": "https://agentic-rag-cv.local",
        "X-Title": "Agentic RAG CV Matcher",
        "Content-Type": "application/json",
    }

    start = time.time()
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            latency_ms = (time.time() - start) * 1000

            if resp.status_code != 200:
                return LLMResponse(
                    content="",
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    success=False,
                    error=f"API error {resp.status_code}: {resp.text[:500]}",
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
