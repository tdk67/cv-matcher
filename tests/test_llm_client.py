"""Tests for the LLM client retry/backoff behavior (no network needed).

Retry-After capping (F2-14) and transient-error retry logic are pure
functions we can exercise without any HTTP. We only test the sync path;
the async twin was removed as dead code in F-08.
"""
from unittest import mock

import pytest

from src.agents.llm_client import _retry_delay_seconds, call_llm_sync
from src.agents.llm_client import LLMResponse, CALL_LLM_OVERRIDE


class TestRetryDelay:
    def test_exponential_backoff_base(self):
        with mock.patch("src.config.settings.llm_retry_backoff_base", 1.0):
            # attempt 1 -> base * 2^1 = 2 (+ jitter in [0, 1))
            d = _retry_delay_seconds(1, None)
            assert 2.0 <= d < 3.0

    def test_retry_after_honored(self):
        with mock.patch("src.config.settings.llm_retry_after_cap_seconds", 60):
            d = _retry_delay_seconds(0, "10")
            assert d == 10.0

    def test_retry_after_capped(self):
        # A hostile 3600s Retry-After must not park a worker for an hour.
        with mock.patch("src.config.settings.llm_retry_after_cap_seconds", 60):
            d = _retry_delay_seconds(0, "3600")
            assert d <= 60.0

    @pytest.mark.live
    def test_call_llm_sync_with_override_no_network(self):
        """The override seam means even a keyed environment never hits HTTP."""
        def fake(prompt, system_prompt, model, temperature, max_tokens, api_key):
            return LLMResponse("ok", "fake-model", 0, 0, 0.0, True)

        old = CALL_LLM_OVERRIDE
        try:
            globals()["CALL_LLM_OVERRIDE"] = fake
            resp = call_llm_sync("hi", api_key="sk-test")
            assert resp.success is True
            assert resp.content == "ok"
        finally:
            globals()["CALL_LLM_OVERRIDE"] = old