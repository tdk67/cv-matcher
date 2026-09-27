"""Pytest bootstrap: ensure sample_data/ exists, define the live-LLM marker.

A fresh clone has no committed binaries under sample_data/pdf (they are
generated, not versioned). tests/test_extractors.py and
tests/test_pipeline.py both depend on real sample PDFs, so generate a
small deterministic set here if none are present yet. This runs once per
session and is a no-op if sample_data/pdf already has PDFs.

Tests that REQUIRE a live OpenRouter key are marked with `@pytest.mark.live`
and are skipped automatically when no key is configured, so the suite is
green on a fresh clone without any credentials.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.cli import generate

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = REPO_ROOT / "sample_data"


def _live_llm_configured() -> bool:
    try:
        from src.config import settings
    except Exception:  # noqa: BLE001 - never crash collection because of config
        return False
    key = (settings.openrouter_api_key or "").strip()
    return bool(key) and key not in ("sk-or-your-key-here", "your_openrouter_api_key_here")


def pytest_configure(config):
    """Generate sample data if missing and register the 'live' marker."""
    pdf_dir = SAMPLE_DATA_DIR / "pdf"
    if not (pdf_dir.exists() and list(pdf_dir.glob("*.pdf"))):
        generate(count=3, output=SAMPLE_DATA_DIR, fmt="all", seed=1)

    config.addinivalue_line(
        "markers",
        "live: test exercises the real LLM and requires an OpenRouter API key "
        "(skipped automatically when none is configured)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip live-LLM tests automatically when no key is configured."""
    if _live_llm_configured():
        return
    skip = pytest.mark.skip(reason="No OpenRouter API key configured; skipping live-LLM test")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip)