"""Pytest bootstrap: ensure sample_data/ exists, define the live-LLM marker.

A fresh clone has no committed binaries under sample_data/pdf (they are
generated, not versioned). tests/test_extractors.py and
tests/test_pipeline.py both depend on real sample PDFs, so generate a
small deterministic set here if none are present yet. This runs once per
session and is a no-op if sample_data/pdf already has PDFs.

Tests that REQUIRE a live OpenRouter key are marked with `@pytest.mark.live`
and are skipped automatically when no key is configured, so the suite is
green on a fresh clone without any credentials.

The `_isolate_runtime_dirs` autouse fixture points every test at a fresh
runtime directory (uploads, ChromaDB persist dir, query log, eval log) so
no test can pollute the developer's `.data/` or another test's state - the
root cause of the intermittent `test_delete_removes_file_from_disk` flake
and the `test_cv.txt` contamination observed in Round-3 review.
"""
from __future__ import annotations

import importlib
import sys
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


# --- Runtime-dir isolation (F3-08) ---------------------------------------

@pytest.fixture(autouse=True)
def _isolate_runtime_dirs(tmp_path, monkeypatch):
    """Point every test at a fresh, per-test runtime directory.

    Covers the three mutable stores the app writes to:
      - settings.upload_dir          (.data/uploads -> tmp .uploads)
      - settings.chroma_persist_dir  (.data/chromadb -> tmp .chromadb)
      - the query log / eval result files (module-level Path objects)

    Also resets the vector-store singleton (it would otherwise stay bound to
    the first test's tmp dir for the rest of the session), the module-level
    JSON file paths that are resolved at import time, and the rate-limit
    accounting.
    """
    import src.vectorstore.store as store_mod
    from src.config import settings

    upload_dir = tmp_path / "uploads"
    chroma_dir = tmp_path / "chromadb"
    upload_dir.mkdir(parents=True)
    chroma_dir.mkdir(parents=True)

    monkeypatch.setattr(settings, "upload_dir", str(upload_dir))
    monkeypatch.setattr(settings, "chroma_persist_dir", str(chroma_dir))

    # Reset the storage singletons so they bind to the fresh tmp dirs.
    store_mod._shared_store_instance = None
    store_mod._shared_store_dir = None

    # Fresh rate-limit accounting per test: buckets are process-global, so
    # without this the TestRateLimit cases would bleed 429s into each other.
    try:
        import src.api.ratelimit as rl
        monkeypatch.setattr(rl, "_hits", {})
    except Exception:  # noqa: BLE001 - never fail collection on import hiccup
        pass

    def _patch_json_paths(module_name: str, attrs: dict[str, str]):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - never fail collection on import hiccup
            return
        for attr, key in attrs.items():
            monkeypatch.setattr(module, attr, Path(tmp_path) / key)

    _patch_json_paths("src.utils.query_log", {"LOG_FILE": "query_log.json"})
    _patch_json_paths(
        "src.api.evaluation",
        {
            "EVAL_FILE": "eval_results.json",
            # DEFAULT_QUESTIONS must reload from the real resources file, not
            # from a wiped tmp dir, so evaluations in tests still work.
        },
    )

    # DEFAULT_QUESTIONS is computed at import time from the real resources
    # file, so it survives the tmp-dir switch untouched - no action needed.

    yield

    # Reset singletons again after the test so later modules importing them
    # fresh are not stuck with an instance bound to a removed tmp dir.
    if "src.vectorstore.store" in sys.modules:
        store_mod._shared_store_instance = None
        store_mod._shared_store_dir = None
    if "src.api.evaluation" in sys.modules:
        import src.api.evaluation as ev
        ev._eval_state = ev._EvalState()