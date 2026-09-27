"""Pytest bootstrap: ensure sample_data/ exists before tests collect.

A fresh clone has no committed binaries under sample_data/pdf (they are
generated, not versioned). tests/test_extractors.py and
tests/test_pipeline.py both depend on real sample PDFs, so generate a
small deterministic set here if none are present yet. This runs once per
session and is a no-op if sample_data/pdf already has PDFs.
"""

from pathlib import Path

from src.data.cli import generate

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = REPO_ROOT / "sample_data"


def pytest_configure(config):
    """Generate a small sample_data/ set before test collection, if missing."""
    pdf_dir = SAMPLE_DATA_DIR / "pdf"
    if pdf_dir.exists() and list(pdf_dir.glob("*.pdf")):
        return
    generate(count=3, output=SAMPLE_DATA_DIR, fmt="all", seed=1)
