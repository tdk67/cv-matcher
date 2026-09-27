"""Synthetic CV data generation package.

Provides deterministic, seeded generation of realistic CV persona data
(TXT / PDF / CSV) for testing the ingestion and RAG pipeline without
releasing any real personal data. The CLI module is imported lazily so
`python -m src.data.cli` does not trigger a circular import.
"""

from src.data.generator import generate_personas  # noqa: F401

__all__ = ["generate_personas"]