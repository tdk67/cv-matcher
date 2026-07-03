"""Tests for document extractors."""
import tempfile
from pathlib import Path

import pytest

from src.ingestion.extractors import (
    extract_text,
    _compute_doc_id,
)


@pytest.fixture
def sample_txt(tmp_path):
    """Create a sample TXT file."""
    p = tmp_path / "test_cv.txt"
    p.write_text("John Doe\nSenior Developer\n\nSkills: Python, Java\n\nExperience:\n- Built APIs at Corp\n")
    return p


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file."""
    p = tmp_path / "team.csv"
    p.write_text("name,role,skills\nAlice,Backend Dev,Python; Java\nBob,Frontend Dev,React; TypeScript\n")
    return p


@pytest.fixture
def sample_pdf():
    """Path to a real sample PDF."""
    pdf_dir = Path("sample_data/pdf")
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        pytest.skip("No sample PDFs available")
    return pdfs[0]


class TestExtractors:
    def test_extract_txt(self, sample_txt):
        result = extract_text(sample_txt)
        assert result["format"] == "txt"
        assert "John Doe" in result["text"]
        assert "Python" in result["text"]
        assert result["page_count"] == 1
        assert len(result["warnings"]) == 0

    def test_extract_csv(self, sample_csv):
        result = extract_text(sample_csv)
        assert result["format"] == "csv"
        assert "Alice" in result["text"]
        assert "Python" in result["text"]

    def test_extract_pdf(self, sample_pdf):
        result = extract_text(sample_pdf)
        assert result["format"] == "pdf"
        assert len(result["text"]) > 100
        assert result["page_count"] >= 1

    def test_doc_id_stability(self, sample_txt):
        """Same file produces same doc_id."""
        r1 = extract_text(sample_txt)
        r2 = extract_text(sample_txt)
        assert r1["doc_id"] == r2["doc_id"]

    def test_doc_id_uniqueness(self, sample_txt, sample_csv):
        """Different files produce different doc_ids."""
        r1 = extract_text(sample_txt)
        r2 = extract_text(sample_csv)
        assert r1["doc_id"] != r2["doc_id"]

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / "test.xyz"
        p.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported format"):
            extract_text(p)

    def test_empty_txt(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        result = extract_text(p)
        assert result["text"] == ""


class TestComputeDocId:
    def test_deterministic(self):
        id1 = _compute_doc_id("test.pdf", "hello world")
        id2 = _compute_doc_id("test.pdf", "hello world")
        assert id1 == id2

    def test_different_content(self):
        id1 = _compute_doc_id("test.pdf", "hello")
        id2 = _compute_doc_id("test.pdf", "world")
        assert id1 != id2

    def test_format(self):
        doc_id = _compute_doc_id("test.pdf", "hello")
        assert "_" in doc_id
        parts = doc_id.split("_")
        assert len(parts) == 2
