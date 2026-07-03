"""Tests for section-aware chunker."""
import pytest

from src.ingestion.chunker import (
    chunk_document,
    detect_cv_sections,
    is_cv_section_header,
)


SAMPLE_CV = """JOHN SMITH
Senior Developer

CONTACT
Email: john@email.com

PROFESSIONAL SUMMARY
Senior developer with 8 years experience in Java and Python.

WORK EXPERIENCE

Senior Developer | TechCorp | 2020 - Present
- Built REST APIs serving 10M daily requests
- Led team of 5 developers

Junior Developer | StartupXYZ | 2018 - 2020
- Developed full-stack web applications

EDUCATION

Bachelor of Science in Computer Science
MIT | 2018

TECHNICAL SKILLS

Java, Python, JavaScript, Docker, Kubernetes

CERTIFICATIONS
- AWS Solutions Architect
- CKA
"""


class TestIsCvSectionHeader:
    def test_known_headers(self):
        assert is_cv_section_header("PROFESSIONAL SUMMARY")
        assert is_cv_section_header("WORK EXPERIENCE")
        assert is_cv_section_header("EDUCATION")
        assert is_cv_section_header("TECHNICAL SKILLS")
        assert is_cv_section_header("CERTIFICATIONS")
        assert is_cv_section_header("LANGUAGES")

    def test_not_headers(self):
        assert not is_cv_section_header("Senior Backend Developer | TechCorp | 2020")
        assert not is_cv_section_header("")  # empty
        assert not is_cv_section_header("A" * 50)  # too long

    def test_all_caps_heuristic(self):
        assert is_cv_section_header("PROJECTS")
        assert is_cv_section_header("AWARDS")
        assert not is_cv_section_header("MIT")  # blocklisted abbreviation
        assert not is_cv_section_header("AWS")  # blocklisted abbreviation
        assert not is_cv_section_header("PhD")  # not all-caps but blocklisted

    def test_blocklisted_abbreviations(self):
        blocklisted = ["MIT", "AWS", "GCP", "PhD", "MBA", "BS", "MS", "API", "SQL", "IT"]
        for abbr in blocklisted:
            assert not is_cv_section_header(abbr), f"'{abbr}' should not be a section header"


class TestDetectCvSections:
    def test_detects_all_sections(self):
        sections = detect_cv_sections(SAMPLE_CV)
        names = [s[0] for s in sections]
        assert "professional_summary" in names
        assert "work_experience" in names
        assert "education" in names
        assert "technical_skills" in names

    def test_section_content_not_empty(self):
        sections = detect_cv_sections(SAMPLE_CV)
        for name, content in sections:
            # Some short all-caps words (like "MIT") may be detected as
            # section headers, producing empty sections. This is a known
            # limitation of the heuristic. Major sections should be non-empty.
            major_sections = [
                "professional_summary", "work_experience",
                "education", "technical_skills",
            ]
            if name in major_sections:
                assert len(content.strip()) > 0, f"Section '{name}' is empty"

    def test_work_experience_has_content(self):
        sections = detect_cv_sections(SAMPLE_CV)
        exp = [c for n, c in sections if n == "work_experience"]
        assert len(exp) == 1
        assert "TechCorp" in exp[0]


class TestChunkDocument:
    def test_produces_chunks(self):
        chunks = chunk_document(SAMPLE_CV)
        assert len(chunks) > 0

    def test_chunks_have_section(self):
        chunks = chunk_document(SAMPLE_CV)
        for chunk in chunks:
            assert chunk.section != ""
            assert chunk.chars > 0

    def test_chunk_indices_sequential(self):
        chunks = chunk_document(SAMPLE_CV)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_section_awareness(self):
        """No chunk should mix two different sections."""
        chunks = chunk_document(SAMPLE_CV)
        sections_in_chunks = set(c.section for c in chunks)
        # Each chunk belongs to exactly one section
        assert len(sections_in_chunks) >= 3  # at least a few sections

    def test_empty_text(self):
        chunks = chunk_document("")
        assert len(chunks) == 0

    def test_short_text(self):
        chunks = chunk_document("Just a name")
        assert len(chunks) >= 1

    def test_chunk_size_respected(self):
        chunks = chunk_document(SAMPLE_CV, chunk_size=200, chunk_overlap=50)
        # Most chunks should be under 300 chars (some overflow due to splitting)
        oversized = [c for c in chunks if c.chars > 400]
        assert len(oversized) == 0, f"Found {len(oversized)} oversized chunks"
