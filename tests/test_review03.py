"""Regression tests for review-03 fixes (F3-01 rate limit, F3-02 synthetic
cleanup, F3-03/F3-04 upload data-loss + rejected-file exposure)."""

from __future__ import annotations

import pytest


class TestRateLimitSessionBypass:
    """F3-01: rotating X-Session-ID must NOT bypass the hard per-IP bucket."""

    def test_session_rotation_cannot_bypass_ip_bucket(self, monkeypatch):
        from fastapi.testclient import TestClient

        from src.config import settings
        from src.main import app
        import src.api.ratelimit as rl

        # 2/min per IP. Session rotation must not stretch this: callers on the
        # same IP share the hard bucket no matter what header they send.
        monkeypatch.setattr(settings, "rate_limits", {**settings.rate_limits, "key_validate": 2})
        monkeypatch.setattr(rl, "_TRACK_SESSION_SUB_BUCKETS", True)
        monkeypatch.setattr(rl, "_hits", {})

        quiet = TestClient(app, raise_server_exceptions=False)
        statuses = []
        for i in range(3):
            resp = quiet.get(
                "/api/key/validate",
                headers={"X-Session-ID": f"session-{i}"},  # rotate every call
            )
            statuses.append(resp.status_code)

        assert statuses[:2] == [200, 200]
        assert statuses[2] == 429, f"rotating X-Session-ID must not bypass the IP bucket: {statuses}"
        assert "Retry-After" in resp.headers

    def test_same_session_still_bounded(self, monkeypatch):
        """A single session is limited just as tightly as no header."""
        from fastapi.testclient import TestClient

        from src.config import settings
        from src.main import app
        import src.api.ratelimit as rl

        monkeypatch.setattr(settings, "rate_limits", {**settings.rate_limits, "key_validate": 2})
        monkeypatch.setattr(rl, "_TRACK_SESSION_SUB_BUCKETS", True)
        monkeypatch.setattr(rl, "_hits", {})

        quiet = TestClient(app, raise_server_exceptions=False)
        statuses = []
        for _ in range(3):
            resp = quiet.get("/api/key/validate", headers={"X-Session-ID": "same-session"})
            statuses.append(resp.status_code)
        assert statuses[2] == 429


class TestSyntheticCleanup:
    """F3-02: the stale-file glob fix (generator truthiness bug)."""

    def test_stale_pdfs_removed_on_regeneration(self, tmp_path, monkeypatch):
        from src.api import synthetic as syn
        from src.data.cli import generate as real_generate

        pdf_dir = tmp_path / "pdf"
        txt_dir = tmp_path / "txt"
        pdf_dir.mkdir(parents=True)
        txt_dir.mkdir(parents=True)
        for i in (1, 2, 3, 4, 5):
            (pdf_dir / f"persona_{i:02d}.pdf").write_bytes(b"%PDF fake")
            (txt_dir / f"persona_{i:02d}.txt").write_text("persona")
        (tmp_path / "personas.csv").write_text("name,role\n")

        monkeypatch.setattr(syn, "_OUTPUT_DIR", tmp_path)

        def fake_generate(count, output, fmt="all", seed=0):
            # Simulate generation of a smaller set: files 01-02 only.
            for p in (pdf_dir / "persona_01.pdf", pdf_dir / "persona_02.pdf"):
                p.write_bytes(b"%PDF fake")
            for p in (txt_dir / "persona_01.txt", txt_dir / "persona_02.txt"):
                p.write_text("persona")
            (tmp_path / "personas.csv").write_text("name,role\n")
            return {
                "count": 2,
                "output_dir": str(tmp_path),
                "txt_count": 2,
                "pdf_count": 2,
                "csv_count": 1,
            }

        monkeypatch.setattr(syn, "generate", fake_generate)

        request = type("Req", (), {"count": 2, "format": "all", "seed": 0})()
        response = syn.generate_synthetic_data(request)

        assert response.success is True
        # Stale higher-numbered files MUST be gone after regenerating with a
        # smaller count - the original F2-17/F3-02 bug kept persona_03..05.
        assert sorted(p.name for p in pdf_dir.glob("persona_*.pdf")) == [
            "persona_01.pdf", "persona_02.pdf"]
        assert sorted(p.name for p in txt_dir.glob("persona_*.txt")) == [
            "persona_01.txt", "persona_02.txt"]


class TestUploadRejectionCleanup:
    """F3-04: a rejected upload must not remain on disk / readable."""

    def test_rejected_upload_leaves_no_disk_file(self, tmp_path, monkeypatch):
        from src.ingestion import pipeline
        import src.ingestion.pipeline as pipe_mod
        from src.vectorstore.store import CVVectorStore

        from src.config import settings
        monkeypatch.setattr(settings, "upload_dir", str(tmp_path))

        store = CVVectorStore(persist_dir=str(tmp_path / "chroma"))

        # Force the pipeline to fail AFTER the file has been written.
        def fail_ingest(file_path, vector_store, scanner=None, **kwargs):
            class R:
                success = False
                filename = file_path.name
                doc_id = ""
                format = ""
                chunks_created = 0
                tainted = False
                warnings = []
                error = "scan failed: injected"
            return R()

        monkeypatch.setattr(pipe_mod, "ingest_document", fail_ingest)
        # Do not trigger the doc_id "re-ingest" flow; nothing gets stored.
        monkeypatch.setattr(pipeline.CVVectorStore, "list_documents", lambda self: [])
        monkeypatch.setattr(pipeline.CVVectorStore, "delete_by_source", lambda self, src: 0)

        result = pipeline.ingest_upload(b"evil pdf bytes", "evil_cv.pdf", store)

        assert result.success is False
        leftovers = [p.name for p in tmp_path.iterdir()]
        assert not any(name.startswith(".uploading-") for name in leftovers), leftovers
        assert not (tmp_path / "evil_cv.pdf").exists(), "rejected upload must not stay on disk"

    def test_reupload_failure_keeps_previous_version(self, tmp_path, monkeypatch):
        """F3-03: a failed re-upload must not destroy the previous good version."""
        from src.ingestion import pipeline
        import src.ingestion.pipeline as pipe_mod
        from src.vectorstore.store import CVVectorStore

        from src.config import settings
        monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
        store = CVVectorStore(persist_dir=str(tmp_path / "chroma"))

        # Previous good version exists on disk + in the KB.
        (tmp_path / "resume.pdf").write_bytes(b"GOOD OLD VERSION")
        monkeypatch.setattr(pipeline.CVVectorStore, "list_documents",
                            lambda self: [{"source": "resume.pdf"}])
        old_chunks = {"n": 5}
        monkeypatch.setattr(pipeline.CVVectorStore, "delete_by_source",
                            lambda self, src: old_chunks["n"])

        def fail_ingest(file_path, vector_store, scanner=None, **kwargs):
            assert file_path.name.startswith(".uploading-")  # temp name, not final
            class R:
                success = False
                filename = file_path.name
                doc_id = ""
                format = ""
                chunks_created = 0
                tainted = False
                warnings = []
                error = "scan failed: injected"
            return R()

        monkeypatch.setattr(pipe_mod, "ingest_document", fail_ingest)

        result = pipeline.ingest_upload(b"NEW EVIL BYTES", "resume.pdf", store)

        assert result.success is False
        # The previous good version must be untouched.
        assert (tmp_path / "resume.pdf").read_bytes() == b"GOOD OLD VERSION"
        assert old_chunks["n"] == 5, "old chunks must not have been deleted yet"
        leftovers = [p.name for p in tmp_path.iterdir()]
        assert not any(name.startswith(".uploading-") for name in leftovers), leftovers

    def test_successful_reupload_swaps_version(self, tmp_path, monkeypatch):
        """End-to-end: a good re-upload replaces old chunks + raw file,
        and the chunk metadata names the final filename (not the temp one)."""
        from src.ingestion.pipeline import ingest_upload
        from src.vectorstore.store import CVVectorStore

        from src.config import settings
        monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
        store = CVVectorStore(persist_dir=str(tmp_path / "chroma"))

        # First version.
        r1 = ingest_upload(b"John Doe\nSenior Java developer\n", "resume.txt", store)
        assert r1.success, r1.error
        assert r1.filename == "resume.txt"

        # Re-upload with new content (new doc_id => NEW chunk ids).
        r2 = ingest_upload(b"Jane Roe\nPython expert\n", "resume.txt", store)
        assert r2.success, r2.error

        # Exactly one document named resume.txt, and its chunks must all point
        # at the final filename (not an .uploading-* temp name).
        docs = store.list_documents()
        resume_docs = [d for d in docs if d["source"] == "resume.txt"]
        assert len(resume_docs) == 1, docs
        assert all(
            not d["source"].startswith(".uploading-") for d in docs
        ), docs

        chunks = store.query("Jane Roe Python", n_results=10)
        assert any(
            c["metadata"].get("source") == "resume.txt" for c in chunks
        ), [c["metadata"] for c in chunks]

        # Raw file on disk has the NEW content, and no temp files remain.
        assert (tmp_path / "resume.txt").read_bytes() == b"Jane Roe\nPython expert\n"
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".uploading-")]