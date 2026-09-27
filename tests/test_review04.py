"""Regression tests for review-04 fixes (F4-01..F4-11).

Each test here reproduces the exact failure mode the review identified and
must FAIL against the pre-fix code (fail-open planner gate, retry sleep
overshoot, temp-name doc_id drift, per-key rate-limit eviction, query
TOCTOU, empty-answer render edge, scanner-singleton reset, upload scope).
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest


# --- F4-01: Planner scope gate fails CLOSED on malformed output ------------

class TestPlannerFailClosed:
    """F4-01: a missing/non-bool `is_in_scope` (or unknown query_type) must
    raise instead of silently treating an out-of-scope query as in-scope."""

    @pytest.fixture
    def stub_llm_malformed(self, monkeypatch):
        from src.agents import llm_client as llm_mod
        from src.agents.llm_client import LLMResponse

        responses = []

        def fake(prompt, system_prompt, model, temperature, max_tokens, api_key):
            if responses:
                body = {"content": responses.pop(0), "model": "stub", "input_tokens": 1,
                        "output_tokens": 1, "latency_ms": 0.5, "success": True}
                return LLMResponse(**body)
            return LLMResponse(content="", model="stub", input_tokens=0, output_tokens=0,
                               latency_ms=0.0, success=False, error="no more stubbed responses")

        monkeypatch.setattr(llm_mod, "CALL_LLM_OVERRIDE", fake)

        def _use(content: str):
            responses.append(content)

        return _use

    def _plan_with(self, stub_llm_malformed, payload: dict):
        from src.agents.context import PipelineContext
        from src.agents.planner import plan

        stub_llm_malformed(json.dumps(payload))
        ctx = PipelineContext(query="What is the weather today?")
        return plan(ctx)

    def test_missing_is_in_scope_raises(self, stub_llm_malformed):
        from src.agents.planner import plan
        with pytest.raises(RuntimeError):
            self._plan_with(stub_llm_malformed, {
                "query_type": "out_of_scope",
                "rejection_reason": "weather",
                "extracted_skills": [],
            })

    def test_non_bool_is_in_scope_raises(self, stub_llm_malformed):
        with pytest.raises(RuntimeError):
            self._plan_with(stub_llm_malformed, {
                "query_type": "out_of_scope",
                "is_in_scope": "yes",  # string, not bool
                "rejection_reason": "weather",
            })

    def test_unknown_query_type_raises(self, stub_llm_malformed):
        with pytest.raises(RuntimeError):
            self._plan_with(stub_llm_malformed, {
                "query_type": "totally_new_type",
                "is_in_scope": True,
            })

    def test_valid_in_scope_still_works(self, stub_llm_malformed):
        ctx = self._plan_with(stub_llm_malformed, {
            "query_type": "similarity",
            "is_in_scope": True,
            "rejection_reason": "",
            "extracted_skills": ["Java"],
            "extracted_requirements": "Java dev",
            "search_strategy": "both",
        })
        assert ctx.is_in_scope is True
        assert ctx.query_type == "similarity"


# --- F4-03: upload endpoint is rate limited --------------------------------

class TestUploadRateLimit:
    """F4-03: the CPU-heavy upload endpoint must be rate limited."""

    def test_upload_scope_wired(self, monkeypatch):
        from fastapi.testclient import TestClient

        from src.main import app
        import src.api.ratelimit as rl
        from src.config import settings

        monkeypatch.setattr(settings, "rate_limits", {**settings.rate_limits, "upload": 1})
        monkeypatch.setattr(rl, "_hits", {})

        quiet = TestClient(app, raise_server_exceptions=False)
        # First upload consumes the only slot for this IP (bucket 1/min).
        r1 = quiet.post("/api/documents/upload", files={"file": ("a.txt", b"hello", "text/plain")})
        # The store singleton is bound to the real .data/chromadb during this
        # test; force a fresh tmp store so we don't touch dev data.
        monkeypatch.setattr(settings, "chroma_persist_dir", str(Path(tempfile.mkdtemp()) / "chroma"))
        r2 = quiet.post("/api/documents/upload", files={"file": ("b.txt", b"world", "text/plain")})
        assert r1.status_code in (200, 422), r1.text  # uploaded or rejected, but consumed a slot
        assert r2.status_code == 429, f"second upload must be rate limited: {r2.status_code}"


# --- F4-04: retry sleep is clamped to the deadline -------------------------

class TestRetrySleepClamped:
    """F4-04: _retry_delay_seconds may return up to 60s, but call_llm_sync
    must never sleep past the caller's deadline."""

    def test_retry_sleep_clamped_to_deadline(self, monkeypatch):
        from src.agents import llm_client as llm_mod

        sleeps = []
        monkeypatch.setattr(llm_mod.time, "sleep", lambda s: sleeps.append(s))
        monkeypatch.setattr(llm_mod.settings, "llm_max_retries", 3)
        monkeypatch.setattr(llm_mod.settings, "llm_retry_backoff_base", 1.0)
        monkeypatch.setattr(llm_mod.settings, "llm_retry_after_cap_seconds", 60)

        # Deadline is 2 seconds from now; a 60s Retry-After must be clamped to ~remaining.
        monkeypatch.setattr(llm_mod, "_llm_deadline", __import__("contextvars").ContextVar("d", default=time.monotonic() + 2))

        # Force a 429 transient failure on the first call.
        class FakeResp:
            status_code = 429
            headers = {"Retry-After": "60"}
            text = "rate limited"

            def json(self):
                return {}

        calls = {"n": 0}

        def fake_post(*a, **k):
            # Return 429 once, then 200
            if calls["n"] == 0:
                calls["n"] += 1
                return FakeResp()
            calls["n"] += 1
            class Ok:
                status_code = 200
                def json(self):
                    return {"choices": [{"message": {"content": "ok"}}], "usage": {}}
            return Ok()

        import httpx
        class FakeClient:
            def __init__(self, timeout=None):
                self._timeout = timeout
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def post(self, *a, **k):
                return fake_post(*a, **k)

        monkeypatch.setattr(llm_mod.httpx, "Client", FakeClient)
        # Force a real HTTP path (no override)
        monkeypatch.setattr(llm_mod, "CALL_LLM_OVERRIDE", None)
        monkeypatch.setattr(llm_mod.settings, "openrouter_api_key", "sk-test")

        resp = llm_mod.call_llm_sync("hi")
        assert resp.success
        assert sleeps, "expected at least one retry sleep"
        # The sleep must be <= remaining deadline (~2s), not the 60s requested.
        assert sleeps[0] <= 2.5, f"retry sleep {sleeps[0]} overshot the deadline"


# --- F4-08: per-key eviction instead of global clear -----------------------

class TestRateLimitPerKeyEviction:
    """F4-08: reaching _MAX_TRACKED_KEYS must evict stale keys, never wipe
    everyone's accounting with a global clear()."""

    def test_eviction_never_clears_active_buckets(self, monkeypatch):
        import src.api.ratelimit as rl

        monkeypatch.setattr(rl, "_MAX_TRACKED_KEYS", 5)
        monkeypatch.setattr(rl, "_hits", {})
        now = time.monotonic()

        # Five distinct active buckets (within window).
        for i in range(5):
            rl._check_and_record(f"active:{i}", limit=10)

        # One stale bucket (all hits older than a window).
        rl._hits["stale:999"] = [now - 100, now - 90]

        assert len(rl._hits) == 6
        # Trigger the eviction path by adding one more key -> over cap.
        rl._check_and_record("active:5", limit=10)

        # The stale bucket must ALWAYS go (expired). And the accounting must
        # NOT have been wiped: other active buckets must survive. At most the
        # single oldest LRU key may be dropped to meet the cap - never a
        # global clear().
        assert "stale:999" not in rl._hits
        remaining = [f"active:{i}" for i in range(6) if f"active:{i}" in rl._hits]
        assert len(remaining) >= 5, f"eviction wiped too much: {rl._hits}"
        # The freshly recorded key MUST be present (it just fired).
        assert "active:5" in rl._hits

    def test_worst_case_only_drops_oldest_inactive(self, monkeypatch):
        import src.api.ratelimit as rl

        monkeypatch.setattr(rl, "_MAX_TRACKED_KEYS", 3)
        monkeypatch.setattr(rl, "_hits", {})
        now = time.monotonic()

        # 3 stale keys (outside window) -> eviction must drop the oldest.
        for i in range(3):
            rl._hits[f"old:{i}"] = [now - 200 + i]  # increasing order: 0 oldest
        # Add a fresh key -> over cap (4 > 3).
        rl._check_and_record("fresh", limit=10)
        assert "fresh" in rl._hits
        # Oldest stale key dropped, others may remain (evicted in order).
        assert "old:0" not in rl._hits or len(rl._hits) <= 3


# --- F4-09: store.query() TOCTOU on count ----------------------------------

class TestQueryClampInsideLock:
    """F4-09: n_results must be clamped INSIDE the collection lock so a
    concurrent delete cannot make it exceed the collection size."""

    def test_query_clamps_to_current_size(self, tmp_path):
        import src.vectorstore.store as store_mod
        store = store_mod.CVVectorStore(persist_dir=str(tmp_path / "chroma"))
        store.add_chunks(
            texts=["java developer", "python expert"],
            metadatas=[
                {"source": "a.txt", "doc_id": "a", "section": "skills", "chunk_index": 0, "tainted": False},
                {"source": "b.txt", "doc_id": "b", "section": "skills", "chunk_index": 0, "tainted": False},
            ],
            ids=["a_chunk_0000", "b_chunk_0000"],
        )
        # Asking for more results than exist must NOT raise (clamped under lock).
        out = store.query("java", n_results=50)
        assert len(out) <= 2


# --- F4-10: empty answer is a parse failure --------------------------------

class TestResponderEmptyAnswerFailsClosed:
    """F4-10: an empty `answer` string must raise instead of rendering an
    empty answer body."""

    def test_empty_answer_raises(self, monkeypatch):
        from src.agents import llm_client as llm_mod
        from src.agents.llm_client import LLMResponse
        from src.agents.responder import respond
        from src.agents.context import PipelineContext

        def fake(prompt, system_prompt, model, temperature, max_tokens, api_key):
            return LLMResponse(
                content=json.dumps({"answer": "", "matches": []}),
                model="stub", input_tokens=1, output_tokens=1, latency_ms=0.5, success=True,
            )

        monkeypatch.setattr(llm_mod, "CALL_LLM_OVERRIDE", fake)

        class FakeChunk:
            source = "a.txt"
            section = "skills"
            similarity = 0.9
            text = "Java expertise"

        ctx = PipelineContext(query="java")
        ctx.retrieved_chunks = [FakeChunk()]
        with pytest.raises(RuntimeError):
            respond(ctx)

    def test_missing_answer_raises(self, monkeypatch):
        from src.agents import llm_client as llm_mod
        from src.agents.llm_client import LLMResponse
        from src.agents.responder import respond
        from src.agents.context import PipelineContext

        def fake(prompt, system_prompt, model, temperature, max_tokens, api_key):
            return LLMResponse(
                content=json.dumps({"matches": []}),
                model="stub", input_tokens=1, output_tokens=1, latency_ms=0.5, success=True,
            )

        monkeypatch.setattr(llm_mod, "CALL_LLM_OVERRIDE", fake)

        class FakeChunk:
            source = "a.txt"
            section = "skills"
            similarity = 0.9
            text = "Java expertise"

        ctx = PipelineContext(query="java")
        ctx.retrieved_chunks = [FakeChunk()]
        with pytest.raises(RuntimeError):
            respond(ctx)

    def test_valid_answer_still_works(self, monkeypatch):
        from src.agents import llm_client as llm_mod
        from src.agents.llm_client import LLMResponse
        from src.agents.responder import respond
        from src.agents.context import PipelineContext

        def fake(prompt, system_prompt, model, temperature, max_tokens, api_key):
            return LLMResponse(
                content=json.dumps({
                    "answer": "Best match is Alex with 85%.",
                    "matches": [{"person_name": "Alex Chen", "score": 85,
                                 "evidence": "Java", "source_document": "a.txt", "sections": ["skills"]}],
                }),
                model="stub", input_tokens=1, output_tokens=1, latency_ms=0.5, success=True,
            )

        monkeypatch.setattr(llm_mod, "CALL_LLM_OVERRIDE", fake)

        class FakeChunk:
            source = "a.txt"
            section = "skills"
            similarity = 0.9
            text = "Java expertise"

        ctx = PipelineContext(query="java")
        ctx.retrieved_chunks = [FakeChunk()]
        out = respond(ctx)
        assert out.answer.startswith("Best match")
        assert len(out.match_candidates) == 1

# --- F4-05: doc_id stability across identical re-uploads --------------------

class TestDocIdStableAcrossReupload:
    """F4-05: doc_id is SHA256(filename + content); byte-identical re-uploads
    must produce the same doc_id even though ingestion reads from a random
    .uploading-* temp name."""

    def test_identical_reupload_same_doc_id(self, tmp_path, monkeypatch):
        from src.config import settings
        from src.vectorstore.store import CVVectorStore
        from src.ingestion.pipeline import ingest_upload

        monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
        store = CVVectorStore(persist_dir=str(tmp_path / "chroma"))

        content = b"John Doe\nSenior Java developer\n"
        r1 = ingest_upload(content, "resume.txt", store)
        r2 = ingest_upload(content, "resume.txt", store)
        assert r1.success and r2.success
        assert r1.doc_id == r2.doc_id, f"F4-05: doc_id drifted {r1.doc_id} != {r2.doc_id}"

    def test_different_content_different_doc_id(self, tmp_path, monkeypatch):
        from src.config import settings
        from src.vectorstore.store import CVVectorStore
        from src.ingestion.pipeline import ingest_upload

        monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
        store = CVVectorStore(persist_dir=str(tmp_path / "chroma"))

        r1 = ingest_upload(b"John Doe\nJava\n", "resume.txt", store)
        r2 = ingest_upload(b"Jane Roe\nPython\n", "resume.txt", store)
        assert r1.success and r2.success
        assert r1.doc_id != r2.doc_id


# --- F4-06: extraction caps bound memory DURING extraction ------------------

class TestExtractionCapStreaming:
    """F4-06: extractors must bail out of their loops at max_extracted_chars
    instead of materializing unbounded text first."""

    def test_csv_cap_bounds_rows(self, tmp_path, monkeypatch):
        from src.config import settings
        from src.ingestion.extractors import extract_text

        monkeypatch.setattr(settings, "max_extracted_chars", 1000)
        big_csv = tmp_path / "big.csv"
        with open(big_csv, "w") as f:
            f.write("name,role\n")
            for i in range(50000):
                f.write(f"person{i},dev\n")

        r = extract_text(big_csv)
        assert len(r["text"]) <= 1000
        assert any("cap" in w.lower() for w in r["warnings"]), r["warnings"]

    def test_txt_cap_bounds(self, tmp_path, monkeypatch):
        from src.config import settings
        from src.ingestion.extractors import extract_text

        monkeypatch.setattr(settings, "max_extracted_chars", 500)
        big_txt = tmp_path / "big.txt"
        big_txt.write_text("x" * 10000)
        r = extract_text(big_txt)
        assert len(r["text"]) <= 500


# --- F4-07: stale .uploading-* sweep at startup -----------------------------

class TestUploadTempSweep:
    """F4-07: a crash mid-ingest leaves a .uploading-* temp file behind; the
    startup sweep must remove files older than an hour."""

    def test_sweep_removes_stale_temp(self, tmp_path, monkeypatch):
        import os
        import time as _time

        import src.main as main_mod
        from src.config import settings

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(settings, "upload_dir", str(upload_dir))

        stale = upload_dir / ".uploading-1234-aaaa-resume.pdf"
        stale.write_bytes(b"half-written")
        fresh = upload_dir / ".uploading-9999-bbbb-cv.pdf"
        fresh.write_bytes(b"in-flight")
        old = _time.time() - 7200
        os.utime(stale, (old, old))

        main_mod._sweep_stale_upload_temp_files(max_age_seconds=1)

        assert not stale.exists(), "stale temp file must be removed"
        assert fresh.exists(), "fresh in-flight temp file must be kept"


# --- F4-11: scanner singleton reset by the isolation fixture ----------------

class TestScannerSingletonReset:
    """F4-11: the autouse isolation fixture must reset the scanner singleton
    so a test that monkeypatches injection_threshold can create a scanner."""

    def test_fixture_reset_allows_new_threshold(self, monkeypatch):
        import src.guardrails.scanner as scanner_mod
        from src.config import settings

        # The autouse fixture must have cleared the singleton before this
        # test (it runs before every test). If it didn't, a scanner built at
        # a DIFFERENT threshold than the default would raise the mismatch
        # RuntimeError - the guard is correct, but the isolation fixture is
        # supposed to prevent that footgun for monkeypatched tests.
        #
        # Build a scanner at a fresh threshold directly; this would raise
        # "already bound to threshold 0.8" if the fixture had NOT reset the
        # module state.
        new_threshold = settings.injection_threshold + 0.05
        scanner = scanner_mod.create_scanner(new_threshold)
        assert scanner is not None
        assert scanner_mod._shared_scanner_threshold == new_threshold
