# FIX LOOP — cv-matcher (Review 01)

Loop owner: **pi** (this environment) — owns git, merges, runs verification.
Reviewer agent: **google-gemini-3.8-flash** (via AdaL) — adversarial QA of each fix round.
Fixer agent: **AdaL (`--agent-mode engineer`, different model than reviewer)** — implements.

> **Safety rules (from /root/notes/adal-coding-agent.md):**
> - Fixer works on the **isolated worktree/branch** only — NEVER on `main`, NEVER pushes.
> - Snapshot (md5 manifest) before every round; revert = `git checkout -- <files>`.
> - Reviewer/fixer: keep agents OUT of repo `.env`; run only stubbed tests (no live LLM).

## Finding order (dependency-first)

### Round A — P0 (foundations + breakage)
- **F-02** LLM client: transient-error retry with exponential backoff + jitter (Retry-After aware; retry 408/429/5xx/transport; NEVER 4xx auth). Config: `llm_max_retries`, `llm_retry_backoff_base`.
- **F-01** Restore OR remove `src.data` synthetic generator + `sample_data` handling, so README steps, `/api/synthetic/generate`, and tests work on a fresh clone. Wire the in-process path (no subprocess) if restored.
- **F-03** Security posture: optional `API_AUTH_TOKEN` (Bearer) enabling when server-side key set; rate limiting on `/api/query/` and `/api/evaluation/start`; CORS restricted; bind localhost by default; README warning.

### Round B — P1 (honesty of guardrails) — DONE (2026-09-27, review-02 round)
- **F-04** Scanner: per-chunk scan status `clean|tainted|error`; errored scans must NOT be recorded as clean; expose in metadata; config `scan_fail_policy` (default: log+flag).
- **F-05** Run PromptInjection scanner on the QUERY path (API/orchestrator) — currently ingestion-only; fix README claim.
- **F-06** `tainted_policy: include|exclude|flag` — tainted chunks flagged/filtered at retrieval.
- **F-11** `CVVectorStore.count()`; stop reaching into `_collection`; mutation lock in store; dedup warning on re-ingest of same `doc_id`.

### Round C — P1/P2 (clean code + eval + tests + docs) — DONE (2026-09-27, review-02 round)
- **F-08** Delete dead code (async `call_llm`, `score_bar`, unused config keys) OR wire it; honor (or remove) `use_llm_planner`/`use_llm_validator`; cap+validate request fields.
- **F-09** Bounded upload read BEFORE size check; `max_extracted_chars` cap; `question` max_length.
- **F-07** Prompt externalization: `min_match_score` enforced from config (single source of truth) — DONE. Agent prompts (planner/responder/validator) remain module constants; externalizing them to `config/prompts/` was NOT implemented — FIX_LOOP claim downgraded to reflect this.
- **F-10** Eval scoring includes `validation_passed` + guard empty-KB runs — DONE. The "optional Judge-agent grade" was NOT implemented: eval remains self-graded by the same-family validator. Claim amended.
- **F-14** Inject LLM callable into agents (test seam); deterministic stubbed agent tests; tighten loose assertions; add coverage config.
- **F-13** README: fix `run.sh`/`src.data.cli` references, add "Limitations & challenges" (required by PDF), troubleshooting, screenshots, TOC.

### Round D — P2 polish — DONE (2026-09-27, review-02 round)
- **F-12** config.json load validation (fail fast + clear error); **F-15** typed ExtractionResult; **F-16** synthetic count caps + enum.

### Extra review-02 hardening (F2-01...F2-17) — DONE (2026-09-27)
- F2-01: README input-scan claim fixed by wiring `scan_query` into the query path (rejects with 400).
- F2-03: `rate_limit("key_validate")` wired on `/api/key/validate` (was dead config + dead import).
- F2-05: rate limiter keyed on hashed X-Session-ID header so Docker's one-IP topology doesn't collapse all users into one bucket; `--proxy-headers` added to docker-entrypoint.
- F2-06: Validator fails CLOSED — missing/non-bool `passed` raises instead of defaulting to True.
- F2-07: scanner exception recorded as status=error; `scan_fail_policy=error` (default) rejects the upload, `log` stores scan_status=error.
- F2-08: retrieval floor now uses `min_match_score` from config; no more "take top 3 anyway" fallback — honest no-match.
- F2-09: conftest auto-skips `@pytest.mark.live` tests without a key; fresh-clone suite green (73 passed, 14 skipped).
- F2-10: eval `passed` includes `validation_passed`; orchestrator early exits no longer set `validation_passed=True`; empty-KB eval fails loud; `max_questions` param on /start.
- F2-11: README file-structure/library/limitations falsehoods purged; new security config documented.
- F2-12: query page shows `detail` errors instead of "Validation warning + No answer".
- F2-13: 500 handler returns generic message (+request id), leaks no exception text.
- F2-14: Retry-After capped at `llm_retry_after_cap_seconds` (default 60).
- F2-15: `secrets.compare_digest`; startup warning when server-side key set without auth.
- F2-16: async `call_llm` deleted; `score_bar` deleted; unused config keys removed; pyyaml dropped.
- F2-17: synthetic endpoint cleans stale files + counts reported by generator; email uniqueness fix; `_extract_excel` reads sheet names before close.
- P3: json_store atomic writes (tmp+rename); `datetime.utcnow()` → timezone-aware; config fail-fast.

### Review-04 fixes (F4-01…F4-11) — DONE (2026-09-28)
- F4-01: Planner fails CLOSED — missing/non-bool `is_in_scope` or unknown `query_type` raises (matches Validator).
- F4-02: honest README rate-limit claims per topology (Docker shared bucket, reverse-proxy per-IP, Cloud shared egress); `--forwarded-allow-ips=127.0.0.1` so XFF is trusted only from loopback; Streamlit Cloud deployment documented (`API_BASE_URL` secret). Public backend live at https://cv-matcher.taskmind-ai.com.
- F4-03: `rate_limits.upload` (default 10/min) wired on `POST /api/documents/upload` + regression test.
- F4-04: retry sleep clamped to the remaining pipeline deadline (both retry paths).
- F4-05: `doc_id` hashes the FINAL filename (not the temp upload name) — identical re-uploads keep the same doc_id, duplicate-ingest warning works.
- F4-06: extraction caps applied DURING extraction (per-format loops bail at `max_extracted_chars`), not post-hoc.
- F4-07: startup sweep removes stale `.uploading-*` temp files older than 1h.
- F4-08: rate-limit key table evicts expired keys + LRU overflow instead of a global `_hits.clear()` (no cross-user reset).
- F4-09: `store.query()` clamps `n_results` inside the collection lock (TOCTOU on count).
- F4-10: Responder treats empty/missing `answer` as a parse failure (fail-closed, no empty answer body under a warning badge).
- F4-11: conftest isolation fixture also resets the scanner singleton.

## Verification gate per round (owner=pi)
1. `git diff --stat` reviewed; no secrets, no `.env`, no stray files.
2. `python -m compileall src app.py` and `pytest tests/ -x -q` pass (live-LLM tests stubbed).
3. Regression greps: no `_collection.` outside `store.py`; no `call_llm(` (async) callers; config keys wired or removed.
4. **Reviewer agent (google-gemini-3.8-flash)** audits the diff and returns `PASS` or reasons to redo (max 3 rounds/finding).
5. Each round logs `reviews/review-01-fix<N>.md` (fixer summary + diff stats + reviewer verdict).

Rounds after Round A may be batched. Do NOT merge to main until reviewer PASS on the full set.