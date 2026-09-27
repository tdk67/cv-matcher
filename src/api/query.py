"""Query API - ask questions against the CV knowledge base."""
from __future__ import annotations

import logging
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.agents.orchestrator import run_pipeline
from src.api.deps import get_api_key
from src.api.ratelimit import rate_limit
from src.config import settings
from src.guardrails.scanner import create_scanner
from src.utils.filenames import repair_mojibake_filename
from src.utils.query_log import log_query, QueryLogEntry

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    max_retries: int | None = Field(default=None, ge=1, le=5)


class MatchResult(BaseModel):
    person_name: str
    score: float
    evidence: str
    source_document: str
    sections: list[str]
    matched_chunks: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    matches: list[MatchResult]
    validation_passed: bool
    validation_feedback: str | None
    retry_count: int
    query_type: str
    latency_ms: float
    out_of_scope: bool
    rejection_reason: str | None
    injection_detected: bool = False


def _strip_diacritics(s: str) -> str:
    """Lowercase a string with accents/diacritics folded to their base letters."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()


def _normalize_filename_key(name: str) -> str:
    """Normalize a filename to a bare alnum key for matching citations to chunks."""
    decoded = repair_mojibake_filename(name)
    base = decoded.rsplit(".", 1)[0]
    return "".join(c for c in _strip_diacritics(base) if c.isalnum())


# A compiled, globally-shared PromptInjection scanner for the query path.
# Pre-warmed in main.lifespan() along with the ingestion scanner; made
# importable here (rather than importing create_scanner's singleton) so
# tests can monkeypatch it deterministically.
_query_scanner = None


def _get_query_scanner():
    global _query_scanner
    if _query_scanner is None:
        _query_scanner = create_scanner(settings.injection_threshold)
    return _query_scanner


def scan_query(query: str) -> bool:
    """Scan a user query for prompt injection before it reaches the pipeline.

    Returns True if an injection attempt was detected. A scanner error is
    treated as a security boundary failure: the query is REJECTED (the
    caller returns a 400) rather than silently proceeding - fail-closed,
    because the prompt-injection scanner is the only defense on this path.
    """
    try:
        _, is_clean, _ = _get_query_scanner().scan(query)
        return not is_clean
    except Exception as exc:  # noqa: BLE001
        logger.error("Query prompt-injection scan failed (rejecting query): %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Security scanner temporarily unavailable; please retry.",
        ) from exc


@router.post("/", response_model=QueryResponse, dependencies=[Depends(rate_limit("query"))])
def query_knowledge_base(request: QueryRequest, api_key: str | None = Depends(get_api_key)):
    """Ask a question. The agentic pipeline plans, retrieves, generates, and validates."""
    if len(request.question) > settings.max_query_length:
        raise HTTPException(
            status_code=400,
            detail=f"Question too long (max {settings.max_query_length} characters).",
        )

    # Run the PromptInjection scanner on the query path too (not only at
    # ingestion): a tainted query must not be sent to the LLM context.
    injection_detected = scan_query(request.question)
    if injection_detected:
        logger.warning("Query rejected: prompt-injection attempt detected: %r", request.question[:200])
        raise HTTPException(
            status_code=400,
            detail="Query rejected: detected a prompt-injection attempt.",
        )

    ctx = run_pipeline(
        query=request.question,
        max_retries=request.max_retries,
        api_key=api_key,
    )

    # Group retrieved chunks by normalized source filename once, instead of
    # re-normalizing every chunk's source for every citation below.
    chunks_by_source: dict[str, list[str]] = defaultdict(list)
    for c in ctx.retrieved_chunks:
        chunks_by_source[_normalize_filename_key(c.source)].append(c.text)

    # Convert match candidates to response format
    matches = []
    for candidate in ctx.match_candidates:
        source_doc = candidate.source_document
        target_key = _normalize_filename_key(source_doc)

        # Prioritize chunks containing candidate name to highlight the correct CV record
        p_words = [_strip_diacritics(w) for w in candidate.person_name.split() if len(w) > 2]
        matching_chunks, other_chunks = [], []
        for text in chunks_by_source.get(target_key, []):
            text_clean = _strip_diacritics(text)
            if p_words and any(w in text_clean for w in p_words):
                matching_chunks.append(text)
            else:
                other_chunks.append(text)

        matches.append(MatchResult(
            person_name=candidate.person_name,
            score=candidate.score,
            evidence=candidate.evidence,
            source_document=source_doc,
            sections=candidate.sections,
            matched_chunks=matching_chunks + other_chunks,
        ))

    # Log query for dashboard stats
    top_score = max((m.score for m in matches), default=0.0)
    log_query(QueryLogEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        query=request.question,
        query_type=ctx.query_type,
        match_count=len(matches),
        top_score=top_score,
        validation_passed=ctx.validation_passed,
        retry_count=ctx.retry_count,
        latency_ms=ctx.total_latency_ms,
    ))

    return QueryResponse(
        answer=ctx.answer,
        matches=matches,
        validation_passed=ctx.validation_passed,
        validation_feedback=ctx.validation_feedback if ctx.validation_feedback else None,
        retry_count=ctx.retry_count,
        query_type=ctx.query_type,
        latency_ms=ctx.total_latency_ms,
        out_of_scope=not ctx.is_in_scope,
        rejection_reason=ctx.rejection_reason if ctx.rejection_reason else None,
        injection_detected=injection_detected,
    )