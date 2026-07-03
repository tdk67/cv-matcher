"""Query API — ask questions against the CV knowledge base."""

import unicodedata
from collections import defaultdict
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from src.agents.orchestrator import run_pipeline
from src.utils.filenames import repair_mojibake_filename
from src.utils.query_log import log_query, QueryLogEntry

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    max_retries: int | None = None
    use_llm_planner: bool = True
    use_llm_validator: bool = True


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


def _strip_diacritics(s: str) -> str:
    """Lowercase a string with accents/diacritics folded to their base letters."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()


def _normalize_filename_key(name: str) -> str:
    """Normalize a filename to a bare alnum key for matching citations to chunks."""
    decoded = repair_mojibake_filename(name)
    base = decoded.rsplit(".", 1)[0]
    return "".join(c for c in _strip_diacritics(base) if c.isalnum())


@router.post("/", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest):
    """Ask a question. The agentic pipeline plans, retrieves, generates, and validates."""
    ctx = run_pipeline(
        query=request.question,
        max_retries=request.max_retries,
    )

    # Group retrieved chunks by normalized source filename once, instead of
    # re-normalizing every chunk's source for every citation below.
    chunks_by_source: dict[str, list[str]] = defaultdict(list)
    for c in ctx.retrieved_chunks:
        chunks_by_source[_normalize_filename_key(c.source)].append(c.text)

    # Convert match candidates to response format
    matches = []
    for citation in ctx.citations:
        if not isinstance(citation, dict):
            continue

        source_doc = citation.get("source_document", "")
        target_key = _normalize_filename_key(source_doc)

        # Prioritize chunks containing candidate name to highlight the correct CV record
        p_words = [_strip_diacritics(w) for w in citation.get("person_name", "").split() if len(w) > 2]
        matching_chunks, other_chunks = [], []
        for text in chunks_by_source.get(target_key, []):
            text_clean = _strip_diacritics(text)
            if p_words and any(w in text_clean for w in p_words):
                matching_chunks.append(text)
            else:
                other_chunks.append(text)

        matches.append(MatchResult(
            person_name=citation.get("person_name", "Unknown"),
            score=citation.get("score", 0),
            evidence=citation.get("evidence", ""),
            source_document=source_doc,
            sections=citation.get("sections", []),
            matched_chunks=matching_chunks + other_chunks,
        ))

    # Log query for dashboard stats
    top_score = max((m.score for m in matches), default=0.0)
    log_query(QueryLogEntry(
        timestamp=datetime.utcnow().isoformat(),
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
    )
