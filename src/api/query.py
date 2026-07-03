"""Query API — ask questions against the CV knowledge base."""

from fastapi import APIRouter
from pydantic import BaseModel

from src.agents.orchestrator import run_pipeline
from src.utils.query_log import log_query, QueryLogEntry
from datetime import datetime

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


@router.post("/", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest):
    """Ask a question. The agentic pipeline plans, retrieves, generates, and validates."""
    ctx = run_pipeline(
        query=request.question,
        max_retries=request.max_retries,
    )

    # Convert match candidates to response format
    matches = []
    for citation in ctx.citations:
        if isinstance(citation, dict):
            source_doc = citation.get("source_document", "")
            # Prioritize chunks containing candidate name to highlight the correct CV record
            p_name = citation.get("person_name", "").lower()
            import unicodedata
            def clean_str(s):
                return "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()
            p_words = [clean_str(w) for w in p_name.split() if len(w) > 2]
            
            def clean_filename(name):
                try:
                    decoded = name.encode('cp437').decode('utf-8')
                except Exception:
                    try:
                        decoded = name.encode('cp1252').decode('utf-8')
                    except Exception:
                        decoded = name
                base = decoded.rsplit(".", 1)[0]
                base_norm = "".join(c for c in unicodedata.normalize('NFD', base) if unicodedata.category(c) != 'Mn')
                return "".join(c for c in base_norm.lower() if c.isalnum())

            target_clean = clean_filename(source_doc)
            matching_chunks = []
            other_chunks = []
            for c in ctx.retrieved_chunks:
                if clean_filename(c.source) == target_clean:
                    c_clean = clean_str(c.text)
                    if p_words and any(w in c_clean for w in p_words):
                        matching_chunks.append(c.text)
                    else:
                        other_chunks.append(c.text)
            doc_chunks = matching_chunks + other_chunks
            
            matches.append(MatchResult(
                person_name=citation.get("person_name", "Unknown"),
                score=citation.get("score", 0),
                evidence=citation.get("evidence", ""),
                source_document=source_doc,
                sections=citation.get("sections", []),
                matched_chunks=doc_chunks,
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
