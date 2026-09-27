"""Retriever Agent - searches the knowledge base based on Planner's strategy.

Responsibilities:
- Build effective search queries from extracted requirements
- Execute similarity search (and optionally keyword search)
- Rank and filter results
- Return top chunks with scores
"""

from src.agents.context import PipelineContext, RetrievedChunk
from src.config import settings
from src.vectorstore.store import CVVectorStore, get_vector_store


def retrieve(ctx: PipelineContext, vector_store: CVVectorStore | None = None) -> PipelineContext:
    """Run the Retriever agent.

    Takes the Planner's output and searches ChromaDB for relevant chunks.
    """
    if vector_store is None:
        vector_store = get_vector_store(persist_dir=settings.chroma_persist_dir)

    if vector_store.is_empty():
        ctx.retrieved_chunks = []
        return ctx

    # Build search query from planner output
    search_query = _build_search_query(ctx)

    # Execute similarity search - get more results than needed for ranking
    n_results = min(15, vector_store.count())
    raw_results = vector_store.query(search_query, n_results=n_results)

    # Convert to RetrievedChunk objects
    chunks = []
    for r in raw_results:
        meta = r.get("metadata", {})
        chunks.append(RetrievedChunk(
            text=r["text"],
            source=meta.get("source", "unknown"),
            section=meta.get("section", "unknown"),
            similarity=r.get("similarity", 0.0),
            metadata=meta,
        ))

    # Rank: group by source document, pick best chunks per person
    ranked = _rank_chunks(chunks, ctx)

    ctx.retrieved_chunks = ranked
    return ctx


def _build_search_query(ctx: PipelineContext) -> str:
    """Build an effective search query from the planner's extracted requirements."""
    # For keyword queries, use the original query
    if ctx.query_type == "keyword":
        return ctx.query

    # For similarity/complex queries, combine query with extracted skills
    parts = [ctx.query]
    if ctx.extracted_skills:
        parts.append("Skills: " + ", ".join(ctx.extracted_skills))

    return " ".join(parts)


def _rank_chunks(
    chunks: list[RetrievedChunk],
    ctx: PipelineContext,
) -> list[RetrievedChunk]:
    """Rank chunks by relevance.

    Strategy:
    1. Drop tainted (prompt-injection flagged) and scan-errored chunks when
       `tainted_policy` is "exclude" (default) - they never reach the
       Responder's context.
    2. Filter out low-similarity chunks below `min_match_score` from config
       (single source of truth; the old hardcoded 0.15 floor is gone).
    3. Group by source document
    4. Return top chunks, ensuring diversity across sections
    """
    if not chunks:
        return []

    # Guardrails: by default, tainted/errored chunks never reach the LLM context.
    if settings.tainted_policy == "exclude":
        chunks = [
            c for c in chunks
            if c.metadata.get("scan_status") in (None, "clean") and not c.metadata.get("tainted", False)
        ]

    if not chunks:
        return []

    # Enforce the configured minimum similarity (single source of truth).
    min_score = settings.min_match_score
    filtered = [c for c in chunks if c.similarity >= min_score]

    # Sort by similarity descending first, so the fallback below picks the
    # genuinely best chunks rather than insertion order.
    chunks.sort(key=lambda c: c.similarity, reverse=True)
    if not filtered:
        # Honest "no relevant candidate" signal: return nothing instead of
        # silently serving low-relevance CVs. The orchestrator then reports
        # that no matching candidates were found.
        return []

    # Deduplicate: if two chunks from same section look identical, keep the best
    seen = set()
    deduped = []
    for chunk in filtered:
        key = f"{chunk.source}:{chunk.section}:{chunk.text[:100]}"
        if key not in seen:
            seen.add(key)
            deduped.append(chunk)

    return deduped[:10]  # Return top 10 chunks
