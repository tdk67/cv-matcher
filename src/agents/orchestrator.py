"""Pipeline Orchestrator - runs the full agentic pipeline with retry loop.

Flow: Planner -> Retriever -> Responder -> Validator -> (retry if failed)
"""

import time

from src.agents.context import PipelineContext
from src.agents.planner import plan
from src.agents.retriever import retrieve
from src.agents.responder import respond
from src.agents.validator import validate
from src.config import settings
from src.vectorstore.store import CVVectorStore, get_vector_store


def run_pipeline(
    query: str,
    vector_store: CVVectorStore | None = None,
    max_retries: int | None = None,
    api_key: str | None = None,
    deadline_seconds: float | None = None,
) -> PipelineContext:
    """Execute the full agentic RAG pipeline.

    Steps:
    1. Planner classifies query and extracts requirements
    2. Retriever searches ChromaDB for relevant chunks
    3. Responder generates answer with citations
    4. Validator checks quality; if failed, Responder retries with feedback

    Args:
        query: User's question
        vector_store: Pre-initialized store (optional, creates one if None)
        max_retries: Override max retry attempts
        api_key: Per-request OpenRouter key (falls back to settings.openrouter_api_key)
        deadline_seconds: Optional wall-clock budget for the whole pipeline.
            Defaults to settings.pipeline_deadline_seconds when None. The
            deadline is enforced inside call_llm_sync (between LLM attempts)
            and checked here between pipeline steps, so a query cannot run
            for minutes while the Streamlit client (120s timeout) has long
            abandoned it.

    Returns:
        PipelineContext with all results
    """
    from src.agents.llm_client import set_llm_deadline

    if max_retries is None:
        max_retries = settings.max_retry_attempts

    budget = settings.pipeline_deadline_seconds if deadline_seconds is None else deadline_seconds

    ctx = PipelineContext(query=query, max_retries=max_retries)
    start_time = time.time()
    deadline = time.monotonic() + budget if budget and budget > 0 else None
    set_llm_deadline(deadline)

    # Step 1: Plan
    ctx = plan(ctx, api_key=api_key)

    # Early exit: out of scope. This is a *correct rejection*, not an
    # answer that passed quality validation - recording validation_passed
    # here would conflate "rejection was correct" with "answer passed
    # validation" in the evaluation ledger.
    if not ctx.is_in_scope:
        ctx.answer = (
            f"I'm sorry, but I can only answer questions about the uploaded CVs "
            f"and team expertise. {ctx.rejection_reason}"
        )
        ctx.validation_passed = False
        ctx.total_latency_ms = (time.time() - start_time) * 1000
        return ctx

    # Early exit: empty knowledge base
    if vector_store is None:
        vector_store = get_vector_store(persist_dir=settings.chroma_persist_dir)
    if vector_store.is_empty():
        ctx.answer = (
            "The knowledge base is empty. Please upload CV documents first, "
            "then try your query again."
        )
        ctx.validation_passed = False
        ctx.total_latency_ms = (time.time() - start_time) * 1000
        return ctx

    # Wall-clock budget guard (checked between every step so a pipeline
    # cannot overshoot the budget by many LLM calls).
    def _abort_if_timed_out():
        if deadline is not None and time.monotonic() > deadline:
            ctx.answer = (
                "The request took too long and was stopped before a full answer "
                "could be generated. Please try a more specific question or try "
                "again later."
            )
            ctx.validation_passed = False
            ctx.timed_out = True
            ctx.deadline_reason = "pipeline deadline exceeded"
            ctx.total_latency_ms = (time.time() - start_time) * 1000
            return True
        return False

    # Step 2: Retrieve
    ctx = retrieve(ctx, vector_store)

    # Early exit: no results
    if not ctx.retrieved_chunks:
        ctx.answer = (
            "No matching candidates found for your query. "
            "The knowledge base may not contain relevant expertise, "
            "or your search criteria may be too specific. "
            "Try broadening your search."
        )
        ctx.validation_passed = False
        ctx.total_latency_ms = (time.time() - start_time) * 1000
        return ctx

    # Steps 3-4: Respond + Validate with retry loop
    for attempt in range(max_retries):
        if _abort_if_timed_out():
            return ctx
        ctx.retry_count = attempt

        # Respond
        ctx = respond(ctx, api_key=api_key)

        if _abort_if_timed_out():
            return ctx

        # Validate
        ctx = validate(ctx, api_key=api_key)

        if ctx.validation_passed:
            break

        # Retry feedback is already in ctx.validation_feedback
        # Next iteration of respond() will use it

    ctx.total_latency_ms = (time.time() - start_time) * 1000
    return ctx
