"""Evaluation API — run test queries and report results."""

import json
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from src.agents.orchestrator import run_pipeline
from src.config import settings
from src.vectorstore.store import CVVectorStore, get_vector_store

router = APIRouter()

EVAL_FILE = Path(settings.eval_file_path)
DEFAULT_QUESTIONS = settings.default_eval_questions


class EvalResultItem(BaseModel):
    question: str
    category: str
    query_type: str
    is_in_scope: bool
    has_match: bool
    top_score: float
    validation_passed: bool
    retry_count: int
    latency_ms: float
    answer_preview: str
    passed: bool  # overall pass/fail


class EvalRunResult(BaseModel):
    timestamp: str
    total_questions: int
    passed: int
    failed: int
    pass_rate: float
    avg_latency_ms: float
    failure_modes: dict[str, int]
    results: list[EvalResultItem]


class EvalResponse(BaseModel):
    runs: list[EvalRunResult]
    latest: EvalRunResult | None


@router.get("/results", response_model=EvalResponse)
async def get_eval_results():
    """Load persisted evaluation results."""
    if EVAL_FILE.exists():
        try:
            data = json.loads(EVAL_FILE.read_text())
            runs = [EvalRunResult(**r) for r in data.get("runs", [])]
            return EvalResponse(runs=runs, latest=runs[-1] if runs else None)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    return EvalResponse(runs=[], latest=None)


@router.post("/run", response_model=EvalRunResult)
async def run_evaluation():
    """Run the full evaluation suite against the knowledge base."""
    store = get_vector_store(persist_dir=settings.chroma_persist_dir)
    results = []

    for item in DEFAULT_QUESTIONS:
        start = time.time()
        ctx = run_pipeline(query=item["question"], vector_store=store)
        latency_ms = (time.time() - start) * 1000

        # Determine pass/fail
        has_match = len(ctx.retrieved_chunks) > 0
        top_score = max((c.similarity for c in ctx.retrieved_chunks), default=0.0) * 100

        if item["category"] == "adversarial":
            # Should be rejected
            passed = not ctx.is_in_scope
        elif item["category"] == "edge":
            # Should handle gracefully (may or may not find matches)
            passed = ctx.is_in_scope
        else:
            # Should find matches
            passed = ctx.is_in_scope and has_match

        results.append(EvalResultItem(
            question=item["question"],
            category=item["category"],
            query_type=ctx.query_type,
            is_in_scope=ctx.is_in_scope,
            has_match=has_match,
            top_score=top_score,
            validation_passed=ctx.validation_passed,
            retry_count=ctx.retry_count,
            latency_ms=latency_ms,
            answer_preview=ctx.answer[:200],
            passed=passed,
        ))

    passed_count = sum(1 for r in results if r.passed)
    total = len(results)

    # Count failure modes
    failure_modes = {}
    for r in results:
        if not r.passed:
            mode = "adversarial_not_rejected" if not r.is_in_scope is False else "no_match_found"
            failure_modes[mode] = failure_modes.get(mode, 0) + 1

    run_result = EvalRunResult(
        timestamp=datetime.utcnow().isoformat(),
        total_questions=total,
        passed=passed_count,
        failed=total - passed_count,
        pass_rate=round(passed_count / total * 100, 1) if total > 0 else 0.0,
        avg_latency_ms=round(sum(r.latency_ms for r in results) / total, 1) if total > 0 else 0.0,
        failure_modes=failure_modes,
        results=results,
    )

    # Persist
    EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if EVAL_FILE.exists():
        try:
            existing = json.loads(EVAL_FILE.read_text()).get("runs", [])
        except (json.JSONDecodeError, ValueError):
            existing = []

    existing.append(json.loads(run_result.model_dump_json()))
    EVAL_FILE.write_text(json.dumps({"runs": existing}, indent=2, default=str))

    return run_result
