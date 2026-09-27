"""Evaluation API - run test queries in the background and report progress.

The evaluation suite runs every default question through the full agentic
pipeline sequentially (each up to 3 LLM calls, with retries), which can take
minutes. POST /start kicks the run off in a background thread and returns
immediately; GET /progress reports how far it's gotten so the frontend can
show a live progress bar instead of blocking on one long HTTP request.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.agents.orchestrator import run_pipeline
from src.api.deps import get_api_key
from src.api.ratelimit import rate_limit
from src.config import settings
from src.utils.json_store import append_to_json_list, read_json
from src.vectorstore.store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()

EVAL_FILE = Path(settings.eval_file_path)
DEFAULT_QUESTIONS = settings.default_eval_questions
MAX_STORED_RUNS = 200


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


class EvalProgress(BaseModel):
    status: str  # "idle" | "running" | "done" | "error"
    total: int
    completed: int
    current_question: str | None = None
    error: str | None = None
    result: EvalRunResult | None = None


class _EvalState:
    """In-process, thread-safe state for the currently running (or last) evaluation run."""

    def __init__(self):
        self._lock = threading.Lock()
        self.status = "idle"
        self.total = 0
        self.completed = 0
        self.current_question: str | None = None
        self.error: str | None = None
        self.result: EvalRunResult | None = None

    def try_start(self, total: int) -> bool:
        """Transition to 'running' unless a run is already in progress. Returns whether it started."""
        with self._lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.total = total
            self.completed = 0
            self.current_question = None
            self.error = None
            self.result = None
            return True

    def set_current_question(self, question: str) -> None:
        with self._lock:
            self.current_question = question

    def mark_completed(self) -> None:
        with self._lock:
            self.completed += 1

    def mark_done(self, result: EvalRunResult) -> None:
        with self._lock:
            self.status = "done"
            self.current_question = None
            self.result = result

    def mark_error(self, error: str) -> None:
        with self._lock:
            self.status = "error"
            self.error = error

    def snapshot(self) -> EvalProgress:
        with self._lock:
            return EvalProgress(
                status=self.status,
                total=self.total,
                completed=self.completed,
                current_question=self.current_question,
                error=self.error,
                result=self.result,
            )


_eval_state = _EvalState()


def _score_result(item: dict, ctx) -> EvalResultItem:
    """Build an EvalResultItem from a completed pipeline run."""
    has_match = len(ctx.retrieved_chunks) > 0
    top_score = max((c.similarity for c in ctx.retrieved_chunks), default=0.0) * 100

    if item["category"] == "adversarial":
        passed = not ctx.is_in_scope  # should have been rejected
    elif item["category"] == "edge":
        passed = ctx.is_in_scope  # should be handled gracefully either way
    else:
        passed = ctx.is_in_scope and has_match  # should find matches

    return EvalResultItem(
        question=item["question"],
        category=item["category"],
        query_type=ctx.query_type,
        is_in_scope=ctx.is_in_scope,
        has_match=has_match,
        top_score=top_score,
        validation_passed=ctx.validation_passed,
        retry_count=ctx.retry_count,
        latency_ms=ctx.total_latency_ms,
        answer_preview=ctx.answer[:200],
        passed=passed,
    )


def _run_evaluation_job(api_key: str | None):
    """Background-thread worker. Updates _eval_state as each question completes."""
    store = get_vector_store(persist_dir=settings.chroma_persist_dir)
    results: list[EvalResultItem] = []

    try:
        for item in DEFAULT_QUESTIONS:
            _eval_state.set_current_question(item["question"])

            start = time.time()
            try:
                ctx = run_pipeline(query=item["question"], vector_store=store, api_key=api_key)
                result_item = _score_result(item, ctx)
            except Exception as e:
                # One bad LLM response (truncated JSON, rate limit, timeout)
                # shouldn't abort the whole batch - record it and move on.
                logger.exception(f"Evaluation question failed: {item['question']!r}")
                result_item = EvalResultItem(
                    question=item["question"],
                    category=item["category"],
                    query_type="error",
                    is_in_scope=False,
                    has_match=False,
                    top_score=0.0,
                    validation_passed=False,
                    retry_count=0,
                    latency_ms=(time.time() - start) * 1000,
                    answer_preview=f"Pipeline error: {e}"[:200],
                    passed=False,
                )

            results.append(result_item)
            _eval_state.mark_completed()

        passed_count = sum(1 for r in results if r.passed)
        total_q = len(results)

        # Count failure modes. Only questions EXPECTED to be rejected
        # (category == "adversarial") but weren't count as "adversarial_not_rejected";
        # everything else that failed is either a pipeline error or a plain no-match.
        failure_modes: dict[str, int] = {}
        for r in results:
            if r.passed:
                continue
            if r.query_type == "error":
                mode = "pipeline_error"
            elif r.category == "adversarial":
                mode = "adversarial_not_rejected"
            else:
                mode = "no_match_found"
            failure_modes[mode] = failure_modes.get(mode, 0) + 1

        run_result = EvalRunResult(
            timestamp=datetime.utcnow().isoformat(),
            total_questions=total_q,
            passed=passed_count,
            failed=total_q - passed_count,
            pass_rate=round(passed_count / total_q * 100, 1) if total_q > 0 else 0.0,
            avg_latency_ms=round(sum(r.latency_ms for r in results) / total_q, 1) if total_q > 0 else 0.0,
            failure_modes=failure_modes,
            results=results,
        )

        append_to_json_list(EVAL_FILE, run_result.model_dump(mode="json"), max_entries=MAX_STORED_RUNS)
        _eval_state.mark_done(run_result)

    except Exception as e:
        logger.exception("Evaluation job crashed")
        _eval_state.mark_error(str(e))


@router.get("/results", response_model=EvalResponse)
async def get_eval_results():
    """Load persisted evaluation results."""
    raw_runs = read_json(EVAL_FILE, default=[])
    try:
        runs = [EvalRunResult(**r) for r in raw_runs]
    except (ValueError, KeyError, TypeError) as e:
        logger.warning(f"Corrupt evaluation results in {EVAL_FILE}, ignoring: {e}")
        runs = []
    return EvalResponse(runs=runs, latest=runs[-1] if runs else None)


@router.post("/start", response_model=EvalProgress, dependencies=[Depends(rate_limit("evaluation_start"))])
async def start_evaluation(api_key: str | None = Depends(get_api_key)):
    """Kick off the evaluation suite in the background. Poll GET /progress for status."""
    if _eval_state.try_start(total=len(DEFAULT_QUESTIONS)):
        threading.Thread(target=_run_evaluation_job, args=(api_key,), daemon=True).start()
    return _eval_state.snapshot()


@router.get("/progress", response_model=EvalProgress)
async def get_evaluation_progress():
    """Report progress of the currently running (or most recently finished) evaluation."""
    return _eval_state.snapshot()
