"""Document-level guardrail scanning using LLM Guard.

Scans each chunk for prompt injection during ingestion.
Documents with any flagged chunks are marked as tainted.
Chunks whose scan errored are recorded as `scan_error` - they are NOT
silently marked clean (that would be fail-open for a safety control).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from llm_guard.input_scanners import PromptInjection

from src.config import settings

logger = logging.getLogger(__name__)


class ScanStatus(str, Enum):
    """Per-chunk scan outcome. Clean implies the scan ran and passed."""

    CLEAN = "clean"
    TAINTED = "tainted"  # injection detected
    ERROR = "error"  # scanner itself failed


@dataclass
class ScanResult:
    """Result of scanning a single chunk.

    `status` is the authoritative outcome: ``clean``, ``tainted`` or
    ``error``. The legacy ``is_clean`` field is kept for backward
    compatibility with existing callers/tests and is derived from the new
    status: it is only True for a real, successful clean scan (an errored
    scan is NOT clean).
    """

    status: ScanStatus
    risk_score: float
    injection_detected: bool
    error: str | None = None

    @property
    def is_clean(self) -> bool:
        """True only when the scan actually ran and found nothing."""
        return self.status == ScanStatus.CLEAN


@dataclass
class DocumentScanSummary:
    """Summary of scanning all chunks in a document."""

    total_chunks: int
    clean_chunks: int
    tainted_chunks: int
    errored_chunks: int
    is_tainted: bool
    max_risk_score: float
    chunk_results: list[ScanResult] = field(default_factory=list)

    @property
    def scan_failed(self) -> bool:
        """True if any chunk's scan errored and no taint was detected."""
        return self.errored_chunks > 0 and self.tainted_chunks == 0


_shared_scanner = None


def create_scanner(threshold: float = 0.8) -> PromptInjection:
    """Create or return the shared singleton PromptInjection scanner."""
    global _shared_scanner
    if _shared_scanner is None:
        _shared_scanner = PromptInjection(threshold=threshold)
    return _shared_scanner


def scan_chunks(
    chunks: list[str],
    scanner=None,
    threshold: float = 0.8,
) -> DocumentScanSummary:
    """Scan a list of chunks for prompt injection.

    Args:
        chunks: List of text chunks to scan
        scanner: Pre-initialized scanner (optional, creates one if None)
        threshold: Injection detection threshold (default 0.8)

    Returns:
        DocumentScanSummary with per-chunk and aggregate results.
        A chunk whose scan raised is recorded with status=error - it is
        never silently marked clean.
    """
    if scanner is None:
        scanner = create_scanner(threshold)

    results = []
    clean_count = 0
    tainted_count = 0
    errored_count = 0
    max_risk = 0.0

    for chunk_text in chunks:
        try:
            _, is_clean, risk_score = scanner.scan(chunk_text)
            injection_detected = not is_clean
            if injection_detected:
                tainted_count += 1
            else:
                clean_count += 1
            if risk_score > max_risk:
                max_risk = risk_score
            results.append(ScanResult(
                status=ScanStatus.TAINTED if injection_detected else ScanStatus.CLEAN,
                risk_score=risk_score,
                injection_detected=injection_detected,
            ))
        except Exception as e:  # noqa: BLE001 - scanner is third-party; any failure is reported as error
            errored_count += 1
            if settings.scan_fail_policy == "log":
                logger.warning("Prompt injection scan failed for a chunk (recorded as error): %s", e)
            else:
                logger.error("Prompt injection scan failed for a chunk (policy=%s): %s", settings.scan_fail_policy, e)
            results.append(ScanResult(
                status=ScanStatus.ERROR,
                risk_score=0.0,
                injection_detected=False,
                error=str(e),
            ))

    return DocumentScanSummary(
        total_chunks=len(chunks),
        clean_chunks=clean_count,
        tainted_chunks=tainted_count,
        errored_chunks=errored_count,
        is_tainted=tainted_count > 0,
        max_risk_score=max_risk,
        chunk_results=results,
    )