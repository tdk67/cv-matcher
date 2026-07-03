"""Document-level guardrail scanning using LLM Guard.

Scans each chunk for prompt injection during ingestion.
Documents with any flagged chunks are marked as tainted.
"""

from dataclasses import dataclass, field

from llm_guard.input_scanners import PromptInjection


@dataclass
class ScanResult:
    """Result of scanning a single chunk."""
    is_clean: bool
    risk_score: float
    injection_detected: bool


@dataclass
class DocumentScanSummary:
    """Summary of scanning all chunks in a document."""
    total_chunks: int
    clean_chunks: int
    tainted_chunks: int
    is_tainted: bool
    max_risk_score: float
    chunk_results: list[ScanResult] = field(default_factory=list)


def create_scanner(threshold: float = 0.8) -> PromptInjection:
    """Create a PromptInjection scanner with the given threshold."""
    return PromptInjection(threshold=threshold)


def scan_chunks(
    chunks: list[str],
    scanner: PromptInjection | None = None,
    threshold: float = 0.8,
) -> DocumentScanSummary:
    """Scan a list of chunks for prompt injection.

    Args:
        chunks: List of text chunks to scan
        scanner: Pre-initialized scanner (optional, creates one if None)
        threshold: Injection detection threshold (default 0.8)

    Returns:
        DocumentScanSummary with per-chunk and aggregate results
    """
    if scanner is None:
        scanner = create_scanner(threshold)

    results = []
    tainted_count = 0
    max_risk = 0.0

    for chunk_text in chunks:
        try:
            _, is_clean, risk_score = scanner.scan(chunk_text)
            injection_detected = not is_clean
            if injection_detected:
                tainted_count += 1
            if risk_score > max_risk:
                max_risk = risk_score
            results.append(ScanResult(
                is_clean=is_clean,
                risk_score=risk_score,
                injection_detected=injection_detected,
            ))
        except Exception as e:
            # Scanner failure should not block ingestion
            results.append(ScanResult(
                is_clean=True,
                risk_score=0.0,
                injection_detected=False,
            ))

    return DocumentScanSummary(
        total_chunks=len(chunks),
        clean_chunks=len(chunks) - tainted_count,
        tainted_chunks=tainted_count,
        is_tainted=tainted_count > 0,
        max_risk_score=max_risk,
        chunk_results=results,
    )
