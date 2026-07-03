"""Query log for tracking stats.

Simple JSON file-backed query log for dashboard statistics.
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

LOG_FILE = Path("data/query_log.json")


@dataclass
class QueryLogEntry:
    timestamp: str
    query: str
    query_type: str
    match_count: int
    top_score: float
    validation_passed: bool
    retry_count: int
    latency_ms: float


def log_query(entry: QueryLogEntry) -> None:
    """Append a query entry to the log."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if LOG_FILE.exists():
        try:
            existing = json.loads(LOG_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            existing = []

    existing.append(asdict(entry))

    # Keep last 1000 entries
    if len(existing) > 1000:
        existing = existing[-1000:]

    LOG_FILE.write_text(json.dumps(existing, indent=2, default=str))


def get_query_stats() -> dict:
    """Get aggregated query statistics."""
    if not LOG_FILE.exists():
        return {"total_queries": 0, "avg_match_score": 0.0, "avg_latency_ms": 0.0}

    try:
        entries = json.loads(LOG_FILE.read_text())
    except (json.JSONDecodeError, ValueError):
        return {"total_queries": 0, "avg_match_score": 0.0, "avg_latency_ms": 0.0}

    if not entries:
        return {"total_queries": 0, "avg_match_score": 0.0, "avg_latency_ms": 0.0}

    scores = [e.get("top_score", 0) for e in entries if e.get("top_score", 0) > 0]
    latencies = [e.get("latency_ms", 0) for e in entries if e.get("latency_ms", 0) > 0]

    return {
        "total_queries": len(entries),
        "avg_match_score": sum(scores) / len(scores) if scores else 0.0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }
