"""Query log for tracking stats.

Simple JSON file-backed query log for dashboard statistics.
Atomic write (tmp + rename) so a crash mid-write can't corrupt the file.
"""

from pathlib import Path
from dataclasses import dataclass, asdict

from src.utils.json_store import append_to_json_list, read_json

LOG_FILE = Path(".data/query_log.json")
MAX_LOG_ENTRIES = 1000


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
    append_to_json_list(LOG_FILE, asdict(entry), max_entries=MAX_LOG_ENTRIES)


def get_query_stats() -> dict:
    """Get aggregated query statistics."""
    entries = read_json(LOG_FILE, default=[])

    if not entries:
        return {"total_queries": 0, "avg_match_score": 0.0, "avg_latency_ms": 0.0}

    scores = [e.get("top_score", 0) for e in entries if e.get("top_score", 0) > 0]
    latencies = [e.get("latency_ms", 0) for e in entries if e.get("latency_ms", 0) > 0]

    return {
        "total_queries": len(entries),
        "avg_match_score": sum(scores) / len(scores) if scores else 0.0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }
