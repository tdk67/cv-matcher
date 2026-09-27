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
    out_of_scope: bool = False


def log_query(entry: QueryLogEntry) -> None:
    """Append a query entry to the log."""
    append_to_json_list(LOG_FILE, asdict(entry), max_entries=MAX_LOG_ENTRIES)


def get_query_stats() -> dict:
    """Get aggregated query statistics.

    F3-16: out-of-scope rejections log `validation_passed=False` by design
    (they are correct rejections, not failed validations). The dashboard
    reports avg_match_score ONLY over answered queries and exposes a
    separate out_of_scope_rejections counter so the aggregates cannot be
    misread as "failed validations".
    """
    entries = read_json(LOG_FILE, default=[])

    if not entries:
        return {
            "total_queries": 0,
            "avg_match_score": 0.0,
            "avg_latency_ms": 0.0,
            "out_of_scope_rejections": 0,
        }

    # avg_match_score over queries that actually produced matches; correct
    # out-of-scope rejections must NOT drag it down as if they failed.
    scores = [e.get("top_score", 0) for e in entries if e.get("top_score", 0) > 0]
    latencies = [e.get("latency_ms", 0) for e in entries if e.get("latency_ms", 0) > 0]

    return {
        "total_queries": len(entries),
        "avg_match_score": sum(scores) / len(scores) if scores else 0.0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "out_of_scope_rejections": sum(
            1 for e in entries if e.get("out_of_scope", False)
        ),
    }
