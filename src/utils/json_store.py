"""JSON-file storage helpers with corruption-tolerant reads.

Shared by anything that persists app state as a JSON file on disk
(query log, evaluation results) so the read/repair/write pattern lives
in one place instead of being copy-pasted per caller.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Global reentrant lock to ensure thread-safe read/write operations
_file_lock = threading.RLock()


def read_json(path: Path, default: Any) -> Any:
    """Read a JSON file, returning `default` if missing or corrupt."""
    with _file_lock:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Corrupt JSON file {path}, using default: {e}")
            return default


def write_json(path: Path, data: Any) -> None:
    """Write data as JSON, creating parent directories as needed."""
    with _file_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def append_to_json_list(path: Path, entry: Any, max_entries: int | None = None) -> None:
    """Append an entry to a top-level JSON list file, optionally capping its size."""
    with _file_lock:
        existing = read_json(path, default=[])
        existing.append(entry)
        if max_entries is not None and len(existing) > max_entries:
            existing = existing[-max_entries:]
        write_json(path, existing)
