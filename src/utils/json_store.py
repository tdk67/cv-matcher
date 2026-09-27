"""JSON-file storage helpers with corruption-tolerant reads and atomic writes.

Shared by anything that persists app state as a JSON file on disk
(query log, evaluation results) so the read/repair/write pattern lives
in one place instead of being copy-pasted per caller.

Writes go through a tmp file + os.replace() so a crash mid-write can't
leave a truncated/corrupt target file.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import tempfile
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
    """Write data as JSON (atomically: tmp + rename), creating parent dirs as needed."""
    with _file_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, indent=2, default=str)
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
        except Exception:
            # Clean up the temp file on failure; never leave stray tmp files.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def append_to_json_list(path: Path, entry: Any, max_entries: int | None = None) -> None:
    """Append an entry to a top-level JSON list file, optionally capping its size."""
    with _file_lock:
        existing = read_json(path, default=[])
        existing.append(entry)
        if max_entries is not None and len(existing) > max_entries:
            existing = existing[-max_entries:]
        write_json(path, existing)
