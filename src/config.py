"""Application configuration via environment variables.

Configuration is merged from three sources, in increasing priority:
1. Hardcoded defaults (below)
2. `config.json` in the repository root
3. Environment variables / `.env`

A corrupt or unreadable `config.json` fails fast with a clear error instead
of silently booting with defaults (previous behavior hid misconfiguration
and made debugging deployments confusing).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Load defaults from config.json if it exists
_CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"


def _load_defaults(config_file: Path) -> dict:
    """Read + validate config.json; fail fast (SystemExit) with a clear error.

    A corrupt or unreadable config file must never silently boot with
    hardcoded defaults - that hides misconfiguration and makes deployment
    debugging confusing.
    """
    if not config_file.exists():
        return {}
    try:
        raw = config_file.read_text(encoding="utf-8")
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError("config.json top-level value must be a JSON object")
        return loaded
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - fail fast with a clear message
        raise SystemExit(
            f"FATAL: config.json is corrupt or unreadable: {exc}.\n"
            "Fix config.json (or delete it to use built-in defaults) and restart."
        ) from exc


defaults = _load_defaults(_CONFIG_FILE)


def _int_or_default(key: str, default: int) -> int:
    value = defaults.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"FATAL: config.json key {key!r} must be an integer, got {value!r}") from exc


def _float_or_default(key: str, default: float) -> float:
    value = defaults.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"FATAL: config.json key {key!r} must be a number, got {value!r}") from exc


def _str_or_default(key: str, default: str) -> str:
    value = defaults.get(key, default)
    if not isinstance(value, str):
        raise SystemExit(f"FATAL: config.json key {key!r} must be a string, got {value!r}")
    return value


def _list_or_default(key: str, default: list) -> list:
    value = defaults.get(key, default)
    if not isinstance(value, list):
        raise SystemExit(f"FATAL: config.json key {key!r} must be a list, got {value!r}")
    return value


def _dict_or_default(key: str, default: dict) -> dict:
    value = defaults.get(key, default)
    if not isinstance(value, dict):
        raise SystemExit(f"FATAL: config.json key {key!r} must be an object, got {value!r}")
    return value


class Settings(BaseSettings):
    """All configuration is loaded from environment variables / .env file.

    No model names or API keys are hardcoded anywhere in the source.
    """

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenRouter
    openrouter_api_key: str = _str_or_default("openrouter_api_key", "")
    rag_model: str = _str_or_default("rag_model", "google/gemini-2.5-flash")
    validation_model: str = _str_or_default("validation_model", "openai/gpt-4.1-nano")

    # Application
    chroma_persist_dir: str = _str_or_default("chroma_persist_dir", "./.data/chromadb")
    upload_dir: str = _str_or_default("upload_dir", "./.data/uploads")
    max_upload_size_mb: int = _int_or_default("max_upload_size_mb", 50)
    max_extracted_chars: int = _int_or_default("max_extracted_chars", 200_000)
    max_query_length: int = _int_or_default("max_query_length", 500)
    log_level: str = _str_or_default("log_level", "INFO")

    # Guardrails
    injection_threshold: float = _float_or_default("injection_threshold", 0.8)
    min_match_score: float = _float_or_default("min_match_score", 0.5)
    max_retry_attempts: int = _int_or_default("max_retry_attempts", 3)
    # What to do with chunks whose scan errored: "error" (never serve) or
    # "log" (record as scan_error, keep upload but exclude from retrieval).
    scan_fail_policy: str = _str_or_default("scan_fail_policy", "error")
    # How to treat tainted (prompt-injection flagged) chunks at retrieval:
    #   "exclude" - drop from retrieval context (default)
    #   "include" - keep (flag them for the UI only)
    tainted_policy: str = _str_or_default("tainted_policy", "exclude")

    # LLM transient-error retry
    llm_max_retries: int = _int_or_default("llm_max_retries", 3)
    llm_retry_backoff_base: float = _float_or_default("llm_retry_backoff_base", 1.0)
    # Cap for an upstream-provided Retry-After so a hostile/misconfigured
    # server can't park a worker thread for an unbounded time.
    llm_retry_after_cap_seconds: int = _int_or_default("llm_retry_after_cap_seconds", 60)
    # Wall-clock budget for one full pipeline (planner + up to N responder +
    # validator rounds). The Streamlit client abandons at 120s; the backend
    # aborts at this budget so it cannot keep burning token/quota on a
    # request the frontend has already given up on. <= 0 disables.
    pipeline_deadline_seconds: float = _float_or_default("pipeline_deadline_seconds", 100.0)

    # Security
    api_auth_token: str = _str_or_default("api_auth_token", "")
    cors_origins: list[str] = _list_or_default(
        "cors_origins", ["http://localhost:8501", "http://127.0.0.1:8501"]
    )
    rate_limits: dict = _dict_or_default(
        "rate_limits", {"query": 30, "evaluation_start": 2, "key_validate": 10}
    )

    # Evaluation Config
    eval_file_path: str = _str_or_default("eval_file_path", ".data/evaluation/eval_results.json")
    eval_questions_path: str = "tests/resources/default_questions.json"

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)

    @property
    def default_eval_questions(self) -> list[dict]:
        """Load default questions from the test resources json file.

        A missing/corrupt questions file is a configuration error: an
        evaluation with zero questions would otherwise report a successful
        "done" with a 0% pass rate, which silently masks unconfigured state.
        """
        resource_path = Path(__file__).resolve().parent.parent / self.eval_questions_path
        if not resource_path.exists():
            raise SystemExit(
                f"FATAL: evaluation questions file not found: {resource_path}\n"
                "(set eval_questions_path in config.json or restore the file)."
            )
        try:
            questions = json.loads(resource_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"FATAL: evaluation questions file is corrupt: {exc}") from exc
        if not isinstance(questions, list) or not questions:
            raise SystemExit(
                f"FATAL: evaluation questions file must be a non-empty JSON list: {resource_path}"
            )
        return questions

    @field_validator("scan_fail_policy")
    @classmethod
    def _validate_scan_fail_policy(cls, v: str) -> str:
        if v not in ("error", "log"):
            raise ValueError(f"scan_fail_policy must be 'error' or 'log', got {v!r}")
        return v

    @field_validator("tainted_policy")
    @classmethod
    def _validate_tainted_policy(cls, v: str) -> str:
        if v not in ("exclude", "include"):
            raise ValueError(f"tainted_policy must be 'exclude' or 'include', got {v!r}")
        return v

    def warn_if_server_keyed_without_auth(self) -> None:
        """Emit a startup warning for the exact 'free LLM proxy' scenario.

        If a server-side OpenRouter key is set but API auth is off, anyone
        who can reach the API can spend that key on full LLM completions.
        """
        if self.openrouter_api_key and not self.api_auth_token:
            logger.warning(
                "OPENROUTER_API_KEY is set but api_auth_token is empty: the backend "
                "will spend YOUR key for any caller who reaches /api/query or "
                "/api/evaluation/start. Set api_auth_token (or an API_AUTH_TOKEN env) "
                "for any non-local deployment."
            )


settings = Settings()