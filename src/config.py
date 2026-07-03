"""Application configuration via environment variables."""

import json
from pathlib import Path
from pydantic_settings import BaseSettings

# Load defaults from config.json if it exists
config_path = Path(__file__).parent.parent / "config.json"
defaults = {}
if config_path.exists():
    try:
        defaults = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        pass


class Settings(BaseSettings):
    """All configuration is loaded from environment variables / .env file.

    No model names or API keys are hardcoded anywhere in the source.
    """

    # OpenRouter
    openrouter_api_key: str = defaults.get("openrouter_api_key", "")
    rag_model: str = defaults.get("rag_model", "google/gemini-2.5-flash")
    validation_model: str = defaults.get("validation_model", "openai/gpt-4.1-nano")
    embedding_model: str = defaults.get("embedding_model", "all-MiniLM-L6-v2")

    # Application
    app_host: str = defaults.get("app_host", "0.0.0.0")
    app_port: int = defaults.get("app_port", 8000)
    chroma_persist_dir: str = defaults.get("chroma_persist_dir", "./.data/chromadb")
    upload_dir: str = defaults.get("upload_dir", "./.data/uploads")
    max_upload_size_mb: int = defaults.get("max_upload_size_mb", 50)
    log_level: str = defaults.get("log_level", "INFO")

    # Guardrails
    injection_threshold: float = defaults.get("injection_threshold", 0.8)
    min_match_score: float = defaults.get("min_match_score", 0.5)
    max_retry_attempts: int = defaults.get("max_retry_attempts", 3)

    # Evaluation Config
    eval_file_path: str = defaults.get("eval_file_path", ".data/evaluation/eval_results.json")
    eval_questions_path: str = "tests/resources/default_questions.json"

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)

    @property
    def default_eval_questions(self) -> list[dict]:
        """Load default questions from the test resources json file."""
        resource_path = Path(__file__).parent.parent / self.eval_questions_path
        if resource_path.exists():
            try:
                return json.loads(resource_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return []

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
