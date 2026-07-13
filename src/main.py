"""Agentic RAG CV Matcher - FastAPI Backend."""

import logging
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.api import documents, query, dashboard, evaluation, synthetic, apikey


def _configure_logging() -> None:
    """Attach a file + stream handler to the root logger.

    Without this, every `logging.error(...)` call in the app (pipeline,
    scanner, etc.) falls back to Python's "last resort" stderr handler:
    unformatted, filterless, and easy to lose in whatever terminal happens
    to be running uvicorn. That's why upload failures were only ever
    visible as an error toast in Streamlit and never showed up anywhere a
    developer would look for logs.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (e.g. --reload re-import, or tests)

    root.setLevel(settings.log_level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    log_dir = Path(".data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)


_configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure data directories exist on startup and pre-warm models sequentially."""
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    Path(".data/evaluation").mkdir(parents=True, exist_ok=True)

    # Pre-warm scanner and vector store to avoid threading/meta-tensor issues in parallel uploads
    try:
        from src.guardrails.scanner import create_scanner
        from src.vectorstore.store import get_vector_store
        create_scanner(settings.injection_threshold)
        store = get_vector_store(settings.chroma_persist_dir)
        
        # Pre-warm ChromaDB default embedding model to download and cache it before any parallel requests
        logger.info("Pre-warming ChromaDB embedding function...")
        store.add_chunks(
            texts=["warmup"],
            metadatas=[{"source": "warmup", "doc_id": "warmup", "section": "warmup", "chunk_index": 0, "tainted": False}],
            ids=["warmup_id"]
        )
        store.delete_by_source("warmup")
        logger.info("ChromaDB embedding function warmed up successfully.")
    except Exception as e:
        logger.error(f"Failed to pre-warm startup models: {str(e)}")

    yield


app = FastAPI(
    title="Agentic RAG CV Matcher",
    description="Agentic RAG system for CV expertise matching with guardrails.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])
app.include_router(synthetic.router, prefix="/api/synthetic", tags=["synthetic"])
app.include_router(apikey.router, prefix="/api/key", tags=["key"])


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Log the full traceback for any exception that slips past route-level
    handling, so a failure is never visible only as a Streamlit error toast."""
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}
