"""Agentic RAG CV Matcher — FastAPI Backend."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.api import documents, query, dashboard, evaluation, synthetic


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure data directories exist on startup."""
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    Path("data/evaluation").mkdir(parents=True, exist_ok=True)
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


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}
