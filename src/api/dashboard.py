"""Dashboard API - stats for the Streamlit frontend."""

from fastapi import APIRouter
from pydantic import BaseModel

from src.config import settings
from src.vectorstore.store import CVVectorStore, get_vector_store
from src.utils.query_log import get_query_stats

router = APIRouter()


class DashboardStats(BaseModel):
    total_documents: int
    total_chunks: int
    documents_by_format: dict[str, int]
    chunks_by_section: dict[str, int]
    tainted_documents: int
    total_queries: int
    avg_match_score: float
    avg_latency_ms: float


@router.get("/stats", response_model=DashboardStats)
def get_dashboard_stats():
    """Return aggregated stats for the Streamlit dashboard."""
    store = get_vector_store(persist_dir=settings.chroma_persist_dir)
    kb_stats = store.get_stats()
    q_stats = get_query_stats()

    return DashboardStats(
        total_documents=kb_stats["total_documents"],
        total_chunks=kb_stats["total_chunks"],
        documents_by_format=kb_stats["documents_by_format"],
        chunks_by_section=kb_stats["sections_represented"],
        tainted_documents=kb_stats["tainted_documents"],
        total_queries=q_stats["total_queries"],
        avg_match_score=q_stats["avg_match_score"],
        avg_latency_ms=q_stats["avg_latency_ms"],
    )
