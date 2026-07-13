"""Document management API - upload, list, remove documents."""

from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.vectorstore.store import CVVectorStore, get_vector_store
from src.ingestion.pipeline import ingest_upload, remove_document
from src.utils.filenames import repair_mojibake_filename

router = APIRouter()


def _get_store() -> CVVectorStore:
    return get_vector_store(persist_dir=settings.chroma_persist_dir)


class DocumentInfo(BaseModel):
    id: str
    filename: str
    format: str
    chunk_count: int
    tainted: bool
    sections: list[str]
    doc_id: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class UploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    chunks_created: int
    tainted: bool
    warnings: list[str]
    error: str | None = None


class RemoveResponse(BaseModel):
    removed: bool
    filename: str
    chunks_removed: int


@router.get("/", response_model=DocumentListResponse)
def list_documents():
    """List all ingested documents with metadata."""
    store = _get_store()
    docs = store.list_documents()
    stats = store.get_stats()

    return DocumentListResponse(
        documents=[
            DocumentInfo(
                id=d["doc_id"],
                filename=d["source"],
                format=d["source"].rsplit(".", 1)[-1] if "." in d["source"] else "unknown",
                chunk_count=d["chunk_count"],
                tainted=d["tainted"],
                sections=d["sections"],
                doc_id=d["doc_id"],
            )
            for d in docs
        ],
        total_documents=stats["total_documents"],
        total_chunks=stats["total_chunks"],
    )


@router.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document (PDF, TXT, CSV, Excel)."""
    filename = Path(repair_mojibake_filename(file.filename)).name

    allowed_formats = {".pdf", ".txt", ".csv", ".xlsx"}
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {suffix}. Allowed: {', '.join(sorted(allowed_formats))}",
        )

    content = file.file.read()
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {settings.max_upload_size_mb}MB",
        )

    store = _get_store()
    result = ingest_upload(content, filename, store)

    if not result.success and result.error:
        raise HTTPException(status_code=422, detail=result.error)

    return UploadResponse(
        success=result.success,
        document_id=result.doc_id,
        filename=result.filename,
        chunks_created=result.chunks_created,
        tainted=result.tainted,
        warnings=result.warnings,
        error=result.error,
    )


@router.delete("/{filename}", response_model=RemoveResponse)
def remove_document_endpoint(filename: str):
    """Remove a document and all its chunks from the knowledge base."""
    # Sanitize to prevent path traversal
    safe_filename = Path(filename).name
    decoded_filename = repair_mojibake_filename(safe_filename)

    store = _get_store()
    result = remove_document(decoded_filename, store)
    if not result["removed"] and decoded_filename != filename:
        result = remove_document(filename, store)

    if not result["removed"]:
        raise HTTPException(status_code=404, detail=f"Document not found: {filename}")
    return RemoveResponse(**result)


@router.get("/{filename}/content")
def get_document_content(filename: str):
    """Retrieve full text content of a document by filename."""
    # Sanitize filename and enforce path containment
    safe_filename = Path(filename).name
    upload_dir = settings.upload_path.resolve()
    file_path = (upload_dir / safe_filename).resolve()

    if not file_path.is_relative_to(upload_dir):
        raise HTTPException(status_code=403, detail="Access denied")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Document file not found: {filename}")

    try:
        from src.ingestion.extractors import extract_text
        res = extract_text(file_path)
        return {"filename": safe_filename, "text": res["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read document: {str(e)}")
