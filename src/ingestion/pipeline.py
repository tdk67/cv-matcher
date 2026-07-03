"""Document ingestion pipeline.

Extract → Chunk → Scan → Store

Handles file upload, format detection, text extraction, section-aware
chunking, LLM Guard scanning, and ChromaDB storage.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings
from src.ingestion.extractors import extract_text
from src.ingestion.chunker import chunk_document
from src.guardrails.scanner import scan_chunks, create_scanner
from src.vectorstore.store import CVVectorStore


@dataclass
class IngestionResult:
    """Result of ingesting a single document."""
    success: bool
    filename: str
    doc_id: str
    format: str
    chunks_created: int
    tainted: bool
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def ingest_document(
    file_path: Path,
    vector_store: CVVectorStore,
    scanner=None,
) -> IngestionResult:
    """Ingest a single document through the full pipeline.

    Steps:
    1. Extract text from file (PDF/TXT/CSV/Excel)
    2. Section-aware chunking
    3. LLM Guard injection scanning per chunk
    4. Store chunks in ChromaDB with metadata

    Args:
        file_path: Path to the document file
        vector_store: ChromaDB store instance
        scanner: Pre-initialized LLM Guard scanner (optional)

    Returns:
        IngestionResult with stats and warnings
    """
    try:
        # Step 1: Extract text
        extracted = extract_text(file_path)

        if not extracted["text"].strip():
            return IngestionResult(
                success=False,
                filename=file_path.name,
                doc_id=extracted["doc_id"],
                format=extracted["format"],
                chunks_created=0,
                tainted=False,
                warnings=["Document contains no extractable text"],
                error="empty_document",
            )

        # Step 2: Chunk
        chunks = chunk_document(extracted["text"])

        if not chunks:
            return IngestionResult(
                success=False,
                filename=file_path.name,
                doc_id=extracted["doc_id"],
                format=extracted["format"],
                chunks_created=0,
                tainted=False,
                warnings=["Document produced no chunks after splitting"],
                error="no_chunks",
            )

        # Step 3: Scan for prompt injection
        chunk_texts = [c.text for c in chunks]
        if scanner is None:
            scanner = create_scanner(settings.injection_threshold)
        scan_summary = scan_chunks(chunk_texts, scanner, settings.injection_threshold)

        warnings = list(extracted["warnings"])
        if scan_summary.is_tainted:
            warnings.append(
                f"Document contains {scan_summary.tainted_chunks} chunk(s) with "
                f"potential prompt injection (max risk: {scan_summary.max_risk_score:.2f}). "
                f"Ingested with caution."
            )

        # Step 4: Store in ChromaDB
        ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{extracted['doc_id']}_chunk_{i:04d}"
            ids.append(chunk_id)
            metadatas.append({
                "source": file_path.name,
                "doc_id": extracted["doc_id"],
                "section": chunk.section,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "tainted": scan_summary.chunk_results[i].injection_detected
                if i < len(scan_summary.chunk_results) else False,
                "format": extracted["format"],
            })

        vector_store.add_chunks(
            texts=chunk_texts,
            metadatas=metadatas,
            ids=ids,
        )

        return IngestionResult(
            success=True,
            filename=file_path.name,
            doc_id=extracted["doc_id"],
            format=extracted["format"],
            chunks_created=len(chunks),
            tainted=scan_summary.is_tainted,
            warnings=warnings,
        )

    except Exception as e:
        return IngestionResult(
            success=False,
            filename=file_path.name,
            doc_id="",
            format="",
            chunks_created=0,
            tainted=False,
            warnings=[],
            error=f"Ingestion failed: {str(e)}",
        )


def ingest_upload(
    file_content: bytes,
    filename: str,
    vector_store: CVVectorStore,
    scanner=None,
) -> IngestionResult:
    """Ingest an uploaded file (from FastAPI UploadFile).

    Saves to upload dir, then runs the standard ingestion pipeline.
    If a document with the same name exists, removes the old one first.
    """
    upload_dir = settings.upload_path
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename

    # Write uploaded content
    file_path.write_bytes(file_content)

    # Check if document already exists — remove old version
    existing_docs = vector_store.list_documents()
    for doc in existing_docs:
        if doc["source"] == filename:
            vector_store.delete_by_source(filename)
            break

    return ingest_document(file_path, vector_store, scanner)


def remove_document(filename: str, vector_store: CVVectorStore) -> dict:
    """Remove a document from the knowledge base.

    Returns:
        {"removed": bool, "filename": str, "chunks_removed": int}
    """
    chunks_removed = vector_store.delete_by_source(filename)
    return {
        "removed": chunks_removed > 0,
        "filename": filename,
        "chunks_removed": chunks_removed,
    }
