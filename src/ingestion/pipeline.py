"""Document ingestion pipeline.

Extract -> Chunk -> Scan -> Store

Handles file upload, format detection, text extraction, section-aware
chunking, LLM Guard scanning, and ChromaDB storage.
"""

import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings
from src.ingestion.extractors import extract_text
from src.ingestion.chunker import chunk_document
from src.guardrails.scanner import scan_chunks, create_scanner
from src.vectorstore.store import CVVectorStore

logger = logging.getLogger(__name__)


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
    final_filename: str | None = None,
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
        final_filename: Logical document name used for the `doc_id` hash and
            chunk metadata. Defaults to `file_path.name`. The upload flow
            ingests from a random temp name (`.uploading-<pid>-<rand>-<name>`)
            and passes the real name here so that
            `doc_id = sha256(final_filename + content)` stays stable across
            re-uploads of identical content (F4-05) and the duplicate-ingest
            warning in `add_chunks` works.

    Returns:
        IngestionResult with stats and warnings
    """
    display_name = Path(final_filename).name if final_filename else file_path.name
    try:
        # Step 1: Extract text (capped so a small file that expands to
        # megabytes of text can't turn into a CPU/memory DoS via thousands
        # of chunks and DeBERTa scans).
        extracted = extract_text(file_path)
        extracted["text"] = extracted["text"][: settings.max_extracted_chars]
        if len(extracted["text"]) == settings.max_extracted_chars:
            extracted["warnings"] = list(extracted["warnings"]) + [
                f"Extracted text exceeds {settings.max_extracted_chars} characters; truncated."
            ]

        # doc_id must hash the FINAL filename (not the temp upload name), so
        # identical re-uploads produce identical ids and the add_chunks
        # duplicate warning fires (F4-05). The chunks are still tagged with
        # `file_path.name` (the temp name) so the upload swap in
        # `ingest_upload` -> `replace_source` can atomically delete the old
        # version (tagged `safe_filename`) and rename the new chunks into
        # place under one lock.
        from src.ingestion.extractors import _compute_doc_id
        extracted["doc_id"] = _compute_doc_id(display_name, extracted["text"])

        if not extracted["text"].strip():
            return IngestionResult(
                success=False,
                filename=display_name,
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
                filename=display_name,
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
        if scan_summary.errored_chunks > 0:
            policy_note = (
                "chunks excluded from retrieval"
                if settings.scan_fail_policy == "error"
                else "recorded as scan_error"
            )
            warnings.append(
                f"{scan_summary.errored_chunks} chunk(s) could not be scanned "
                f"(scanner error); {policy_note}."
            )

        # With the strict policy, a scanner failure rejects the whole upload:
        # nothing is stored (no half-tainted state in the vector store) and
        # the error surfaces to the caller.
        if scan_summary.errored_chunks > 0 and settings.scan_fail_policy == "error":
            return IngestionResult(
                success=False,
                filename=display_name,
                doc_id=extracted["doc_id"],
                format=extracted["format"],
                chunks_created=0,
                tainted=scan_summary.is_tainted,
                warnings=warnings,
                error="scan_failed: prompt-injection scanner error(s) while ingesting; upload rejected (scan_fail_policy=error)",
            )

        # Step 4: Store in ChromaDB
        ids = []
        metadatas = []
        statuses = [
            scan_summary.chunk_results[i].status.value
            if i < len(scan_summary.chunk_results) else "clean"
            for i in range(len(chunks))
        ]
        for i, chunk in enumerate(chunks):
            chunk_id = f"{extracted['doc_id']}_chunk_{i:04d}"
            ids.append(chunk_id)
            metadatas.append({
                "source": file_path.name,
                "doc_id": extracted["doc_id"],
                "section": chunk.section,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "tainted": statuses[i] == "tainted",
                "scan_status": statuses[i],
                "format": extracted["format"],
            })

        vector_store.add_chunks(
            texts=chunk_texts,
            metadatas=metadatas,
            ids=ids,
        )

        return IngestionResult(
            success=True,
            filename=display_name,
            doc_id=extracted["doc_id"],
            format=extracted["format"],
            chunks_created=len(chunks),
            tainted=scan_summary.is_tainted,
            warnings=warnings,
        )

    except Exception as e:
        logger.exception(f"Ingestion failed for {display_name}: {str(e)}")
        return IngestionResult(
            success=False,
            filename=display_name,
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

    Saves to a temporary name, runs the standard ingestion pipeline, and
    only on success atomically replaces any previous version of the same
    document (vector chunks AND the raw file). This makes a failed
    re-upload harmless: the previous good version stays queryable and its
    raw file stays on disk; a rejected upload (e.g. prompt-injection scan
    error under scan_fail_policy=error, empty text, extraction crash) leaves
    no stray file in the upload dir.
    """
    upload_dir = settings.upload_path
    safe_filename = Path(filename).name
    tmp_path = upload_dir / f".uploading-{os.getpid()}-{secrets.token_hex(4)}-{safe_filename}"

    try:
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Write to a temp name FIRST: the old document must stay intact if
        # the new bytes fail to extract/chunk/scan (no data-loss path).
        tmp_path.write_bytes(file_content)

        result = ingest_document(tmp_path, vector_store, scanner, final_filename=safe_filename)

        if not result.success:
            # Rejected upload: nothing may be left in the upload dir. The raw
            # file is unlinked so `GET /content` cannot serve text from a
            # rejected (potentially injection-laden) document, and the file
            # does not silently accumulate on disk. (An ingestion failure
            # stores nothing, so the KB is untouched too.)
            tmp_path.unlink(missing_ok=True)
            return result

        # Success: atomically swap the KB from the old version to the new
        # one. The new chunks are tagged with the temp source name, so the
        # previous version (tagged safe_filename) is removed and the new
        # chunks renamed - all under one lock - before the raw file moves
        # into place. If this raises, the whole upload reports failure.
        vector_store.replace_source(tmp_path.name, safe_filename)

        final_path = upload_dir / safe_filename
        tmp_path.replace(final_path)

        return IngestionResult(
            success=True,
            filename=safe_filename,
            doc_id=result.doc_id,
            format=result.format,
            chunks_created=result.chunks_created,
            tainted=result.tainted,
            warnings=result.warnings,
        )
    except Exception as e:
        # Best-effort rewind: if the KB swap partially applied, remove any
        # temp-tagged chunks so nothing orphaned stays behind.
        try:
            vector_store.delete_by_source(tmp_path.name)
        except Exception:
            pass
        tmp_path.unlink(missing_ok=True)
        logger.exception(f"Failed to save/prepare upload for {safe_filename}: {str(e)}")
        return IngestionResult(
            success=False,
            filename=safe_filename,
            doc_id="",
            format="",
            chunks_created=0,
            tainted=False,
            warnings=[],
            error=f"Upload failed: {str(e)}",
        )


def remove_document(filename: str, vector_store: CVVectorStore) -> dict:
    """Remove a document from the knowledge base.

    Returns:
        {"removed": bool, "filename": str, "chunks_removed": int}
    """
    chunks_removed = vector_store.delete_by_source(filename)

    # Also delete the raw file from the uploads directory
    file_path = settings.upload_path / filename
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.warning(f"Failed to delete raw file {filename} from disk: {e}")

    return {
        "removed": chunks_removed > 0,
        "filename": filename,
        "chunks_removed": chunks_removed,
    }
