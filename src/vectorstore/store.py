"""ChromaDB vector store for CV knowledge base.

Handles collection management, document storage, similarity search,
and document removal by metadata filter.
"""

import logging
import threading
from pathlib import Path
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


# Collection name
COLLECTION_NAME = "cv_knowledge_base"

_shared_store_instance = None
_shared_store_dir: str | None = None


def get_vector_store(persist_dir: str = "./.data/chromadb") -> "CVVectorStore":
    """Return a shared singleton instance of CVVectorStore.

    The singleton exists so the default embedding model loads once and is
    reused across requests. Once created it is bound to ONE persist_dir;
    requesting a different directory is a footgun (silent wrong-dir reads),
    so it raises instead of silently returning a store attached to the old
    directory.
    """
    global _shared_store_instance, _shared_store_dir
    if _shared_store_instance is None:
        _shared_store_instance = CVVectorStore(persist_dir=persist_dir)
        _shared_store_dir = persist_dir
        return _shared_store_instance
    if _shared_store_dir != persist_dir:
        raise RuntimeError(
            f"get_vector_store: singleton already bound to {_shared_store_dir!r}, "
            f"cannot also serve {persist_dir!r} (restart the process or construct "
            "CVVectorStore directly for the second directory)"
        )
    return _shared_store_instance


@dataclass
class StoredChunk:
    """A chunk stored in ChromaDB with full metadata."""
    id: str
    text: str
    source: str        # filename
    doc_id: str        # content hash
    section: str       # CV section name
    chunk_index: int
    tainted: bool


class CVVectorStore:
    """ChromaDB-backed vector store for CV documents."""

    def __init__(self, persist_dir: str = "./.data/chromadb"):
        self._mutex = threading.RLock()
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return the number of chunks in the collection.

        Public, thread-safe entry point so callers do not reach into the
        private ChromaDB `_collection` attribute.
        """
        with self._mutex:
            return self._collection.count()

    def add_chunks(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> int:
        """Add chunks to the collection.

        Args:
            texts: Chunk text content
            metadatas: List of metadata dicts (source, doc_id, section, etc.)
            ids: Unique chunk IDs

        Returns:
            Number of chunks added
        """
        if not texts:
            return 0

        with self._mutex:
            # Re-ingesting the same document produces the same deterministic
            # chunk ids (doc_id is a content hash) - warn loudly instead of
            # silently upserting duplicates after a user error.
            existing_ids = set(self._collection.get(ids=ids, include=[]).get("ids", [])) if ids else set()
            dupes = existing_ids.intersection(ids)
            if dupes:
                logger.warning(
                    "add_chunks: %d chunk id(s) already exist (re-ingest of the same "
                    "document?); they will be overwritten: %s",
                    len(dupes),
                    sorted(dupes)[:5],
                )
            self._collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
        return len(texts)

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """Similarity search against the knowledge base.

        Args:
            query_text: Natural language query
            n_results: Number of results to return
            where: Optional metadata filter (ChromaDB where clause)

        Returns:
            List of result dicts with text, metadata, distance
        """
        if self.count() == 0:
            return []

        with self._mutex:
            # F4-09: clamp n_results INSIDE the lock. ChromaDB rejects
            # n_results > collection size with an error; count() was checked
            # just above, but a concurrent delete between the two locked
            # sections could make n_results exceed the now-smaller collection.
            size = self._collection.count()
            if size == 0:
                return []
            query_params = {
                "query_texts": [query_text],
                "n_results": min(n_results, size),
            }
            if where:
                query_params["where"] = where
            results = self._collection.query(**query_params)

        output = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # Convert cosine distance to similarity score (0-1)
                similarity = 1.0 - distance
                output.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": max(0.0, similarity),
                })

        return output

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source file.

        Args:
            source: The source filename to delete

        Returns:
            Number of chunks deleted (approximate)
        """
        # Get count before deletion
        with self._mutex:
            before = self._collection.count()
            self._collection.delete(where={"source": source})
            after = self._collection.count()
        return before - after

    def replace_source(self, old_source: str, new_source: str) -> int:
        """Atomically swap one source's chunks for another.

        Deletes every chunk currently tagged `new_source` (the previous
        version of the document) and renames the freshly ingested chunks
        tagged `old_source` to `new_source` - all under the same lock, so no
        other reader can observe the intermediate state. Used by the upload
        pipeline's ingest-to-temp-then-swap flow.

        Returns:
            Number of chunks now tagged `new_source` (0 if none matched).
        """
        with self._mutex:
            self._collection.delete(where={"source": new_source})
            existing = self._collection.get(where={"source": old_source}, include=["metadatas"])
            ids = existing.get("ids", [])
            if not ids:
                return 0
            metadatas = []
            for meta in existing.get("metadatas", [])[: len(ids)]:
                fixed = dict(meta)
                fixed["source"] = new_source
                metadatas.append(fixed)
            self._collection.update(
                ids=ids,
                metadatas=metadatas,
            )
        return len(ids)

    def list_documents(self) -> list[dict]:
        """List all unique documents with metadata and chunk counts.

        Returns:
            List of document summaries:
            [{
                "source": str,
                "doc_id": str,
                "chunk_count": int,
                "tainted": bool,
                "sections": list[str],
            }]
        """
        if self.count() == 0:
            return []

        with self._mutex:
            all_data = self._collection.get(include=["metadatas"])
        docs = {}

        for meta in all_data["metadatas"]:
            source = meta.get("source", "unknown")
            if source not in docs:
                docs[source] = {
                    "source": source,
                    "doc_id": meta.get("doc_id", ""),
                    "chunk_count": 0,
                    "tainted": meta.get("tainted", False),
                    "sections": set(),
                }
            docs[source]["chunk_count"] += 1
            section = meta.get("section", "unknown")
            docs[source]["sections"].add(section)
            if meta.get("tainted", False):
                docs[source]["tainted"] = True

        # Convert sets to sorted lists
        for doc in docs.values():
            doc["sections"] = sorted(doc["sections"])

        return sorted(docs.values(), key=lambda d: d["source"])

    def get_stats(self) -> dict:
        """Get aggregate statistics about the knowledge base."""
        docs = self.list_documents()
        total_chunks = self.count()
        total_docs = len(docs)
        tainted = sum(1 for d in docs if d["tainted"])

        sections = {}
        for doc in docs:
            for section in doc["sections"]:
                sections[section] = sections.get(section, 0) + 1

        formats = {}
        for doc in docs:
            ext = doc["source"].rsplit(".", 1)[-1] if "." in doc["source"] else "unknown"
            formats[ext] = formats.get(ext, 0) + 1

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "tainted_documents": tainted,
            "documents_by_format": formats,
            "sections_represented": sections,
        }

    def is_empty(self) -> bool:
        """Check if the knowledge base has any documents."""
        return self.count() == 0
