"""
RAG Engine Module
-----------------
Implements the Retrieval-Augmented Generation pipeline.
Uses ChromaDB as the vector store and Sentence-Transformers for embeddings.

Key design decisions:
- Recursive character chunking for robustness across document types
- all-MiniLM-L6-v2 embeddings for a good size/quality tradeoff (~80MB, 384-dim)
- Persistent local ChromaDB for reproducibility and offline use
- Metadata-aware retrieval to keep source attribution traceable

Credits:
- ChromaDB (https://www.trychroma.com/) — Apache 2.0
- Sentence-Transformers (Reimers & Gurevych, 2019) — Apache 2.0
- LangChain text splitter utility — MIT
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed — RAGEngine will run in degraded mode.")


class TextSplitter:
    """
    A simplified recursive character splitter inspired by LangChain's
    RecursiveCharacterTextSplitter. Splits on paragraph, sentence, word
    boundaries in that order.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split(self, text: str) -> List[str]:
        """Split text into chunks that respect natural boundaries."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return self._hard_split(text)
        sep = separators[0]
        rest = separators[1:]
        if sep == "":
            return self._hard_split(text)

        parts = text.split(sep)
        chunks, buffer = [], ""
        for part in parts:
            candidate = buffer + (sep if buffer else "") + part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                if len(part) > self.chunk_size:
                    chunks.extend(self._recursive_split(part, rest))
                    buffer = ""
                else:
                    buffer = part
        if buffer:
            chunks.append(buffer)
        return self._apply_overlap(chunks)

    def _hard_split(self, text: str) -> List[str]:
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if self.chunk_overlap <= 0 or len(chunks) < 2:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = overlapped[-1][-self.chunk_overlap :]
            overlapped.append(tail + chunks[i])
        return overlapped


class RAGEngine:
    """
    Main RAG orchestrator. Wraps ChromaDB for vector storage and provides
    document ingestion and similarity search.
    """

    def __init__(
        self,
        collection_name: str = "studymate",
        persist_dir: str = "./data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.splitter = TextSplitter(chunk_size, chunk_overlap)

        os.makedirs(persist_dir, exist_ok=True)

        if CHROMA_AVAILABLE:
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
            self.collection = self.client.get_or_create_collection(
                name=collection_name, embedding_function=self.embedder
            )
        else:
            # Fallback: in-memory dict (for unit tests without chromadb)
            self.client = None
            self.collection = _InMemoryCollection()

    def add_document(self, text: str, metadata: Optional[Dict] = None) -> int:
        """
        Chunk a document, embed each chunk, and store it.
        Returns the number of chunks added.
        """
        if not text or not text.strip():
            logger.warning("Skipping empty document")
            return 0

        metadata = metadata or {}
        chunks = self.splitter.split(text)
        if not chunks:
            return 0

        ids, texts, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            doc_id = self._make_id(metadata.get("source", "doc"), i, chunk)
            ids.append(doc_id)
            texts.append(chunk)
            metadatas.append({**metadata, "chunk_index": i, "chunk_count": len(chunks)})

        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
        logger.info(f"Indexed {len(chunks)} chunks from {metadata.get('source', 'unknown')}")
        return len(chunks)

    def query(self, query_text: str, top_k: int = 4) -> List[Dict]:
        """Retrieve top-k most similar chunks for a query."""
        if not query_text.strip():
            return []
        results = self.collection.query(query_texts=[query_text], n_results=top_k)
        return self._format_results(results)

    def _format_results(self, raw) -> List[Dict]:
        """Normalize ChromaDB's nested response into flat dicts."""
        formatted = []
        documents = raw.get("documents", [[]])[0] if raw.get("documents") else []
        metadatas = raw.get("metadatas", [[]])[0] if raw.get("metadatas") else []
        distances = raw.get("distances", [[]])[0] if raw.get("distances") else []
        for i, doc in enumerate(documents):
            formatted.append({
                "text": doc,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "score": 1.0 - distances[i] if i < len(distances) else None,
            })
        return formatted

    @staticmethod
    def _make_id(source: str, chunk_idx: int, text: str) -> str:
        """Deterministic ID that prevents duplicate insertion of the same chunk."""
        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        return f"{source}::{chunk_idx}::{h}"

    def clear(self):
        """Delete the current collection (used for tests and resets)."""
        if self.client is not None:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception as e:
                logger.warning(f"Could not delete collection: {e}")
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, embedding_function=self.embedder
            )


class _InMemoryCollection:
    """Minimal fallback used only for unit tests when chromadb is absent."""

    def __init__(self):
        self._store: List[Dict] = []

    def add(self, ids, documents, metadatas):
        for i, d, m in zip(ids, documents, metadatas):
            self._store.append({"id": i, "text": d, "metadata": m})

    def query(self, query_texts, n_results=4):
        # Naive keyword overlap — only for testing
        q = query_texts[0].lower()
        scored = []
        for item in self._store:
            overlap = sum(1 for w in q.split() if w in item["text"].lower())
            scored.append((overlap, item))
        scored.sort(key=lambda x: -x[0])
        top = scored[:n_results]
        return {
            "documents": [[x[1]["text"] for x in top]],
            "metadatas": [[x[1]["metadata"] for x in top]],
            "distances": [[1.0 / (1 + x[0]) for x in top]],
        }

    def peek(self, limit=10):
        """Return up to `limit` items — mirrors ChromaDB's peek() API."""
        items = self._store[:limit]
        return {
            "ids": [x["id"] for x in items],
            "documents": [x["text"] for x in items],
            "metadatas": [x["metadata"] for x in items],
        }
