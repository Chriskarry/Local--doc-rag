"""
Local RAG engine: ingest documents -> chunk -> embed (Ollama) -> ChromaDB,
then retrieve the most relevant chunks for a question.

Everything runs on your machine. No API keys, no cloud.
Requires in Ollama: an embedding model (default: nomic-embed-text).
"""

import hashlib
import os

import chromadb
import ollama
from pypdf import PdfReader

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
COLLECTION_NAME = "documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4


# ---------------------------------------------------------------- ingestion

def read_text_file(path: str) -> str:
    """Extract raw text from a PDF, TXT, or Markdown file."""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence cuts."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            for sep in ("\n\n", "\n", ". "):
                cut = text.rfind(sep, start + 200, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # always make progress
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the local Ollama embedding model."""
    resp = ollama.embed(model=EMBED_MODEL, input=texts)
    return resp["embeddings"]


# ---------------------------------------------------------------- storage

_client = None


def _collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client.get_or_create_collection(COLLECTION_NAME)


def ingest_file(path: str, filename: str) -> int:
    """Chunk + embed a file into the vector store. Returns chunk count."""
    chunks = chunk_text(read_text_file(path))
    if not chunks:
        return 0
    col = _collection()
    # Stable ids per file, so re-uploading replaces instead of duplicating.
    digest = hashlib.md5(filename.encode()).hexdigest()[:8]
    ids = [f"{digest}-{i}" for i in range(len(chunks))]
    col.upsert(
        ids=ids,
        documents=chunks,
        metadatas=[{"source": filename} for _ in chunks],
        embeddings=embed(chunks),
    )
    return len(chunks)


def indexed_count() -> int:
    return _collection().count()


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Return the k most relevant chunks for a question."""
    col = _collection()
    count = col.count()
    if count == 0:
        return []
    q_emb = embed([question])[0]
    res = col.query(query_embeddings=[q_emb], n_results=min(k, count))
    return [
        {"text": doc, "source": meta.get("source", "?"), "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def clear_all() -> None:
    """Wipe the whole document collection (fresh start)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    _client.delete_collection(COLLECTION_NAME)
    _client = None
