"""
Step 1: RAG pipeline for Daejeon elderly-welfare information.

Responsibilities
  1. Load documents from data/welfare_docs (.pdf via pdfplumber, .md/.txt directly)
  2. Chunk long text into 200-500 char overlapping pieces
  3. Embed chunks and store them in ChromaDB (persisted on disk)
  4. Retrieve the top-K most relevant chunks for a user question

If ChromaDB is unavailable, a small in-memory cosine store is used so the
retrieval logic can still be demonstrated offline.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from . import config
from .embeddings import EmbeddingModel


# --------------------------------------------------------------------------
# Document loading
# --------------------------------------------------------------------------
def load_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF using pdfplumber (page by page)."""
    import pdfplumber
    parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def load_documents(docs_dir: Path = config.WELFARE_DOCS_DIR) -> List[dict]:
    """Load every supported file in docs_dir. Returns [{source, text}]."""
    docs: List[dict] = []
    for path in sorted(docs_dir.glob("*")):
        if path.suffix.lower() == ".pdf":
            text = load_text_from_pdf(path)
        elif path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
        else:
            continue
        if text.strip():
            docs.append({"source": path.name, "text": text})
    return docs


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_text(text: str,
               chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks of ~chunk_size characters.

    Tries to break on paragraph / sentence boundaries first, then falls back
    to a sliding character window so no chunk exceeds the size limit.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    # Split on blank lines (paragraphs) as the natural welfare-item boundary.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= chunk_size:
            buffer = f"{buffer}\n{para}".strip()
        else:
            if buffer:
                chunks.append(buffer)
            # A single paragraph longer than chunk_size -> sliding window.
            if len(para) > chunk_size:
                chunks.extend(_sliding_window(para, chunk_size, overlap))
                buffer = ""
            else:
                buffer = para
    if buffer:
        chunks.append(buffer)
    return chunks


def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
    step = max(size - overlap, 1)
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------
@dataclass
class Retrieved:
    text: str
    source: str
    score: float


class _InMemoryStore:
    """Fallback cosine-similarity store used when ChromaDB is unavailable."""

    def __init__(self):
        self._vecs: List[List[float]] = []
        self._docs: List[str] = []
        self._meta: List[dict] = []

    def add(self, ids, embeddings, documents, metadatas):
        self._vecs.extend(embeddings)
        self._docs.extend(documents)
        self._meta.extend(metadatas)

    def query(self, embedding, top_k):
        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5 or 1.0
            nb = sum(y * y for y in b) ** 0.5 or 1.0
            return dot / (na * nb)
        scored = sorted(
            ((cos(embedding, v), d, m) for v, d, m in zip(self._vecs, self._docs, self._meta)),
            key=lambda t: t[0], reverse=True,
        )[:top_k]
        return [Retrieved(text=d, source=m.get("source", "?"), score=float(s))
                for s, d, m in scored]

    def count(self):
        return len(self._docs)


class RagPipeline:
    def __init__(self, embed_model: EmbeddingModel | None = None):
        self.embedder = embed_model or EmbeddingModel()
        self.backend = "chroma"
        self._collection = self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            # Reset so re-indexing is idempotent for the demo.
            try:
                client.delete_collection(config.CHROMA_COLLECTION)
            except Exception:
                pass
            return client.create_collection(
                config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
        except Exception as exc:
            print(f"[rag] ChromaDB unavailable ({exc.__class__.__name__}); "
                  f"using in-memory store.")
            self.backend = "in-memory"
            return _InMemoryStore()

    def index(self, docs_dir: Path = config.WELFARE_DOCS_DIR) -> int:
        """Load -> chunk -> embed -> store. Returns number of chunks indexed."""
        documents = load_documents(docs_dir)
        chunks, metadatas, ids = [], [], []
        for doc in documents:
            for i, chunk in enumerate(chunk_text(doc["text"])):
                chunks.append(chunk)
                metadatas.append({"source": doc["source"], "chunk": i})
                ids.append(f"{doc['source']}::{i}")
        if not chunks:
            return 0
        # Fit the embedder on the full corpus first (matters for the TF-IDF
        # fallback; a no-op for bge-m3).
        self.embedder.fit(chunks)
        embeddings = [list(map(float, v)) for v in self.embedder.encode(chunks)]
        self._collection.add(ids=ids, embeddings=embeddings,
                             documents=chunks, metadatas=metadatas)
        return len(chunks)

    def retrieve(self, question: str, top_k: int = config.RAG_TOP_K) -> List[Retrieved]:
        q_emb = list(map(float, self.embedder.encode([question])[0]))
        if self.backend == "chroma":
            res = self._collection.query(query_embeddings=[q_emb], n_results=top_k)
            out = []
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                                       res["distances"][0]):
                out.append(Retrieved(text=doc, source=meta.get("source", "?"),
                                     score=1.0 - float(dist)))  # cosine dist -> sim
            return out
        return self._collection.query(q_emb, top_k)


if __name__ == "__main__":
    rag = RagPipeline()
    n = rag.index()
    print(f"backend={rag.backend} embedder={rag.embedder.backend} chunks={n}\n")
    for q in ["기초연금은 어떻게 신청하나요?",
              "혼자 사는데 응급상황이 걱정돼요",
              "우울하고 외로울 때 상담받고 싶어요"]:
        print(f"Q: {q}")
        for r in rag.retrieve(q):
            print(f"  [{r.score:.3f}] ({r.source}) {r.text[:60].replace(chr(10),' ')}...")
        print()
