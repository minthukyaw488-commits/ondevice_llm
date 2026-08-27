"""
Minimal RAG prototype (original reference).

This is the small, self-contained example the project started from:
ChromaDB + sentence-transformers (bge-m3) + Ollama (llama3.1), all local.

The full, extended system lives in the `src/` package (PDF loading, chunking,
STT, abnormal-signal detection, integration, UI). Keep this file as a compact
reference for how the core RAG loop works on its own.

Run:  python rag_minimal_example.py
Requires: `ollama serve` running with `ollama pull llama3.1`, and the bge-m3
model downloadable via sentence-transformers.
"""
import chromadb
from sentence_transformers import SentenceTransformer

# 1) Three dummy welfare documents (the original prototype's test data).
DOCS = [
    "기초연금은 만 65세 이상 어르신에게 매월 지급되며 주민센터에서 신청합니다.",
    "독거노인 응급안전안심서비스는 화재·가스 감지기를 무료로 설치해 줍니다.",
    "치매안심센터에서는 무료 치매 조기검진과 상담을 제공합니다.",
]

# 2) Embed with bge-m3 and store in ChromaDB.
embedder = SentenceTransformer("BAAI/bge-m3")
client = chromadb.Client()
col = client.create_collection("welfare_minimal")
col.add(
    ids=[str(i) for i in range(len(DOCS))],
    embeddings=[embedder.encode(d).tolist() for d in DOCS],
    documents=DOCS,
)

# 3) Retrieve for a question.
question = "기초연금은 어떻게 신청하나요?"
hit = col.query(query_embeddings=[embedder.encode(question).tolist()], n_results=1)
context = hit["documents"][0][0]
print("retrieved:", context)

# 4) Answer with a local LLM via Ollama (no cloud API).
import json
import urllib.request

payload = {
    "model": "llama3.1",
    "prompt": f"참고 자료: {context}\n\n질문: {question}\n\n존댓말로 짧게 답하세요.",
    "stream": False,
}
req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as resp:
    print("answer:", json.loads(resp.read())["response"].strip())
