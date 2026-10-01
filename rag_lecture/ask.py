"""
교수님 강의(Week 05 · STEP 09) 방식 RAG — 저장한 벡터로 질문하기 (대전 복지 데이터).

강의 slide 67(ask.py)과 동일: 질문 하나만 임베딩 → 코사인 유사도로 상위 3개
조각을 찾아 → "자료만 근거로" 프롬프트에 넣어 → Ollama 로컬 모델이 답한다.

  python rag_lecture/ask.py "기초연금은 어떻게 신청하나요?"
  python rag_lecture/ask.py                      # 기본 질문으로 실행
"""
import json
import os
import sys

import numpy as np
import requests

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "exaone3.5:2.4b")   # 강의는 llama3.1:8b
# 상위 몇 조각을 근거로 쓸지. 자료가 많을수록(수백~수천 조각) 3개로는 정답
# 조각이 밀려날 수 있어 기본을 5로. RAG_TOPK 로 조절.
TOPK = int(os.environ.get("RAG_TOPK", "5"))
HERE = os.path.dirname(os.path.abspath(__file__))

VECS = np.load(os.path.join(HERE, "vectors.npy"))
CHUNKS = json.load(open(os.path.join(HERE, "chunks.json"), encoding="utf-8"))
_src_path = os.path.join(HERE, "sources.json")
SOURCES = (json.load(open(_src_path, encoding="utf-8"))
           if os.path.exists(_src_path) else [""] * len(CHUNKS))


def embed(t: str):
    r = requests.post(OLLAMA + "/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": t})
    r.raise_for_status()
    return np.array(r.json()["embedding"])


def search(q: str, k: int = TOPK):
    """질문과 가장 가까운 조각 k개 — 코사인 유사도 (강의 slide 67)."""
    qv = embed(q)
    sims = VECS @ qv / (np.linalg.norm(VECS, axis=1) * np.linalg.norm(qv) + 1e-9)
    idx = np.argsort(sims)[::-1][:k]
    return [(CHUNKS[i], SOURCES[i], float(sims[i])) for i in idx]


def ask(q: str):
    hits = search(q)
    ctx = "\n\n".join(h[0] for h in hits)
    # "자료만 근거로" — 이 한 줄이 환각을 막는 핵심 (강의 slide 60).
    prompt = ("아래 자료만 근거로 한국어로 짧고 공손하게 답하세요.\n"
              "자료에 없으면 '자료에 없습니다. 가까운 주민센터에 문의하세요'라고만 "
              "답하세요.\n\n"
              f"[자료]\n{ctx}\n\n[질문] {q}")
    r = requests.post(OLLAMA + "/api/generate",
                      json={"model": CHAT_MODEL, "prompt": prompt, "stream": False})
    r.raise_for_status()
    return r.json()["response"], hits


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "기초연금은 어떻게 신청하나요?"
    answer, hits = ask(q)
    print("\n질문:", q)
    print("\n답변:", answer.strip())
    print("\n근거 (유사도 · 출처):")
    for c, src, s in hits:
        print(f"  [{s:.3f}] ({src}) {c[:80].strip()}…")
