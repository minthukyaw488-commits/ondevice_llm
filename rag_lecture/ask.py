"""
교수님 강의(Week 05 · STEP 09) 방식 RAG — 저장한 벡터로 질문하기 (대전 복지 데이터).

강의 slide 67(ask.py)과 동일: 질문 하나만 임베딩 → 코사인 유사도로 상위 3개
조각을 찾아 → "자료만 근거로" 프롬프트에 넣어 → Ollama 로컬 모델이 답한다.

  python rag_lecture/ask.py "기초연금은 어떻게 신청하나요?"
  python rag_lecture/ask.py                      # 기본 질문으로 실행
"""
import json
import os
import re
import sys

import numpy as np
import requests

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "exaone3.5:2.4b")   # 강의는 llama3.1:8b
# 상위 몇 조각을 근거로 쓸지. 자료가 많을수록(수백~수천 조각) 3개로는 정답
# 조각이 밀려날 수 있어 기본을 5로. RAG_TOPK 로 조절.
TOPK = int(os.environ.get("RAG_TOPK", "5"))
# 하이브리드 재랭킹: 순수 코사인은 자료가 많으면 키워드가 겹치는 엉뚱한 조각을
# 정답 조각 위로 올리기도 한다(예: '기초연금' 질문에 '장기요양' 조각). 코사인
# 점수에 '질문 핵심어가 조각에 실제로 들어있는 비율'을 더해 정답 조각을 끌어올린다.
# RAG_HYBRID=0 으로 끄면 강의 그대로의 순수 코사인이 된다.
HYBRID = os.environ.get("RAG_HYBRID", "1") not in ("0", "false", "False", "")
LEX_WEIGHT = float(os.environ.get("RAG_LEX_WEIGHT", "0.3"))  # 키워드 보너스 가중치
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


# 질문에서 '핵심어'만 뽑는다. 길이 2+ 한글/영문/숫자 토큰을 뽑고, 뒤에 붙은
# 조사/어미(은/를/하나요 등)를 떼어 어간만 남긴다. 조사를 떼지 않으면 '기초연금은'이
# 문서의 '기초연금'과 substring으로 매칭되지 않기 때문이다.
_TOKEN = re.compile(r"[가-힣]{2,}|[A-Za-z]{2,}|\d+")
_VERB = ("하나요", "하세요", "합니다", "할까요", "드릴까요", "받나요", "되나요",
         "싶어요", "싶습니다", "해요", "하고", "받을", "알려")
_JOSA2 = ("에서", "에게", "으로", "부터", "까지", "이나", "라도", "처럼", "마다")
_JOSA1 = ("은", "는", "이", "가", "을", "를", "에", "도", "와", "과", "의", "로", "만")
_STOP = {"어떻게", "무엇", "뭐가", "어디", "그리고", "대해", "대한", "주세요",
         "궁금", "방법", "좀", "요즘", "그냥", "있는", "있나요", "곳이", "곳"}


def _stem(t: str) -> str:
    for e in _VERB:                       # 동사 어미: 신청하나요 -> 신청
        if t.endswith(e) and len(t) - len(e) >= 2:
            return t[:-len(e)]
    for j in _JOSA2:                       # 2글자 조사: 집에서 -> 집
        if t.endswith(j) and len(t) - 2 >= 2:
            return t[:-2]
    for j in _JOSA1:                       # 1글자 조사: 기초연금은 -> 기초연금
        if t.endswith(j) and len(t) - 1 >= 2:
            return t[:-1]
    return t


def _keywords(q: str):
    out = []
    for t in _TOKEN.findall(q):
        s = _stem(t)
        if len(s) >= 2 and s not in _STOP:
            out.append(s)
    return out


def search(q: str, k: int = TOPK):
    """질문과 가장 관련 있는 조각 k개.

    강의의 코사인 유사도에 더해(HYBRID), 질문 핵심어가 실제로 들어있는 조각에
    가중치를 주는 간단한 하이브리드 재랭킹을 적용한다. 자료가 수천 조각일 때
    키워드만 겹치는 엉뚱한 조각이 정답을 밀어내는 문제를 완화한다.
    """
    qv = embed(q)
    sims = VECS @ qv / (np.linalg.norm(VECS, axis=1) * np.linalg.norm(qv) + 1e-9)
    kws = _keywords(q)
    if HYBRID and kws:
        # 각 조각에 들어있는 질문 핵심어의 비율(0~1) → 코사인에 가중 합산.
        bonus = np.array([sum(kw in CHUNKS[i] for kw in kws) / len(kws)
                          for i in range(len(CHUNKS))])
        score = sims + LEX_WEIGHT * bonus
    else:
        score = sims
    idx = np.argsort(score)[::-1][:k]
    # 근거 표시는 원래 코사인 유사도를 그대로 보여준다.
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
