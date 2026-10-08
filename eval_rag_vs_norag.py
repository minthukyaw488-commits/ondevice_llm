"""
RAG 효과 정량 평가 — 'LLM만' vs 'RAG + LLM' 정답률 비교.

compare_rag.py 가 질문 하나를 눈으로 비교한다면, 이 스크립트는 라벨링된
질문 세트 전체를 두 방식으로 돌려 숫자로 비교한다:

  · 정보 질문 — 사실 정확도(fact hit): 정답 키워드를 담았는가
  · 범위 밖 질문 — 안전 거절(declined): 지어내지 않고 주민센터로 안내했는가

같은 모델, 같은 질문. 차이는 '검색 근거(RAG)를 주었는가' 뿐이다.

  LLM_BACKEND=ollama LLM_MODEL=exaone3.5:2.4b python eval_rag_vs_norag.py
  EVAL_LIMIT=12 python eval_rag_vs_norag.py        # 문항 수 제한(속도)
"""
from __future__ import annotations
import os
os.environ.setdefault("USE_AGENT", "0")
os.environ.setdefault("USE_LLM_SIGNAL", "0")

from src.pipeline import WelfareAssistant
from eval_qa import QA_CASES, score_answer
from compare_rag import no_rag_answer, rag_answer

LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))


def main():
    print("파이프라인 로딩...\n")
    bot = WelfareAssistant()
    cases = QA_CASES[:LIMIT] if LIMIT else QA_CASES
    n_in = sum(c.scope == "in" for c in cases)
    n_out = sum(c.scope == "out" for c in cases)
    print(f"답변 모델 : {type(bot.llm).__name__}:{bot.llm.model}")
    print(f"평가 문항 : {len(cases)}개 (정보 {n_in} / 범위밖 {n_out})\n")

    # in: 사실 정확도(fact) / out: 안전 거절(declined)
    nr = {"fact": 0, "decl": 0}   # no-RAG
    rg = {"fact": 0, "decl": 0}   # RAG
    for qa in cases:
        a1 = no_rag_answer(bot, qa.q)
        s1 = score_answer(qa, a1)
        a2, _, _ = rag_answer(bot, qa.q)
        s2 = score_answer(qa, a2)
        if qa.scope == "in":
            nr["fact"] += int(s1["fact"]); rg["fact"] += int(s2["fact"])
            m1, m2 = ("O" if s1["fact"] else "X"), ("O" if s2["fact"] else "X")
        else:
            nr["decl"] += int(s1["declined"]); rg["decl"] += int(s2["declined"])
            m1, m2 = ("O" if s1["declined"] else "X"), ("O" if s2["declined"] else "X")
        print(f"[{qa.scope:3}] LLM만={m1}  RAG={m2}  {qa.q}")

    def pct(x, t):
        return f"{x/t:.0%}" if t else "-"

    print("\n" + "=" * 66)
    print(" RAG 미사용(LLM만)  vs  RAG 사용(RAG+LLM)")
    print("=" * 66)
    print(f"  정보 질문 사실 정확도 : LLM만 {pct(nr['fact'], n_in):>4}  →  "
          f"RAG {pct(rg['fact'], n_in):>4}")
    print(f"  범위 밖 안전 거절     : LLM만 {pct(nr['decl'], n_out):>4}  →  "
          f"RAG {pct(rg['decl'], n_out):>4}")
    print("  (정보=정답 키워드 포함, 범위밖=지어내지 않고 주민센터 안내)")


if __name__ == "__main__":
    main()
