"""
RAG 품질 지표 — Context Precision & Faithfulness (RAGAS 방식, LLM-judge).

evaluate.py 가 검색의 Hit@k(근거를 찾았나)를 재는 반면, 이 스크립트는 RAG
평가 프레임워크의 나머지 두 축을 LLM 심판으로 측정한다:

  1) Context Precision — 검색해 온 상위 조각 중 '질문과 실제로 관련 있는'
     조각의 비율 (= 노이즈가 얼마나 적은가).
  2) Faithfulness      — 생성된 답변이 '검색된 근거에만' 기반하는가
     (= 지어내지 않았는가 / 환각 없음).

심판(judge)은 답변 LLM(config.LLM_BACKEND, 기본 로컬 EXAONE)을 그대로 쓴다.

  LLM_BACKEND=ollama LLM_MODEL=exaone3.5:2.4b python eval_rag.py
  EVAL_LIMIT=10 python eval_rag.py      # 질문 수 제한(기본 15, 속도용)
"""
from __future__ import annotations
import os
os.environ.setdefault("USE_AGENT", "0")        # 결정형 RAG 경로로 측정
os.environ.setdefault("USE_LLM_SIGNAL", "0")

from src import config
from src.llm import ANSWER_SYSTEM_PROMPT, build_answer_prompt
from src.pipeline import WelfareAssistant
from eval_qa import QA_CASES

LIMIT = int(os.environ.get("EVAL_LIMIT", "15"))


def judge_yes(llm, prompt: str) -> bool:
    """LLM 심판에게 예/아니오를 물어 '예'면 True."""
    r = (llm.generate(prompt, system="당신은 꼼꼼한 평가자입니다. '예' 또는 "
                      "'아니오' 한 단어로만 답하세요.") or "").strip()
    head = r.replace(" ", "")[:6]
    return ("예" in head) and ("아니오" not in head)


def context_precision(llm, q: str, chunks) -> float:
    """검색된 조각 중 질문과 관련 있는 비율 (0~1)."""
    if not chunks:
        return 0.0
    rel = 0
    for c in chunks:
        p = (f"[질문] {q}\n\n[자료]\n{c[:600]}\n\n"
             "이 자료가 위 질문에 답하는 데 관련이 있습니까? '예'/'아니오'로만.")
        rel += int(judge_yes(llm, p))
    return rel / len(chunks)


def faithfulness(llm, answer: str, chunks) -> bool:
    """답변이 근거에만 기반하는가(지어내지 않았는가)."""
    ctx = "\n\n".join(c[:500] for c in chunks)
    p = (f"[자료]\n{ctx}\n\n[답변]\n{answer}\n\n"
         "위 답변의 모든 내용이 자료에 근거합니까? 자료에 없는 내용을 "
         "지어냈으면 '아니오', 모두 자료에 근거하면 '예'로만 답하세요.")
    return judge_yes(llm, p)


def main():
    print("파이프라인 로딩...\n")
    bot = WelfareAssistant()
    cases = [c for c in QA_CASES if c.scope == "in"][:LIMIT]
    print(f"RAG     : {bot.rag.backend} / {bot.rag.embedder.backend}")
    print(f"심판 LLM: {type(bot.llm).__name__}:{bot.llm.model}")
    print(f"평가 문항: {len(cases)}개 (정보 질문)\n")

    prec_sum, faith_sum, answered = 0.0, 0, 0
    for i, qa in enumerate(cases, 1):
        retrieved = bot.rag.retrieve(qa.q)
        top = retrieved[0].score if retrieved else 0.0
        chunks = [r.text for r in retrieved]
        # 관련성 게이트에 막히면(근거 없음) 답변 없이 건너뜀
        if not retrieved or top < config.RAG_MIN_RELEVANCE:
            print(f"[{i:2}] (게이트: 근거 없음) {qa.q}")
            continue
        prec = context_precision(bot.llm, qa.q, chunks)
        answer = bot.llm.generate(build_answer_prompt(qa.q, chunks),
                                  system=ANSWER_SYSTEM_PROMPT).strip()
        faith = faithfulness(bot.llm, answer, chunks)
        prec_sum += prec
        faith_sum += int(faith)
        answered += 1
        print(f"[{i:2}] precision={prec:.0%}  faithful={'O' if faith else 'X'}  {qa.q}")

    print("\n" + "=" * 60)
    print(" RAG 품질 지표 (LLM-judge)")
    print("=" * 60)
    if answered:
        print(f"  Context Precision (평균) : {prec_sum/answered:.0%}  "
              f"(검색 조각 중 관련 있는 비율 — 노이즈 적을수록 높음)")
        print(f"  Faithfulness (근거 충실)  : {faith_sum}/{answered} = "
              f"{faith_sum/answered:.0%}  (답변이 근거에만 기반 — 환각 없음)")
    else:
        print("  답변된 문항이 없습니다(모두 게이트에 막힘).")
    print(f"  * 심판: {type(bot.llm).__name__}:{bot.llm.model} (로컬 LLM)")


if __name__ == "__main__":
    main()
