"""
Answer-quality evaluation for the welfare Q&A assistant (text-to-text).

`evaluate.py` measures RETRIEVAL (Hit@k) and the abnormal-signal classifier.
This script measures the thing the elderly user actually reads: the generated
ANSWER. It runs the full pipeline (RAG + local LLM) on a labelled question set
and scores each answer on four axes:

  1. 정확성(fact) : answer mentions at least one expected key fact for the topic
  2. 한국어(ko)   : no English leakage (no run of >=2 latin letters)
  3. 간결성(len)  : short, 1-2 sentences (<= LEN_LIMIT chars)
  4. 근거/거절    : out-of-scope questions are declined to the 주민센터, not made up

Runs fully locally. With Ollama present it grades the real model; with the
offline fallback it still runs (the fallback echoes retrieved text, so fact/ko
stay meaningful) - so you get report numbers in any environment.

Usage:  python eval_qa.py           # from the env that has the models
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List

from src.pipeline import WelfareAssistant

LEN_LIMIT = 220          # chars; the system prompt asks for 1-2 sentences
LATIN_RUN = re.compile(r"[A-Za-z]{2,}")   # >=2 latin letters in a row = leakage


@dataclass
class QA:
    q: str
    facts: List[str]     # answer should mention ANY one of these (in-scope)
    scope: str = "in"    # "in" = must answer with facts; "out" = must decline


# Labelled set grounded in data/welfare_docs/daejeon_welfare_sample.md
QA_CASES: List[QA] = [
    QA("기초연금은 어떻게 신청하나요?", ["주민센터", "국민연금공단"]),
    QA("기초연금은 얼마까지 받을 수 있나요?", ["33만", "월", "지급"]),
    QA("노인맞춤돌봄서비스를 받고 싶어요", ["주민센터", "생활지원사", "방문"]),
    QA("혼자 사는데 응급상황이 걱정돼요", ["응급", "119", "감지기", "버튼"]),
    QA("치매 검진 무료로 받고 싶어요", ["치매안심센터", "무료", "검진"]),
    QA("무료로 밥 먹을 수 있는 곳 있나요?", ["급식", "도시락", "경로식당"]),
    QA("겨울 난방비 지원 받을 수 있나요?", ["바우처", "난방", "주민센터"]),
    QA("우울하고 외로울 때 상담받고 싶어요", ["정신건강", "상담", "109"]),
    QA("집에서 혈압 건강 체크 받고 싶어요", ["방문건강", "보건소", "간호사"]),
    QA("노인 일자리를 구하고 싶어요", ["일자리", "시니어클럽", "노인복지관"]),
    QA("지하철 무료로 탈 수 있나요?", ["무임", "지하철", "경로우대", "65세"]),
    QA("응급안전 장비는 돈이 드나요?", ["무료", "설치"]),
    # out-of-scope: the assistant must decline to the 주민센터, not fabricate
    QA("비행기표 싸게 사는 방법 알려줘", ["주민센터", "문의"], scope="out"),
    QA("주식 투자는 어떻게 시작하나요?", ["주민센터", "문의"], scope="out"),
]


def score_answer(qa: QA, answer: str) -> dict:
    a = answer.replace(" ", "")
    ko = LATIN_RUN.search(answer) is None
    concise = len(answer) <= LEN_LIMIT
    if qa.scope == "in":
        fact = any(f.replace(" ", "") in a for f in qa.facts)
        declined = False
        ok = fact and ko and concise
    else:                                  # out-of-scope: correct = a clean refusal
        declined = any(f.replace(" ", "") in a for f in qa.facts)
        fact = declined
        ok = declined and ko and concise
    return {"ok": ok, "fact": fact, "ko": ko, "concise": concise,
            "declined": declined, "len": len(answer)}


def main():
    print("Loading pipeline (indexing welfare docs + LLM)...\n")
    bot = WelfareAssistant()
    print(f"RAG backend : {bot.rag.backend} / {bot.rag.embedder.backend}")
    llm_desc = f"{bot.llm.model}" if bot.llm.available else "offline fallback (echo)"
    print(f"LLM backend : {llm_desc}\n")

    n = len(QA_CASES)
    agg = {"ok": 0, "fact": 0, "ko": 0, "concise": 0}
    total_len = 0
    print("=" * 70)
    print(" ANSWER QUALITY  (질문 -> 생성된 답변 평가)")
    print("=" * 70)
    for qa in QA_CASES:
        # Fresh history each question so the signal layer can't perturb answers.
        bot.reset_conversation()
        ans = bot.ask_text(qa.q).answer.strip()
        s = score_answer(qa, ans)
        for k in agg:
            agg[k] += int(s[k])
        total_len += s["len"]
        tag = "OUT" if qa.scope == "out" else "in "
        flags = (f"{'✓fact' if s['fact'] else '✗fact'} "
                 f"{'✓ko' if s['ko'] else '✗ko'} "
                 f"{'✓len' if s['concise'] else '✗len'}")
        mark = "✓" if s["ok"] else "✗"
        print(f"\n[{mark}][{tag}] {qa.q}   ({flags})")
        print(f"      → {ans[:160]}{'…' if len(ans) > 160 else ''}")

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"  전체 정답률(pass)   : {agg['ok']}/{n} = {agg['ok']/n:.0%}")
    print(f"  정확성(fact hit)    : {agg['fact']}/{n} = {agg['fact']/n:.0%}")
    print(f"  한국어 준수(no eng) : {agg['ko']}/{n} = {agg['ko']/n:.0%}")
    print(f"  간결성(<= {LEN_LIMIT}자)  : {agg['concise']}/{n} = {agg['concise']/n:.0%}")
    print(f"  평균 답변 길이      : {total_len/n:.0f}자")
    if not bot.llm.available:
        print("\n  주의: 지금은 오프라인 폴백(검색결과 요약)입니다. Ollama가 켜진 Mac에서\n"
              "  실행하면 실제 모델 답변 품질이 측정됩니다.")


if __name__ == "__main__":
    main()
