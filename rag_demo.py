"""
RAG 데모 — 같은 질문을 'RAG 없이(LLM만)'와 'RAG 사용'으로 나란히 보여준다.

발표용 시연 스크립트. 같은 모델·같은 질문인데, 검색 근거(RAG)를 주면 대전
공공문서에 근거해 정확히(출처와 함께) 답하고, 주지 않으면 일반 지식으로
막연하거나 지어낸다는 것을 한눈에 비교한다.

  python rag_demo.py                      # 효과가 잘 드러나는 기본 질문 3개
  python rag_demo.py "질문 하나"           # 직접 질문

로컬 모델로 실행:
  LLM_BACKEND=ollama LLM_MODEL=exaone3.5:2.4b python rag_demo.py
"""
import os
os.environ.setdefault("USE_AGENT", "0")
os.environ.setdefault("USE_LLM_SIGNAL", "0")

import sys

from src import config
from src.llm import ANSWER_SYSTEM_PROMPT, OFF_DOMAIN_REPLY, build_answer_prompt
from src.pipeline import WelfareAssistant

# RAG 없이 물을 때: 문서를 주지 않으므로 모델은 자기 지식으로만 답한다.
NO_RAG_SYSTEM = ("당신은 복지 상담 도우미입니다. 질문에 한국어로 "
                 "2~3문장으로 답하세요.")

# RAG 효과가 잘 드러나는 기본 질문 (대전 고유 정보 + 범위 밖 1개).
DEFAULT_QUESTIONS = [
    "대전 재가노인 식사배달 대상은 누구인가요?",       # 실제 CSV: 60세·저소득
    "대전 노인맞춤돌봄서비스 수행기관은 어디인가요?",   # 실제 CSV: 구별 기관명
    "주식 투자로 돈 버는 방법 알려줘",                 # 범위 밖 → 거절해야 정답
]


def answer_without_rag(bot, q: str) -> str:
    if not bot.llm.available:
        return "(LLM 미연결 — 설정을 확인하세요)"
    return bot.llm.generate(q, system=NO_RAG_SYSTEM).strip()


def answer_with_rag(bot, q: str):
    hits = bot.rag.retrieve(q)
    top = hits[0].score if hits else 0.0
    if not hits or top < config.RAG_MIN_RELEVANCE:
        return OFF_DOMAIN_REPLY, []                 # 근거 없음 → 안전 안내
    contexts = [h.text for h in hits]
    ans = bot.llm.generate(build_answer_prompt(q, contexts),
                           system=ANSWER_SYSTEM_PROMPT).strip()
    if not ans:
        ans = "(검색된 근거에 해당 내용이 부족합니다)"
    return ans, sorted(set(h.source for h in hits))


def main():
    questions = sys.argv[1:] or DEFAULT_QUESTIONS
    print("파이프라인 로딩 (대전 공공문서 RAG)...\n")
    bot = WelfareAssistant()
    print(f"답변 모델 : {type(bot.llm).__name__}:{bot.llm.model}")
    print(f"RAG 인덱스 : {bot.rag.backend} / {bot.rag.embedder.backend}\n")

    for q in questions:
        print("=" * 74)
        print(f"[질문] {q}\n")
        print("[RAG 미사용 — LLM만]")
        print(f"  {answer_without_rag(bot, q)}\n")
        ans, srcs = answer_with_rag(bot, q)
        print("[RAG 사용 — 대전 공공문서 근거]")
        print(f"  {ans}")
        if srcs:
            print(f"  출처: {', '.join(srcs)}")
        print()


if __name__ == "__main__":
    main()
