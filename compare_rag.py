"""
RAG 효과 시연 — 같은 질문을 'RAG 없이(LLM만)'와 'RAG 사용'으로 나란히 답해 비교.

발표용: 같은 모델, 같은 질문인데 RAG(대전 공공문서 근거)가 있으면 정확하고
출처가 있으며, 없으면 일반 지식으로 부정확하거나 지어낸다는 것을 보여준다.

  python compare_rag.py                       # 기본 질문 세트
  python compare_rag.py "기초연금은 어떻게 신청하나요?"   # 질문 하나

같은 RAG 인덱스(대전 공공문서, 실데이터)를 쓰며, 답변 모델은 config의
LLM_BACKEND를 따른다(API 또는 로컬 Ollama).
"""
from __future__ import annotations
import sys

from src import config
from src.llm import ANSWER_SYSTEM_PROMPT, build_answer_prompt
from src.pipeline import WelfareAssistant

# RAG 없이 물을 때의 시스템 프롬프트: 문서를 주지 않으므로 모델은 자기 지식으로만
# 답한다(=일반 챗봇). 대전 구체 정보가 없거나 환각이 나는 걸 그대로 보여준다.
NO_RAG_SYSTEM = ("당신은 복지 상담 도우미입니다. 사용자의 질문에 한국어로 "
                 "2~3문장으로 답하세요.")

DEFAULT_QUESTIONS = [
    "대전 기초연금은 어떻게 신청하나요?",
    "혼자 사는데 응급상황이 걱정돼요. 도움받을 수 있나요?",
    "비행기표 싸게 사는 방법 알려줘",          # 범위 밖 — RAG는 거절해야 정답
]


def no_rag_answer(bot: WelfareAssistant, q: str) -> str:
    """검색 없이 모델에게 직접 질문 (RAG 미사용)."""
    if not bot.llm.available:
        return "(LLM 미연결 — 로컬/ API 설정을 확인하세요)"
    return bot.llm.generate(q, system=NO_RAG_SYSTEM).strip()


def rag_answer(bot: WelfareAssistant, q: str):
    """검색 → 관련성 게이트 → 근거 기반 답변 (RAG 사용)."""
    retrieved = bot.rag.retrieve(q)
    top = retrieved[0].score if retrieved else 0.0
    if not retrieved or top < config.RAG_MIN_RELEVANCE:
        from src.llm import OFF_DOMAIN_REPLY
        return OFF_DOMAIN_REPLY, []               # 자료에 없음 → 안전하게 안내
    contexts = [r.text for r in retrieved]
    answer = bot.llm.generate(build_answer_prompt(q, contexts),
                              system=ANSWER_SYSTEM_PROMPT).strip()
    return answer, sorted(set(r.source for r in retrieved))


def main():
    questions = sys.argv[1:] or DEFAULT_QUESTIONS
    print("파이프라인 로딩 (대전 공공문서 인덱싱)...\n")
    bot = WelfareAssistant()
    print(f"RAG 인덱스 : {bot.rag.backend} / {bot.rag.embedder.backend}")
    print(f"답변 모델  : {type(bot.llm).__name__}:{bot.llm.model} "
          f"({'on' if bot.llm.available else 'off'})\n")

    for q in questions:
        print("=" * 74)
        print(f"❓ 질문: {q}\n")
        print("─" * 74)
        print("❌ RAG 미사용 (LLM만, 문서 근거 없음)")
        print(f"   {no_rag_answer(bot, q)}\n")
        ans, srcs = rag_answer(bot, q)
        print("✅ RAG 사용 (대전 공공문서 근거)")
        print(f"   {ans}")
        if srcs:
            print(f"   📄 근거: {', '.join(srcs)}")
        print()


if __name__ == "__main__":
    main()
