"""
교수님 강의(Week 05 · STEP 09) 방식 RAG — 대화형 챗봇 (대전 복지 데이터).

ask.py 는 질문 하나를 명령줄로 받아 한 번 답하고 끝납니다. 이 파일은 같은
RAG(임베딩·검색·"자료만 근거로" 프롬프트)를 그대로 쓰되, 벡터를 한 번만
불러두고 질문을 계속 받는 챗봇처럼 동작합니다. 어르신이 이어서 묻는 상황을
시연하기에 좋습니다.

  python rag_lecture/chat.py

종료: 'exit' / 'quit' / '종료' / 'q' 입력 또는 Ctrl-C.
"""
import os
import sys

# ask.py 의 검색·답변 로직을 그대로 재사용한다(벡터는 import 시 한 번만 로드).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ask  # noqa: E402  (search, ask, CHAT_MODEL, CHUNKS 재사용)

EXIT_WORDS = {"exit", "quit", "q", "종료", "끝", "그만"}


def main():
    print("=" * 60)
    print(" 대전 독거노인 복지 상담 챗봇 (교수님 강의 RAG 방식)")
    print("=" * 60)
    print(f"  검색 자료 : {len(ask.CHUNKS)}개 조각")
    print(f"  임베딩    : {ask.EMBED_MODEL}   |   답변 : {ask.CHAT_MODEL}")
    print("  질문을 입력하세요. (종료: exit / 종료 / q)")
    print("-" * 60)

    while True:
        try:
            q = input("\n👵 어르신 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n상담을 종료합니다. 건강하세요. 🙏")
            return

        if not q:
            continue
        if q.lower() in EXIT_WORDS:
            print("\n상담을 종료합니다. 건강하세요. 🙏")
            return

        try:
            answer, hits = ask.ask(q)
        except Exception as e:                       # 모델/서버 오류도 끊기지 않게
            print(f"  (오류: {e} — Ollama 실행 여부를 확인하세요)")
            continue

        top_sim = hits[0][2] if hits else 0.0
        print(f"\n🤖 도우미 > {answer.strip()}")
        print(f"\n   근거 (상위 유사도 {top_sim:.3f}):")
        for c, src, s in hits:
            print(f"     [{s:.3f}] ({src}) {c[:60].strip()}…")


if __name__ == "__main__":
    main()
