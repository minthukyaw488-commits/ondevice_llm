"""
Step 5 (terminal demo): interactive welfare assistant.

Usage:
    python demo_cli.py                # text chat
    python demo_cli.py --audio a.wav  # transcribe an audio file, then chat once

Type Korean questions. If an abnormal signal is detected, the social-worker
alert is printed. Type 'quit' / '종료' to exit.
"""
import sys

from src.pipeline import WelfareAssistant


def banner(bot: WelfareAssistant):
    print("=" * 62)
    print(" 대전 독거노인 복지 안내 · 이상신호 감지 데모 (GPT-4o + 대전 RAG)")
    print("=" * 62)
    print(f" RAG        : {bot.rag.backend} / {bot.rag.embedder.backend}")
    print(f" LLM        : {type(bot.llm).__name__}:{bot.llm.model} "
          f"({'실행 중' if bot.llm.available else '미실행 → 템플릿 대체'})")
    print(f" 감정 분석  : {bot.detector.sentiment.backend}")
    print(" 질문을 입력하세요. 종료하려면 'quit' 또는 '종료'.\n")


def show(res):
    print(f"\n🤖 {res.answer}")
    print(f"   📄 근거 문서: {', '.join(sorted(set(res.sources)))}")
    if res.alert:
        print("\n" + "🚨" * 20)
        print(res.alert)
        print("🚨" * 20)
    print()


def main():
    bot = WelfareAssistant(user_name="데모 어르신")
    banner(bot)

    if len(sys.argv) >= 3 and sys.argv[1] == "--audio":
        res = bot.ask_audio(sys.argv[2])
        print(f"👵 (음성 인식): {res.question}")
        show(res)
        return

    while True:
        try:
            q = input("👵 질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "종료"} or not q:
            break
        show(bot.ask_text(q))
    print("데모를 종료합니다.")


if __name__ == "__main__":
    main()
