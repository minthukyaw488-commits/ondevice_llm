"""
Step 4: Full integration pipeline.

    (voice or text)
         |
      STT (Whisper)            [voice input only]
         |
   +-----+------------------------------+
   |                                     |
 RAG search                     Abnormal-signal analysis
 (welfare chunks)               (sentiment + symptom repetition)
   |                                     |
   +-----------------+-------------------+
                     |
              Local LLM (Ollama)
                     |
        answer  +  (if abnormal) social-worker alert summary

Everything runs locally. Conversation content never leaves the device.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from . import config
from .abnormal_signal import AbnormalSignalDetector, SignalResult
from .alerts import AlertDispatcher
from .llm import ANSWER_SYSTEM_PROMPT, LocalLLM, build_answer_prompt
from .rag_pipeline import RagPipeline


@dataclass
class TurnResult:
    question: str
    answer: str
    sources: List[str]
    signal: SignalResult
    alert: Optional[str]        # social-worker summary, only if abnormal


class WelfareAssistant:
    """One long-lived object per elderly user (keeps conversation history)."""

    def __init__(self, user_name: str = "어르신"):
        self.user_name = user_name
        self.rag = RagPipeline()
        self.rag.index()                         # build the welfare index once
        self.llm = LocalLLM()
        self.detector = AbnormalSignalDetector()  # accumulates history
        self.alerts = AlertDispatcher()           # local log (+ opt-in channels)

    # -- main entry points -------------------------------------------------
    def ask_text(self, question: str) -> TurnResult:
        # Layer 1 (RAG) and Layer 2 (abnormal signal) run on the same input.
        retrieved = self.rag.retrieve(question)
        contexts = [r.text for r in retrieved]
        sources = [r.source for r in retrieved]

        signal = self.detector.add_utterance(question)

        prompt = build_answer_prompt(question, contexts)
        answer = self.llm.generate(prompt, system=ANSWER_SYSTEM_PROMPT)

        alert = None
        if signal.is_abnormal:
            alert = self._build_alert(signal)
            urgency = "긴급" if signal.crisis else "주의"
            self.alerts.dispatch(self.user_name, urgency, alert)  # log + notify
        return TurnResult(question, answer, sources, signal, alert)

    def reset_conversation(self) -> None:
        """Clear the conversation history for a new session/demo.

        Reuses the already-loaded sentiment model so no reload is needed.
        """
        self.detector = AbnormalSignalDetector(sentiment=self.detector.sentiment)

    def ask_audio(self, audio_path: str) -> TurnResult:
        from .stt import SpeechToText
        text = SpeechToText().transcribe_file(audio_path)
        return self.ask_text(text)

    # -- social-worker alert ----------------------------------------------
    def _build_alert(self, signal: SignalResult) -> str:
        """Generate a concise Korean alert summary for the 사회복지사."""
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        urgency = "긴급" if signal.crisis else "주의"
        lines = [
            f"[{urgency}] 독거노인 이상신호 알림 - {self.user_name}",
            f"발생시각: {stamp}",
            f"대화 횟수: {len(self.detector.history)}회 / 평균 부정감정: "
            f"{signal.avg_negative:.2f}",
            "감지 근거:",
        ]
        lines += [f"  - {r}" for r in signal.reasons]
        if signal.symptom_counts:
            top = ", ".join(f"{k}({v}회)" for k, v in signal.symptom_counts.items())
            lines.append(f"반복 호소 증상: {top}")
        recent = self.detector.history[-3:]
        lines.append("최근 발화: " + " / ".join(f'"{u}"' for u in recent))
        action = ("자살예방상담(109) 연계 및 즉시 방문 확인 권장"
                  if signal.crisis else "안부 확인 방문 또는 상담 권장")
        lines.append(f"권장 조치: {action}")
        return "\n".join(lines)


if __name__ == "__main__":
    print("Initialising assistant (indexing welfare docs)...\n")
    bot = WelfareAssistant(user_name="김OO 어르신")
    print(f"RAG={bot.rag.backend}/{bot.rag.embedder.backend} | "
          f"LLM(Ollama)={'on' if bot.llm.available else 'off (fallback)'} | "
          f"sentiment={bot.detector.sentiment.backend}\n")

    conversation = [
        "기초연금은 어떻게 신청하나요?",
        "무릎이 아픈데 도움받을 수 있는 곳이 있나요?",
        "요즘 무릎도 아프고 잠도 안 와요.",
        "무릎이 계속 아프고 혼자라 너무 외롭고 우울해요.",
    ]
    for q in conversation:
        res = bot.ask_text(q)
        print("=" * 70)
        print(f"👵 질문: {res.question}")
        print(f"🤖 답변: {res.answer[:220]}")
        print(f"   📄 근거: {res.sources}")
        if res.alert:
            print("\n🚨 사회복지사 알림 -----------------------------")
            print(res.alert)
        print()
