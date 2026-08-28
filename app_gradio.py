"""
Gradio web demo (voice + text), two roles.

  👵 어르신 : a chat with ONE input bar that takes both typed text and a
             microphone recording (gr.MultimodalTextbox). With voice mode on,
             the answer is spoken back (local TTS) - voice-to-voice.
  🧑‍⚕️ 관리자 : a background monitor (status, anomaly metrics, alert log) that
             the elderly user never sees; auto-refreshes on a timer.

The heavy pipeline runs in a separate process (src/worker_server.py) via
AssistantClient, so no ML libraries load inside this UI process.

Run:  python -m app_gradio           (or: python app_gradio.py)
"""
from __future__ import annotations
import json
from pathlib import Path

import gradio as gr

from src import config
from src.tts import TextToSpeech
from src.worker import AssistantClient

client = AssistantClient()      # starts the worker subprocess (loads models once)
tts = TextToSpeech()
_reply_n = 0


# --------------------------------------------------------------------------
# Elderly chat handlers
# --------------------------------------------------------------------------
def on_submit(message: dict, history: list, voice_on: bool):
    """message = {"text": str, "files": [paths]} from MultimodalTextbox."""
    global _reply_n
    text = (message or {}).get("text", "").strip()
    files = (message or {}).get("files", [])

    if files:                                   # a microphone recording
        r = client.ask_audio(files[-1])
    elif text:
        r = client.ask(text)
    else:
        return history, None, gr.MultimodalTextbox(value=None)

    if "error" in r:
        history = history + [{"role": "assistant", "content": f"⚠️ {r['error']}"}]
        return history, None, gr.MultimodalTextbox(value=None)

    history = history + [
        {"role": "user", "content": r["question"]},
        {"role": "assistant", "content": r["answer"]},
    ]

    reply_audio = None
    if voice_on:
        _reply_n += 1
        out = Path("data/audio") / f"reply_{_reply_n}.m4a"
        out.parent.mkdir(parents=True, exist_ok=True)
        reply_audio = tts.synthesize(r["answer"], str(out))

    return history, reply_audio, gr.MultimodalTextbox(value=None)


def reset_chat():
    client.reset()
    return [], None, gr.MultimodalTextbox(value=None)


# --------------------------------------------------------------------------
# Admin monitor
# --------------------------------------------------------------------------
def _read_alerts(limit: int = 5) -> list:
    path = config.ALERT_LOG_DIR / "alert_log.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[-limit:][::-1]


def refresh_monitor():
    s = client.state()
    if s.get("crisis"):
        status = "### 🚨 긴급 · 즉시 확인"
    elif s.get("is_abnormal"):
        status = "### ⚠️ 주의 · 이상신호"
    else:
        status = "### ✅ 정상"

    lines = [status, ""]
    lines.append(f"**평균 부정 감정:** {s.get('avg_negative', 0.0):.2f}  ·  "
                 f"대화 {s.get('history_len', 0)}회 누적")
    counts = s.get("symptom_counts", {})
    if counts:
        lines.append("**반복 호소 증상:** " +
                     ", ".join(f"{k}({v}회)" for k, v in counts.items()))
    m = s.get("metrics", {})
    if m:
        lines.append(f"**이상징후 지표:** 대화빈도 -{m['freq_drop']:.0%} · "
                     f"감정 +{m['sentiment_shift']:.2f} · 외로움 +{m['keyword_shift']:.0%} · "
                     f"응답길이 -{m['length_drop']:.0%}")
    status_md = "\n\n".join(lines)

    alerts = _read_alerts()
    if alerts:
        blocks = []
        for a in alerts:
            blocks.append(f"```\n{a['alert']}\n```")
        alert_md = "#### 🔔 사회복지사 알림 로그\n" + "\n".join(blocks)
    else:
        alert_md = "#### 🔔 사회복지사 알림 로그\n아직 알림이 없습니다."
    return status_md, alert_md


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def build():
    with gr.Blocks(title="독거노인 복지 안내 · 이상신호 감지") as demo:
        gr.Markdown("# 👵 대전 독거노인 복지 안내 · 이상신호 감지\n"
                    "On-device LLM + RAG · 모든 처리는 로컬에서 실행됩니다 (외부 전송 없음)")

        with gr.Tab("👵 어르신 (사용자)"):
            voice_on = gr.Checkbox(value=True, label="🔊 음성으로 답변 듣기 (voice-to-voice)")
            chatbot = gr.Chatbot(height=440, label="대화")
            reply_audio = gr.Audio(autoplay=True, visible=True, label="음성 답변",
                                   interactive=False)
            box = gr.MultimodalTextbox(
                sources=["microphone"], file_types=["audio"],
                placeholder="궁금한 복지 서비스를 말씀하거나 입력하세요…",
                label="", show_label=False)
            gr.Examples(
                examples=[{"text": "기초연금은 어떻게 신청하나요?", "files": []},
                          {"text": "치매 검진 무료로 받고 싶어요", "files": []},
                          {"text": "요즘 무릎이 아파요", "files": []},
                          {"text": "너무 외롭고 우울해요", "files": []}],
                inputs=box)
            clear = gr.Button("🔄 대화 초기화", size="sm")

            box.submit(on_submit, [box, chatbot, voice_on], [chatbot, reply_audio, box])
            clear.click(reset_chat, None, [chatbot, reply_audio, box])

        with gr.Tab("🧑‍⚕️ 관리자 (사회복지사)"):
            gr.Markdown("어르신 화면에는 보이지 않는 배경 모니터링입니다. (자동 새로고침)")
            status_md = gr.Markdown("### ✅ 정상")
            alerts_md = gr.Markdown("#### 🔔 사회복지사 알림 로그\n아직 알림이 없습니다.")
            timer = gr.Timer(2.0)
            timer.tick(refresh_monitor, None, [status_md, alerts_md])

    return demo


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
