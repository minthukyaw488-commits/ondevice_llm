"""
Step 5 (web demo): Streamlit UI for the welfare assistant.

Run:  streamlit run app.py

Features
  - text or microphone-file input (voice)
  - welfare answer with source documents
  - live abnormal-signal panel + social-worker alert log

Everything runs locally; no data leaves the machine.
"""
import streamlit as st

from src.pipeline import WelfareAssistant

st.set_page_config(page_title="독거노인 복지 안내 · 이상신호 감지", page_icon="👵")


@st.cache_resource
def get_bot():
    # Cached so the welfare index is built only once per session.
    return WelfareAssistant(user_name="데모 어르신")


bot = get_bot()

st.title("👵 대전 독거노인 복지 안내 · 이상신호 감지")
st.caption("On-device LLM + RAG · 모든 처리는 로컬에서 실행됩니다 (외부 전송 없음)")

with st.sidebar:
    st.header("시스템 상태")
    st.write(f"**RAG**: {bot.rag.backend} / {bot.rag.embedder.backend}")
    st.write(f"**LLM (Ollama)**: {'실행 중' if bot.llm.available else '미실행 → 템플릿 대체'}")
    st.write(f"**감정 분석**: {bot.detector.sentiment.backend}")
    st.divider()
    st.subheader("🚨 이상신호 알림 로그")
    for a in st.session_state.get("alerts", []):
        st.error(a)
    if not st.session_state.get("alerts"):
        st.write("아직 감지된 이상신호가 없습니다.")

if "alerts" not in st.session_state:
    st.session_state["alerts"] = []
if "chat" not in st.session_state:
    st.session_state["chat"] = []

# --- input ---------------------------------------------------------------
tab_text, tab_voice = st.tabs(["⌨️ 텍스트 입력", "🎤 음성 파일 입력"])

question = None
with tab_text:
    q = st.text_input("어르신 질문을 입력하세요", key="text_q",
                      placeholder="예) 기초연금은 어떻게 신청하나요?")
    if st.button("질문하기", key="ask_text") and q.strip():
        question = q.strip()

with tab_voice:
    audio = st.file_uploader("음성 파일 업로드 (wav/mp3)", type=["wav", "mp3", "m4a"])
    if st.button("음성 인식 후 질문", key="ask_voice") and audio is not None:
        from pathlib import Path
        from src.stt import SpeechToText
        tmp = Path("data/audio") / audio.name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(audio.read())
        question = SpeechToText().transcribe_file(tmp)
        st.info(f"음성 인식 결과: {question}")

# --- process -------------------------------------------------------------
if question:
    res = bot.ask_text(question)
    st.session_state["chat"].append(("user", res.question))
    st.session_state["chat"].append(("bot", res.answer, sorted(set(res.sources))))
    if res.alert:
        st.session_state["alerts"].insert(0, res.alert)

# --- conversation --------------------------------------------------------
st.divider()
for msg in st.session_state["chat"]:
    if msg[0] == "user":
        with st.chat_message("user"):
            st.write(msg[1])
    else:
        with st.chat_message("assistant"):
            st.write(msg[1])
            st.caption("📄 근거 문서: " + ", ".join(msg[2]))

if st.session_state["alerts"]:
    st.divider()
    st.subheader("🚨 최신 사회복지사 알림")
    st.error(st.session_state["alerts"][0])
