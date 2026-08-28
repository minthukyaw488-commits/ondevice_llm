"""
Web demo: two-role interface.

  👵 어르신 (user)      : a clean voice/text chatbot. Answers only - no monitoring
                          is shown, so the elderly user just has a friendly chat.
  🧑‍⚕️ 관리자 (social worker): a background dashboard that watches the same
                          conversation for abnormal signals and shows the alerts.

The heavy pipeline runs in a separate process (src/worker_server.py). Launch
with the SAME interpreter that has the models installed:

    python -m streamlit run app.py
"""
import streamlit as st

from src import config
from src.worker import AssistantClient

CSS = """
<style>
  .block-container{padding-top:2rem; max-width:900px}
  .hero h1{font-size:1.7rem; margin:0 0 .2rem}
  .hero p{color:#6b7a7e; margin:0}
  .stChatMessage{font-size:1.05rem}
  .pill{display:inline-block; padding:.5rem 1rem; border-radius:999px;
        font-weight:700; font-size:1.05rem}
  .pill.ok{background:#e6f4ec; color:#256b47}
  .pill.warn{background:#fbeede; color:#9a5a12}
  .pill.crit{background:#fbe3e1; color:#b23a3a}
  .badge{display:inline-block; padding:.15rem .55rem; border-radius:6px;
         font-size:.72rem; font-weight:700; margin-right:.4rem}
  .badge.warn{background:#fbeede; color:#9a5a12}
  .badge.crit{background:#fbe3e1; color:#b23a3a}
  .alertbox{border-left:4px solid #b23a3a; background:rgba(178,58,58,.06);
            padding:.7rem .9rem; border-radius:8px; font-size:.82rem;
            white-space:pre-wrap; margin-bottom:.6rem; font-family:monospace}
  .sym{display:flex; justify-content:space-between; padding:.25rem 0;
       border-bottom:1px solid rgba(128,128,128,.15); font-size:.95rem}
  .sym b{color:#c67a2c}
</style>
"""

SAMPLES = [
    "기초연금은 어떻게 신청하나요?",
    "치매 검진 무료로 받고 싶어요",
    "요즘 무릎이 아파요",
    "너무 외롭고 우울해요",
]


@st.cache_resource
def get_client():
    return AssistantClient()


def _store(r: dict):
    """Record a turn result into session state (chat + alert log + monitor)."""
    st.session_state["chat"].append(("user", r["question"]))
    st.session_state["chat"].append(("bot", r["answer"], sorted(set(r["sources"]))))
    if r.get("alert"):
        urgency = "긴급" if r.get("crisis") else "주의"
        st.session_state["alerts"].insert(0, (urgency, r["alert"]))
    st.session_state["state"] = r


def ask_text(client, question: str):
    r = client.ask(question)
    if "error" in r:
        st.error(r["error"])
        return
    _store(r)


def ask_audio(client, wav_bytes: bytes):
    from pathlib import Path
    tmp = Path("data/audio") / "mic_input.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(wav_bytes)
    r = client.ask_audio(str(tmp))
    if "error" in r:
        st.error(r["error"])
        return
    _store(r)


# ---- 어르신 (user) view --------------------------------------------------
def user_view(client):
    st.markdown('<div class="hero"><h1>👵 복지 도우미</h1>'
                '<p>궁금한 복지 서비스를 말씀하거나 입력해 주세요.</p></div>',
                unsafe_allow_html=True)
    st.write("")

    for msg in st.session_state["chat"]:
        if msg[0] == "user":
            with st.chat_message("user"):
                st.write(msg[1])
        else:
            with st.chat_message("assistant"):
                st.write(msg[1])
    if not st.session_state["chat"]:
        cols = st.columns(2)
        for i, s in enumerate(SAMPLES):
            if cols[i % 2].button(s, key=f"s{i}", use_container_width=True):
                ask_text(client, s)
                st.rerun()

    # Live microphone (records in the browser, sent to local Whisper).
    audio = st.audio_input("🎤 마이크로 말씀하세요", key="mic")
    if audio is not None:
        data = audio.getvalue()
        fp = hash(data)
        if data and st.session_state.get("last_mic") != fp:
            st.session_state["last_mic"] = fp
            with st.spinner("음성을 인식하는 중…"):
                ask_audio(client, data)
            st.rerun()

    if q := st.chat_input("여기에 입력하세요…"):
        ask_text(client, q)
        st.rerun()


# ---- 관리자 (social worker) view ----------------------------------------
def admin_view(client):
    st.markdown('<div class="hero"><h1>🧑‍⚕️ 관리자 · 이상신호 모니터</h1>'
                '<p>어르신 화면에는 보이지 않는 배경 모니터링입니다.</p></div>',
                unsafe_allow_html=True)
    st.write("")
    s = st.session_state["state"]

    left, right = st.columns([1, 1])
    with left:
        if s.get("crisis"):
            st.markdown('<span class="pill crit">🚨 긴급 · 즉시 확인</span>', unsafe_allow_html=True)
        elif s.get("is_abnormal"):
            st.markdown('<span class="pill warn">⚠️ 주의 · 이상신호</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill ok">✅ 정상</span>', unsafe_allow_html=True)
        st.write("")
        st.metric("평균 부정 감정", f"{s.get('avg_negative', 0.0):.2f}")
        st.progress(min(s.get("avg_negative", 0.0), 1.0))
        st.caption(f"대화 {s.get('history_len', 0)}회 누적")

    with right:
        st.markdown("**반복 호소 증상**")
        counts = s.get("symptom_counts", {})
        if counts:
            for grp, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                flag = " ⚠️" if cnt >= config.SYMPTOM_REPEAT_THRESHOLD else ""
                st.markdown(f'<div class="sym"><span>{grp}</span>'
                            f'<span><b>{cnt}회</b>{flag}</span></div>', unsafe_allow_html=True)
        else:
            st.caption("감지된 증상 없음")

    metrics = s.get("metrics", {})
    if metrics:
        st.write("")
        st.markdown("**이상징후 지표 (최근 vs 평소)**")
        c = st.columns(4)
        c[0].metric("대화 빈도", f"-{metrics['freq_drop']:.0%}")
        c[1].metric("감정 변화", f"+{metrics['sentiment_shift']:.2f}")
        c[2].metric("외로움 표현", f"+{metrics['keyword_shift']:.0%}")
        c[3].metric("응답 길이", f"-{metrics['length_drop']:.0%}")

    st.divider()
    st.markdown("#### 🔔 사회복지사 알림 로그")
    if st.session_state["alerts"]:
        for urgency, text in st.session_state["alerts"]:
            cls = "crit" if urgency == "긴급" else "warn"
            st.markdown(f'<span class="badge {cls}">{urgency}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="alertbox">{text}</div>', unsafe_allow_html=True)
    else:
        st.caption("아직 알림이 없습니다.")

    with st.expander("💬 대화 내용 (모니터링용)"):
        for msg in st.session_state["chat"]:
            who = "어르신" if msg[0] == "user" else "도우미"
            st.markdown(f"**{who}:** {msg[1]}")


def main():
    st.set_page_config(page_title="독거노인 복지 안내 · 이상신호 감지",
                       page_icon="👵", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.spinner("모델을 불러오는 중입니다… (최초 실행은 다소 걸립니다)"):
        client = get_client()

    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("alerts", [])
    st.session_state.setdefault("state", client.state())

    with st.sidebar:
        view = st.radio("화면 선택", ["👵 어르신 (사용자)", "🧑‍⚕️ 관리자 (사회복지사)"])
        st.divider()
        if view.startswith("🧑"):
            m = client.meta
            st.subheader("⚙️ 시스템 상태")
            st.caption(f"RAG · {m.get('rag_backend')} / {m.get('embedder')}")
            st.caption(f"LLM · {'Ollama' if m.get('llm') else '템플릿 대체'}")
            st.caption(f"감정 · {m.get('sentiment')}")
            st.caption(f"알림 · {', '.join(m.get('alert_channels', []))}")
            st.metric("누적 알림", f"{len(st.session_state['alerts'])}건")
        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state["state"] = client.reset()
            st.session_state["chat"] = []
            st.session_state["alerts"] = []
            st.session_state.pop("last_mic", None)
            st.rerun()

    if view.startswith("👵"):
        user_view(client)
    else:
        admin_view(client)


# Skip execution when re-imported by a multiprocessing spawn bootstrap.
if __name__ != "__mp_main__":
    main()
