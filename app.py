"""
Step 5 (web demo): Streamlit UI for the welfare assistant.

Run:  streamlit run app.py

The heavy pipeline (RAG + models) runs in a SEPARATE process (src/worker.py);
this script only exchanges plain dicts with it over a queue. That keeps torch /
chromadb / whisper off Streamlit's worker thread, which otherwise crashes with
a bus error on macOS.

Left panel  : conversation (text, sample questions, voice file upload)
Right panel : live abnormal-signal monitor (status, sentiment, symptoms, alerts)
Everything runs locally; no data leaves the machine.
"""
import streamlit as st

from src import config
from src.worker import AssistantClient

st.set_page_config(page_title="독거노인 복지 안내 · 이상신호 감지",
                   page_icon="👵", layout="wide")

st.markdown("""
<style>
  .block-container{padding-top:2.2rem; max-width:1200px}
  .hero h1{font-size:1.7rem; margin:0 0 .2rem}
  .hero p{color:#6b7a7e; margin:0}
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
       border-bottom:1px solid rgba(128,128,128,.15); font-size:.9rem}
  .sym b{color:#c67a2c}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_client():
    # Started once; keeps the model-loaded worker process alive across reruns.
    return AssistantClient()


with st.spinner("모델을 불러오는 중입니다… (최초 실행은 다소 걸립니다)"):
    client = get_client()

st.session_state.setdefault("chat", [])
st.session_state.setdefault("alerts", [])
st.session_state.setdefault("state", client.state())

SAMPLES = [
    "기초연금은 어떻게 신청하나요?",
    "치매 검진 무료로 받고 싶어요",
    "혼자 사는데 응급상황이 걱정돼요",
    "요즘 무릎이 아파요",
    "무릎이 계속 아프고 잠도 안 와요",
    "너무 외롭고 우울해요",
]


def handle(question: str):
    r = client.ask(question)
    if "error" in r:
        st.error(r["error"])
        return
    st.session_state["chat"].append(("user", r["question"]))
    st.session_state["chat"].append(("bot", r["answer"], sorted(set(r["sources"]))))
    if r["alert"]:
        urgency = "긴급" if r["crisis"] else "주의"
        st.session_state["alerts"].insert(0, (urgency, r["alert"]))
    st.session_state["state"] = r


# ---- header --------------------------------------------------------------
st.markdown('<div class="hero"><h1>👵 대전 독거노인 복지 안내 · 이상신호 감지</h1>'
            '<p>On-device LLM + RAG · 모든 처리는 로컬에서 실행됩니다 (외부 전송 없음)</p></div>',
            unsafe_allow_html=True)
st.write("")

# ---- sidebar -------------------------------------------------------------
with st.sidebar:
    m = client.meta
    st.subheader("⚙️ 시스템 상태")
    st.caption(f"RAG · {m.get('rag_backend')} / {m.get('embedder')}")
    st.caption(f"LLM · {'Ollama 실행 중' if m.get('llm') else '미실행 → 템플릿 대체'}")
    st.caption(f"감정 분석 · {m.get('sentiment')}")
    st.caption(f"알림 채널 · {', '.join(m.get('alert_channels', []))}")
    st.divider()
    st.subheader("판정 임계값")
    st.caption(f"증상 반복 ≥ {config.SYMPTOM_REPEAT_THRESHOLD}회")
    st.caption(f"부정 감정 평균 ≥ {config.NEGATIVE_SENTIMENT_THRESHOLD}")
    st.caption("위기 키워드 → 즉시 긴급")
    st.divider()
    if st.button("🔄 대화 초기화", use_container_width=True):
        st.session_state["state"] = client.reset()
        st.session_state["chat"] = []
        st.session_state["alerts"] = []
        st.rerun()

# ---- main: two columns ---------------------------------------------------
left, right = st.columns([1.4, 1])

with left:
    st.markdown("#### 💬 대화")
    st.caption("예시 질문을 클릭하거나 직접 입력하세요.")
    cols = st.columns(3)
    for i, s in enumerate(SAMPLES):
        if cols[i % 3].button(s, key=f"s{i}", use_container_width=True):
            handle(s)
            st.rerun()

    if q := st.chat_input("어르신 질문을 입력하세요…"):
        handle(q)
        st.rerun()

    with st.expander("🎤 음성 파일로 질문 (wav/mp3)"):
        audio = st.file_uploader("음성 업로드", type=["wav", "mp3", "m4a"],
                                 label_visibility="collapsed")
        if st.button("음성 인식 후 질문") and audio is not None:
            from pathlib import Path
            tmp = Path("data/audio") / audio.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(audio.read())
            r = client.ask_audio(str(tmp))
            if "error" in r:
                st.error(r["error"])
            else:
                st.info(f"음성 인식 결과: {r['question']}")
                st.session_state["chat"].append(("user", r["question"]))
                st.session_state["chat"].append(("bot", r["answer"],
                                                sorted(set(r["sources"]))))
                if r["alert"]:
                    urg = "긴급" if r["crisis"] else "주의"
                    st.session_state["alerts"].insert(0, (urg, r["alert"]))
                st.session_state["state"] = r
                st.rerun()

    st.divider()
    if not st.session_state["chat"]:
        st.caption("아직 대화가 없습니다. 위에서 질문을 시작하세요.")
    for msg in st.session_state["chat"]:
        if msg[0] == "user":
            with st.chat_message("user"):
                st.write(msg[1])
        else:
            with st.chat_message("assistant"):
                st.write(msg[1])
                if msg[2]:
                    st.caption("📄 근거 문서: " + ", ".join(msg[2]))

with right:
    st.markdown("#### 🩺 이상신호 모니터")
    s = st.session_state["state"]

    if s.get("crisis"):
        st.markdown('<span class="pill crit">🚨 긴급 · 즉시 확인</span>', unsafe_allow_html=True)
    elif s.get("is_abnormal"):
        st.markdown('<span class="pill warn">⚠️ 주의 · 이상신호</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill ok">✅ 정상</span>', unsafe_allow_html=True)

    st.write("")
    st.metric("평균 부정 감정", f"{s.get('avg_negative', 0.0):.2f}",
              help=f"기준 {config.NEGATIVE_SENTIMENT_THRESHOLD} 이상이면 플래그")
    st.progress(min(s.get("avg_negative", 0.0), 1.0))
    st.caption(f"대화 {s.get('history_len', 0)}회 누적")

    st.write("")
    st.markdown("**반복 호소 증상**")
    counts = s.get("symptom_counts", {})
    if counts:
        for grp, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            flag = " ⚠️" if cnt >= config.SYMPTOM_REPEAT_THRESHOLD else ""
            st.markdown(f'<div class="sym"><span>{grp}</span>'
                        f'<span><b>{cnt}회</b>{flag}</span></div>', unsafe_allow_html=True)
    else:
        st.caption("감지된 증상 없음")

    st.write("")
    st.markdown("**🔔 사회복지사 알림 로그**")
    if st.session_state["alerts"]:
        for urgency, text in st.session_state["alerts"]:
            cls = "crit" if urgency == "긴급" else "warn"
            st.markdown(f'<span class="badge {cls}">{urgency}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="alertbox">{text}</div>', unsafe_allow_html=True)
    else:
        st.caption("아직 알림이 없습니다.")
