"""
대전광역시 복지포털 (demo) - Streamlit.

A public-service style welfare portal for elderly residents of Daejeon:
official-style header + nav, a prominent voice AI welfare assistant, welfare
search, large service-category cards, popular services, announcements,
accessibility controls, and an official footer. A hidden 관리자 view holds the
abnormal-signal monitor.

Run with the SAME interpreter that has the models installed:
    python -m streamlit run app.py
"""
import streamlit as st

from src import config
from src.tts import TextToSpeech
from src.worker import AssistantClient

st.set_page_config(page_title="대전광역시 복지포털", page_icon="🏛️", layout="wide",
                   initial_sidebar_state="expanded")

# --- service categories & content ----------------------------------------
CATEGORIES = [
    ("💰", "기초연금", "매월 연금 지원", "기초연금은 어떻게 신청하나요?"),
    ("🤝", "노인맞춤돌봄", "생활지원사 방문·안부", "노인맞춤돌봄서비스를 받고 싶어요"),
    ("🩺", "방문건강관리", "간호사 가정 방문", "집에서 혈압 건강관리 받고 싶어요"),
    ("🧠", "치매안심", "무료 검진·상담", "치매 검진 무료로 받고 싶어요"),
    ("🚨", "응급안전안심", "화재·응급 감지기", "혼자 사는데 응급상황이 걱정돼요"),
    ("👷", "노인일자리", "일자리·사회활동", "노인 일자리를 구하고 싶어요"),
]
POPULAR = ["기초연금", "치매 검진", "노인일자리", "에너지 바우처", "경로우대", "방문간호"]
NOTICES = [
    ("2026-08-20", "2026년 기초연금 선정기준액 인상 안내"),
    ("2026-08-14", "겨울철 에너지바우처 신청 접수 시작"),
    ("2026-08-05", "치매안심센터 무료 조기검진 프로그램 운영"),
    ("2026-07-28", "노인맞춤돌봄서비스 신규 대상자 모집"),
]


@st.cache_resource
def get_client():
    return AssistantClient()


@st.cache_resource
def get_tts():
    return TextToSpeech()


# --------------------------------------------------------------------------
# Styling (navy public-service palette, large type, high contrast)
# --------------------------------------------------------------------------
def inject_css():
    scale = st.session_state.get("font_scale", 1.0)
    hc = st.session_state.get("high_contrast", False)
    base = 17 * scale
    ink = "#0a0a0a" if hc else "#1b2733"
    navy = "#0b2e63"
    navy2 = "#103a7d"
    accent = "#1a56b0"
    line = "#000000" if hc else "#d6dde6"
    card_bd = "#000000" if hc else "#e2e8f2"
    st.markdown(f"""
    <style>
      #MainMenu, header[data-testid="stHeader"], footer {{visibility:hidden; height:0}}
      .block-container{{padding:0 !important; max-width:100% !important}}
      html, body, [class*="css"]{{font-size:{base}px;
        font-family:'Malgun Gothic','Noto Sans KR',sans-serif; color:{ink}}}
      .gov-wrap{{max-width:1120px; margin:0 auto; padding:0 20px}}

      /* top utility bar */
      .util{{background:{navy}; color:#dfe7f5; font-size:.8rem}}
      .util .gov-wrap{{display:flex; justify-content:flex-end; gap:18px; padding:7px 20px}}

      /* main header + nav */
      .gnav{{background:#fff; border-bottom:3px solid {navy}}}
      .gnav .row{{display:flex; align-items:center; justify-content:space-between; padding:16px 20px}}
      .brand{{display:flex; align-items:center; gap:12px; font-weight:800; color:{navy};
        font-size:1.5rem; letter-spacing:-.02em}}
      .brand .seal{{width:40px;height:40px;border-radius:50%;background:{navy};
        display:flex;align-items:center;justify-content:center;color:#fff;font-size:1.3rem}}
      .menu{{display:flex; gap:28px; font-weight:700; color:{ink}; font-size:1.05rem}}
      .menu span{{padding:6px 0; border-bottom:3px solid transparent; cursor:pointer}}
      .menu span:hover{{border-bottom-color:{accent}; color:{accent}}}

      /* hero */
      .hero{{background:linear-gradient(135deg,{navy} 0%,{navy2} 60%,{accent} 130%); color:#fff}}
      .hero .gov-wrap{{padding:40px 20px 30px}}
      .hero h1{{font-size:2.2rem; font-weight:800; margin:0 0 8px; letter-spacing:-.02em}}
      .hero p{{font-size:1.15rem; color:#dbe6fb; margin:0 0 22px}}

      /* section */
      .sec{{padding:36px 0}}
      .sec h2{{font-size:1.6rem; font-weight:800; color:{navy}; margin:0 0 4px}}
      .sec .sub{{color:#5b6b7d; margin:0 0 22px; font-size:1rem}}
      .sec.alt{{background:#f2f5fa}}

      /* category cards (rendered via buttons) */
      div[data-testid="stButton"] > button{{
        width:100%; min-height:118px; border:2px solid {card_bd}; border-radius:16px;
        background:#fff; color:{ink}; font-size:1.15rem; font-weight:800;
        box-shadow:0 2px 10px rgba(11,46,99,.06); line-height:1.5; white-space:pre-line;
        transition:all .15s}}
      div[data-testid="stButton"] > button:hover{{border-color:{accent};
        box-shadow:0 8px 22px rgba(11,46,99,.16); transform:translateY(-2px); color:{navy}}}

      /* AI assistant card */
      .ai-card{{background:#fff; border:2px solid {card_bd}; border-radius:18px;
        padding:10px 18px 4px; box-shadow:0 10px 30px rgba(11,46,99,.10)}}
      .ai-head{{display:flex; align-items:center; gap:10px; font-weight:800; color:{navy};
        font-size:1.3rem; padding:8px 2px}}
      .stChatMessage{{font-size:1.05rem}}

      /* chips */
      .chips{{display:flex; flex-wrap:wrap; gap:10px}}
      .chip{{background:#fff; border:2px solid {card_bd}; border-radius:999px;
        padding:9px 18px; font-weight:700; color:{navy}; font-size:1rem}}

      /* notices */
      .notice{{display:flex; gap:16px; padding:14px 4px; border-bottom:1px solid {line};
        font-size:1.05rem}}
      .notice .date{{color:#7c8ba0; font-variant-numeric:tabular-nums; min-width:100px}}
      .notice .t{{font-weight:600; color:{ink}}}
      .newtag{{background:#e12; color:#fff; font-size:.7rem; font-weight:800;
        border-radius:5px; padding:1px 6px; margin-left:8px; vertical-align:middle}}

      /* footer */
      .foot{{background:#0a2652; color:#c6d3ea; margin-top:20px}}
      .foot .gov-wrap{{padding:30px 20px}}
      .foot .cols{{display:flex; gap:40px; flex-wrap:wrap; margin-bottom:18px;
        font-size:.95rem}}
      .foot b{{color:#fff; display:block; margin-bottom:8px; font-size:1rem}}
      .foot .copy{{border-top:1px solid #24457e; padding-top:16px; font-size:.85rem;
        color:#8ea6cf}}

      /* hands-free voice orb */
      .orb{{width:170px;height:170px;border-radius:50%;margin:26px auto;
        background:radial-gradient(circle at 50% 35%,#cdd6ff,#1a56b0 70%,{navy});
        box-shadow:0 14px 44px rgba(11,46,99,.35)}}
      .orb.listen{{animation:breathe 2.4s ease-in-out infinite}}
      .orb.think{{animation:spin 1.1s linear infinite}}
      .orb.speak{{animation:pulse .7s ease-in-out infinite}}
      @keyframes breathe{{0%,100%{{transform:scale(1);opacity:.9}}50%{{transform:scale(1.06);opacity:1}}}}
      @keyframes pulse{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.12)}}}}
      @keyframes spin{{to{{transform:rotate(360deg)}}}}
      .vstatus{{text-align:center;color:{navy};font-weight:800;font-size:1.25rem;margin-bottom:6px}}
    </style>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# AI assistant helpers
# --------------------------------------------------------------------------
def _speak(answer: str):
    if not st.session_state.get("voice_mode"):
        return None
    from pathlib import Path
    n = st.session_state.get("reply_n", 0) + 1
    st.session_state["reply_n"] = n
    out = Path("data/audio") / f"reply_{n}.m4a"
    out.parent.mkdir(parents=True, exist_ok=True)
    return get_tts().synthesize(answer, str(out))


def _store(r: dict):
    audio = _speak(r["answer"])
    st.session_state["chat"].append(("user", r["question"]))
    st.session_state["chat"].append(("bot", r["answer"], audio))
    st.session_state["latest_audio"] = audio
    if r.get("alert"):
        urgency = "긴급" if r.get("crisis") else "주의"
        st.session_state["alerts"].insert(0, (urgency, r["alert"]))
    st.session_state["state"] = r


def ask_text(client, q: str):
    r = client.ask(q)
    if "error" in r:
        st.error(r["error"]); return
    _store(r)


def ask_audio(client, wav: bytes):
    from pathlib import Path
    tmp = Path("data/audio") / "mic_input.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(wav)
    r = client.ask_audio(str(tmp))
    if "error" in r:
        st.error(r["error"]); return
    _store(r)


# --------------------------------------------------------------------------
# Portal sections
# --------------------------------------------------------------------------
def header_nav():
    st.markdown("""
    <div class="util"><div class="gov-wrap">
      <span>로그인</span><span>회원가입</span><span>사이트맵</span><span>English</span>
    </div></div>
    <div class="gnav"><div class="gov-wrap"><div class="row">
      <div class="brand"><span class="seal">🏛️</span> 대전광역시 복지포털</div>
      <div class="menu">
        <span>복지서비스</span><span>건강·의료</span><span>어르신 돌봄</span>
        <span>공지사항</span><span>상담·문의</span>
      </div>
    </div></div></div>
    """, unsafe_allow_html=True)


def hero(client):
    st.markdown("""
    <div class="hero"><div class="gov-wrap">
      <h1>어르신, 무엇을 도와드릴까요?</h1>
      <p>말씀하거나 입력하시면 복지 서비스를 쉽게 안내해 드립니다. 모든 상담은 안전하게 보호됩니다.</p>
    </div></div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="gov-wrap sec">', unsafe_allow_html=True)
        st.markdown('<div class="ai-head">🎙️ 음성 복지 상담 (AI 도우미)</div>',
                    unsafe_allow_html=True)

        if st.button("🎙️ 음성으로 대화하기 (핸즈프리 · 버튼 없이 그냥 말하세요)",
                     use_container_width=True, type="primary"):
            st.session_state["mode"] = "voice"; st.rerun()
        st.caption("↑ 마이크 버튼을 누를 필요 없이, 말하면 자동으로 알아듣고 답합니다.")

        # welfare search
        with st.form("search", clear_on_submit=True):
            c1, c2 = st.columns([5, 1])
            q = c1.text_input("복지 검색", placeholder="예) 기초연금 신청 방법",
                              label_visibility="collapsed")
            go = c2.form_submit_button("🔍 검색", use_container_width=True)
        if go and q.strip():
            ask_text(client, q.strip()); st.rerun()

        # conversation
        for msg in st.session_state["chat"][-6:]:
            if msg[0] == "user":
                with st.chat_message("user"):
                    st.write(msg[1])
            else:
                with st.chat_message("assistant"):
                    st.write(msg[1])
                    audio = msg[2] if len(msg) > 2 else None
                    if audio:
                        st.audio(audio, format="audio/mp4",
                                 autoplay=(audio == st.session_state.get("latest_audio")))

        mic = st.audio_input("🎤 마이크로 말씀하세요", key="mic",
                             label_visibility="collapsed")
        if mic is not None:
            data = mic.getvalue(); fp = hash(data)
            if data and st.session_state.get("last_mic") != fp:
                st.session_state["last_mic"] = fp
                with st.spinner("음성을 인식하는 중…"):
                    ask_audio(client, data)
                st.rerun()
        if p := st.chat_input("여기에 입력하세요…"):
            ask_text(client, p); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def category_cards(client):
    st.markdown('<div class="sec alt"><div class="gov-wrap">', unsafe_allow_html=True)
    st.markdown('<h2>복지 서비스 바로가기</h2>'
                '<p class="sub">필요하신 서비스를 선택하시면 AI 도우미가 안내해 드립니다.</p>',
                unsafe_allow_html=True)
    for row in range(0, len(CATEGORIES), 3):
        cols = st.columns(3, gap="medium")
        for col, (icon, name, desc, q) in zip(cols, CATEGORIES[row:row + 3]):
            if col.button(f"{icon}\n{name}\n{desc}", key=f"cat_{name}"):
                ask_text(client, q); st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)


def popular_and_notices():
    st.markdown('<div class="sec"><div class="gov-wrap">', unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<h2>자주 찾는 서비스</h2><p class="sub">많이 이용하시는 서비스입니다.</p>',
                    unsafe_allow_html=True)
        chips = "".join(f'<span class="chip">{p}</span>' for p in POPULAR)
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<h2>공지사항</h2><p class="sub">복지 관련 최신 소식입니다.</p>',
                    unsafe_allow_html=True)
        rows = ""
        for i, (date, title) in enumerate(NOTICES):
            new = '<span class="newtag">NEW</span>' if i == 0 else ""
            rows += (f'<div class="notice"><span class="date">{date}</span>'
                     f'<span class="t">{title}{new}</span></div>')
        st.markdown(rows, unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)


def footer():
    st.markdown("""
    <div class="foot"><div class="gov-wrap">
      <div class="cols">
        <div><b>대전광역시 복지포털</b>대전광역시 서구 둔산로 100<br>
             노인복지과 042-000-0000</div>
        <div><b>바로가기</b>기초연금 · 노인맞춤돌봄<br>치매안심센터 · 방문건강관리</div>
        <div><b>상담전화</b>복지상담 129<br>자살예방상담 109 (24시간)</div>
        <div><b>이용안내</b>개인정보처리방침<br>웹 접근성 안내 · 이용약관</div>
      </div>
      <div class="copy">본 서비스는 모든 처리를 기기 내에서 수행하며 대화 내용을 외부로 전송하지 않습니다.
        © 2026 Daejeon Welfare Portal (demo).</div>
    </div></div>
    """, unsafe_allow_html=True)


def voice_conversation(client):
    """Hands-free ChatGPT-style voice chat inside the portal (no record/send)."""
    import queue
    import time
    from pathlib import Path
    try:
        import av
        import numpy as np
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
        from src.vad_stream import UtteranceSegmenter
    except Exception:
        st.error("음성 대화 기능 패키지가 필요합니다: "
                 "pip install streamlit-webrtc webrtcvad 'setuptools<81'")
        return

    st.markdown('<div class="gov-wrap sec">', unsafe_allow_html=True)
    st.markdown('<h2>🎙️ 음성 대화 (AI 도우미)</h2>'
                '<p class="sub">마이크를 켜고 그냥 말씀하세요. 말이 끝나면 자동으로 답합니다. '
                '(🎧 이어폰 사용 권장 · 실험 기능)</p>', unsafe_allow_html=True)

    class VP:
        def __init__(self):
            self.seg = UtteranceSegmenter(silence_ms=800, min_speech_ms=300)
            self.res = av.AudioResampler(format="s16", layout="mono", rate=16000)
            self.utterances = queue.Queue()

        def recv(self, frame):
            for f in self.res.resample(frame):
                wav = self.seg.add_pcm(f.to_ndarray().astype(np.int16).tobytes())
                if wav:
                    self.utterances.put(wav)
            return frame

    orb = st.empty()
    status = st.empty()
    ctx = webrtc_streamer(
        key="portal_voice", mode=WebRtcMode.SENDONLY, audio_processor_factory=VP,
        media_stream_constraints={
            "audio": {"echoCancellation": True, "noiseSuppression": True},
            "video": False},
        async_processing=True)
    chat_ph = st.empty()
    audio_ph = st.empty()
    tts = get_tts()
    st.session_state.setdefault("vchat", [])

    def render():
        with chat_ph.container():
            for role, text in st.session_state["vchat"][-8:]:
                with st.chat_message("user" if role == "user" else "assistant"):
                    st.write(text)

    def orbset(state, label):
        orb.markdown(f'<div class="orb {state}"></div>', unsafe_allow_html=True)
        status.markdown(f'<div class="vstatus">{label}</div>', unsafe_allow_html=True)

    render()
    st.markdown('</div>', unsafe_allow_html=True)

    if not ctx.state.playing:
        orbset("listen", "위의 START를 눌러 음성 대화를 시작하세요.")
        return

    orbset("listen", "듣고 있어요…")
    while ctx.state.playing:
        proc = ctx.audio_processor
        if proc is None:
            time.sleep(0.1)
            continue
        try:
            wav = proc.utterances.get(timeout=0.5)
        except queue.Empty:
            continue

        orbset("think", "생각 중…")
        Path("data/audio").mkdir(parents=True, exist_ok=True)
        utt = Path("data/audio") / "portal_utt.wav"
        utt.write_bytes(wav)
        r = client.ask_audio(str(utt))
        if "error" in r:
            st.session_state["vchat"].append(("assistant", f"⚠️ {r['error']}"))
            render(); orbset("listen", "듣고 있어요…"); continue

        st.session_state["vchat"].append(("user", r["question"]))
        st.session_state["vchat"].append(("assistant", r["answer"]))
        st.session_state["state"] = r
        if r.get("alert"):
            st.session_state["alerts"].insert(
                0, ("긴급" if r.get("crisis") else "주의", r["alert"]))
        render()

        n = st.session_state.get("reply_n", 0) + 1
        st.session_state["reply_n"] = n
        audio = tts.synthesize(r["answer"], str(Path("data/audio") / f"vreply_{n}.m4a"))
        if audio:
            orbset("speak", "말하는 중…")
            audio_ph.audio(audio, format="audio/mp4", autoplay=True)
        orbset("listen", "듣고 있어요…")


def admin_view(client):
    st.markdown('<div class="gov-wrap sec">', unsafe_allow_html=True)
    st.markdown('<h2>🧑‍⚕️ 관리자 · 이상신호 모니터</h2>'
                '<p class="sub">어르신 화면에는 보이지 않는 배경 모니터링입니다.</p>',
                unsafe_allow_html=True)
    s = st.session_state["state"]
    c1, c2, c3 = st.columns(3)
    status = ("🚨 긴급" if s.get("crisis") else
              "⚠️ 주의" if s.get("is_abnormal") else "✅ 정상")
    c1.metric("상태", status)
    c2.metric("평균 부정감정", f"{s.get('avg_negative', 0.0):.2f}")
    c3.metric("대화 횟수", f"{s.get('history_len', 0)}회")
    counts = s.get("symptom_counts", {})
    if counts:
        st.write("**반복 호소 증상:** " + ", ".join(f"{k}({v}회)" for k, v in counts.items()))
    m = s.get("metrics", {})
    if m:
        cc = st.columns(4)
        cc[0].metric("대화 빈도", f"-{m['freq_drop']:.0%}")
        cc[1].metric("감정 변화", f"+{m['sentiment_shift']:.2f}")
        cc[2].metric("외로움 표현", f"+{m['keyword_shift']:.0%}")
        cc[3].metric("응답 길이", f"-{m['length_drop']:.0%}")
    st.divider()
    st.markdown("#### 🔔 사회복지사 알림 로그")
    if st.session_state["alerts"]:
        for urgency, text in st.session_state["alerts"]:
            st.error(f"[{urgency}]\n\n{text}") if urgency == "긴급" else st.warning(text)
    else:
        st.caption("아직 알림이 없습니다.")
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    with st.spinner("복지포털을 준비하는 중입니다…"):
        client = get_client()
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("alerts", [])
    st.session_state.setdefault("state", client.state())
    st.session_state.setdefault("font_scale", 1.0)
    st.session_state.setdefault("voice_mode", False)

    with st.sidebar:
        st.subheader("♿ 화면 설정")
        st.caption("글자 크기")
        b1, b2, b3 = st.columns(3)
        if b1.button("가", use_container_width=True):
            st.session_state["font_scale"] = 1.0; st.rerun()
        if b2.button("가+", use_container_width=True):
            st.session_state["font_scale"] = 1.18; st.rerun()
        if b3.button("가++", use_container_width=True):
            st.session_state["font_scale"] = 1.38; st.rerun()
        st.session_state["high_contrast"] = st.toggle(
            "고대비 모드", value=st.session_state.get("high_contrast", False))
        st.session_state["voice_mode"] = st.toggle(
            "🔊 음성으로 답변 듣기", value=st.session_state.get("voice_mode", False),
            help=f"로컬 TTS: {get_tts().backend or '사용 불가'}")
        st.divider()
        view = st.radio("화면", ["🏛️ 복지포털 (어르신)", "🧑‍⚕️ 관리자"])
        st.caption("상담 방식")
        if st.button("💬 텍스트·검색", use_container_width=True):
            st.session_state["mode"] = "text"; st.rerun()
        if st.button("🎙️ 음성 대화 (핸즈프리)", use_container_width=True):
            st.session_state["mode"] = "voice"; st.rerun()
        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state["state"] = client.reset()
            st.session_state["chat"] = []
            st.session_state["alerts"] = []
            st.session_state.pop("last_mic", None)
            st.rerun()

    inject_css()

    if view.startswith("🧑"):
        header_nav()
        admin_view(client)
        footer()
    elif st.session_state.get("mode") == "voice":
        header_nav()
        voice_conversation(client)   # hands-free ChatGPT-style voice loop
    else:
        header_nav()
        hero(client)
        category_cards(client)
        popular_and_notices()
        footer()


if __name__ != "__mp_main__":
    main()
