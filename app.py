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
    navy = "#0f2f6f"                       # brand
    navy2 = "#0a2350"                      # deep
    accent = "#2f6bff"                     # vivid CTA blue
    ink = "#000000" if hc else "#15233d"
    muted = "#334155" if hc else "#5c6b86"
    bg = "#ffffff" if hc else "#eef2f9"    # cool light ground
    surface = "#ffffff"
    surface2 = "#f5f8fd"
    line = "#000000" if hc else "#e6ebf4"
    card_bd = "#000000" if hc else "#e8edf7"
    st.markdown(f"""
    <style>
      @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

      #MainMenu, header[data-testid="stHeader"], footer {{visibility:hidden; height:0}}
      .block-container{{padding:0 !important; max-width:100% !important}}
      .stApp{{background:{bg}}}
      html, body, [class*="css"], .stApp{{font-size:{base}px;
        font-family:'Pretendard','Pretendard Variable',-apple-system,'Noto Sans KR',sans-serif;
        color:{ink}; -webkit-font-smoothing:antialiased}}
      .gov-wrap{{max-width:1140px; margin:0 auto; padding:0 24px}}

      /* top utility bar */
      .util{{background:{navy2}; color:#c7d4ee; font-size:.82rem}}
      .util .gov-wrap{{display:flex; justify-content:flex-end; gap:22px; padding:9px 24px}}
      .util span{{opacity:.85; cursor:pointer}} .util span:hover{{opacity:1;color:#fff}}

      /* header + nav */
      .gnav{{background:rgba(255,255,255,.9); backdrop-filter:saturate(1.4) blur(8px);
        border-bottom:1px solid {line}; position:sticky; top:0; z-index:50}}
      .gnav .row{{display:flex; align-items:center; justify-content:space-between; padding:16px 24px}}
      .brand{{display:flex; align-items:center; gap:13px; font-weight:800; color:{navy};
        font-size:1.45rem; letter-spacing:-.03em}}
      .brand .seal{{width:42px;height:42px;border-radius:13px;
        background:linear-gradient(140deg,{accent},{navy}); color:#fff;
        display:flex;align-items:center;justify-content:center;font-size:1.25rem;
        box-shadow:0 6px 16px -4px {accent}66}}
      .menu{{display:flex; gap:8px; font-weight:600; font-size:1.02rem}}
      .menu span{{padding:9px 15px; border-radius:11px; color:{muted}; cursor:pointer;
        transition:all .15s}}
      .menu span:hover{{background:{surface2}; color:{navy}}}

      /* hero */
      .hero .gov-wrap{{padding:30px 24px 34px}}
      .hero-inner{{border-radius:26px; padding:48px 46px; color:#fff; position:relative;
        overflow:hidden;
        background:
          radial-gradient(130% 120% at 12% 8%, {accent}55, transparent 55%),
          radial-gradient(120% 130% at 92% 100%, #7aa0ff44, transparent 50%),
          linear-gradient(135deg, {navy} 0%, {navy2} 100%);
        box-shadow:0 30px 60px -24px {navy}80}}
      .hero .eyebrow{{display:inline-block; font-size:.82rem; font-weight:700;
        letter-spacing:.14em; text-transform:uppercase; color:#bcd0ff;
        background:#ffffff1f; padding:6px 14px; border-radius:999px; margin-bottom:18px}}
      .hero h1{{font-size:2.55rem; font-weight:800; margin:0 0 12px; letter-spacing:-.035em;
        line-height:1.18}}
      .hero p{{font-size:1.18rem; color:#d7e2fb; margin:0; max-width:60ch; line-height:1.6}}

      /* section */
      .sec{{padding:44px 0}}
      .sec.alt{{background:{surface2}}}
      .sec h2{{font-size:1.72rem; font-weight:800; color:{ink}; margin:0 0 6px;
        letter-spacing:-.03em}}
      .sec .sub{{color:{muted}; margin:0 0 26px; font-size:1.05rem}}

      /* category cards (Streamlit buttons) */
      div[data-testid="stButton"] > button{{
        width:100%; min-height:132px; border:1px solid {card_bd}; border-radius:20px;
        background:{surface}; color:{ink}; font-size:1.18rem; font-weight:700;
        line-height:1.55; white-space:pre-line; letter-spacing:-.01em;
        box-shadow:0 1px 2px rgba(16,35,61,.05), 0 16px 32px -18px {navy}33;
        transition:transform .18s cubic-bezier(.2,.7,.3,1), box-shadow .18s, border-color .18s}}
      div[data-testid="stButton"] > button:hover{{border-color:{accent};
        box-shadow:0 1px 2px rgba(16,35,61,.05), 0 24px 44px -18px {accent}55;
        transform:translateY(-4px); color:{navy}}}
      div[data-testid="stButton"] > button:active{{transform:translateY(-1px)}}
      /* primary buttons (CTA / voice) */
      .stButton button[kind="primary"], .stButton button[kind="primaryFormSubmit"]{{
        min-height:0; background:linear-gradient(135deg,{accent},{navy});
        border:none; color:#fff; box-shadow:0 12px 24px -10px {accent}88}}
      .stButton button[kind="primary"]:hover{{transform:translateY(-2px); color:#fff;
        box-shadow:0 18px 32px -10px {accent}aa}}

      /* AI assistant */
      .ai-head{{display:flex; align-items:center; gap:11px; font-weight:800; color:{navy};
        font-size:1.42rem; padding:6px 2px 12px; letter-spacing:-.02em}}
      .stChatMessage{{font-size:1.06rem; border-radius:16px}}
      .stTextInput input{{border-radius:12px !important; border:1px solid {card_bd} !important;
        font-size:1.05rem !important}}
      [data-testid="stChatInput"]{{border-radius:14px}}

      /* chips */
      .chips{{display:flex; flex-wrap:wrap; gap:11px}}
      .chip{{background:{surface}; border:1px solid {card_bd}; border-radius:999px;
        padding:11px 20px; font-weight:600; color:{navy}; font-size:1.02rem;
        box-shadow:0 1px 2px rgba(16,35,61,.04); transition:all .15s}}
      .chip:hover{{border-color:{accent}; color:{accent}; transform:translateY(-2px)}}

      /* notices */
      .notice{{display:flex; gap:16px; align-items:center; padding:15px 12px;
        border-bottom:1px solid {line}; font-size:1.05rem; border-radius:12px;
        transition:background .15s}}
      .notice:hover{{background:{surface2}}}
      .notice .date{{color:#8493ac; font-variant-numeric:tabular-nums; min-width:104px;
        font-size:.95rem}}
      .notice .t{{font-weight:600; color:{ink}}}
      .newtag{{background:{accent}; color:#fff; font-size:.68rem; font-weight:800;
        border-radius:6px; padding:2px 7px; margin-left:9px; vertical-align:middle;
        letter-spacing:.05em}}

      /* footer */
      .foot{{background:linear-gradient(180deg,{navy2},#071a3d); color:#b9c8e6; margin-top:24px}}
      .foot .gov-wrap{{padding:40px 24px 34px}}
      .foot .cols{{display:flex; gap:48px; flex-wrap:wrap; margin-bottom:22px; font-size:.98rem;
        line-height:1.9}}
      .foot b{{color:#fff; display:block; margin-bottom:10px; font-size:1.02rem; font-weight:700}}
      .foot .copy{{border-top:1px solid #ffffff1a; padding-top:18px; font-size:.86rem; color:#8195b8}}

      /* hands-free voice orb */
      .orb{{width:184px;height:184px;border-radius:50%;margin:30px auto;
        background:radial-gradient(circle at 42% 32%,#dfe7ff,{accent} 62%,{navy});
        box-shadow:0 20px 60px -12px {accent}88, inset 0 -10px 30px #0a235066}}
      .orb.listen{{animation:breathe 2.6s ease-in-out infinite}}
      .orb.think{{animation:spin 1.1s linear infinite}}
      .orb.speak{{animation:pulse .7s ease-in-out infinite}}
      @keyframes breathe{{0%,100%{{transform:scale(1);opacity:.92}}50%{{transform:scale(1.07);opacity:1}}}}
      @keyframes pulse{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.13)}}}}
      @keyframes spin{{to{{transform:rotate(360deg)}}}}
      .vstatus{{text-align:center;color:{navy};font-weight:800;font-size:1.3rem;margin-bottom:6px;
        letter-spacing:-.02em}}

      @media (prefers-reduced-motion:reduce){{
        div[data-testid="stButton"] > button, .chip{{transition:none}}
        .orb{{animation:none !important}}
      }}
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
    <div class="hero"><div class="gov-wrap"><div class="hero-inner">
      <span class="eyebrow">대전광역시 · 어르신 복지</span>
      <h1>어르신, 무엇을<br>도와드릴까요?</h1>
      <p>말씀하거나 입력하시면 복지 서비스를 쉽게 안내해 드립니다.
         모든 상담은 기기 안에서만 처리되어 안전하게 보호됩니다.</p>
    </div></div></div>
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
    if st.button("◀ 복지포털 홈으로 돌아가기", use_container_width=True, type="primary"):
        st.session_state["mode"] = "text"
        st.rerun()
    st.caption("대화 중이면 먼저 아래 **STOP**을 누른 뒤 이 버튼을 눌러 주세요.")
    st.markdown('<h2>🎙️ 음성 대화 (AI 도우미)</h2>'
                '<p class="sub">마이크를 켜고 그냥 말씀하세요. 말이 끝나면 자동으로 답합니다. '
                '(🎧 이어폰 사용 권장 · 실험 기능)</p>', unsafe_allow_html=True)

    class VP:
        def __init__(self):
            self.seg = UtteranceSegmenter(aggressiveness=3, silence_ms=800,
                                          min_speech_ms=400)
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

    # Common Whisper hallucinations on silence/noise - ignore these.
    NOISE = {"", ".", "..", "네", "음", "아", "감사합니다", "시청해주셔서 감사합니다",
             "구독과 좋아요", "다음 영상에서 만나요", "고맙습니다", "수고하셨습니다"}

    def drain(proc):
        try:
            while True:
                proc.utterances.get_nowait()
        except queue.Empty:
            pass

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

        Path("data/audio").mkdir(parents=True, exist_ok=True)
        utt = Path("data/audio") / "portal_utt.wav"
        utt.write_bytes(wav)

        # Transcribe first and reject noise / the AI's own echoed voice.
        text = (client.transcribe(str(utt)).get("text") or "").strip()
        if len(text) < 2 or text in NOISE:
            continue                                   # keep listening, no reply

        orbset("think", "생각 중…")
        r = client.ask(text)
        if "error" in r:
            st.session_state["vchat"].append(("assistant", f"⚠️ {r['error']}"))
            render(); drain(proc); orbset("listen", "듣고 있어요…"); continue

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
            # Half-duplex: while the reply plays, discard whatever the mic hears
            # (the AI's own voice) so it does not answer itself.
            cooldown = min(max(len(r["answer"]) * 0.09, 2.0), 12.0)
            end = time.time() + cooldown
            while time.time() < end:
                drain(proc)
                time.sleep(0.15)
        drain(proc)                                    # final flush before listening
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
