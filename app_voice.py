"""
Hands-free voice conversation (experimental) - ChatGPT-voice style.

You just talk: the mic streams continuously, VAD detects when you finish a
sentence, Whisper transcribes it, the local LLM answers, and the answer is
spoken back (TTS) - then it listens again. No record/send button per turn.

    python -m streamlit run app_voice.py     (use the env with the models)

Notes
  - All local: STT (Whisper) + LLM (Ollama) + TTS run on-device.
  - Latency depends on the LLM. For snappy turns use a small model, e.g.
    `ollama pull llama3.2:1b` and set LLM_MODEL in src/config.py.
  - Use headphones so the AI's own voice is not picked up by the mic.
"""
import queue
import time
from pathlib import Path

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.tts import TextToSpeech
from src.vad_stream import UtteranceSegmenter
from src.worker import AssistantClient

st.set_page_config(page_title="음성 대화 · 복지 도우미", page_icon="🎙️", layout="centered")

CSS = """
<style>
  .orb{width:180px;height:180px;border-radius:50%;margin:32px auto;
       background:radial-gradient(circle at 50% 35%, #cdd6ff, #7c8cff 65%, #5566ee);
       box-shadow:0 12px 40px rgba(90,110,240,.35);}
  .orb.listen{animation:breathe 2.4s ease-in-out infinite}
  .orb.think{animation:spin 1.1s linear infinite}
  .orb.speak{animation:pulse 0.7s ease-in-out infinite}
  @keyframes breathe{0%,100%{transform:scale(1);opacity:.9}50%{transform:scale(1.06);opacity:1}}
  @keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.12)}}
  @keyframes spin{to{transform:rotate(360deg)}}
  .vstatus{text-align:center;color:#5566ee;font-weight:600;font-size:1.05rem}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_client():
    return AssistantClient()


@st.cache_resource
def get_tts():
    return TextToSpeech()


class VoiceProcessor:
    """Runs in the WebRTC audio thread: resample -> VAD -> emit utterance WAVs."""

    def __init__(self):
        self.segmenter = UtteranceSegmenter(silence_ms=800, min_speech_ms=300)
        self.resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        self.utterances: "queue.Queue[bytes]" = queue.Queue()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        for f in self.resampler.resample(frame):
            wav = self.segmenter.add_pcm(f.to_ndarray().astype(np.int16).tobytes())
            if wav:
                self.utterances.put(wav)
        return frame


st.markdown("### 🎙️ 음성 대화 · 복지 도우미")
st.caption("마이크를 켜고 그냥 말씀하세요. 말이 끝나면 자동으로 답합니다. (실험 기능)")

client = get_client()
tts = get_tts()
st.session_state.setdefault("vchat", [])

ctx = webrtc_streamer(
    key="voice",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=VoiceProcessor,
    media_stream_constraints={
        "audio": {"echoCancellation": True, "noiseSuppression": True},
        "video": False,
    },
    async_processing=True,
)

orb = st.empty()
status = st.empty()
chat_ph = st.empty()
audio_ph = st.empty()


def render_chat():
    with chat_ph.container():
        for role, text in st.session_state["vchat"][-8:]:
            with st.chat_message("user" if role == "user" else "assistant"):
                st.write(text)


def set_orb(state: str, label: str):
    orb.markdown(f'<div class="orb {state}"></div>', unsafe_allow_html=True)
    status.markdown(f'<div class="vstatus">{label}</div>', unsafe_allow_html=True)


render_chat()

if ctx.state.playing:
    set_orb("listen", "듣고 있어요…")
    reply_n = 0
    while ctx.state.playing:
        proc = ctx.audio_processor
        if proc is None:
            time.sleep(0.1)
            continue
        try:
            wav = proc.utterances.get(timeout=0.5)
        except queue.Empty:
            continue

        set_orb("think", "생각 중…")
        Path("data/audio").mkdir(parents=True, exist_ok=True)
        utt_path = Path("data/audio") / "voice_utt.wav"
        utt_path.write_bytes(wav)

        r = client.ask_audio(str(utt_path))
        if "error" in r:
            st.session_state["vchat"].append(("assistant", f"⚠️ {r['error']}"))
            render_chat()
            set_orb("listen", "듣고 있어요…")
            continue

        st.session_state["vchat"].append(("user", r["question"]))
        st.session_state["vchat"].append(("assistant", r["answer"]))
        render_chat()

        reply_n += 1
        out = Path("data/audio") / f"vreply_{reply_n}.m4a"
        audio = tts.synthesize(r["answer"], str(out))
        if audio:
            set_orb("speak", "말하는 중…")
            audio_ph.audio(audio, format="audio/mp4", autoplay=True)

        set_orb("listen", "듣고 있어요…")
else:
    set_orb("listen", "위의 START를 눌러 대화를 시작하세요.")
