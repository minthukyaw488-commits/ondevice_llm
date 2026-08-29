"""
Voice Activity Detection (VAD) utterance segmenter for hands-free voice mode.

Feeds on a stream of 16 kHz mono int16 PCM and, using webrtcvad, decides when a
complete spoken utterance has ended (a burst of speech followed by enough
trailing silence). Returns that utterance as WAV bytes, ready for Whisper.

This is the "auto-detect start/stop of speech" piece that turns push-to-record
into a continuous conversation (no button press per turn).
"""
from __future__ import annotations
import io
import wave
from typing import Optional

import webrtcvad


class UtteranceSegmenter:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30,
                 aggressiveness: int = 2, silence_ms: int = 800,
                 min_speech_ms: int = 300):
        self.sr = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2   # int16 = 2 bytes
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.vad = webrtcvad.Vad(aggressiveness)                    # 0..3 (strict)
        self._pending = bytearray()      # bytes not yet aligned to a frame
        self._speech = bytearray()       # accumulated speech pcm
        self._num_speech = 0
        self._trailing_silence = 0
        self._in_speech = False

    def add_pcm(self, pcm: bytes) -> Optional[bytes]:
        """Add PCM bytes; return WAV bytes when an utterance just completed."""
        self._pending.extend(pcm)
        completed = None
        while len(self._pending) >= self.frame_bytes:
            frame = bytes(self._pending[:self.frame_bytes])
            del self._pending[:self.frame_bytes]
            speech = self.vad.is_speech(frame, self.sr)
            if speech:
                self._in_speech = True
                self._speech.extend(frame)
                self._num_speech += 1
                self._trailing_silence = 0
            elif self._in_speech:
                self._speech.extend(frame)          # keep a little trailing audio
                self._trailing_silence += 1
                if self._trailing_silence >= self.silence_frames:
                    if self._num_speech >= self.min_speech_frames:
                        completed = self._to_wav(bytes(self._speech))
                    self._reset()
                    break
        return completed

    def _reset(self):
        self._speech = bytearray()
        self._num_speech = 0
        self._trailing_silence = 0
        self._in_speech = False

    def _to_wav(self, pcm: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(pcm)
        return buf.getvalue()


if __name__ == "__main__":
    # Sanity check: pure silence should never emit an utterance.
    seg = UtteranceSegmenter()
    silence = b"\x00\x00" * 160          # one 10ms-ish chunk of silence
    emitted = any(seg.add_pcm(silence) for _ in range(200))
    print("silence emits utterance:", emitted, "(expected False)")
    print("frame_bytes:", seg.frame_bytes, "silence_frames:", seg.silence_frames)
