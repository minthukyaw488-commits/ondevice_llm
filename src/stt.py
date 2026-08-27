"""
Step 2: Speech-to-Text (STT) using Whisper - fully local.

Primary backend: faster-whisper (CTranslate2) running the Whisper model
on-device. Nothing is sent to any cloud service.

If the model cannot be loaded (e.g. offline, model not downloaded), a
fallback transcriber returns a clear placeholder so the rest of the pipeline
(RAG + abnormal-signal analysis) can still be demonstrated from text input.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from . import config


class SpeechToText:
    def __init__(self, model_size: str = config.WHISPER_MODEL,
                 language: str = config.WHISPER_LANGUAGE):
        self.model_size = model_size
        self.language = language
        self.backend = "unavailable"
        self._model = None
        try:
            from faster_whisper import WhisperModel
            # int8 on CPU keeps it light for on-device use.
            self._model = WhisperModel(model_size, device="cpu",
                                       compute_type="int8")
            self.backend = f"faster-whisper:{model_size}"
        except Exception as exc:
            print(f"[stt] Whisper unavailable ({exc.__class__.__name__}). "
                  f"Use text input, or install faster-whisper + download the "
                  f"'{model_size}' model to enable voice.")

    def transcribe_file(self, audio_path: str | Path) -> str:
        """Transcribe a WAV/MP3 file to Korean text."""
        audio_path = str(audio_path)
        if self._model is None:
            return f"[STT unavailable - could not transcribe {Path(audio_path).name}]"
        segments, _info = self._model.transcribe(audio_path, language=self.language)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def transcribe_array(self, samples, sample_rate: int = 16000) -> str:
        """Transcribe a raw float32 numpy audio array (for microphone input)."""
        if self._model is None:
            return "[STT unavailable]"
        # faster-whisper resamples internally; it expects 16 kHz mono float32.
        segments, _info = self._model.transcribe(samples, language=self.language)
        return " ".join(seg.text.strip() for seg in segments).strip()


def record_microphone(seconds: int = 5, sample_rate: int = 16000):
    """Record `seconds` of mono audio from the default microphone.

    Returns a float32 numpy array. Requires the `sounddevice` package and a
    working microphone (available on the deployment device, not in CI).
    """
    import numpy as np
    import sounddevice as sd
    print(f"[stt] recording {seconds}s ... speak now")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)


if __name__ == "__main__":
    import sys
    stt = SpeechToText()
    print("backend:", stt.backend)
    if len(sys.argv) > 1:
        print("transcript:", stt.transcribe_file(sys.argv[1]))
    else:
        print("Pass an audio file path to transcribe, or use record_microphone().")
