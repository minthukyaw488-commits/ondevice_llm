"""
Local Text-to-Speech for voice-to-voice replies.

Fully on-device (no cloud): the answer text is spoken aloud so the elderly user
can hear the reply. Backends, in order of preference:

  1. macOS `say` (+ afconvert) - built-in Korean voice ("Yuna"), best on Mac.
  2. pyttsx3               - cross-platform offline TTS.
  3. none                 - graceful: no audio, the text answer still shows.

synthesize() returns a path to a browser-playable audio file, or None.
"""
from __future__ import annotations
import os
import platform
import shutil
import subprocess
from typing import Optional


class TextToSpeech:
    def __init__(self, voice: str = "Yuna"):
        self.voice = voice
        self.backend = self._detect()

    def _detect(self) -> Optional[str]:
        if platform.system() == "Darwin" and shutil.which("say"):
            return "macos-say"
        try:
            import pyttsx3  # noqa: F401
            return "pyttsx3"
        except Exception:
            return None

    def synthesize(self, text: str, out_path: str) -> Optional[str]:
        """Speak `text` into an audio file at out_path (.m4a/.aiff). None on failure."""
        text = (text or "").strip()
        if not text or self.backend is None:
            return None
        try:
            if self.backend == "macos-say":
                return self._macos_say(text, out_path)
            if self.backend == "pyttsx3":
                return self._pyttsx3(text, out_path)
        except Exception as exc:
            print(f"[tts] synthesis failed ({exc.__class__.__name__}); "
                  f"continuing without audio.")
        return None

    def _macos_say(self, text: str, out_path: str) -> str:
        aiff = out_path + ".aiff"
        # Try the Korean voice; fall back to the default voice if it is missing.
        try:
            subprocess.run(["say", "-v", self.voice, "-o", aiff, text],
                           check=True, capture_output=True)
        except subprocess.CalledProcessError:
            subprocess.run(["say", "-o", aiff, text], check=True, capture_output=True)
        # Convert AIFF -> m4a (AAC) so browsers can play it inline.
        if shutil.which("afconvert"):
            subprocess.run(["afconvert", aiff, out_path, "-f", "m4af", "-d", "aac"],
                           check=True, capture_output=True)
            os.remove(aiff)
            return out_path
        return aiff

    def _pyttsx3(self, text: str, out_path: str) -> str:
        import pyttsx3
        engine = pyttsx3.init()
        for v in engine.getProperty("voices"):
            langs = " ".join(str(x) for x in getattr(v, "languages", []))
            if "ko" in (v.id + v.name + langs).lower():
                engine.setProperty("voice", v.id)
                break
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return out_path


if __name__ == "__main__":
    tts = TextToSpeech()
    print("backend:", tts.backend)
    p = tts.synthesize("안녕하세요, 무엇을 도와드릴까요?",
                       os.path.join("data", "audio", "tts_test.m4a"))
    print("wrote:", p)
