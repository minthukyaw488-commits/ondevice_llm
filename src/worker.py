"""
Run the WelfareAssistant in a separate process.

Why: Streamlit runs the app script in a worker thread. On macOS several native
libraries (torch / chromadb / faster-whisper) crash with a bus error when
initialised or called off the main thread. Running the assistant in its own
process (where it owns the main thread) side-steps that entire class of crash.
Streamlit only exchanges plain picklable dicts over a queue - no native code in
its thread. This mirrors how the LLM already runs as a separate local process.
"""
from __future__ import annotations
import multiprocessing as mp
from typing import Optional


def _turn_dict(res, bot) -> dict:
    return {
        "question": res.question,
        "answer": res.answer,
        "sources": list(res.sources),
        "alert": res.alert,
        **_state_dict(bot),
    }


def _state_dict(bot) -> dict:
    sig = bot.detector.evaluate()
    return {
        "is_abnormal": sig.is_abnormal,
        "crisis": sig.crisis,
        "avg_negative": sig.avg_negative,
        "symptom_counts": dict(sig.symptom_counts),
        "history_len": len(bot.detector.history),
    }


def _log(msg: str) -> None:
    print(f"[worker] {msg}", flush=True)


def run_worker(req_q: "mp.Queue", resp_q: "mp.Queue") -> None:
    """Child-process entry point. Builds the assistant, then serves requests."""
    try:
        _log("starting… importing pipeline")
        from src.pipeline import WelfareAssistant
        _log("building assistant (loading models + indexing welfare docs)…")
        bot = WelfareAssistant(user_name="데모 어르신")   # loads + indexes once
        _log("assistant ready")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        # Tell the parent instead of letting it block forever on ready.
        resp_q.put({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return

    resp_q.put({"ready": True, "meta": {
        "rag_backend": bot.rag.backend,
        "embedder": bot.rag.embedder.backend,
        "llm": bot.llm.available,
        "sentiment": bot.detector.sentiment.backend,
        "alert_channels": bot.alerts.channels,
    }})

    while True:
        req = req_q.get()
        cmd = req.get("cmd")
        if cmd == "stop":
            break
        try:
            if cmd == "ask":
                resp_q.put(_turn_dict(bot.ask_text(req["text"]), bot))
            elif cmd == "ask_audio":
                resp_q.put(_turn_dict(bot.ask_audio(req["path"]), bot))
            elif cmd == "reset":
                bot.reset_conversation()
                resp_q.put(_state_dict(bot))
            elif cmd == "state":
                resp_q.put(_state_dict(bot))
            else:
                resp_q.put({"error": f"unknown command: {cmd}"})
        except Exception as exc:  # never let the worker die on one bad request
            resp_q.put({"error": f"{type(exc).__name__}: {exc}"})


class AssistantClient:
    """Parent-side handle. Starts the worker once and does blocking RPC."""

    def __init__(self):
        ctx = mp.get_context("spawn")     # fresh main thread in the child
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._proc = ctx.Process(target=run_worker,
                                 args=(self._req_q, self._resp_q), daemon=True)
        self._proc.start()
        ready = self._get(timeout=600)    # wait until models are loaded
        if not ready.get("ready"):
            raise RuntimeError(
                "Assistant worker failed to start: "
                + ready.get("error", "unknown error")
                + " (see the [worker] lines in the terminal)")
        self.meta = ready.get("meta", {})

    def _get(self, timeout: float) -> dict:
        """Wait for a response, but fail fast if the worker process dies."""
        import queue
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                return self._resp_q.get(timeout=1.0)
            except queue.Empty:
                if not self._proc.is_alive():
                    raise RuntimeError(
                        f"Assistant worker process died (exit code "
                        f"{self._proc.exitcode}) - likely a native crash while "
                        f"loading models. See the [worker] lines in the terminal.")
        raise RuntimeError(
            f"No response from the assistant worker within {timeout:.0f}s. "
            f"Check the terminal for [worker] logs.")

    def _rpc(self, req: dict) -> dict:
        self._req_q.put(req)
        return self._get(timeout=180)

    def ask(self, text: str) -> dict:
        return self._rpc({"cmd": "ask", "text": text})

    def ask_audio(self, path: str) -> dict:
        return self._rpc({"cmd": "ask_audio", "path": path})

    def reset(self) -> dict:
        return self._rpc({"cmd": "reset"})

    def state(self) -> dict:
        return self._rpc({"cmd": "state"})


if __name__ == "__main__":
    # Smoke test the IPC plumbing.
    c = AssistantClient()
    print("meta:", c.meta)
    print("ask :", {k: v for k, v in c.ask("기초연금 신청 방법").items()
                    if k in ("answer", "sources", "is_abnormal")})
    print("state:", c.state())
