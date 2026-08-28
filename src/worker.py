"""
Client handle for the assistant worker process.

The heavy pipeline (RAG + torch models + chromadb) runs in a SEPARATE process
(src/worker_server.py) launched as a clean subprocess - it never imports
Streamlit. This matters on macOS: when the worker was started via
multiprocessing "spawn", the child re-imported app.py and thus Streamlit
before torch, and the native-library mix crashed with a bus error (SIGBUS).
A plain subprocess that only imports the pipeline avoids that entirely.

Communication is line-delimited JSON over the subprocess's stdin/stdout; the
worker's stderr is inherited so its [worker] logs appear in the terminal.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys

from . import config


class AssistantClient:
    def __init__(self):
        env = dict(os.environ)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        # Launch from the project root so `-m src.worker_server` resolves.
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "src.worker_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,      # JSON responses only
            stderr=None,                 # inherit: [worker] logs go to terminal
            text=True,
            bufsize=1,                   # line-buffered
            cwd=str(config.ROOT_DIR),
            env=env,
        )
        ready = self._read()             # blocks until models load (or EOF)
        if not ready.get("ready"):
            raise RuntimeError(
                "Assistant worker failed to start: "
                + ready.get("error", "unknown error")
                + " (see the [worker] lines in the terminal)")
        self.meta = ready.get("meta", {})

    def _read(self) -> dict:
        """Read one JSON response line. Detects a dead worker via EOF."""
        line = self._proc.stdout.readline()
        if line == "":                   # pipe closed -> the worker died
            code = self._proc.poll()
            raise RuntimeError(
                f"Assistant worker process exited (code {code}) - likely a "
                f"native crash while loading models. See the [worker] lines "
                f"in the terminal.")
        return json.loads(line)

    def _rpc(self, req: dict) -> dict:
        self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        return self._read()

    def ask(self, text: str) -> dict:
        return self._rpc({"cmd": "ask", "text": text})

    def ask_audio(self, path: str) -> dict:
        return self._rpc({"cmd": "ask_audio", "path": path})

    def reset(self) -> dict:
        return self._rpc({"cmd": "reset"})

    def state(self) -> dict:
        return self._rpc({"cmd": "state"})


if __name__ == "__main__":
    c = AssistantClient()
    print("meta:", c.meta)
    print("ask :", {k: v for k, v in c.ask("기초연금 신청 방법").items()
                    if k in ("answer", "sources", "is_abnormal")})
    print("state:", c.state())
