"""
Local LLM client (Ollama).

Ollama runs the model on-device at http://localhost:11434. This is a local
process, NOT a cloud API - it satisfies the strict "no external LLM" rule.

If the Ollama server is not running, a transparent template responder is used
so the full pipeline can still be demonstrated. The template simply relays the
retrieved welfare text; it is clearly marked so it is never mistaken for a
real model answer.
"""
from __future__ import annotations
import json
import urllib.error
import urllib.request
from typing import List

from . import config


ANSWER_SYSTEM_PROMPT = (
    "당신은 대전광역시 독거노인을 돕는 친절한 복지 안내 도우미입니다. "
    "아래 '참고 자료'에 있는 내용만 근거로 답하세요. "
    "반드시 1~2문장으로 아주 짧고 간단하게, 공손한 존댓말로 답하세요. "
    "자료에 없으면 '가까운 주민센터에 문의하세요'라고만 답하세요."
)


class LocalLLM:
    def __init__(self, model: str = config.LLM_MODEL, host: str = config.OLLAMA_HOST):
        self.model = model
        self.host = host.rstrip("/")
        self.available = self._ping()

    def _ping(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=3):
                return True
        except Exception:
            return False

    def generate(self, prompt: str, system: str = "") -> str:
        if not self.available:
            return self._fallback(prompt)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": "30m",          # keep the model loaded -> no reload latency
            "options": {
                "num_predict": 130,       # short answers -> much less to generate
                "num_ctx": 2048,          # smaller context -> faster prompt eval
                "temperature": 0.3,
                "top_k": 20,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=data,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
                return json.loads(resp.read())["response"].strip()
        except (urllib.error.URLError, KeyError, TimeoutError) as exc:
            return self._fallback(prompt, error=str(exc))

    @staticmethod
    def _fallback(prompt: str, error: str = "") -> str:
        note = "[로컬 LLM(Ollama) 미실행 - 아래는 검색된 자료 요약입니다]"
        # The prompt already carries the retrieved context; return it plainly.
        return f"{note}\n{prompt.split('참고 자료:', 1)[-1].strip()[:600]}"


def build_answer_prompt(question: str, contexts: List[str]) -> str:
    """Combine the question with retrieved welfare chunks into an LLM prompt."""
    joined = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return f"질문: {question}\n\n참고 자료:\n{joined}\n\n답변:"


if __name__ == "__main__":
    llm = LocalLLM()
    print("Ollama available:", llm.available, f"(model={llm.model})")
    prompt = build_answer_prompt(
        "기초연금은 어떻게 신청하나요?",
        ["기초연금은 만 65세 이상이고 소득 기준 이하인 어르신에게 매월 지급되며, "
         "주민센터나 국민연금공단 지사에서 신청합니다."])
    print(llm.generate(prompt, system=ANSWER_SYSTEM_PROMPT))
