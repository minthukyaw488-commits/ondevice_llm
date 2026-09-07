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
    "당신은 대전광역시 독거노인을 돕는 친절한 복지 안내 도우미입니다.\n"
    "규칙:\n"
    "1) 아래 '참고 자료'에 실제로 적힌 내용만 근거로 답합니다. "
    "자료에 없는 금액·전화번호·기관명·조건을 절대 지어내지 않습니다.\n"
    "2) 참고 자료에서 질문과 관련된 내용을 찾을 수 없으면, 다른 말은 하지 말고 "
    "'가까운 주민센터(행정복지센터)에 문의하시면 자세히 안내받으실 수 있습니다.'라고만 답합니다.\n"
    "3) 반드시 한국어로만 답합니다. 영어 단어·알파벳·웹주소(URL)를 쓰지 않으며, "
    "참고 자료에 영문 기관명이나 주소가 있어도 한글로 옮기거나 생략합니다.\n"
    "3-1) 질문의 핵심 주제가 참고 자료의 주제와 다르면(예: 반려동물·자동차·주식·"
    "여행 등 노인복지와 무관한 질문) 억지로 자료를 갖다 붙이지 말고, 2)의 안내 문장만 "
    "답합니다.\n"
    "4) 핵심만 2~3문장(200자 이내)으로 짧고 쉽게, 공손한 존댓말로 답합니다. "
    "여러 시설·기관을 길게 나열하지 말고, '어디서/어떻게 신청'을 우선 안내합니다.\n"
    "5) 비용(무료/유료·금액)이나 자격을 물으면, 참고 자료에 명시된 경우에만 그대로 "
    "답합니다. 자료에 '무료'나 금액이 없으면 '국가 보조금' 같은 추측을 하지 말고 "
    "주민센터 문의로 안내합니다.\n"
    "예시)\n"
    "질문: 기초연금은 어떻게 신청하나요?\n"
    "답변: 만 65세 이상이시면 신분증과 통장 사본을 가지고 가까운 주민센터나 "
    "국민연금공단 지사에서 신청하실 수 있습니다."
)


# Warm, chatbot-like replies for greetings / small talk (no RAG, no refusal).
SMALLTALK_SYSTEM_PROMPT = (
    "당신은 대전광역시 독거노인을 돕는 따뜻하고 친근한 복지 도우미입니다. "
    "어르신의 인사나 가벼운 안부에 1~2문장으로 다정하고 공손하게 한국어로 답합니다. "
    "사실이나 숫자를 지어내지 말고, 자연스럽게 안부를 나눈 뒤 '복지 관련해 궁금하신 점을 "
    "편하게 말씀해 주세요'처럼 부드럽게 안내합니다. 영어 단어나 이모지는 쓰지 않습니다."
)

# Shown when retrieval finds nothing relevant (off-topic or not in the docs):
# warm, not a cold refusal, and it steers the user back to what we can help with.
OFF_DOMAIN_REPLY = (
    "저는 대전 어르신 복지 안내를 도와드리고 있어요. 방금 말씀은 제가 가진 자료로는 "
    "정확히 알려드리기 어렵네요. 기초연금, 돌봄, 건강, 일자리 같은 복지 관련이라면 편하게 "
    "물어봐 주시고, 자세한 문의는 가까운 주민센터(행정복지센터)로 연락하시면 됩니다."
)


# Smaller models answer far faster on-device. Prefer these for low latency;
# fall back to whatever is installed so nothing breaks if they are absent.
FAST_MODELS = ["exaone3.5:2.4b", "qwen2.5:1.5b", "llama3.2:1b", "qwen2.5:0.5b",
               "gemma2:2b", "llama3.2:3b"]


class LocalLLM:
    def __init__(self, model: str = config.LLM_MODEL, host: str = config.OLLAMA_HOST):
        self.host = host.rstrip("/")
        self.model = model
        self.available = self._ping_and_pick(model)

    def _ping_and_pick(self, preferred: str) -> bool:
        """Ping Ollama and choose the model: preferred if present, else the
        fastest installed model, else the first available one."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                installed = [m["name"] for m in json.loads(resp.read()).get("models", [])]
        except Exception:
            return False
        names = set(installed) | {n.split(":")[0] for n in installed}
        if preferred in names:
            self.model = preferred
        else:
            self.model = next((m for m in FAST_MODELS if m in names),
                              installed[0] if installed else preferred)
        return True

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
                "repeat_penalty": 1.3,    # stop small models looping the same phrase
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
    if contexts:
        joined = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    else:
        joined = "(관련 자료 없음)"
    return (f"질문: {question}\n\n"
            f"참고 자료 (아래 내용에만 근거해 답하세요):\n{joined}\n\n"
            f"위 자료만 사용해 한국어 1~2문장으로 답하세요. 자료에 없으면 "
            f"주민센터 문의 안내만 하세요.\n답변:")


if __name__ == "__main__":
    llm = LocalLLM()
    print("Ollama available:", llm.available, f"(model={llm.model})")
    prompt = build_answer_prompt(
        "기초연금은 어떻게 신청하나요?",
        ["기초연금은 만 65세 이상이고 소득 기준 이하인 어르신에게 매월 지급되며, "
         "주민센터나 국민연금공단 지사에서 신청합니다."])
    print(llm.generate(prompt, system=ANSWER_SYSTEM_PROMPT))
