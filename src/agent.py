"""
Phase 1: a bounded, local tool-calling agent (experimental).

The deterministic pipeline (src/pipeline.py) runs a fixed sequence
    route -> retrieve -> rerank -> gate -> answer.
This agent instead lets the LLM *decide* what to do: on each step it picks a
tool, sees the result, and repeats up to AGENT_MAX_STEPS before answering. It
is deliberately kept safe for an on-device welfare setting:

  * Local only        - same Ollama model, no cloud (privacy preserved).
  * Bounded loop      - at most AGENT_MAX_STEPS tool calls (latency + safety).
  * Whitelisted tools - the model can only call the tools registered below.
  * Relevance gate    - if no tool ever finds relevant evidence, it refers the
                        user to the 주민센터 instead of making something up.
  * Graceful fallback - greetings, clearly off-domain questions, a missing LLM,
                        or any invalid tool call fall back to the proven
                        deterministic pipeline, so behaviour never gets worse.

This is Phase 1 of the agentic roadmap (Phase 2: multi-intent planner,
Phase 3: task agents that draft applications / send alerts on confirmation).
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from . import config
from .llm import ANSWER_SYSTEM_PROMPT, OFF_DOMAIN_REPLY, build_answer_prompt
from .pipeline import TurnResult, WelfareAssistant
from .router import is_off_domain, is_smalltalk


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@dataclass
class ToolResult:
    text: str                 # observation shown back to the agent
    sources: List[str]        # document sources gathered (for citation)
    max_score: float = 0.0    # best retrieval score seen (for the gate)


@dataclass
class Tool:
    name: str
    description: str
    args: str                 # human-readable arg spec for the prompt
    run: Callable[..., ToolResult]


class WelfareAgent:
    """A bounded tool-calling wrapper around WelfareAssistant."""

    def __init__(self, assistant: WelfareAssistant | None = None,
                 max_steps: int = config.AGENT_MAX_STEPS):
        self.bot = assistant or WelfareAssistant()
        self.max_steps = max_steps
        self.model = config.AGENT_MODEL
        self.tools: Dict[str, Tool] = self._build_tools()

    # -- tool registry -----------------------------------------------------
    def _build_tools(self) -> Dict[str, Tool]:
        tools = [
            Tool("search_welfare_docs",
                 "복지 제도·서비스 내용을 문서에서 검색합니다 (기초연금·돌봄·급식 등).",
                 '{"query": "검색어"}',
                 self._t_search),
            Tool("lookup_facility",
                 "대전 자치구별 시설·서비스 현황을 찾습니다 (식사배달·복지관·경로당 등).",
                 '{"district": "구 이름(선택)", "service_type": "시설/서비스 유형"}',
                 self._t_facility),
        ]
        return {t.name: t for t in tools}

    def _t_search(self, query: str = "", **_) -> ToolResult:
        hits = self.bot.rag.retrieve(str(query))
        return self._pack(hits)

    def _t_facility(self, district: str = "", service_type: str = "", **_) -> ToolResult:
        # Structured facility data is indexed as text, so a targeted query
        # retrieves the matching rows. Combine district + type into the query.
        q = " ".join(x for x in (str(district), str(service_type), "현황") if x.strip())
        hits = self.bot.rag.retrieve(q)
        return self._pack(hits)

    @staticmethod
    def _pack(hits) -> ToolResult:
        if not hits:
            return ToolResult("검색 결과가 없습니다.", [], 0.0)
        top = hits[0].score
        lines, sources = [], []
        for h in hits:
            lines.append(f"({h.source}) {h.text[:300]}")
            if h.source not in sources:
                sources.append(h.source)
        return ToolResult("\n".join(lines), sources, top)

    # -- main entry --------------------------------------------------------
    def ask(self, question: str) -> TurnResult:
        # Small talk / off-domain / no local LLM -> proven deterministic path
        # (it also records the abnormal signal and dispatches alerts).
        if (not self.bot.llm.available or is_smalltalk(question)
                or is_off_domain(question)):
            return self.bot.ask_text(question)

        # Welfare question -> run the agent loop. Record the signal once here.
        signal = self.bot.detector.add_utterance(question)
        try:
            answer, sources = self._run_loop(question)
        except Exception as exc:               # never fail worse than the pipeline
            print(f"[agent] loop error ({exc.__class__.__name__}); using pipeline.")
            answer, sources = self._rag_answer(question)
        if answer is None:                     # agent gave up -> deterministic RAG
            answer, sources = self._rag_answer(question)
        return TurnResult(question, answer, sources, signal,
                          self.bot._maybe_alert(signal))

    # -- the ReAct-style loop ---------------------------------------------
    def _run_loop(self, question: str) -> Tuple[str | None, List[str]]:
        scratch: List[str] = []
        gathered: List[str] = []
        sources: List[str] = []
        best_score = 0.0

        for step in range(self.max_steps):
            action = self._decide(question, scratch)
            if action is None:                 # unparseable -> bail to fallback
                return None, []
            if "answer" in action:
                # Gate: don't let the agent answer with no real evidence.
                if best_score < config.RAG_MIN_RELEVANCE and not gathered:
                    return OFF_DOMAIN_REPLY, []
                return str(action["answer"]).strip(), sources

            name = action.get("tool")
            tool = self.tools.get(name)
            if tool is None:
                scratch.append(f"관찰: 알 수 없는 도구 '{name}'. 등록된 도구만 사용하세요.")
                continue
            args = action.get("args") or {}
            result = tool.run(**args) if isinstance(args, dict) else tool.run()
            best_score = max(best_score, result.max_score)
            if result.text and result.text != "검색 결과가 없습니다.":
                gathered.append(result.text)
                for s in result.sources:
                    if s not in sources:
                        sources.append(s)
            scratch.append(
                f"행동: {name}({json.dumps(args, ensure_ascii=False)})\n"
                f"관찰: {result.text[:600]}")

        # Ran out of steps: synthesize a grounded answer from what we gathered.
        if best_score < config.RAG_MIN_RELEVANCE and not gathered:
            return OFF_DOMAIN_REPLY, []
        prompt = build_answer_prompt(question, gathered)
        answer = self.bot.llm.generate(prompt, system=ANSWER_SYSTEM_PROMPT)
        return answer, sources

    def _decide(self, question: str, scratch: List[str]) -> dict | None:
        """Ask the LLM for the next action as strict JSON."""
        prompt = self._agent_prompt(question, scratch)
        raw = self.bot.llm.generate(prompt, system=_AGENT_SYSTEM, model=self.model,
                                    num_predict=200, temperature=0.2)
        return _parse_action(raw)

    def _agent_prompt(self, question: str, scratch: List[str]) -> str:
        tool_desc = "\n".join(
            f"- {t.name}{t.args}: {t.description}" for t in self.tools.values())
        history = "\n".join(scratch) if scratch else "(아직 없음)"
        return (
            f"어르신 질문: {question}\n\n"
            f"사용 가능한 도구:\n{tool_desc}\n\n"
            f"지금까지의 행동/관찰:\n{history}\n\n"
            "다음 행동을 JSON 한 줄로만 출력하세요. 두 가지 형식 중 하나입니다:\n"
            '  도구 호출: {"tool": "도구이름", "args": {...}}\n'
            '  최종 답변: {"answer": "어르신께 드릴 한국어 답변"}\n'
            "규칙: 관찰에 근거가 충분하면 answer를 내고, 부족하면 도구를 호출하세요. "
            "설명 없이 JSON만 출력합니다."
        )

    # -- deterministic fallback (same logic as pipeline, no re-signal) ------
    def _rag_answer(self, question: str) -> Tuple[str, List[str]]:
        hits = self.bot.rag.retrieve(question)
        top = hits[0].score if hits else 0.0
        if not hits or top < config.RAG_MIN_RELEVANCE:
            return OFF_DOMAIN_REPLY, []
        contexts = [h.text for h in hits]
        prompt = build_answer_prompt(question, contexts)
        answer = self.bot.llm.generate(prompt, system=ANSWER_SYSTEM_PROMPT)
        return answer, [h.source for h in hits]


_AGENT_SYSTEM = (
    "당신은 대전 독거노인 복지 안내 에이전트입니다. 도구를 사용해 근거를 모은 뒤 "
    "한국어로 정확히 답합니다. 자료에 없는 내용은 지어내지 말고, 반드시 지정된 JSON "
    "형식으로만 응답합니다."
)


def _parse_action(raw: str) -> dict | None:
    """Extract the first JSON object from the model output. Robust to extra text."""
    if not raw:
        return None
    # Grab the first balanced {...} block.
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                blob = raw[start:i + 1]
                try:
                    obj = json.loads(blob)
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


if __name__ == "__main__":
    print("Initialising agent (indexing welfare docs)...\n")
    agent = WelfareAgent()
    mode = "on" if agent.bot.llm.available else "off (deterministic fallback)"
    print(f"agent model={agent.model} | LLM={mode} | max_steps={agent.max_steps}\n")
    for q in ["기초연금은 어떻게 신청하나요?",
              "서구에 재가노인 식사배달 되는 곳 알려줘",
              "우울하고 외로워요"]:
        res = agent.ask(q)
        print("=" * 70)
        print(f"👵 {res.question}")
        print(f"🤖 {res.answer[:240]}")
        print(f"   📄 {res.sources}")
        if res.alert:
            print("🚨 (사회복지사 알림 생성됨)")
        print()
