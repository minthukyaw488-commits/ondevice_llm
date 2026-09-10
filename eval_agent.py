"""
Compare the deterministic pipeline vs the agentic (planner) pipeline on the
same labelled QA set, using the same scoring as eval_qa.py.

Both share ONE WelfareAssistant (one index, one LLM), so the only difference is
routing: WelfareAssistant.ask_text (fixed) vs WelfareAgent.ask (plan/tool loop).

Usage:
    conda activate welfare
    python eval_agent.py                      # full set (slow: agent = many LLM calls)
    EVAL_LIMIT=30 python eval_agent.py        # quick balanced subset (15 in / 15 out)
    AGENT_MODEL=llama3.1:latest EVAL_LIMIT=30 python eval_agent.py   # better planner
"""
from __future__ import annotations
import os
import time
from typing import List

from eval_qa import QA, load_cases, run_qa_eval
from src.agent import WelfareAgent
from src.pipeline import WelfareAssistant


def sample_balanced(cases: List[QA], limit: int) -> List[QA]:
    ins = [c for c in cases if c.scope == "in"]
    outs = [c for c in cases if c.scope == "out"]
    half = limit // 2
    return ins[:half] + outs[:limit - half]


def _pct(part: int, total: int) -> str:
    return f"{part/total*100:4.0f}%" if total else "  - "


def _scope_pct(r: dict, scope: str) -> str:
    p, t = r["scope"][scope]
    return f"{p}/{t} ({p/t*100:.0f}%)" if t else "-"


def _row(name: str, det, ag) -> str:
    return f"  {name:<22} {det:>16}   {ag:>16}"


def main() -> None:
    cases = load_cases()
    limit = int(os.environ.get("EVAL_LIMIT", "0"))
    if limit:
        cases = sample_balanced(cases, limit)
    n = len(cases)

    print("Loading pipeline (indexing welfare docs + LLM)...\n")
    assistant = WelfareAssistant()
    agent = WelfareAgent(assistant=assistant)     # reuse the same index + LLM
    print(f"RAG={assistant.rag.backend}/{assistant.rag.embedder.backend} | "
          f"answer LLM={assistant.llm.model if assistant.llm.available else 'offline'} | "
          f"agent model={agent.model} | agent mode={agent.mode}")
    print(f"평가 문항: {n}개 "
          f"(정보질문 {sum(c.scope=='in' for c in cases)} / "
          f"범위밖 {sum(c.scope=='out' for c in cases)})\n")

    print("→ 결정형(deterministic) 실행 중...")
    t0 = time.time()
    det = run_qa_eval(assistant, cases)
    det_wall = time.time() - t0

    print("→ 에이전트(agentic planner) 실행 중... (LLM 호출이 많아 느립니다)")
    t0 = time.time()
    ag = run_qa_eval(agent, cases)
    ag_wall = time.time() - t0

    print("\n" + "=" * 62)
    print(f"  {'지표':<22} {'결정형':>16}   {'에이전트':>16}")
    print("-" * 62)
    print(_row("전체 정답률(pass)", _pct(det['ok'], n), _pct(ag['ok'], n)))
    print(_row("정보질문(in)", _scope_pct(det, 'in'), _scope_pct(ag, 'in')))
    print(_row("범위밖 거절(out)", _scope_pct(det, 'out'), _scope_pct(ag, 'out')))
    print(_row("사실 정확도(fact)", _pct(det['fact'], n), _pct(ag['fact'], n)))
    print(_row("한국어 준수(ko)", _pct(det['ko'], n), _pct(ag['ko'], n)))
    print(_row("간결성(<=220자)", _pct(det['concise'], n), _pct(ag['concise'], n)))
    print(_row("평균 답변 길이", f"{det['avg_len']:.0f}자", f"{ag['avg_len']:.0f}자"))
    print(_row("평균 응답(초/문항)", f"{det['avg_latency']:.2f}s", f"{ag['avg_latency']:.2f}s"))
    print(_row("총 실행 시간", f"{det_wall:.0f}s", f"{ag_wall:.0f}s"))
    print("=" * 62)
    print("\n해석: 에이전트는 복합 질문 분해·도구 선택이 가능하지만 LLM 호출이 많아 "
          "느립니다. 소형 모델일수록 분해/도구선택이 불안정해 결정형보다 낮을 수 있으며, "
          "planner에 더 큰 모델(예: llama3.1)을 쓰면 개선됩니다.")


if __name__ == "__main__":
    main()
