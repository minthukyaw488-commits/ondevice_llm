"""
Model comparison for the welfare Q&A assistant.

Runs the SAME labelled question set (eval_qa.QA_CASES) through several LLMs on
the SAME 대전 RAG index, so the only thing that changes is the answer model.
Prints a side-by-side table of the metrics that matter for this project and a
recommendation, so the choice of model can be justified with numbers.

For a fair, fast, apples-to-apples comparison of GENERATION quality this runs
the deterministic RAG path (USE_AGENT off) and turns the LLM signal judge off,
so each question makes exactly one answer call per model.

Usage:
  # Groq models (needs OPENAI_API_KEY=gsk_..., OPENAI_BASE_URL=groq):
  export LLM_BACKEND=openai
  export OPENAI_API_KEY=gsk_...
  export OPENAI_BASE_URL=https://api.groq.com/openai/v1
  python compare_models.py

  # Pick which models (comma-separated); prefix ollama: for a local model:
  COMPARE_MODELS="qwen/qwen3.8-27b,openai/gpt-oss-20b,ollama:exaone3.5:2.4b" python compare_models.py

  # Slow down if you hit Groq rate limits (429):
  COMPARE_DELAY=2 python compare_models.py
"""
from __future__ import annotations
import os
# Fair, fast comparison: one answer call per question, no agent / no LLM judge.
os.environ.setdefault("USE_AGENT", "0")
os.environ.setdefault("USE_LLM_SIGNAL", "0")

import time

from src import config
from src.llm import OpenAILLM, LocalLLM
from src.pipeline import WelfareAssistant
from eval_qa import QA_CASES, run_qa_eval

# Candidate models. Groq text models by default; prefix "ollama:" for a local
# model (compared as an on-device baseline). Override with COMPARE_MODELS.
DEFAULT_MODELS = "qwen/qwen3.8-27b,openai/gpt-oss-120b,openai/gpt-oss-20b"
MODELS = [m.strip() for m in
          os.environ.get("COMPARE_MODELS", DEFAULT_MODELS).split(",") if m.strip()]
DELAY = float(os.environ.get("COMPARE_DELAY", "0"))


def _make_llm(name: str):
    """Build the LLM for a candidate. 'ollama:<model>' -> local; else the
    configured API backend (Groq/OpenAI) with this model name."""
    if name.startswith("ollama:"):
        return LocalLLM(model=name.split(":", 1)[1])
    return OpenAILLM(model=name)


def _with_delay(llm):
    """Wrap generate() with a sleep so free-tier rate limits aren't tripped."""
    if DELAY <= 0:
        return llm
    orig = llm.generate

    def slow(*a, **k):
        time.sleep(DELAY)
        return orig(*a, **k)

    llm.generate = slow
    return llm


def main():
    print("Loading pipeline (indexing welfare docs once)...\n")
    bot = WelfareAssistant()
    bot.use_agent = False                     # deterministic path for fairness
    cases = QA_CASES
    n_in = sum(c.scope == "in" for c in cases)
    n_out = sum(c.scope == "out" for c in cases)
    print(f"RAG        : {bot.rag.backend} / {bot.rag.embedder.backend}")
    print(f"평가 문항  : {len(cases)}개 (정보 {n_in} / 범위밖 {n_out})")
    print(f"비교 모델  : {', '.join(MODELS)}")
    print(f"설정       : USE_AGENT=off, LLM_signal=off, delay={DELAY}s\n")

    results = []
    for name in MODELS:
        llm = _make_llm(name)
        if not getattr(llm, "available", False):
            print(f"  ⏭  {name}: 사용 불가(키 없음/모델 미접속) → 건너뜀")
            continue
        bot.llm = _with_delay(llm)
        print(f"  ▶ 평가 중: {name} ...", flush=True)
        t0 = time.time()
        r = run_qa_eval(bot, cases)
        r["_name"] = name
        r["_wall"] = time.time() - t0
        results.append(r)

    if not results:
        print("\n평가된 모델이 없습니다. OPENAI_API_KEY / OPENAI_BASE_URL 또는 "
              "Ollama 실행 상태를 확인하세요.")
        return

    # --- comparison table ------------------------------------------------
    def pct(x, tot):
        return f"{x/tot:.0%}" if tot else "-"

    print("\n" + "=" * 92)
    print(" 모델 비교 (동일 대전 RAG · 동일 문항)")
    print("=" * 92)
    header = (f"{'모델':28} {'정보정답':>7} {'거절':>6} {'한국어':>7} "
              f"{'간결':>6} {'전체':>6} {'응답(s)':>8} {'길이':>6}")
    print(header)
    print("-" * 92)
    for r in results:
        n = r["n"]
        ip, it = r["scope"]["in"]
        op, ot = r["scope"]["out"]
        print(f"{r['_name']:28} {pct(ip, it):>7} {pct(op, ot):>6} "
              f"{pct(r['ko'], n):>7} {pct(r['concise'], n):>6} "
              f"{pct(r['ok'], n):>6} {r['avg_latency']:>8.2f} {r['avg_len']:>5.0f}자")
    print("=" * 92)

    # --- recommendation: best info-accuracy, tie-break 한국어 then latency ---
    def key(r):
        ip, it = r["scope"]["in"]
        return (ip / it if it else 0, r["ko"] / r["n"], -r["avg_latency"])

    best = max(results, key=key)
    ip, it = best["scope"]["in"]
    print(f"\n추천 모델 → {best['_name']}")
    print(f"  · 정보 질문 정답률 {pct(ip, it)}, 한국어 준수 {pct(best['ko'], best['n'])}, "
          f"평균 응답 {best['avg_latency']:.2f}초")
    print("  · 기준: 정보 정답률 > 한국어 준수 > 응답속도 (동일 RAG 기준)")


if __name__ == "__main__":
    main()
