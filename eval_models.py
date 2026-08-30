"""
Compare local LLMs on the welfare Q&A task, side by side.

Runs the same labelled question set (eval_qa.QA_CASES) through several Ollama
models and prints one comparison table: answer quality (pass / fact / Korean /
concise) plus speed (avg latency). The RAG index is built once and only the LLM
is swapped between models, so it is fast and fair.

Usage:
    python eval_models.py                                  # all installed FAST_MODELS
    python eval_models.py exaone3.5:2.4b qwen2.5:1.5b      # specific models

Only models already pulled in Ollama are tested; missing ones are skipped with
a note (pull them first with e.g. `ollama pull qwen2.5:1.5b`).
"""
from __future__ import annotations
import sys

from eval_qa import LEN_LIMIT, run_qa_eval
from src.llm import FAST_MODELS, LocalLLM
from src.pipeline import WelfareAssistant


def installed_models(host: str) -> set:
    """Names currently pulled in Ollama (both 'name:tag' and bare 'name')."""
    probe = LocalLLM(model="__none__", host=host)
    if not probe.available:
        return set()
    import json
    import urllib.request
    with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=3) as resp:
        names = [m["name"] for m in json.loads(resp.read()).get("models", [])]
    return set(names) | {n.split(":")[0] for n in names}


def main():
    wanted = sys.argv[1:] or FAST_MODELS
    print("Loading pipeline once (indexing welfare docs)…\n")
    bot = WelfareAssistant()
    print(f"RAG backend : {bot.rag.backend} / {bot.rag.embedder.backend}")

    have = installed_models(bot.llm.host)
    if not have:
        print("\n[!] Ollama에 연결할 수 없습니다. `ollama serve` 후 다시 실행하세요.")
        return
    todo = [m for m in wanted if m in have]
    skipped = [m for m in wanted if m not in have]
    if skipped:
        print(f"건너뜀(미설치): {', '.join(skipped)}  →  ollama pull <model>")
    if not todo:
        print("\n[!] 테스트할 설치된 모델이 없습니다.")
        return
    print(f"비교할 모델: {', '.join(todo)}\n")

    rows = []
    for m in todo:
        print(f"── 평가 중: {m} …", flush=True)
        bot.llm = LocalLLM(model=m, host=bot.llm.host)   # swap LLM, keep RAG index
        if bot.llm.model != m:                            # safety: fell back
            print(f"   (건너뜀: {m} 로드 실패)")
            continue
        r = run_qa_eval(bot, verbose=False)
        rows.append((m, r))

    if not rows:
        print("\n[!] 평가된 모델이 없습니다.")
        return
    print("\n" + "=" * 78)
    print(" 모델 비교  (welfare Q&A, 높을수록 좋음 / 시간은 낮을수록 좋음)")
    print("=" * 78)
    hdr = f"{'model':22} {'pass':>6} {'fact':>6} {'한국어':>7} {'간결':>6} {'평균자':>7} {'초/응답':>8}"
    print(hdr)
    print("-" * 78)
    # sort by pass rate desc, then latency asc
    for m, r in sorted(rows, key=lambda x: (-x[1]["ok"], x[1]["avg_latency"])):
        nn = r["n"]
        print(f"{m:22} {r['ok']/nn:>6.0%} {r['fact']/nn:>6.0%} {r['ko']/nn:>7.0%} "
              f"{r['concise']/nn:>6.0%} {r['avg_len']:>7.0f} {r['avg_latency']:>8.2f}")
    print("-" * 78)
    print(f"  기준: pass = fact+한국어+간결 모두 충족 | 간결 = {LEN_LIMIT}자 이하 | 문항 {rows[0][1]['n']}개")
    print("  권장: pass율이 가장 높으면서 응답 시간이 수용 가능한 모델을 선택하세요.")


if __name__ == "__main__":
    main()
