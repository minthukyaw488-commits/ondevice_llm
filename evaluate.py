"""
Evaluation harness for the welfare assistant.

Produces quantitative numbers for the project report:

  1. RAG retrieval  - does the right welfare document come back for a question?
                      Metrics: Hit@1, Hit@3 over labelled question/topic pairs.
  2. Abnormal signal - are risky conversations flagged and normal ones not?
                      Metrics: accuracy, precision, recall, F1 on the "abnormal"
                      class over labelled sample conversations.

Runs fully locally. Uses whatever backend is available (real models where they
load, offline fallbacks otherwise), so it works in any environment.

Usage:  python evaluate.py
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from src.abnormal_signal import AbnormalSignalDetector, SentimentAnalyzer
from src.rag_pipeline import RagPipeline


# --------------------------------------------------------------------------
# Labelled evaluation data
# --------------------------------------------------------------------------
# (question, keyword that MUST appear in a correctly retrieved chunk)
RAG_CASES: List[Tuple[str, str]] = [
    ("기초연금은 어떻게 신청하나요?", "기초연금"),
    ("혼자 사는데 응급상황이 걱정돼요", "응급안전"),
    ("치매 검진 무료로 받고 싶어요", "치매안심센터"),
    ("무료로 밥 먹을 수 있는 곳 있나요?", "급식"),
    ("겨울 난방비 지원 받을 수 있나요?", "바우처"),
    ("우울하고 외로울 때 상담받고 싶어요", "정신건강"),
    ("집에서 혈압 건강 체크 받고 싶어요", "방문건강"),
    ("일자리를 구하고 싶어요", "일자리"),
    ("지하철 무료로 탈 수 있나요?", "경로우대"),
    ("생활이 어려운데 돌봄 서비스 받고 싶어요", "맞춤돌봄"),
]

# (is_abnormal, [utterances in order])
SIGNAL_CASES: List[Tuple[bool, List[str]]] = [
    (False, ["기초연금 신청 방법 알려주세요", "지하철은 무료로 타나요?", "고맙습니다"]),
    (False, ["치매 검진 무료인가요?", "방문 건강관리 신청은 어떻게 하나요?"]),
    (False, ["오늘 날씨가 참 좋네요", "산책하기 좋은 날이에요"]),
    (False, ["복지관에 어떤 프로그램이 있나요?", "노인 일자리도 신청하고 싶어요"]),
    (True,  ["요즘 무릎이 아파요", "어제도 무릎이 아팠어요", "무릎이 계속 아프고 힘들어요"]),
    (True,  ["허리가 아파요", "허리가 아파서 잠도 못 자요", "또 허리가 아프네요"]),
    (True,  ["너무 외롭고 쓸쓸해요", "매일 우울하고 힘들어요", "사는 게 의욕이 없어요"]),
    (True,  ["사는 게 너무 힘들고 죽고 싶어요"]),
]


@dataclass
class Metrics:
    def rate(self, num, den):
        return num / den if den else 0.0


# --------------------------------------------------------------------------
# RAG evaluation
# --------------------------------------------------------------------------
def evaluate_rag(rag: RagPipeline) -> dict:
    hit1 = hit3 = 0
    rows = []
    for question, keyword in RAG_CASES:
        results = rag.retrieve(question, top_k=3)
        texts = [r.text for r in results]
        in_top1 = bool(texts) and keyword in texts[0]
        in_top3 = any(keyword in t for t in texts)
        hit1 += in_top1
        hit3 += in_top3
        rows.append((question, keyword, in_top1, in_top3))
    n = len(RAG_CASES)
    return {"n": n, "hit1": hit1, "hit3": hit3,
            "hit1_rate": hit1 / n, "hit3_rate": hit3 / n, "rows": rows}


# --------------------------------------------------------------------------
# Abnormal-signal evaluation
# --------------------------------------------------------------------------
def evaluate_signal() -> dict:
    tp = tn = fp = fn = 0
    rows = []
    shared_sentiment = SentimentAnalyzer()      # load the model only once
    for is_abnormal, utterances in SIGNAL_CASES:
        # Fresh history per conversation, but reuse the loaded sentiment model.
        det = AbnormalSignalDetector(sentiment=shared_sentiment)
        result = None
        for u in utterances:
            result = det.add_utterance(u)
        pred = result.is_abnormal
        if is_abnormal and pred:
            tp += 1
        elif is_abnormal and not pred:
            fn += 1
        elif not is_abnormal and pred:
            fp += 1
        else:
            tn += 1
        rows.append((is_abnormal, pred, utterances[-1]))
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / total if total else 0.0,
            "precision": precision, "recall": recall, "f1": f1, "rows": rows}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def main():
    print("Loading pipeline (indexing welfare docs)...\n")
    rag = RagPipeline()
    rag.index()
    print(f"RAG backend : {rag.backend} / {rag.embedder.backend}\n")

    print("=" * 64)
    print(" 1) RAG RETRIEVAL")
    print("=" * 64)
    r = evaluate_rag(rag)
    for q, kw, h1, h3 in r["rows"]:
        mark = "✓@1" if h1 else ("·@3" if h3 else "✗  ")
        print(f"  [{mark}] {q}  (기대: {kw})")
    print(f"\n  Hit@1 = {r['hit1']}/{r['n']} = {r['hit1_rate']:.0%}    "
          f"Hit@3 = {r['hit3']}/{r['n']} = {r['hit3_rate']:.0%}")

    print("\n" + "=" * 64)
    print(" 2) ABNORMAL-SIGNAL DETECTION")
    print("=" * 64)
    s = evaluate_signal()
    for truth, pred, last in s["rows"]:
        ok = "✓" if truth == pred else "✗"
        t = "이상" if truth else "정상"
        p = "이상" if pred else "정상"
        print(f"  [{ok}] 정답={t} 예측={p}  ...\"{last}\"")
    print(f"\n  Confusion:  TP={s['tp']} TN={s['tn']} FP={s['fp']} FN={s['fn']}")
    print(f"  Accuracy={s['accuracy']:.0%}  Precision={s['precision']:.0%}  "
          f"Recall={s['recall']:.0%}  F1={s['f1']:.2f}")

    print("\n" + "=" * 64)
    print(" SUMMARY")
    print("=" * 64)
    print(f"  RAG   : Hit@1 {r['hit1_rate']:.0%} | Hit@3 {r['hit3_rate']:.0%}")
    print(f"  Signal: Acc {s['accuracy']:.0%} | P {s['precision']:.0%} | "
          f"R {s['recall']:.0%} | F1 {s['f1']:.2f}")
    print("\n  Note: numbers improve further with real bge-m3 + the HF Korean\n"
          "  sentiment model (semantic retrieval vs. the TF-IDF/lexicon fallback).")


if __name__ == "__main__":
    main()
