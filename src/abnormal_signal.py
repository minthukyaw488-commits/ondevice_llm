"""
Step 3: Abnormal-signal detection layer.

Two independent signals are combined by simple rule-based thresholds:

  (A) Sentiment  - negative emotion probability per utterance, averaged over
                   the recent conversation. Primary backend is a public Korean
                   sentiment model from Hugging Face (no auth needed); a local
                   lexicon fallback keeps it working offline.
  (B) Symptom repetition - the same physical/emotional complaint mentioned
                   >= N times across the conversation history.

If either threshold is crossed (or a crisis keyword appears), an abnormal
signal is raised and a summary is generated for the social worker (사회복지사).

All processing is local. Conversation content never leaves the device.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from . import config


# --------------------------------------------------------------------------
# (A) Sentiment analysis
# --------------------------------------------------------------------------
# Local lexicon fallback: negative / positive Korean cue words. Rough, but
# lets the abnormal-signal logic run and be tested without a model download.
_NEG_WORDS = ["아프", "힘들", "외로", "혼자", "쓸쓸", "우울", "슬프", "죽고 싶",
              "살기 싫", "못 자", "잠이 안", "입맛이 없", "답답", "숨이 차",
              "어지러", "무섭", "불안", "지치", "의욕이 없", "희망이 없", "싫"]
_POS_WORDS = ["좋", "행복", "고맙", "감사", "괜찮", "즐겁", "기쁘", "편안",
              "든든", "다행"]


class SentimentAnalyzer:
    def __init__(self, model_name: str = config.SENTIMENT_MODEL):
        self.model_name = model_name
        self.backend = "fallback-lexicon"
        self._pipe = None
        try:
            from transformers import pipeline
            # device=-1 forces CPU. On Apple Silicon the MPS backend crashes
            # (bus error) when run from Streamlit's worker thread; CPU avoids it.
            self._pipe = pipeline("sentiment-analysis", model=model_name, device=-1)
            self.backend = f"transformers:{model_name}"
        except Exception as exc:
            print(f"[sentiment] model unavailable ({exc.__class__.__name__}); "
                  f"using local Korean lexicon fallback.")

    def negative_score(self, text: str) -> float:
        """Return probability that `text` is negative, in [0, 1].

        The sentiment model is binary and has no neutral class, so plain
        info-seeking questions can score as strongly negative. To avoid false
        alarms we only trust a high negative score when the utterance actually
        contains an emotional/distress cue word; otherwise it is capped at
        neutral (0.5).
        """
        if self._pipe is not None:
            res = self._pipe(text[:512])[0]
            label = res["label"].lower()
            score = float(res["score"])
            # Map various label schemes to a negative probability.
            if any(k in label for k in ("neg", "label_0", "1 star", "부정")):
                neg = score
            elif any(k in label for k in ("pos", "label_1", "5 star", "긍정")):
                neg = 1.0 - score
            else:
                neg = 0.5
        else:
            neg = self._lexicon_score(text)

        # Emotional-cue gate: no distress word -> treat as neutral, not negative.
        if not self._has_negative_cue(text):
            return min(neg, 0.5)
        return neg

    @staticmethod
    def _has_negative_cue(text: str) -> bool:
        return any(w in text for w in _NEG_WORDS)

    @staticmethod
    def _lexicon_score(text: str) -> float:
        neg = sum(text.count(w) for w in _NEG_WORDS)
        pos = sum(text.count(w) for w in _POS_WORDS)
        if neg == 0 and pos == 0:
            return 0.5                      # neutral
        return neg / (neg + pos)


# --------------------------------------------------------------------------
# (B) Symptom keyword tracking + combined decision
# --------------------------------------------------------------------------
@dataclass
class SignalResult:
    is_abnormal: bool
    reasons: List[str]
    avg_negative: float
    symptom_counts: Dict[str, int]
    crisis: bool = False


@dataclass
class AbnormalSignalDetector:
    """Tracks a running conversation and decides when to alert a social worker."""

    sentiment: SentimentAnalyzer = field(default_factory=SentimentAnalyzer)
    repeat_threshold: int = config.SYMPTOM_REPEAT_THRESHOLD
    neg_threshold: float = config.NEGATIVE_SENTIMENT_THRESHOLD
    sentiment_min_utterances: int = config.SENTIMENT_MIN_UTTERANCES

    history: List[str] = field(default_factory=list)
    _neg_scores: List[float] = field(default_factory=list)
    _symptom_counts: Counter = field(default_factory=Counter)

    def add_utterance(self, text: str) -> SignalResult:
        """Record one elderly utterance and re-evaluate the abnormal signal."""
        self.history.append(text)
        self._neg_scores.append(self.sentiment.negative_score(text))

        # Count symptom groups mentioned in this utterance (once per group).
        for group, variants in config.SYMPTOM_KEYWORDS.items():
            if any(v in text for v in variants):
                self._symptom_counts[group] += 1

        return self.evaluate()

    def evaluate(self) -> SignalResult:
        reasons: List[str] = []

        avg_neg = (sum(self._neg_scores) / len(self._neg_scores)
                   if self._neg_scores else 0.0)
        # Only fire the sentiment rule once we have enough conversation to
        # judge a sustained mood (avoids single-question false positives).
        if (len(self._neg_scores) >= self.sentiment_min_utterances
                and avg_neg >= self.neg_threshold):
            reasons.append(f"부정 감정 평균 {avg_neg:.2f} (기준 {self.neg_threshold})")

        repeated = {g: c for g, c in self._symptom_counts.items()
                    if c >= self.repeat_threshold}
        for g, c in repeated.items():
            reasons.append(f"'{g}' 관련 호소 {c}회 반복 (기준 {self.repeat_threshold})")

        crisis = any(kw in " ".join(self.history) for kw in config.CRISIS_KEYWORDS)
        if crisis:
            reasons.append("위기 신호 키워드 감지 (즉시 확인 필요)")

        return SignalResult(
            is_abnormal=bool(reasons),
            reasons=reasons,
            avg_negative=avg_neg,
            symptom_counts=dict(self._symptom_counts),
            crisis=crisis,
        )


if __name__ == "__main__":
    det = AbnormalSignalDetector()
    print("sentiment backend:", det.sentiment.backend, "\n")
    convo = [
        "안녕하세요, 오늘 날씨가 좋네요.",
        "요즘 무릎이 너무 아파요.",
        "어제도 무릎이 아파서 잠을 못 잤어요.",
        "무릎도 아프고 혼자 있으니 너무 외롭고 우울해요.",
    ]
    for utt in convo:
        res = det.add_utterance(utt)
        flag = "🚨 ABNORMAL" if res.is_abnormal else "정상"
        print(f"- {utt}\n    avg_neg={res.avg_negative:.2f} symptoms={res.symptom_counts} -> {flag}")
    print("\nfinal reasons:", det.evaluate().reasons)
