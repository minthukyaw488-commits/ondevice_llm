"""
Cross-encoder reranker (BAAI/bge-reranker-v2-m3), run locally.

A bi-encoder (bge-m3) embeds the question and each chunk separately, so a chunk
that merely shares keywords ("기초연금수급자" in a 급식 passage) can outrank the
chunk that actually answers the question. A cross-encoder reads the pair
(question, chunk) together and scores true relevance, giving much better
ordering. The pipeline retrieves a larger candidate pool by embedding, then
reranks down to the final top-K.

If the reranker model can't be loaded (offline / not downloaded), reranking is
skipped and the embedding order is kept - the pipeline never breaks.
"""
from __future__ import annotations
import math
from typing import List, Tuple

from . import config


class Reranker:
    def __init__(self, model_name: str = config.RERANK_MODEL):
        self.model_name = model_name
        self.model = None
        self.available = False
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            self.available = True
        except Exception as exc:                     # offline, not installed, etc.
            print(f"[rerank] '{self.model_name}' unavailable ({exc.__class__.__name__}); "
                  f"keeping embedding order. It loads automatically where available.")
            self.available = False

    def rerank(self, query: str, docs: List[str]) -> List[Tuple[int, float]]:
        """Return [(original_index, relevance 0-1)] sorted most-relevant first.

        Falls back to the input order (score 0.0) when the model is unavailable.
        """
        if not self.available or not docs:
            return [(i, 0.0) for i in range(len(docs))]
        scores = self.model.predict([[query, d] for d in docs])
        order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
        # bge-reranker outputs a logit; sigmoid maps it to a 0-1 relevance.
        return [(i, 1.0 / (1.0 + math.exp(-float(scores[i])))) for i in order]
