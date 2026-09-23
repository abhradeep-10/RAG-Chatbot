"""Cross-encoder reranker: scores (query, passage) pairs jointly - far more precise than
bi-encoder cosine, and the score is usable as a calibrated-ish relevance gate."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...   # higher = more relevant, in [0, 1]


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self._model = CrossEncoder(model_name, device=device, max_length=512)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        raw = np.asarray(self._model.predict([(query, t) for t in texts], show_progress_bar=False),
                         dtype=np.float64).reshape(-1)
        # sentence-transformers applies a sigmoid for single-label models in recent versions;
        # if we still received raw logits, squash them so thresholds are on a [0, 1] scale.
        if raw.min() < 0.0 or raw.max() > 1.0:
            raw = 1.0 / (1.0 + np.exp(-raw))
        return raw.tolist()
