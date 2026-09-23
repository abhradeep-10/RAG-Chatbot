"""Hybrid retrieval: dense (bi-encoder) + sparse (BM25) -> Reciprocal Rank Fusion
-> optional cross-encoder rerank -> relevance gate."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from ragbot.config import Settings
from ragbot.retrieval.embeddings import Embedder
from ragbot.retrieval.reranker import Reranker
from ragbot.retrieval.store import DocumentIndex
from ragbot.schemas import RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[_\-'][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an the and or of to in on for with by at from as is are was were be been it its this that "
    "these those what which who whom how does do did can could should would will may about into "
    "than then there their they them i you we our your he she his her not no".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d)). Uses ranks only, so the
    incomparable scales of cosine similarity and BM25 never need to be normalised."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


@dataclass
class RetrievalResult:
    candidates: list[RetrievedChunk]   # full ranked candidate list (for observability / eval)
    selected: list[RetrievedChunk]     # what goes to the LLM (empty if the gate failed)
    gate_passed: bool
    gate_reason: str


class HybridRetriever:
    def __init__(self, index: DocumentIndex, embedder: Embedder, reranker: Reranker | None,
                 settings: Settings):
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.s = settings
        corpus = [tokenize(c.embed_text) or ["_empty_"] for c in index.chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def retrieve(self, query: str) -> RetrievalResult:
        s = self.s
        chunks = self.index.chunks
        if not chunks or not query.strip():
            return RetrievalResult([], [], False, "empty index or empty query")

        # dense: exact cosine (vectors are normalised, so dot product == cosine)
        q_vec = self.embedder.embed_query(query)
        dense = self.index.embeddings @ q_vec
        dense_order = np.argsort(-dense)[: s.dense_top_k].tolist()

        # sparse: BM25 over heading-prefixed chunk text (exact terms, IDs, numbers, file names)
        q_tokens = tokenize(query)
        sparse = self._bm25.get_scores(q_tokens) if (q_tokens and self._bm25) else np.zeros(len(chunks))
        sparse_order = [i for i in np.argsort(-sparse)[: s.sparse_top_k].tolist() if sparse[i] > 0]

        fused = rrf_fuse([dense_order, sparse_order], s.rrf_k)
        d_rank = {i: r for r, i in enumerate(dense_order)}
        s_rank = {i: r for r, i in enumerate(sparse_order)}
        ranked = sorted(fused, key=lambda i: -fused[i])[: max(s.dense_top_k, s.sparse_top_k)]
        cands = [
            RetrievedChunk(chunk=chunks[i], dense_score=float(dense[i]), dense_rank=d_rank.get(i),
                           sparse_score=float(sparse[i]), sparse_rank=s_rank.get(i), fused_score=fused[i])
            for i in ranked
        ]
        if not cands:
            return RetrievalResult([], [], False, "no candidates")

        if self.reranker is not None:
            scores = self.reranker.score(query, [c.chunk.embed_text for c in cands])
            for c, sc in zip(cands, scores):
                c.rerank_score = float(sc)
            cands.sort(key=lambda c: -(c.rerank_score or 0.0))
            best = cands[0].rerank_score or 0.0
            passed = best >= s.min_rerank_score
            selected = [c for c in cands if (c.rerank_score or 0.0) >= s.min_rerank_score][: s.final_top_k]
            reason = f"best rerank score {best:.3f} (threshold {s.min_rerank_score})"
        else:
            best = float(dense.max())
            passed = best >= s.min_dense_score
            selected = cands[: s.final_top_k]
            reason = f"best dense cosine {best:.3f} (threshold {s.min_dense_score})"

        return RetrievalResult(cands, selected if passed else [], passed, reason)
