"""Embedding models. Heavy imports are lazy so tests can run with fakes and no torch."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...   # (n, d) float32, L2-normalised
    def embed_query(self, text: str) -> np.ndarray: ...              # (d,)   float32, L2-normalised


class SentenceTransformerEmbedder:
    """BGE-style bi-encoder. Queries get an instruction prefix (asymmetric retrieval),
    passages do not - this is how bge-*-v1.5 was trained."""

    def __init__(self, model_name: str, device: str = "cpu", query_instruction: str = ""):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.query_instruction = query_instruction
        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=32, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._model.encode([self.query_instruction + text], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)[0]
        return np.asarray(vec, dtype=np.float32)
