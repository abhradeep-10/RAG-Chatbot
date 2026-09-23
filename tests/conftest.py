"""Shared fixtures: synthetic PDFs, and deterministic fakes for the embedder, reranker and LLM
so the test-suite runs offline in seconds (no model downloads, no LLM server)."""
from __future__ import annotations

import json
import re
import zlib

import fitz
import numpy as np
import pytest

from ragbot.config import Settings
from ragbot.retrieval.store import IndexStore
from ragbot.service import RAGService

# (text, font size, bold)
HANDBOOK_PAGES = [
    [("Grant Program Handbook", 20, True),
     ("Eligibility Requirements", 15, True),
     ("Applicants must be residents of the state for at least two years.", 11, False),
     ("Applicants must hold a valid business registration certificate.", 11, False),
     ("The maximum amount allowed is 50 units per applicant.", 11, False)],
    [("International Applicants", 15, True),
     ("International applicants must provide a sponsor letter from a partner.", 11, False),
     ("International applicants are limited to 20 units per year.", 11, False),
     ("Responsibilities", 15, True),
     ("The program office is responsible for reviewing every application.", 11, False)],
    [("Appeals Process", 15, True),
     ("Appeals must be filed within 30 days of a funding decision.", 11, False),
     ("The appeals committee meets once per month to review appeals.", 11, False)],
]


def make_pdf(pages, header: str | None = None) -> bytes:
    doc = fitz.open()
    for n, items in enumerate(pages, start=1):
        page = doc.new_page()
        if header:
            page.insert_text((72, 30), header.format(n=n), fontsize=9, fontname="helv")
        y = 130
        for text, size, bold in items:
            page.insert_text((72, y), text, fontsize=size, fontname="hebo" if bold else "helv")
            y += size * 2
    data = doc.tobytes()
    doc.close()
    return data


class FakeEmbedder:
    """Hashed bag-of-words; cosine == lexical overlap. Deterministic and fast."""
    name = "fake-embedder"
    dim = 4096

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if len(tok) > 2:
                v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def embed_query(self, text):
        return self._vec(text)


class FakeReranker:
    def __init__(self, fn):
        self.fn = fn

    def score(self, query, texts):
        return [self.fn(query, t) for t in texts]


class FakeLLM:
    """Either a queue of responses (str or Exception) or a callable(messages, json_mode) -> str."""

    def __init__(self, responses=None, fn=None):
        self.responses = list(responses or [])
        self.fn = fn
        self.calls: list[tuple[list[dict], bool]] = []

    def complete(self, messages, *, json_mode=False, max_tokens=None):
        self.calls.append((messages, json_mode))
        if self.fn is not None:
            return self.fn(messages, json_mode)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def answer_json(citations=("C1",), answer="Test answer.", answerable=True) -> str:
    return json.dumps({"answerable": answerable, "answer": answer, "citations": list(citations)})


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, index_dir=tmp_path / "indexes", use_reranker=False, min_dense_score=0.2,
                    chunk_target_tokens=120, chunk_overlap_tokens=20, min_chunk_tokens=5,
                    max_history_turns=4)


@pytest.fixture
def handbook_pdf() -> bytes:
    return make_pdf(HANDBOOK_PAGES)


@pytest.fixture
def make_service(settings):
    def _make(llm=None, reranker=None, s: Settings | None = None) -> RAGService:
        s = s or settings
        return RAGService(s, embedder=FakeEmbedder(), reranker=reranker, llm=llm or FakeLLM(),
                          store=IndexStore(s.index_dir))
    return _make
