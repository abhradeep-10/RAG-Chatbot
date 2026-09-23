import numpy as np

from ragbot.retrieval.retriever import HybridRetriever, rrf_fuse, tokenize
from ragbot.retrieval.store import DocumentIndex
from ragbot.schemas import Chunk, DocumentInfo
from tests.conftest import FakeEmbedder, FakeReranker

TEXTS = [
    ("Eligibility", "Applicants must be residents for two years and hold a business registration."),
    ("Limits", "The maximum amount allowed is 50 units per applicant."),
    ("International", "International applicants must provide a sponsor letter."),
    ("Appeals", "Appeals must be filed within 30 days of the decision."),
]


def _index():
    emb = FakeEmbedder()
    chunks = [Chunk(chunk_id=f"d:{i:04d}", doc_id="d", ord=i, text=t, section_path=[s],
                    page_start=i + 1, page_end=i + 1) for i, (s, t) in enumerate(TEXTS)]
    info = DocumentInfo(doc_id="0123456789abcdef", filename="t.pdf", n_pages=4, n_chunks=4,
                        embedding_model=emb.name, created_at="now")
    return DocumentIndex(info, chunks, emb.embed_documents([c.embed_text for c in chunks])), emb


def test_rrf_prefers_items_ranked_well_by_both_lists():
    scores = rrf_fuse([[0, 1, 2], [2, 0]], k=60)
    order = sorted(scores, key=lambda i: -scores[i])
    assert order == [0, 2, 1]


def test_tokenize_drops_stopwords_and_keeps_identifiers():
    assert tokenize("What is the AI_USAGE.md file?") == ["ai_usage", "md", "file"]


def test_dense_plus_sparse_finds_relevant_chunk(settings):
    index, emb = _index()
    r = HybridRetriever(index, emb, None, settings).retrieve("What is the maximum amount allowed?")
    assert r.gate_passed
    assert r.selected[0].chunk.section_path == ["Limits"]


def test_gate_fails_for_unrelated_query(settings):
    index, emb = _index()
    r = HybridRetriever(index, emb, None, settings).retrieve("zebra xylophone quantum")
    assert not r.gate_passed
    assert r.selected == []


def test_reranker_controls_order_and_gate(settings):
    index, emb = _index()
    settings.min_rerank_score = 0.5
    reranker = FakeReranker(lambda q, t: 0.9 if "Appeals" in t else 0.1)
    r = HybridRetriever(index, emb, reranker, settings).retrieve("filed within 30 days")
    assert r.gate_passed
    assert [c.chunk.section_path[0] for c in r.selected] == ["Appeals"]   # below-threshold passages dropped

    low = FakeReranker(lambda q, t: 0.01)
    r2 = HybridRetriever(index, emb, low, settings).retrieve("filed within 30 days")
    assert not r2.gate_passed and r2.selected == []


def test_empty_query_returns_nothing(settings):
    index, emb = _index()
    r = HybridRetriever(index, emb, None, settings).retrieve("   ")
    assert not r.gate_passed and r.candidates == []
