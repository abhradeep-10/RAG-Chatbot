import pytest

from ragbot.ingestion.chunker import approx_tokens, chunk_document
from ragbot.ingestion.pdf_parser import Block, ParsedDocument, parse_pdf


def _doc(blocks):
    return ParsedDocument(doc_id="0123456789abcdef", filename="t.pdf", n_pages=5, blocks=blocks)


def _sentences(n, tag):
    return " ".join(f"Sentence {i} about {tag} has exactly nine words here." for i in range(n))


def test_chunks_never_cross_sections():
    blocks = [
        Block(1, "Alpha", True, ("Alpha",)),
        Block(1, _sentences(3, "alpha"), False, ("Alpha",)),
        Block(2, "Beta", True, ("Beta",)),
        Block(2, _sentences(3, "beta"), False, ("Beta",)),
    ]
    chunks = chunk_document(_doc(blocks), target_tokens=200, overlap_tokens=20, min_tokens=5)
    assert len(chunks) == 2
    assert all(("alpha" in c.text) != ("beta" in c.text) for c in chunks)
    assert chunks[0].section_path == ["Alpha"] and chunks[1].section_path == ["Beta"]


def test_long_section_is_split_with_sentence_overlap():
    blocks = [Block(1, _sentences(30, "gamma"), False, ("Gamma",))]
    chunks = chunk_document(_doc(blocks), target_tokens=60, overlap_tokens=15, min_tokens=5)
    assert len(chunks) > 3
    for c in chunks:
        assert approx_tokens(c.text) <= 60 + 15
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = prev.text.split(". ")[-1].strip()
        assert last_sentence.rstrip(".") in nxt.text      # overlap carried forward


def test_page_span_tracked_across_pages():
    blocks = [
        Block(2, "First part of the section on page two.", False, ("Delta",)),
        Block(3, "Second part of the same section on page three.", False, ("Delta",)),
    ]
    [chunk] = chunk_document(_doc(blocks), target_tokens=200, overlap_tokens=20, min_tokens=5)
    assert (chunk.page_start, chunk.page_end) == (2, 3)
    assert chunk.pages_label == "Pages 2-3"


def test_embed_text_contains_heading_path():
    blocks = [Block(4, "The limit is 50 units for everyone involved here.", False, ("Rules", "Limits"))]
    [chunk] = chunk_document(_doc(blocks), target_tokens=200, overlap_tokens=20, min_tokens=5)
    assert chunk.embed_text.startswith("Rules > Limits\n")
    assert chunk.section == "Rules > Limits"


def test_tiny_sibling_sections_are_merged_under_parent():
    blocks = [
        Block(1, "Take-home assignment", False, ("Brief", "Format")),
        Block(1, "Approximately 2-3 hours", False, ("Brief", "Expected effort")),
    ]
    [chunk] = chunk_document(_doc(blocks), target_tokens=200, overlap_tokens=20, min_tokens=40)
    assert chunk.section_path == ["Brief"]
    assert "Expected effort: Approximately 2-3 hours" in chunk.text


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_document(_doc([]), target_tokens=50, overlap_tokens=50)


def test_end_to_end_on_pdf(handbook_pdf):
    parsed = parse_pdf(handbook_pdf, "h.pdf", max_bytes=10**7, max_pages=100)
    chunks = chunk_document(parsed, target_tokens=120, overlap_tokens=20, min_tokens=5)
    assert chunks
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    target = next(c for c in chunks if "maximum amount" in c.text)
    assert target.page_start == 1 and "Eligibility Requirements" in target.section
