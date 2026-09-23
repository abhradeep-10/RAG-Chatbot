import fitz
import pytest

from ragbot.errors import EmptyDocumentError, PDFParseError
from ragbot.ingestion.pdf_parser import parse_pdf
from tests.conftest import HANDBOOK_PAGES, make_pdf

LIMITS = dict(max_bytes=10 * 1024 * 1024, max_pages=100)


def test_extracts_headings_sections_and_pages(handbook_pdf):
    doc = parse_pdf(handbook_pdf, "handbook.pdf", **LIMITS)
    assert doc.n_pages == 3
    headings = [b.text for b in doc.blocks if b.is_heading]
    assert "Eligibility Requirements" in headings
    assert "International Applicants" in headings

    body = [b for b in doc.blocks if not b.is_heading]
    max_amount = next(b for b in body if "maximum amount" in b.text)
    assert max_amount.page == 1
    assert max_amount.section_path == ("Grant Program Handbook", "Eligibility Requirements")

    intl = next(b for b in body if "sponsor letter" in b.text)
    assert intl.page == 2
    # same-level heading replaces the previous sibling in the path
    assert intl.section_path == ("Grant Program Handbook", "International Applicants")


def test_running_header_is_removed():
    data = make_pdf(HANDBOOK_PAGES, header="Grant Handbook v2 - Page {n}")
    doc = parse_pdf(data, "h.pdf", **LIMITS)
    assert not any("Grant Handbook v2" in b.text for b in doc.blocks)


def test_corrupt_pdf_raises():
    with pytest.raises(PDFParseError):
        parse_pdf(b"%PDF-1.7\nthis is not really a pdf", "bad.pdf", **LIMITS)


def test_non_pdf_bytes_rejected():
    with pytest.raises(PDFParseError):
        parse_pdf(b"hello world, definitely text", "x.pdf", **LIMITS)


def test_empty_upload_rejected():
    with pytest.raises(PDFParseError):
        parse_pdf(b"", "x.pdf", **LIMITS)


def test_too_large_rejected(handbook_pdf):
    with pytest.raises(PDFParseError):
        parse_pdf(handbook_pdf, "x.pdf", max_bytes=100, max_pages=100)


def test_too_many_pages_rejected(handbook_pdf):
    with pytest.raises(PDFParseError):
        parse_pdf(handbook_pdf, "x.pdf", max_bytes=10 * 1024 * 1024, max_pages=2)


def test_pdf_without_text_raises_empty_document():
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    with pytest.raises(EmptyDocumentError):
        parse_pdf(data, "scan.pdf", **LIMITS)
