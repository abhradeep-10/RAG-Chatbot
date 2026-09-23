"""PDF -> structured blocks (headings + paragraphs), each tagged with page and heading path.

Structure recovery uses layout signals from PyMuPDF:
  * font size relative to the dominant body size  -> heading level
  * all-bold short title-like lines                 -> lowest heading level
  * lines repeated in the top/bottom page margin on many pages -> running header/footer, dropped
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from ragbot.errors import EmptyDocumentError, PDFParseError

log = logging.getLogger(__name__)

HEADING_SIZE_RATIO = 1.15
MAX_LEVEL = 3
EDGE_ZONE = 0.10          # top/bottom fraction of the page treated as header/footer area
MIN_TOTAL_CHARS = 50
_PAGE_NUMBER_RE = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.I)
_BULLET_RE = re.compile(r"^([-\u2022\u2013\u25aa\u25cf*]|\d+[.)])\s")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


@dataclass
class Line:
    page: int            # 1-based
    text: str
    size: float
    bold: bool
    block_no: int
    y0: float
    y1: float
    page_height: float


@dataclass
class Block:
    page: int
    text: str
    is_heading: bool
    section_path: tuple[str, ...] = ()


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    n_pages: int
    blocks: list[Block] = field(default_factory=list)


def compute_doc_id(data: bytes) -> str:
    """Content hash: the same PDF always maps to the same index (idempotent ingestion)."""
    return hashlib.sha256(data).hexdigest()[:16]


def parse_pdf(data: bytes, filename: str, *, max_bytes: int, max_pages: int) -> ParsedDocument:
    _validate_bytes(data, max_bytes)
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  
        raise PDFParseError(f"cannot open PDF: {exc}", "The file could not be read as a PDF.") from exc

    try:
        if doc.needs_pass:
            raise PDFParseError("encrypted PDF", "The PDF is password-protected and cannot be read.")
        n_pages = doc.page_count
        if n_pages == 0:
            raise EmptyDocumentError("PDF has no pages", "The PDF contains no pages.")
        if n_pages > max_pages:
            raise PDFParseError(f"{n_pages} pages > limit {max_pages}",
                                f"The PDF has {n_pages} pages; the limit is {max_pages}.")
        lines = _extract_lines(doc)
    finally:
        doc.close()

    lines = _drop_headers_footers(lines, n_pages)
    if sum(len(l.text) for l in lines) < MIN_TOTAL_CHARS:
        raise EmptyDocumentError(
            "no extractable text",
            "No extractable text was found. The PDF may consist of scanned images (OCR is not supported yet).",
        )
    blocks = _build_blocks(lines)
    log.info("parsed %s: %d pages, %d lines, %d blocks", filename, n_pages, len(lines), len(blocks))
    return ParsedDocument(doc_id=compute_doc_id(data), filename=filename, n_pages=n_pages, blocks=blocks)


def _validate_bytes(data: bytes, max_bytes: int) -> None:
    if not data:
        raise PDFParseError("empty upload", "The uploaded file is empty.")
    if len(data) > max_bytes:
        raise PDFParseError(f"file too large: {len(data)} bytes",
                            f"The file is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    if b"%PDF-" not in data[:1024]:
        raise PDFParseError("missing %PDF- header", "The file does not look like a PDF.")


def _clean(text: str) -> str:
    text = _CONTROL_RE.sub("", text.replace("\u00a0", " "))
    return re.sub(r"\s+", " ", text).strip()


def _is_bold(span: dict) -> bool:
    font = str(span.get("font", "")).lower()
    return bool(span.get("flags", 0) & 16) or "bold" in font or "black" in font or "heavy" in font


def _extract_lines(doc: "fitz.Document") -> list[Line]:
    lines: list[Line] = []
    failed_pages = 0
    for pno in range(doc.page_count):
        try:
            page = doc.load_page(pno)
            height = float(page.rect.height) or 1.0
            layout = page.get_text("dict", sort=True)
        except Exception as exc:  # one bad page must not kill the whole document
            failed_pages += 1
            log.warning("skipping page %d: %s", pno + 1, exc)
            continue
        for b_no, block in enumerate(layout.get("blocks", [])):
            if block.get("type", 0) != 0:        # 1 = image block
                continue
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = _clean("".join(s["text"] for s in spans))
                if not text:
                    continue
                dominant = max(spans, key=lambda s: len(s["text"].strip()))
                _, y0, _, y1 = line["bbox"]
                lines.append(Line(
                    page=pno + 1, text=text, size=round(float(dominant["size"]), 1),
                    bold=all(_is_bold(s) for s in spans), block_no=b_no,
                    y0=float(y0), y1=float(y1), page_height=height,
                ))
    if failed_pages == doc.page_count:
        raise PDFParseError("all pages failed to parse", "The PDF could not be parsed.")
    return lines


def _in_edge_zone(line: Line) -> bool:
    return line.y0 < line.page_height * EDGE_ZONE or line.y1 > line.page_height * (1 - EDGE_ZONE)


def _norm_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _drop_headers_footers(lines: list[Line], n_pages: int) -> list[Line]:
    """Remove page numbers and running headers/footers (same text, digits normalised,
    in the margin zone of at least half the pages)."""
    repeated: set[str] = set()
    if n_pages >= 2:
        pages_by_key: dict[str, set[int]] = defaultdict(set)
        for l in lines:
            if _in_edge_zone(l) and len(l.text) <= 150:
                pages_by_key[_norm_key(l.text)].add(l.page)
        threshold = max(2, math.ceil(0.5 * n_pages))
        repeated = {k for k, pages in pages_by_key.items() if len(pages) >= threshold}
    return [
        l for l in lines
        if not (_in_edge_zone(l) and (_PAGE_NUMBER_RE.match(l.text) or _norm_key(l.text) in repeated))
    ]


def _body_font_size(lines: list[Line]) -> float:
    weights: Counter[float] = Counter()
    for l in lines:
        weights[l.size] += len(l.text)
    return weights.most_common(1)[0][0] if weights else 10.0


def _heading_level(line: Line, body: float, size_level: dict[float, int],
                   bold_level: int, allow_bold: bool) -> int:
    t = line.text
    if not any(c.isalpha() for c in t):
        return 0
    words = t.split()
    if len(t) > 120 or len(words) > 15 or t.endswith((",", ";")):
        return 0
    if line.size >= body * HEADING_SIZE_RATIO:
        return size_level.get(line.size, MAX_LEVEL)
    if (allow_bold and line.bold and len(words) <= 12 and (t[0].isupper() or t[0].isdigit())
            and not t.endswith(".")):
        return bold_level
    return 0


def _join_lines(texts: list[str]) -> str:
    out = ""
    for t in texts:
        if not out:
            out = t
        elif _BULLET_RE.match(t):
            out += "\n" + t                       # keep list items on their own line
        elif out.endswith("-") and t[:1].islower():
            out = out[:-1] + t                    # de-hyphenate a word broken across lines
        else:
            out += " " + t
    return out


def _build_blocks(lines: list[Line]) -> list[Block]:
    body = _body_font_size(lines)
    heading_sizes = sorted({l.size for l in lines if l.size >= body * HEADING_SIZE_RATIO}, reverse=True)
    size_level = {s: min(i + 1, MAX_LEVEL) for i, s in enumerate(heading_sizes)}
    bold_level = min(len(heading_sizes) + 1, MAX_LEVEL)
    total_chars = sum(len(l.text) for l in lines) or 1
    allow_bold = sum(len(l.text) for l in lines if l.bold) / total_chars < 0.5  # bold-everywhere docs

    blocks: list[Block] = []
    para: list[Line] = []
    stack: list[tuple[int, str]] = []
    last_heading: Line | None = None

    def path() -> tuple[str, ...]:
        return tuple(t for _, t in stack)

    def flush() -> None:
        if para:
            blocks.append(Block(para[0].page, _join_lines([l.text for l in para]), False, path()))
            para.clear()

    for line in lines:
        level = _heading_level(line, body, size_level, bold_level, allow_bold)
        if level:
            flush()
            continues_heading = (
                last_heading is not None and stack and stack[-1][0] == level
                and last_heading.page == line.page and last_heading.block_no == line.block_no
            )
            if continues_heading:                 # heading wrapped over two lines
                merged = f"{stack[-1][1]} {line.text}"
                stack[-1] = (level, merged)
                blocks[-1] = Block(line.page, merged, True, path())
            else:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, line.text))
                blocks.append(Block(line.page, line.text, True, path()))
            last_heading = line
            continue
        last_heading = None
        if para and (line.page != para[-1].page or line.block_no != para[-1].block_no):
            flush()
        para.append(line)
    flush()
    return blocks
