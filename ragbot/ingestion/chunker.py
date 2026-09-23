"""Structure-aware chunking.

1. Body text is grouped by heading path; a chunk never crosses a section boundary,
   so every chunk has exactly one section label for its citation.
2. Text is split into sentence units (paragraph starts remembered), then greedily
   packed up to ``target_tokens`` with a sentence-level overlap of ``overlap_tokens``.
3. Tiny adjacent sibling sections (label/value pairs such as "Expected effort:
   2-3 hours") are merged under their parent heading, keeping the child heading inline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ragbot.ingestion.pdf_parser import Block, ParsedDocument
from ragbot.schemas import Chunk

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")


def approx_tokens(text: str) -> int:
    """~1.3 sub-word tokens per English word for BERT/BPE tokenizers. Cheap and deterministic;
    exact counts are unnecessary because the target leaves headroom below the 512 limit."""
    return max(1, round(len(text.split()) * 1.3))


@dataclass
class _Unit:
    text: str
    page: int
    tokens: int
    new_para: bool


def chunk_document(doc: ParsedDocument, *, target_tokens: int = 350, overlap_tokens: int = 60,
                   min_tokens: int = 40) -> list[Chunk]:
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens")
    groups = _group_by_section(doc.blocks, target_tokens)
    groups = _merge_small_sections(groups, min_tokens, target_tokens)

    chunks: list[Chunk] = []
    for path, units in groups:
        for piece in _pack(units, target_tokens, overlap_tokens, min_tokens):
            n = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}:{n:04d}", doc_id=doc.doc_id, ord=n,
                text=_join(piece), section_path=list(path),
                page_start=min(u.page for u in piece), page_end=max(u.page for u in piece),
            ))
    return chunks


def _join(units: list[_Unit]) -> str:
    parts: list[str] = []
    for i, u in enumerate(units):
        if i:
            parts.append("\n" if u.new_para else " ")
        parts.append(u.text)
    return "".join(parts)


def _hard_split(text: str, target: int) -> list[str]:
    words = text.split()
    max_words = max(1, int(target / 1.3))
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _split_block(block: Block, target: int) -> list[_Unit]:
    units: list[_Unit] = []
    for part in block.text.split("\n"):
        part = part.strip()
        if not part:
            continue
        first = True
        for sentence in _SENTENCE_RE.split(part):
            sentence = sentence.strip()
            if not sentence:
                continue
            for piece in _hard_split(sentence, target):
                units.append(_Unit(piece, block.page, approx_tokens(piece), first))
                first = False
    return units


def _group_by_section(blocks: list[Block], target: int) -> list[tuple[tuple[str, ...], list[_Unit]]]:
    groups: list[tuple[tuple[str, ...], list[_Unit]]] = []
    cur_path: tuple[str, ...] | None = None
    cur_units: list[_Unit] = []
    for b in blocks:
        if b.is_heading:
            continue  # heading text lives in section_path (and in the embedded text prefix)
        if b.section_path != cur_path:
            if cur_units:
                groups.append((cur_path or (), cur_units))
            cur_path, cur_units = b.section_path, []
        cur_units.extend(_split_block(b, target))
    if cur_units:
        groups.append((cur_path or (), cur_units))
    return groups


def _merge_small_sections(groups, min_tokens: int, target: int):
    out: list[dict] = []
    for path, units in groups:
        size = sum(u.tokens for u in units)
        if size >= min_tokens or not path:
            out.append({"path": path, "units": units, "small": False, "size": size})
            continue
        head = path[-1]
        inline = [_Unit(f"{head}: {units[0].text}", units[0].page,
                        units[0].tokens + approx_tokens(head), True)] + units[1:]
        size += approx_tokens(head)
        parent = path[:-1]
        prev = out[-1] if out else None
        if prev and prev["small"] and prev["parent"] == parent and prev["size"] + size <= target:
            prev["units"].extend(inline)
            prev["size"] += size
            prev["path"] = parent        # merged chunk is labelled with the common parent
        else:
            out.append({"path": path, "units": inline, "small": True, "size": size, "parent": parent})
    return [(g["path"], g["units"]) for g in out]


def _pack(units: list[_Unit], target: int, overlap: int, min_tokens: int) -> list[list[_Unit]]:
    pieces: list[list[_Unit]] = []
    cur: list[_Unit] = []
    cur_tok = 0
    n_carry = 0  # how many leading units of `cur` are overlap from the previous chunk
    for u in units:
        if cur and cur_tok + u.tokens > target and len(cur) > n_carry:
            pieces.append(cur)
            carry: list[_Unit] = []
            t = 0
            for prev in reversed(cur):
                if t + prev.tokens > overlap:
                    break
                carry.insert(0, prev)
                t += prev.tokens
            cur, cur_tok, n_carry = list(carry), t, len(carry)
        cur.append(u)
        cur_tok += u.tokens
    if len(cur) > n_carry:
        fresh = cur[n_carry:]
        fresh_tok = sum(u.tokens for u in fresh)
        if pieces and fresh_tok < min_tokens and sum(u.tokens for u in pieces[-1]) + fresh_tok <= int(target * 1.25):
            pieces[-1].extend(fresh)   # avoid a tiny tail chunk that is mostly overlap
        else:
            pieces.append(cur)
    return pieces
