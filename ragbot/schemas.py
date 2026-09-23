"""Data models shared by ingestion, retrieval, the service layer and the API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ragbot.config import get_settings



class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    ord: int
    text: str
    section_path: list[str] = Field(default_factory=list)
    page_start: int
    page_end: int

    @property
    def section(self) -> str:
        return " > ".join(self.section_path) if self.section_path else "(untitled section)"

    @property
    def embed_text(self) -> str:
        """Text used for embedding and BM25: the heading path is prepended so that a
        passage like "the limit is 50 units" is still found for "eligibility limit"."""
        return f"{self.section}\n{self.text}" if self.section_path else self.text

    @property
    def pages_label(self) -> str:
        if self.page_start == self.page_end:
            return f"Page {self.page_start}"
        return f"Pages {self.page_start}-{self.page_end}"


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: float | None = None
    dense_rank: int | None = None
    sparse_score: float | None = None
    sparse_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    n_pages: int
    n_chunks: int
    embedding_model: str
    created_at: str


# --------------------------------------------------------------------------- API
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    doc_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    question: str
    history: list[ChatTurn] = Field(default_factory=list, max_length=100)

    @field_validator("question")
    @classmethod
    def _check_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        limit = get_settings().max_question_chars
        if len(v) > limit:
            raise ValueError(f"question is too long ({len(v)} characters; limit is {limit})")
        return v


class Citation(BaseModel):
    chunk_id: str
    page_start: int
    page_end: int
    section: str
    label: str          # e.g. 'Page 47 - "Eligibility Requirements"'
    snippet: str


class CandidateDebug(BaseModel):
    rank: int
    chunk_id: str
    page_start: int
    page_end: int
    section: str
    dense_score: float | None
    sparse_score: float | None
    fused_score: float
    rerank_score: float | None
    selected: bool
    preview: str


class RetrievalDebug(BaseModel):
    standalone_query: str
    query_rewritten: bool
    gate_passed: bool
    gate_reason: str
    candidates: list[CandidateDebug] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


Status = Literal["answered", "not_found", "unsupported", "error"]


class ChatResponse(BaseModel):
    answer: str
    status: Status
    citations: list[Citation] = Field(default_factory=list)
    debug: RetrievalDebug | None = None
