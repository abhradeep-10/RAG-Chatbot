"""Application service: orchestrates ingestion and question answering.
Both the FastAPI app and the Streamlit UI are thin layers over this class."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from ragbot.config import Settings, get_settings
from ragbot.errors import EmptyDocumentError, IndexMismatchError, LLMError, LLMOutputError, RAGError
from ragbot.ingestion.chunker import chunk_document
from ragbot.ingestion.pdf_parser import compute_doc_id, parse_pdf
from ragbot.llm.client import LLMClient, OpenAICompatibleLLM
from ragbot.llm.output_parser import LLMAnswer, parse_llm_answer
from ragbot.llm.prompts import (REPAIR_INSTRUCTION, build_answer_messages, build_context,
                                build_rewrite_messages)
from ragbot.logging_utils import log_event
from ragbot.retrieval.embeddings import Embedder, SentenceTransformerEmbedder
from ragbot.retrieval.reranker import CrossEncoderReranker, Reranker
from ragbot.retrieval.retriever import HybridRetriever, RetrievalResult
from ragbot.retrieval.store import DocumentIndex, IndexStore
from ragbot.schemas import (CandidateDebug, ChatRequest, ChatResponse, ChatTurn, Citation,
                            DocumentInfo, RetrievalDebug, RetrievedChunk)

log = logging.getLogger("ragbot.service")

NOT_FOUND_MESSAGE = "I couldn't find information about this in the document."
UNSUPPORTED_MESSAGE = ("I found related passages, but I couldn't produce an answer that is clearly supported "
                       "by them, so I won't guess. Try rephrasing or asking about a more specific part of the document.")
ERROR_MESSAGE = ("I couldn't generate a reliable answer for this question because the model returned a "
                 "malformed response. Please try again.")

_REWRITE_PREFIX_RE = re.compile(r"^(standalone question|rewritten question|question)\s*:\s*", re.I)
_UNSET = object()


class RAGService:
    def __init__(self, settings: Settings | None = None, *, embedder: Embedder | None = None,
                 reranker: Reranker | None | object = _UNSET, llm: LLMClient | None = None,
                 store: IndexStore | None = None):
        self.settings = settings or get_settings()
        self._embedder = embedder
        self._reranker = reranker
        self._llm = llm
        self.store = store or IndexStore(self.settings.index_dir)
        self._retrievers: dict[str, HybridRetriever] = {}

    # ------------------------------------------------------------------ lazy components
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            s = self.settings
            self._embedder = SentenceTransformerEmbedder(s.embedding_model, s.device, s.query_instruction)
        return self._embedder

    @property
    def reranker(self) -> Reranker | None:
        if self._reranker is _UNSET:
            s = self.settings
            self._reranker = CrossEncoderReranker(s.reranker_model, s.device) if s.use_reranker else None
        return self._reranker  # type: ignore[return-value]

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = OpenAICompatibleLLM(self.settings)
        return self._llm

    def _chunker_params(self) -> dict:
        s = self.settings
        return {"target_tokens": s.chunk_target_tokens, "overlap_tokens": s.chunk_overlap_tokens,
                "min_tokens": s.min_chunk_tokens}

    # ------------------------------------------------------------------ ingestion
    def ingest_pdf(self, data: bytes, filename: str) -> DocumentInfo:
        s = self.settings
        filename = Path(filename or "document.pdf").name[:200]
        doc_id = compute_doc_id(data)
        if data and self.store.exists(doc_id):
            try:
                existing = self.store.load(doc_id)
                if (existing.info.embedding_model == self.embedder.name
                        and existing.chunker_params == self._chunker_params()):
                    return existing.info          # idempotent: same bytes + same config
            except RAGError as exc:
                log.warning("existing index for %s unusable, re-ingesting: %s", doc_id, exc)

        t0 = perf_counter()
        parsed = parse_pdf(data, filename, max_bytes=int(s.max_pdf_mb * 1024 * 1024), max_pages=s.max_pages)
        chunks = chunk_document(parsed, **self._chunker_params())
        if not chunks:
            raise EmptyDocumentError("no chunks produced", "No usable text was found in the document.")
        try:
            embeddings = self.embedder.embed_documents([c.embed_text for c in chunks])
        except Exception as exc:
            raise RAGError(f"embedding failed: {exc}", "Failed to compute embeddings for the document.") from exc
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise RAGError(f"bad embedding shape {embeddings.shape}", "Failed to compute embeddings for the document.")

        info = DocumentInfo(doc_id=parsed.doc_id, filename=filename, n_pages=parsed.n_pages,
                            n_chunks=len(chunks), embedding_model=self.embedder.name,
                            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.store.save(DocumentIndex(info, chunks, embeddings, self._chunker_params()))
        self._retrievers.pop(info.doc_id, None)
        log_event(log, "ingest", doc_id=info.doc_id, filename=filename, pages=info.n_pages,
                  chunks=info.n_chunks, seconds=round(perf_counter() - t0, 2))
        return info

    def list_documents(self) -> list[DocumentInfo]:
        return self.store.list_documents()

    # ------------------------------------------------------------------ question answering
    def chat(self, req: ChatRequest, *, include_debug: bool = True) -> ChatResponse:
        s = self.settings
        timings: dict[str, float] = {}
        t_all = perf_counter()

        index = self.store.load(req.doc_id)
        if index.info.embedding_model != self.embedder.name:
            raise IndexMismatchError(
                f"index built with {index.info.embedding_model}, current embedder {self.embedder.name}",
                "This document was indexed with a different embedding model. Please re-upload it.")

        history = self._trim_history(req.history)

        t = perf_counter()
        standalone, rewritten = self._standalone_query(req.question, history)
        timings["rewrite_ms"] = _ms(t)

        t = perf_counter()
        result = self._retriever(index).retrieve(standalone)
        timings["retrieve_ms"] = _ms(t)

        debug = self._debug(result, standalone, rewritten, timings)

        if not result.gate_passed:        # layer 1 of abstention: nothing relevant enough
            return self._finish(req, ChatResponse(answer=NOT_FOUND_MESSAGE, status="not_found", debug=debug),
                                include_debug, t_all)

        context, label_map = build_context(result.selected, s.max_context_chars)
        messages = build_answer_messages(req.question, standalone, history, context, s.max_history_turn_chars)

        t = perf_counter()
        parsed = self._generate(messages)          # LLMError (API failure) propagates to the caller
        timings["generate_ms"] = _ms(t)

        if parsed is None:
            return self._finish(req, ChatResponse(answer=ERROR_MESSAGE, status="error", debug=debug),
                                include_debug, t_all)

        valid = [c for c in parsed.citations if c in label_map]
        debug.invalid_citations = [c for c in parsed.citations if c not in label_map]

        if not parsed.answerable:         # layer 2: the model abstains
            resp = ChatResponse(answer=parsed.answer or NOT_FOUND_MESSAGE, status="not_found", debug=debug)
        elif not valid:                   # layer 3: an answer with no valid evidence is not shown
            resp = ChatResponse(answer=UNSUPPORTED_MESSAGE, status="unsupported", debug=debug)
        else:
            citations = [self._to_citation(label_map[c]) for c in valid]
            resp = ChatResponse(answer=parsed.answer, status="answered", citations=citations, debug=debug)
        return self._finish(req, resp, include_debug, t_all)

    # ------------------------------------------------------------------ internals
    def _retriever(self, index: DocumentIndex) -> HybridRetriever:
        doc_id = index.info.doc_id
        if doc_id not in self._retrievers:
            self._retrievers[doc_id] = HybridRetriever(index, self.embedder, self.reranker, self.settings)
        return self._retrievers[doc_id]

    def _trim_history(self, history: list[ChatTurn]) -> list[ChatTurn]:
        s = self.settings
        recent = history[-s.max_history_turns:] if s.max_history_turns > 0 else []
        lim = s.max_history_turn_chars
        return [ChatTurn(role=t.role, content=t.content if len(t.content) <= lim else t.content[: lim - 1] + "\u2026")
                for t in recent]

    def _standalone_query(self, question: str, history: list[ChatTurn]) -> tuple[str, bool]:
        """Condense a follow-up into a standalone retrieval query."""
        if not history or not self.settings.enable_query_rewrite:
            return question, False
        try:
            raw = self.llm.complete(build_rewrite_messages(question, history, self.settings.max_history_turn_chars),
                                    json_mode=False, max_tokens=128)
            candidate = _clean_rewrite(raw)
            if not candidate or len(candidate) > max(3 * len(question), 400):
                raise LLMOutputError(f"unusable rewrite: {raw[:200]!r}")
            return candidate, candidate != question
        except LLMError as exc:
            # degrade gracefully: previous user question + follow-up still carries the topic
            log.warning("query rewrite failed, using heuristic fallback: %s", exc)
            last_user = next((t.content for t in reversed(history) if t.role == "user"), "")
            fallback = f"{last_user} {question}".strip()[: 2 * self.settings.max_question_chars]
            return fallback, fallback != question

    def _generate(self, messages: list[dict]) -> LLMAnswer | None:
        """One repair attempt on malformed output; None if still malformed."""
        raw = ""
        try:
            raw = self.llm.complete(messages, json_mode=True)
            return parse_llm_answer(raw)
        except LLMOutputError as exc:
            log.warning("malformed LLM output, attempting repair: %s", exc)
        repair = list(messages)
        if raw:
            repair.append({"role": "assistant", "content": raw[:4000]})
        repair.append({"role": "user", "content": REPAIR_INSTRUCTION})
        try:
            return parse_llm_answer(self.llm.complete(repair, json_mode=True))
        except LLMOutputError as exc:
            log.error("LLM output still malformed after repair: %s", exc)
            return None

    @staticmethod
    def _to_citation(rc: RetrievedChunk) -> Citation:
        c = rc.chunk
        snippet = c.text if len(c.text) <= 300 else c.text[:297] + "..."
        return Citation(chunk_id=c.chunk_id, page_start=c.page_start, page_end=c.page_end,
                        section=c.section, label=f'{c.pages_label} - "{c.section}"', snippet=snippet)

    @staticmethod
    def _debug(result: RetrievalResult, standalone: str, rewritten: bool, timings: dict) -> RetrievalDebug:
        selected_ids = {rc.chunk.chunk_id for rc in result.selected}
        cands = [
            CandidateDebug(rank=i + 1, chunk_id=rc.chunk.chunk_id, page_start=rc.chunk.page_start,
                           page_end=rc.chunk.page_end, section=rc.chunk.section,
                           dense_score=rc.dense_score, sparse_score=rc.sparse_score,
                           fused_score=rc.fused_score, rerank_score=rc.rerank_score,
                           selected=rc.chunk.chunk_id in selected_ids, preview=rc.chunk.text[:200])
            for i, rc in enumerate(result.candidates)
        ]
        return RetrievalDebug(standalone_query=standalone, query_rewritten=rewritten,
                              gate_passed=result.gate_passed, gate_reason=result.gate_reason,
                              candidates=cands, timings_ms=timings)

    def _finish(self, req: ChatRequest, resp: ChatResponse, include_debug: bool, t_all: float) -> ChatResponse:
        if resp.debug is not None:
            resp.debug.timings_ms["total_ms"] = _ms(t_all)
            d = resp.debug
            log_event(log, "query", doc_id=req.doc_id, question=req.question, standalone=d.standalone_query,
                      gate=d.gate_reason, status=resp.status,
                      selected=[(c.chunk_id, c.page_start, c.rerank_score if c.rerank_score is not None else c.fused_score)
                                for c in d.candidates if c.selected],
                      citations=[c.chunk_id for c in resp.citations],
                      invalid_citations=d.invalid_citations, timings=d.timings_ms)
        if not include_debug:
            resp.debug = None
        return resp


def _ms(t0: float) -> float:
    return round((perf_counter() - t0) * 1000, 1)


def _clean_rewrite(raw: str) -> str:
    line = next((l.strip() for l in raw.strip().splitlines() if l.strip()), "")
    line = _REWRITE_PREFIX_RE.sub("", line)
    return line.strip().strip('"').strip("'").strip()
