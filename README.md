# Document Intelligence Chatbot (RAG)

Ask questions about a PDF and get answers grounded in the document, with page + section citations.
Fully open-source stack: PyMuPDF, `bge-small-en-v1.5` embeddings, BM25, `bge-reranker-base`,
and an open-weight LLM (default **Qwen2.5-7B-Instruct via Ollama**) behind an OpenAI-compatible API.
UI: Streamlit. API: FastAPI.

```
PDF ─► parse (layout → headings, pages, header/footer removal) ─► structure-aware chunks
    ─► embeddings (bge-small) + BM25 ─► per-document index on disk (content-hashed doc_id)

question ─► [follow-up? condense with LLM] ─► dense + BM25 ─► RRF ─► cross-encoder rerank
        ─► relevance gate ──(fail)──► "not in the document" (no LLM call)
                          └─(pass)──► LLM (JSON: answerable / answer / passage IDs)
        ─► validate JSON (1 repair retry) ─► map IDs → page/section ─► reject uncited answers
```

## Repository layout

| Path | Responsibility |
|---|---|
| `ragbot/ingestion/` | `pdf_parser.py` (PDF → blocks with pages/heading path), `chunker.py` |
| `ragbot/retrieval/` | `embeddings.py`, `reranker.py`, `store.py` (persistent index), `retriever.py` (hybrid + gate) |
| `ragbot/llm/` | `client.py` (OpenAI-compatible, retries), `prompts.py`, `output_parser.py` |
| `ragbot/service.py` | Orchestration: ingest, rewrite, retrieve, generate, validate citations |
| `ragbot/api.py` | FastAPI: `POST /documents`, `GET /documents`, `POST /chat`, `GET /health` |
| `ui/streamlit_app.py` | Chat UI with sources and a retrieval-debug panel |
| `tests/` | Offline unit/integration tests (fake embedder/reranker/LLM) |
| `evaluation/` | `eval_set.jsonl` (20 items) + `run_eval.py` |
| `DESIGN.md`, `EVALUATION.md`, `AI_USAGE.md` | Design rationale, eval methodology, AI usage |

## Setup

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate           
pip install -r requirements.txt
cp .env.example .env                 
```

### Open-source LLM

Default: [Ollama](https://ollama.com) (CPU works; a GPU makes it fast).

```bash
ollama pull qwen2.5:7b-instruct     
ollama serve                         
```

Any OpenAI-compatible server works by changing `.env` only, e.g. vLLM
(`vllm serve Qwen/Qwen2.5-7B-Instruct` → `RAG_LLM_BASE_URL=http://localhost:8000/v1`),
llama.cpp server, TGI, or a hosted open-weight endpoint (Groq/Together) if you have no GPU.

The embedding model (~130 MB) and reranker (~1.1 GB) download from Hugging Face on first use.
On a slow machine set `RAG_USE_RERANKER=false` (then the gate uses `RAG_MIN_DENSE_SCORE`)
or use `RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`.

## Run

```bash
# UI
streamlit run ui/streamlit_app.py
# → upload the PDF in the sidebar, click "Ingest", ask questions.

# API 
uvicorn ragbot.api:app --reload
curl -F "file=@/path/to/doc.pdf" http://localhost:8000/documents        
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"doc_id":"<doc_id>","question":"What are the eligibility requirements?","history":[]}'

```

`/chat` is stateless: the client sends prior turns in `history`
(`[{"role":"user","content":...},{"role":"assistant","content":...}]`). Add `?debug=false` to omit
retrieval details.

## Tests

```bash
pytest
```
Tests use deterministic fakes, so they need neither model downloads nor an LLM server. They cover
parsing (headings, header removal, corrupt/empty/oversized/non-PDF input), chunking (section
boundaries, overlap, page spans), hybrid retrieval + RRF + gate, JSON output parsing, the service
(abstention, invalid citations, repair retry, LLM outage, follow-up rewriting and its fallback,
history truncation, idempotent ingest) and API status codes (422/404/503).

## Evaluation

```bash
python -m evaluation.run_eval --pdf AI_Engineering_Task_1_RAG.pdf           # rule-based metrics
python -m evaluation.run_eval --pdf AI_Engineering_Task_1_RAG.pdf --judge   # + LLM faithfulness judge
```
The included set is written against the assignment brief itself (so it runs immediately);
swap in questions for the provided PDF by editing `evaluation/eval_set.jsonl`. See `EVALUATION.md`.

## Configuration

All settings live in `ragbot/config.py` and are overridable via `RAG_*` env vars / `.env`
(model names, endpoints, chunk sizes, top-k, thresholds, input limits). No secrets in code.

## Observability

- Every query logs one JSON line (`event: query`): original + rewritten query, gate decision,
  selected chunk IDs/pages/scores, citations, invalid citations, per-stage latency.
- `/chat` returns `debug` with the full ranked candidate list (dense, BM25, fused, rerank scores).
- The Streamlit "Retrieval debug" expander shows the same.
