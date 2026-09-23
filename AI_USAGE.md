# AI usage

## Tools used

- **Claude** - design discussion and drafting code, tests and documentation.
- **Chatgpt** - design discussion and overview and validating some ideas.
- **Ollama + Qwen2.5-3B-Instruct** - the open-source LLM the system itself runs on (not a coding
  assistant, listed here for completeness).

## What I used them for

I designed the pipeline and made the engineering decisions - chunking strategy, hybrid retrieval,
the citation mechanism, the abstention layers and the evaluation methodology - and used Claude to
draft implementations of them, to write boilerplate faster (pydantic schemas, FastAPI wiring, the
Streamlit UI, pytest fakes), and as a sounding board when comparing options.

What I did myself: set up and ran the environment, integrated and debugged the pipeline end to end,
ran the test suite and the evaluation, interpreted the results, and made the changes described
below. Every file was reviewed before it went in; several needed correcting.

## Where AI suggestions were suboptimal and I changed them

**1. Citations written directly by the LLM (incorrect and unsafe - rejected).**
The first draft had the model emit its source as free text, e.g.
`"source": "Page 47 - Eligibility Requirements"`. This defeats the purpose of citations: the model
can produce a page number that looks plausible and is wrong, and nothing downstream can detect it.
I replaced it with ID indirection - passages go to the model as `[C1]`, `[C2]`, the model may only
return those IDs, and the server maps each ID back to the page and section recorded at ingestion.
IDs that were not in the context are dropped and logged, and an answer with no valid citation is
never shown (it becomes `unsupported`). Page numbers in the UI are now impossible to hallucinate.

**2. Fixed-size character chunking (suboptimal - replaced).**
The suggested default was 1000-character chunks with 200-character overlap. On a structured
document this splits sentences mid-clause and merges unrelated sections, which also makes a
section-level citation ambiguous - a chunk spanning two headings has no single correct source.
I replaced it with section-bounded chunks packed from whole sentences (~350 tokens, sentence-level
overlap), with the heading path prepended to the indexed text so a passage like "the maximum is
50 units" is still retrievable for "eligibility limit".

**3. The relevance threshold was pruning evidence (bug I found from the evaluation).**
The retriever applied `min_rerank_score` twice: once to decide whether the document contains an
answer, and again to filter the passages sent to the LLM. On multi-part questions the second
supporting passage usually scores well below the best one, so it was being filtered out and the
answer came back incomplete. The threshold should gate, not prune. It now decides only whether to
answer at all, and the top-k passages all reach the model.

**4. A JSON-only output contract with a 7B model (suboptimal on small models - extended).**
Requiring strict JSON worked in tests but failed in the real evaluation run: two questions returned
`status: error` after 17-21 seconds, because long answers were truncated mid-JSON. Discarding a
usable answer over its formatting is the wrong trade-off. I added a third stage: after the JSON
attempt and one repair retry, the model is asked for a plain-text format
(`ANSWER: ... / SOURCES: C1, C3`) that small models handle reliably. It still requires a citation
label or an explicit `NOT_FOUND`, so the grounding guarantee is unchanged.

**5. Heavyweight model defaults (suboptimal for the target machine - changed).**
The defaults were a 7B LLM and `bge-reranker-base` (~1.1 GB), giving 2-20 s per query on a laptop
CPU. I moved to Qwen2.5-3B-Instruct and `ms-marco-MiniLM-L-6-v2` (~90 MB), which is the right point
on the quality/latency curve for this assignment; the model is a config value, so a stronger one
can be used where the hardware allows.

## A note on the evaluation

My first evaluation run reported a 94% false-refusal rate. Before changing any thresholds I read
the per-item records and noticed a citation referring to page 6 of a research paper - I had pointed
`--pdf` at the wrong file, so the system was being asked 20 questions about a document that did not
contain the answers. The refusals were correct behaviour. I re-ran it against the right document
and added the top retrieved sections to every evaluation record so that this failure mode is
visible immediately rather than being mistaken for a quality problem.
