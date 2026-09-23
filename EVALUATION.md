# Evaluation methodology

A working demo says nothing about failure rates. The evaluation decomposes the pipeline so that a
wrong answer can be attributed to the stage that caused it: **retrieval → generation → citation →
abstention**.

## Dataset (`evaluation/eval_set.jsonl`, 20 items)

Written against the assignment brief PDF itself so it is runnable immediately; the same schema
applies to the provided PDF.

| Category | # | What it stresses |
|---|---|---|
| `factual` | 5 | single-passage lookup, exact identifiers |
| `multi_passage` | 4 | answers needing two sections/pages |
| `follow_up` | 4 | pronouns/ellipsis resolved only via history (gold history supplied) |
| `unanswerable` | 4 | incl. near-misses sharing vocabulary with real content ("maximum PDF size" vs "maximum amount") |
| `similar_sections` | 3 | same topic in several sections (tests in *Engineering requirements* vs *Deliverables*) |

Each item: `question`, `history`, `answerable`, `expected_pages` (gold evidence), `expected_facts`
(key facts that must appear; `a|b` = alternatives).

Gold labels are page-level because pages are stable across chunking changes - chunk IDs are not.
Changing the chunker must not invalidate the dataset.

## Metrics

**Retrieval quality** - on the post-rerank candidate list (top-k, independent of the gate):
- *Hit@k*: at least one top-k chunk overlaps a gold page.
- *MRR@k*: 1 / rank of the first relevant chunk - rewards putting evidence first.
- *Page recall@k*: fraction of gold pages covered - matters for multi-passage questions.

**Answer quality**
- *Fact recall*: fraction of `expected_facts` present in the answer (normalised dashes/case/space).
- *Accuracy*: answered and all facts present.
- *False-refusal rate*: answerable questions the system declined (the cost of being cautious).

**Citation quality**
- *Citation precision*: fraction of cited chunks that overlap gold pages (does the citation point
  at real evidence?).
- *Answers with citations*: should be 100 % by construction (uncited answers become `unsupported`).
- *Faithfulness (LLM judge, `--judge`)*: the judge sees only the **cited** chunks' full text and
  the answer, and must list unsupported claims. This checks that the cited source actually
  supports the answer, not merely that it is on the right page.

**Hallucination / unsupported-answer rate**
- *Unanswerable hallucination rate*: unanswerable items that received an answer (target 0).
- *Judge-unsupported rate*: answered items containing at least one claim the cited passages do not support.

## Reading the results

- Low Hit@k → retrieval/chunking problem (tune chunk size, hybrid weights, rerank depth).
  High Hit@k but low accuracy → generation/prompt problem.
- Failed follow-ups: check `standalone_query` in the report - usually a rewrite problem.
- Refusal behaviour is a threshold trade-off: sweep `RAG_MIN_RERANK_SCORE` and plot
  false-refusal rate vs unanswerable-hallucination rate; pick the operating point deliberately.
- Every run writes `evaluation/results/eval_<timestamp>.json` (per-item records + summary) so runs
  can be diffed across configuration changes - a regression test for quality.

## Caveats

- 20 items give coarse estimates (one item = 5 points); use it for regressions and failure
  analysis, not for precise claims. Grow it from real user questions / logs.
- Substring fact-matching can miss valid paraphrases (false negatives) - the judge and a manual
  look at FAIL rows complement it.
- The judge is an open 7-8B model grading output from a model of the same family:
  self-preference bias and limited reliability. For higher confidence, use a stronger judge
  and spot-check it against human labels.
