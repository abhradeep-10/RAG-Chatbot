"""
Run the evaluation set end-to-end against a PDF.

"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragbot.errors import LLMError, RAGError  # noqa: E402
from ragbot.llm.output_parser import extract_json_object  # noqa: E402
from ragbot.logging_utils import setup_logging  # noqa: E402
from ragbot.schemas import ChatRequest, ChatTurn  # noqa: E402
from ragbot.service import RAGService  # noqa: E402

JUDGE_SYSTEM = """You grade the faithfulness of an answer to its sources.
A claim is supported only if the SOURCES state it (paraphrase is fine; outside knowledge is not).
Return ONLY JSON: {"supported": true|false, "unsupported_claims": ["..."]}
"supported" is true only if EVERY factual claim in the ANSWER is supported by the SOURCES."""


def norm(text: str) -> str:
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def fact_hit(answer: str, fact: str) -> bool:
    return any(norm(alt) in norm(answer) for alt in fact.split("|"))


def overlaps(start: int, end: int, pages: list[int]) -> bool:
    return any(start <= p <= end for p in pages)


def mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def judge(service: RAGService, answer: str, sources: str) -> dict:
    msgs = [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"SOURCES:\n{sources}\n\nANSWER:\n{answer}"}]
    try:
        data = extract_json_object(service.llm.complete(msgs, json_mode=True, max_tokens=300))
        return {"supported": str(data.get("supported")).lower() == "true",
                "unsupported_claims": data.get("unsupported_claims", [])}
    except LLMError as exc:
        return {"error": str(exc)}


def evaluate_item(service, doc_id, item, k, use_judge, chunk_text) -> dict:
    rec = {"id": item["id"], "category": item["category"], "question": item["question"],
           "answerable": item["answerable"]}
    history = [ChatTurn(**t) for t in item.get("history", [])]
    t0 = time.perf_counter()
    try:
        resp = service.chat(ChatRequest(doc_id=doc_id, question=item["question"], history=history))
    except RAGError as exc:
        rec.update(error=exc.user_message, status="error", success=False)
        return rec
    answered = resp.status == "answered"
    rec.update(status=resp.status, answer=resp.answer, latency_s=round(time.perf_counter() - t0, 2),
               standalone_query=resp.debug.standalone_query,
               citations=[c.label for c in resp.citations])

    exp = item.get("expected_pages", [])
    if item["answerable"]:
        ranked = resp.debug.candidates[:k]         
        rel = [overlaps(c.page_start, c.page_end, exp) for c in ranked]
        rec["hit_at_k"] = any(rel)
        rec["mrr"] = next((1.0 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
        covered = {p for p in exp if any(c.page_start <= p <= c.page_end for c in ranked)}
        rec["page_recall_at_k"] = len(covered) / len(exp) if exp else None
        facts = item.get("expected_facts", [])
        hits = [answered and fact_hit(resp.answer, f) for f in facts]
        rec["fact_recall"] = sum(hits) / len(facts) if facts else None
        rec["answer_correct"] = answered and all(hits)
        rec["false_refusal"] = not answered
        if answered and resp.citations:
            good = [overlaps(c.page_start, c.page_end, exp) for c in resp.citations]
            rec["citation_precision"] = sum(good) / len(good)
        rec["success"] = rec["answer_correct"]
    else:
        rec["correct_refusal"] = not answered
        rec["hallucinated"] = answered
        rec["success"] = rec["correct_refusal"]

    if use_judge and answered:
        sources = "\n\n".join(f"[{c.label}]\n{chunk_text.get(c.chunk_id, c.snippet)}" for c in resp.citations)
        rec["judge"] = judge(service, resp.answer, sources)
    return rec


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if "error" not in r]
    ans = [r for r in ok if r["answerable"]]
    unans = [r for r in ok if not r["answerable"]]
    answered = [r for r in ok if r["status"] == "answered"]
    judged = [r for r in answered if "supported" in r.get("judge", {})]
    by_cat: dict[str, list] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r.get("success", False))
    return {
        "n_items": len(records),
        "pipeline_errors": len(records) - len(ok),
        "retrieval": {"hit@k": mean(r["hit_at_k"] for r in ans),
                      "mrr@k": mean(r["mrr"] for r in ans),
                      "page_recall@k": mean(r["page_recall_at_k"] for r in ans)},
        "answer": {"accuracy": mean(r["answer_correct"] for r in ans),
                   "fact_recall": mean(r["fact_recall"] for r in ans),
                   "false_refusal_rate": mean(r["false_refusal"] for r in ans)},
        "citation": {"precision": mean(r.get("citation_precision") for r in ans if r["status"] == "answered"),
                     "answers_with_citations": mean(bool(r["citations"]) for r in answered)},
        "hallucination": {
            "unanswerable_correct_refusal_rate": mean(r["correct_refusal"] for r in unans),
            "unanswerable_hallucination_rate": mean(r["hallucinated"] for r in unans),
            "judge_unsupported_answer_rate": mean((not r["judge"]["supported"]) for r in judged),
        },
        "by_category": {c: {"n": len(v), "success_rate": mean(v)} for c, v in by_cat.items()},
        "mean_latency_s": mean(r.get("latency_s") for r in ok),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True, type=Path)
    p.add_argument("--dataset", type=Path, default=Path(__file__).with_name("eval_set.jsonl"))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--judge", action="store_true", help="LLM-as-judge faithfulness check on answered items")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    setup_logging("WARNING")

    service = RAGService()
    info = service.ingest_pdf(args.pdf.read_bytes(), args.pdf.name)
    chunk_text = {c.chunk_id: c.text for c in service.store.load(info.doc_id).chunks}
    items = [json.loads(l) for l in args.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"Document {info.filename}: {info.n_pages} pages, {info.n_chunks} chunks; {len(items)} eval items\n")
    records = []
    for item in items:
        rec = evaluate_item(service, info.doc_id, item, args.k, args.judge, chunk_text)
        records.append(rec)
        flag = "PASS" if rec.get("success") else "FAIL"
        print(f"{rec['id']:<4} {rec['category']:<17} {rec['status']:<11} {flag}  {rec['question'][:60]}")

    summary = summarize(records)
    print("\n" + json.dumps(summary, indent=2))
    out = args.out or Path(__file__).parent / "results" / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()
