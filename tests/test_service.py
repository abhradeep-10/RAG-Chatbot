import pytest

from ragbot.errors import DocumentNotFoundError, LLMError, PDFParseError
from ragbot.schemas import ChatRequest, ChatTurn
from ragbot.service import ERROR_MESSAGE, NOT_FOUND_MESSAGE, UNSUPPORTED_MESSAGE
from tests.conftest import FakeLLM, answer_json

FOLLOW_UP_HISTORY = [
    ChatTurn(role="user", content="What are the eligibility requirements?"),
    ChatTurn(role="assistant", content="Applicants must be state residents for two years."),
]


def _ingest(service, pdf):
    return service.ingest_pdf(pdf, "handbook.pdf").doc_id


def test_answer_with_citation(make_service, handbook_pdf):
    llm = FakeLLM(fn=lambda m, j: answer_json(["C1"], "The maximum amount allowed is 50 units."))
    svc = make_service(llm)
    doc_id = _ingest(svc, handbook_pdf)
    resp = svc.chat(ChatRequest(doc_id=doc_id, question="What is the maximum amount allowed?"))
    assert resp.status == "answered"
    assert resp.citations[0].page_start == 1
    assert "Eligibility Requirements" in resp.citations[0].section
    assert resp.citations[0].label.startswith("Page 1")
    # retrieved context was actually passed to the LLM
    prompt = llm.calls[-1][0][-1]["content"]
    assert "[C1] Page 1" in prompt and "50 units" in prompt


def test_irrelevant_question_short_circuits_without_llm(make_service, handbook_pdf):
    llm = FakeLLM()
    svc = make_service(llm)
    doc_id = _ingest(svc, handbook_pdf)
    resp = svc.chat(ChatRequest(doc_id=doc_id, question="zebra xylophone quantum"))
    assert resp.status == "not_found" and resp.answer == NOT_FOUND_MESSAGE
    assert resp.debug and not resp.debug.gate_passed
    assert llm.calls == []


def test_model_abstains(make_service, handbook_pdf):
    llm = FakeLLM(fn=lambda m, j: answer_json([], "The document does not say.", answerable=False))
    svc = make_service(llm)
    resp = svc.chat(ChatRequest(doc_id=_ingest(svc, handbook_pdf), question="What is the maximum amount allowed?"))
    assert resp.status == "not_found" and resp.citations == []


def test_hallucinated_citation_is_rejected(make_service, handbook_pdf):
    llm = FakeLLM(fn=lambda m, j: answer_json(["C9"], "Made-up answer."))
    svc = make_service(llm)
    resp = svc.chat(ChatRequest(doc_id=_ingest(svc, handbook_pdf), question="What is the maximum amount allowed?"))
    assert resp.status == "unsupported" and resp.answer == UNSUPPORTED_MESSAGE
    assert resp.debug.invalid_citations == ["C9"]


def test_malformed_output_repaired_once(make_service, handbook_pdf):
    llm = FakeLLM(responses=["garbage, not json", answer_json(["C1"], "50 units.")])
    svc = make_service(llm)
    resp = svc.chat(ChatRequest(doc_id=_ingest(svc, handbook_pdf), question="What is the maximum amount allowed?"))
    assert resp.status == "answered"
    assert len(llm.calls) == 2


def test_malformed_output_twice_returns_error_status(make_service, handbook_pdf):
    llm = FakeLLM(fn=lambda m, j: "still not json")
    svc = make_service(llm)
    resp = svc.chat(ChatRequest(doc_id=_ingest(svc, handbook_pdf), question="What is the maximum amount allowed?"))
    assert resp.status == "error" and resp.answer == ERROR_MESSAGE
    assert len(llm.calls) == 2


def test_llm_api_failure_propagates(make_service, handbook_pdf):
    def boom(m, j):
        raise LLMError("connection refused", "unavailable")
    svc = make_service(FakeLLM(fn=boom))
    doc_id = _ingest(svc, handbook_pdf)
    with pytest.raises(LLMError):
        svc.chat(ChatRequest(doc_id=doc_id, question="What is the maximum amount allowed?"))


def test_follow_up_is_rewritten_before_retrieval(make_service, handbook_pdf):
    def fn(messages, json_mode):
        if not json_mode:
            return "What are the rules for international applicants?"
        return answer_json(["C1"], "They need a sponsor letter.")
    svc = make_service(FakeLLM(fn=fn))
    doc_id = _ingest(svc, handbook_pdf)
    resp = svc.chat(ChatRequest(doc_id=doc_id, question="What about international applicants?",
                                history=FOLLOW_UP_HISTORY))
    assert resp.debug.query_rewritten
    assert resp.debug.standalone_query == "What are the rules for international applicants?"
    assert resp.citations[0].page_start == 2


def test_rewrite_failure_falls_back_to_heuristic(make_service, handbook_pdf):
    def fn(messages, json_mode):
        if not json_mode:
            raise LLMError("timeout")
        return answer_json(["C1"], "Sponsor letter.")
    svc = make_service(FakeLLM(fn=fn))
    doc_id = _ingest(svc, handbook_pdf)
    resp = svc.chat(ChatRequest(doc_id=doc_id, question="What about international applicants?",
                                history=FOLLOW_UP_HISTORY))
    assert resp.status == "answered"
    assert resp.debug.standalone_query.startswith("What are the eligibility requirements?")


def test_history_is_truncated(make_service, handbook_pdf, settings):
    llm = FakeLLM(fn=lambda m, j: "What is the maximum amount allowed?" if not j else answer_json())
    svc = make_service(llm)
    doc_id = _ingest(svc, handbook_pdf)
    history = [ChatTurn(role="user" if i % 2 == 0 else "assistant", content=f"turn-{i:02d}") for i in range(20)]
    svc.chat(ChatRequest(doc_id=doc_id, question="And the maximum?", history=history))
    rewrite_prompt = llm.calls[0][0][-1]["content"]
    assert "turn-19" in rewrite_prompt
    assert "turn-00" not in rewrite_prompt and "turn-15" not in rewrite_prompt   # only last 4 turns


def test_ingest_is_idempotent(make_service, handbook_pdf):
    svc = make_service()
    a = svc.ingest_pdf(handbook_pdf, "a.pdf")
    b = svc.ingest_pdf(handbook_pdf, "a.pdf")
    assert a.doc_id == b.doc_id and len(svc.list_documents()) == 1


def test_bad_pdf_ingest_raises(make_service):
    with pytest.raises(PDFParseError):
        make_service().ingest_pdf(b"not a pdf", "x.pdf")


def test_unknown_document(make_service):
    with pytest.raises(DocumentNotFoundError):
        make_service().chat(ChatRequest(doc_id="0" * 16, question="hi"))


def test_question_validation():
    with pytest.raises(ValueError):
        ChatRequest(doc_id="0" * 16, question="   ")
    with pytest.raises(ValueError):
        ChatRequest(doc_id="0" * 16, question="x" * 5000)
    with pytest.raises(ValueError):
        ChatRequest(doc_id="../../etc", question="hi")
