import pytest
from fastapi.testclient import TestClient

from ragbot.api import app, get_service
from ragbot.errors import LLMError
from tests.conftest import FakeLLM, answer_json


@pytest.fixture
def client_factory(make_service):
    def _make(llm=None):
        svc = make_service(llm)
        app.dependency_overrides[get_service] = lambda: svc
        return TestClient(app)
    yield _make
    app.dependency_overrides.clear()


def _upload(client, data, name="h.pdf"):
    return client.post("/documents", files={"file": (name, data, "application/pdf")})


def test_upload_and_chat(client_factory, handbook_pdf):
    client = client_factory(FakeLLM(fn=lambda m, j: answer_json(["C1"], "50 units.")))
    r = _upload(client, handbook_pdf)
    assert r.status_code == 200
    doc_id = r.json()["doc_id"]
    assert client.get("/documents").json()[0]["doc_id"] == doc_id

    r = client.post("/chat", json={"doc_id": doc_id, "question": "What is the maximum amount allowed?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["citations"][0]["page_start"] == 1
    assert body["debug"]["candidates"]

    r = client.post("/chat?debug=false", json={"doc_id": doc_id, "question": "What is the maximum amount allowed?"})
    assert r.json()["debug"] is None


def test_upload_non_pdf_is_422(client_factory):
    r = _upload(client_factory(), b"just some text", "notes.pdf")
    assert r.status_code == 422 and r.json()["error"] == "pdf_parse_error"


@pytest.mark.parametrize("payload", [
    {"doc_id": "0" * 16, "question": ""},
    {"doc_id": "0" * 16, "question": "x" * 5000},
    {"doc_id": "../secret", "question": "hi"},
    {"doc_id": "0" * 16},
    {"doc_id": "0" * 16, "question": "hi", "history": [{"role": "system", "content": "x"}]},
])
def test_invalid_chat_requests_are_422(client_factory, payload):
    assert client_factory().post("/chat", json=payload).status_code == 422


def test_unknown_document_is_404(client_factory):
    r = client_factory().post("/chat", json={"doc_id": "0" * 16, "question": "hi"})
    assert r.status_code == 404 and r.json()["error"] == "document_not_found"


def test_llm_outage_is_503(client_factory, handbook_pdf):
    def boom(m, j):
        raise LLMError("connection refused", "The language model is unavailable right now.")
    client = client_factory(FakeLLM(fn=boom))
    doc_id = _upload(client, handbook_pdf).json()["doc_id"]
    r = client.post("/chat", json={"doc_id": doc_id, "question": "What is the maximum amount allowed?"})
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]
