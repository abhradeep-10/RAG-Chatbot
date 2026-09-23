"""Lightweight Streamlit UI. Run from the repo root: streamlit run ui/streamlit_app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from ragbot.errors import RAGError  # noqa: E402
from ragbot.logging_utils import setup_logging  # noqa: E402
from ragbot.schemas import ChatRequest, ChatTurn  # noqa: E402
from ragbot.service import RAGService  # noqa: E402

st.set_page_config(page_title="Document Chatbot", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def get_service() -> RAGService:
    setup_logging()
    return RAGService()


service = get_service()
st.session_state.setdefault("histories", {})

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None and st.button("Ingest", type="primary"):
        with st.spinner("Parsing, chunking and embedding..."):
            try:
                info = service.ingest_pdf(uploaded.getvalue(), uploaded.name)
                st.session_state.doc_id = info.doc_id
                st.success(f"Indexed {info.n_chunks} chunks from {info.n_pages} pages.")
            except RAGError as exc:
                st.error(exc.user_message)

    docs = service.list_documents()
    if docs:
        ids = [d.doc_id for d in docs]
        names = {d.doc_id: f"{d.filename} ({d.n_pages} p, {d.n_chunks} chunks)" for d in docs}
        current = st.session_state.get("doc_id")
        st.session_state.doc_id = st.selectbox("Active document", ids, format_func=names.get,
                                               index=ids.index(current) if current in ids else 0)
    show_debug = st.toggle("Show retrieval debug", value=True)
    if st.session_state.get("doc_id") and st.button("Clear conversation"):
        st.session_state.histories.pop(st.session_state.doc_id, None)


def render(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] != "assistant":
            return
        status = msg.get("status")
        if status in ("not_found", "unsupported"):
            st.caption("No supported answer in the document.")
        if msg.get("citations"):
            st.markdown("**Sources**")
            for c in msg["citations"]:
                with st.expander(c["label"]):
                    st.write(c["snippet"])
        dbg = msg.get("debug")
        if show_debug and dbg:
            with st.expander("Retrieval debug"):
                st.write(f"**Search query:** {dbg['standalone_query']}"
                         + ("  _(rewritten from follow-up)_" if dbg["query_rewritten"] else ""))
                st.write(f"**Gate:** {'passed' if dbg['gate_passed'] else 'failed'} - {dbg['gate_reason']}")
                if dbg.get("invalid_citations"):
                    st.warning(f"Model cited unknown passages: {dbg['invalid_citations']}")
                st.dataframe(
                    [{k: c[k] for k in ("rank", "selected", "page_start", "section", "rerank_score",
                                        "dense_score", "sparse_score", "preview")} for c in dbg["candidates"]],
                    use_container_width=True, hide_index=True)
                st.json(dbg["timings_ms"])


# ------------------------------------------------------------------ main
st.title("📄 Document Intelligence Chatbot")
doc_id = st.session_state.get("doc_id")
if not doc_id:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

messages: list[dict] = st.session_state.histories.setdefault(doc_id, [])
for m in messages:
    render(m)

question = st.chat_input("Ask a question about the document")
if question:
    history = [ChatTurn(role=m["role"], content=m["content"]) for m in messages if m.get("status") != "error"]
    try:
        request = ChatRequest(doc_id=doc_id, question=question, history=history)
    except ValidationError as exc:
        st.warning(exc.errors()[0]["msg"])
        st.stop()
    messages.append({"role": "user", "content": question})
    with st.spinner("Searching the document..."):
        try:
            resp = service.chat(request)
            messages.append({
                "role": "assistant", "content": resp.answer, "status": resp.status,
                "citations": [c.model_dump() for c in resp.citations],
                "debug": resp.debug.model_dump() if resp.debug else None,
            })
        except RAGError as exc:
            messages.append({"role": "assistant", "content": exc.user_message, "status": "error"})
    st.rerun()
