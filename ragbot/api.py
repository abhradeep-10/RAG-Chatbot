"""FastAPI layer. Run: uvicorn ragbot.api:app --reload"""
from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from ragbot.errors import (DocumentNotFoundError, IndexMismatchError, InputValidationError, LLMError,
                           PDFParseError, RAGError)
from ragbot.logging_utils import setup_logging
from ragbot.schemas import ChatRequest, ChatResponse, DocumentInfo
from ragbot.service import RAGService

log = logging.getLogger("ragbot.api")
app = FastAPI(title="Document Intelligence Chatbot", version="1.0.0")

_STATUS_CODES: list[tuple[type[RAGError], int]] = [
    (PDFParseError, 422), (InputValidationError, 422), (DocumentNotFoundError, 404),
    (IndexMismatchError, 409), (LLMError, 503),
]


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    setup_logging()
    return RAGService()


@app.exception_handler(RAGError)
async def rag_error_handler(request: Request, exc: RAGError) -> JSONResponse:
    status = next((code for cls, code in _STATUS_CODES if isinstance(exc, cls)), 500)
    log.warning("%s %s -> %d %s: %s", request.method, request.url.path, status, exc.code, exc)
    return JSONResponse(status_code=status, content={"error": exc.code, "detail": exc.user_message})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentInfo)
def upload_document(file: UploadFile = File(...), service: RAGService = Depends(get_service)) -> DocumentInfo:
    limit = int(service.settings.max_pdf_mb * 1024 * 1024)
    data = file.file.read(limit + 1)           # never read more than limit + 1 bytes into memory
    if len(data) > limit:
        raise PDFParseError("upload too large", f"The file is larger than the {service.settings.max_pdf_mb:g} MB limit.")
    return service.ingest_pdf(data, file.filename or "document.pdf")


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents(service: RAGService = Depends(get_service)) -> list[DocumentInfo]:
    return service.list_documents()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, debug: bool = Query(True, description="Include retrieval debug info"),
         service: RAGService = Depends(get_service)) -> ChatResponse:
    return service.chat(req, include_debug=debug)
