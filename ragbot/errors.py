"""Typed errors. ``user_message`` is always safe to show to an end user;"""
from __future__ import annotations


class RAGError(Exception):
    code = "rag_error"

    def __init__(self, message: str, user_message: str | None = None):
        super().__init__(message)
        self.user_message = user_message or "Something went wrong while processing the request."


class InputValidationError(RAGError):
    code = "invalid_input"


class PDFParseError(RAGError):
    code = "pdf_parse_error"


class EmptyDocumentError(PDFParseError):
    code = "empty_document"


class DocumentNotFoundError(RAGError):
    code = "document_not_found"


class IndexMismatchError(RAGError):
    code = "index_mismatch"


class LLMError(RAGError):
    """Model/API failure (unreachable, timeout, rejected request)."""
    code = "llm_error"


class LLMOutputError(LLMError):
    """The model answered, but its output is empty or malformed."""
    code = "llm_output_error"
