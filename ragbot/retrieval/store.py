"""Per-document persistent index: chunks.jsonl + embeddings.npy + meta.json.

Exact (brute-force) cosine search over a numpy matrix. For a single PDF (hundreds to a few
thousand chunks) this is faster than an ANN index, has perfect recall and zero extra
infrastructure. The class boundary makes swapping in FAISS/Qdrant/pgvector a local change.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ragbot.errors import DocumentNotFoundError, RAGError
from ragbot.schemas import Chunk, DocumentInfo

_DOC_ID_RE = re.compile(r"^[0-9a-f]{16}$")


@dataclass
class DocumentIndex:
    info: DocumentInfo
    chunks: list[Chunk]
    embeddings: np.ndarray                 # (n, d) float32, L2-normalised
    chunker_params: dict = field(default_factory=dict)


class IndexStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, DocumentIndex] = {}
        self._lock = threading.Lock()

    def _dir(self, doc_id: str) -> Path:
        if not _DOC_ID_RE.match(doc_id):   # also prevents path traversal
            raise DocumentNotFoundError(f"invalid doc_id {doc_id!r}", "Document not found.")
        return self.root / doc_id

    def exists(self, doc_id: str) -> bool:
        try:
            return (self._dir(doc_id) / "meta.json").exists()
        except DocumentNotFoundError:
            return False

    def save(self, index: DocumentIndex) -> None:
        doc_id = index.info.doc_id
        final = self._dir(doc_id)
        tmp = Path(tempfile.mkdtemp(dir=self.root, prefix=f".{doc_id}-"))
        try:  # write to a temp dir, then rename: readers never see a half-written index
            (tmp / "chunks.jsonl").write_text(
                "\n".join(c.model_dump_json() for c in index.chunks), encoding="utf-8")
            np.save(tmp / "embeddings.npy", index.embeddings.astype(np.float32))
            meta = {"info": index.info.model_dump(), "chunker": index.chunker_params}
            (tmp / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            with self._lock:
                if final.exists():
                    shutil.rmtree(final)
                tmp.rename(final)
                self._cache[doc_id] = index
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def load(self, doc_id: str) -> DocumentIndex:
        with self._lock:
            if doc_id in self._cache:
                return self._cache[doc_id]
        d = self._dir(doc_id)
        if not (d / "meta.json").exists():
            raise DocumentNotFoundError(f"unknown doc_id {doc_id}", "Document not found. Please upload it first.")
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            lines = (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            chunks = [Chunk.model_validate_json(l) for l in lines if l.strip()]
            emb = np.load(d / "embeddings.npy", allow_pickle=False)
        except Exception as exc:
            raise RAGError(f"corrupt index {doc_id}: {exc}",
                           "The stored index for this document is corrupted; please re-upload it.") from exc
        if emb.ndim != 2 or emb.shape[0] != len(chunks):
            raise RAGError(f"index {doc_id}: {emb.shape} embeddings vs {len(chunks)} chunks",
                           "The stored index for this document is corrupted; please re-upload it.")
        index = DocumentIndex(DocumentInfo(**meta["info"]), chunks, emb, meta.get("chunker", {}))
        with self._lock:
            self._cache[doc_id] = index
        return index

    def list_documents(self) -> list[DocumentInfo]:
        docs = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and (d / "meta.json").exists():
                try:
                    docs.append(DocumentInfo(**json.loads((d / "meta.json").read_text(encoding="utf-8"))["info"]))
                except Exception:
                    continue
        return docs
