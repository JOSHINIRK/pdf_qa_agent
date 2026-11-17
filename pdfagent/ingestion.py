"""PDF ingestion: validate -> parse -> chunk -> embed -> store."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .chunking import chunk_document
from .config import Settings
from .llm import LLMError
from .models import Document
from .parsing import PDFParseError, parse_pdf
from .store import DocumentStore

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    filename: str
    status: str  # "added" | "duplicate" | "failed"
    message: str
    document: Document | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("added", "duplicate")


class Ingestor:
    def __init__(self, store: DocumentStore, settings: Settings):
        self.store = store
        self.settings = settings

    def ingest_path(self, path: str | Path) -> IngestResult:
        p = Path(path)
        try:
            data = p.read_bytes()
        except OSError as exc:
            return IngestResult(p.name, "failed", f"Cannot read file: {exc}")
        return self.ingest_bytes(data, p.name)

    def ingest_bytes(self, data: bytes, filename: str) -> IngestResult:
        filename = Path(filename).name  # never trust a client-supplied path
        limit = self.settings.max_upload_mb * 1024 * 1024
        if len(data) > limit:
            return IngestResult(filename, "failed", f"File is larger than {self.settings.max_upload_mb} MB.")
        if b"%PDF-" not in data[:1024]:
            return IngestResult(filename, "failed", "Not a PDF file.")

        sha = hashlib.sha256(data).hexdigest()
        doc_id = sha[:16]
        existing = self.store.get_document(doc_id)
        if existing is not None:
            return IngestResult(filename, "duplicate", f"Already in the library as '{existing.title}'.", existing)

        try:
            parsed = parse_pdf(data, filename)
        except PDFParseError as exc:
            return IngestResult(filename, "failed", str(exc))
        except Exception as exc:  # malformed PDFs can fail deep inside MuPDF
            log.exception("Unexpected parse failure for %s", filename)
            return IngestResult(filename, "failed", f"Could not parse the PDF: {exc}")

        if parsed.num_pages > self.settings.max_pages:
            return IngestResult(filename, "failed", f"PDF has {parsed.num_pages} pages; the limit is {self.settings.max_pages}.")

        doc = Document(
            id=doc_id,
            filename=filename,
            title=parsed.title,
            num_pages=parsed.num_pages,
            sha256=sha,
            sections=parsed.sections,
            structure_source=parsed.structure_source,
            ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            warnings=parsed.warnings,
        )
        chunks = chunk_document(doc, self.settings.chunk_tokens, self.settings.chunk_overlap_tokens)
        if not chunks:
            return IngestResult(filename, "failed", "No text chunks could be produced from this PDF.")

        try:
            self.store.add(doc, chunks, data)
        except LLMError as exc:
            return IngestResult(filename, "failed", str(exc))
        except Exception as exc:
            log.exception("Failed to store %s", filename)
            return IngestResult(filename, "failed", f"Could not index the document: {exc}")

        msg = f"{doc.num_pages} pages, {len(doc.sections)} sections, {len(chunks)} chunks."
        return IngestResult(filename, "added", msg, doc, parsed.warnings)
