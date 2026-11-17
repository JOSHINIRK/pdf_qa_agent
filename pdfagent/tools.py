"""The agent's tools: JSON schemas for the model, and their implementations.

Every passage a tool returns is registered in a SourceRegistry under a short id
("S1", "S2", ...). The model cites those ids; the agent later turns them into
numbered citations with document, section and page references. A citation can
therefore only point at text the model was actually shown.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .llm import ToolCall
from .models import Document, format_pages
from .store import DocumentStore
from .tokens import count_tokens, truncate_tokens

log = logging.getLogger(__name__)

MAX_PAGES_PER_READ = 5
MAX_OUTLINE_ENTRIES = 300


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
        },
    }


_DOC_ID = {"type": "string", "description": "Document id from list_documents (the filename or title also works)."}

TOOL_SPECS: list[dict[str, Any]] = [
    _fn(
        "list_documents",
        "List the documents in the library: id, title, filename, page count and number of sections.",
        {},
        [],
    ),
    _fn(
        "search_documents",
        "Search the documents for passages relevant to a query (semantic + keyword search). Returns passages "
        "with a source_id to cite, the document, section and pages. Write specific queries using the terms the "
        "document would use. For multi-part questions or comparisons, search once per part.",
        {
            "query": {"type": "string", "description": "What to look for."},
            "document_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: restrict the search to these documents.",
            },
            "top_k": {"type": "integer", "minimum": 1, "maximum": 15, "description": "Passages to return (default 6)."},
        },
        ["query"],
    ),
    _fn(
        "get_document_outline",
        "Get the section outline (table of contents) of a document: section ids, titles, levels and page ranges. "
        "Use it for questions about structure, to list what a document covers, or to find the section to read.",
        {"document_id": _DOC_ID},
        ["document_id"],
    ),
    _fn(
        "read_section",
        "Read the full text of one section, including its subsections. Use it to summarise or explain a whole "
        "section, or when search passages are incomplete.",
        {
            "document_id": _DOC_ID,
            "section_id": {"type": "string", "description": "Section id from get_document_outline (e.g. 's7'), or its title."},
        },
        ["document_id", "section_id"],
    ),
    _fn(
        "read_pages",
        f"Read the text of a page range (at most {MAX_PAGES_PER_READ} pages). Use it when the user refers to "
        "specific pages or when a passage continues on the next page.",
        {
            "document_id": _DOC_ID,
            "start_page": {"type": "integer", "minimum": 1},
            "end_page": {"type": "integer", "minimum": 1},
        },
        ["document_id", "start_page", "end_page"],
    ),
]


@dataclass
class Source:
    id: str
    doc_id: str
    doc_title: str
    filename: str
    section: str
    page_start: int
    page_end: int
    text: str

    @property
    def pages(self) -> str:
        return format_pages(self.page_start, self.page_end)


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}
        self._keys: dict[tuple, str] = {}

    def register(self, doc: Document, section: str, page_start: int, page_end: int, text: str) -> Source:
        key = (doc.id, section, page_start, page_end, hash(text))
        if key in self._keys:
            return self._sources[self._keys[key]]
        sid = f"S{len(self._sources) + 1}"
        src = Source(sid, doc.id, doc.title, doc.filename, section, page_start, page_end, text)
        self._sources[sid] = src
        self._keys[key] = sid
        return src

    def get(self, sid: str) -> Source | None:
        return self._sources.get(sid)

    def __contains__(self, sid: str) -> bool:
        return sid in self._sources

    def __len__(self) -> int:
        return len(self._sources)


class ToolError(ValueError):
    """Bad tool input. The message is returned to the model so it can correct itself."""


class ToolExecutor:
    def __init__(self, store: DocumentStore, settings: Settings, scope: list[str] | None = None):
        self.store = store
        self.settings = settings
        self.scope = set(scope) if scope else None
        self.registry = SourceRegistry()
        self._handlers = {
            "list_documents": self._list_documents,
            "search_documents": self._search,
            "get_document_outline": self._outline,
            "read_section": self._read_section,
            "read_pages": self._read_pages,
        }

    def execute(self, call: ToolCall) -> tuple[str, str]:
        """Run a tool call. Returns (JSON result for the model, short summary for the UI)."""
        handler = self._handlers.get(call.name)
        try:
            if handler is None:
                raise ToolError(f"Unknown tool '{call.name}'. Available: {', '.join(self._handlers)}.")
            args = call.parsed_arguments()
            result, summary = handler(**args)
        except TypeError as exc:
            result, summary = {"error": f"Invalid arguments for {call.name}: {exc}"}, "invalid arguments"
        except (ToolError, ValueError) as exc:
            result, summary = {"error": str(exc)}, f"error: {exc}"
        except Exception as exc:  # never let a tool crash the agent loop
            log.exception("Tool %s failed", call.name)
            result, summary = {"error": f"Tool failed: {exc}"}, "tool failed"
        return json.dumps(result, ensure_ascii=False), summary

    # ------------------------------------------------------------------ helpers

    def _visible_docs(self) -> list[Document]:
        docs = self.store.list_documents()
        return [d for d in docs if self.scope is None or d.id in self.scope]

    def _resolve(self, ref: str) -> Document:
        ref = (ref or "").strip()
        docs = self._visible_docs()
        if not docs:
            raise ToolError("The library is empty (or no documents are selected).")
        low = ref.lower()
        for match in (
            lambda d: d.id == ref,
            lambda d: d.id.startswith(ref) and len(ref) >= 6,
            lambda d: d.filename.lower() == low or d.title.lower() == low,
            lambda d: low and (low in d.filename.lower() or low in d.title.lower()),
        ):
            found = [d for d in docs if match(d)]
            if len(found) == 1:
                return found[0]
        known = "; ".join(f"{d.id} ({d.filename})" for d in docs)
        raise ToolError(f"Unknown or ambiguous document '{ref}'. Available documents: {known}")

    def _budget(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Trim passage texts so the whole result stays under max_tool_result_tokens."""
        limit = self.settings.max_tool_result_tokens
        if sum(count_tokens(i["text"]) for i in items) <= limit:
            return items
        per_item = max(80, limit // max(1, len(items)))
        for item in items:
            item["text"], cut = truncate_tokens(item["text"], per_item)
            if cut:
                item["truncated"] = True
        return items

    # ------------------------------------------------------------------ tools

    def _list_documents(self) -> tuple[dict, str]:
        docs = self._visible_docs()
        return (
            {
                "documents": [
                    {
                        "document_id": d.id,
                        "title": d.title,
                        "filename": d.filename,
                        "pages": d.num_pages,
                        "sections": len(d.sections),
                    }
                    for d in docs
                ]
            },
            f"{len(docs)} document(s)",
        )

    def _search(self, query: str, document_ids: list[str] | None = None, top_k: int | None = None) -> tuple[dict, str]:
        if not query or not query.strip():
            raise ToolError("query must not be empty.")
        k = max(1, min(int(top_k or self.settings.search_top_k), 15))
        ids = [self._resolve(r).id for r in document_ids] if document_ids else None
        if ids is None and self.scope is not None:
            ids = sorted(self.scope)
        hits = self.store.search(query, top_k=k, doc_ids=ids, candidates=self.settings.search_candidates)
        passages = []
        for hit in hits:
            c = hit.chunk
            doc = self.store.get_document(c.doc_id)
            if doc is None:
                continue
            src = self.registry.register(doc, c.section_path, c.page_start, c.page_end, c.text)
            passages.append(
                {"source_id": src.id, "document": doc.title, "document_id": doc.id, "section": c.section_path,
                 "pages": src.pages, "text": c.text}
            )
        result: dict[str, Any] = {"query": query, "passages": self._budget(passages)}
        if not passages:
            result["note"] = "No matching passages. Try different wording or check the document outline."
        n_docs = len({p["document_id"] for p in passages})
        return result, f"{len(passages)} passage(s) from {n_docs} document(s)"

    def _outline(self, document_id: str) -> tuple[dict, str]:
        doc = self._resolve(document_id)
        entries = []
        for s in doc.sections[:MAX_OUTLINE_ENTRIES]:
            start, end = doc.page_range(s.id)
            entries.append({"section_id": s.id, "title": s.title, "level": s.level, "pages": format_pages(start, end)})
        result: dict[str, Any] = {
            "document_id": doc.id,
            "title": doc.title,
            "pages": doc.num_pages,
            "structure_source": doc.structure_source,
            "sections": entries,
        }
        if len(doc.sections) > MAX_OUTLINE_ENTRIES:
            result["note"] = f"Outline truncated to the first {MAX_OUTLINE_ENTRIES} of {len(doc.sections)} sections."
        if doc.structure_source == "pages":
            result["note"] = "No headings were detected in this document, so each page is listed as a section."
        return result, f"{len(entries)} section(s) of {doc.filename}"

    def _read_section(self, document_id: str, section_id: str) -> tuple[dict, str]:
        doc = self._resolve(document_id)
        section = doc.section(section_id)
        if section is None:  # the model may pass a title instead of an id
            low = section_id.strip().lower()
            matches = [s for s in doc.sections if s.title.lower() == low] or [
                s for s in doc.sections if low and low in s.title.lower()
            ]
            if len(matches) != 1:
                raise ToolError(f"Unknown section '{section_id}' in {doc.filename}. Use get_document_outline for ids.")
            section = matches[0]
        parts = []
        for s in doc.descendants(section.id):
            if not s.blocks:
                continue
            path = " > ".join(doc.section_path(s.id))
            src = self.registry.register(doc, path, s.page_start, s.page_end, s.text)
            parts.append({"source_id": src.id, "section": path, "pages": src.pages, "text": s.text})
        start, end = doc.page_range(section.id)
        result: dict[str, Any] = {
            "document": doc.title,
            "document_id": doc.id,
            "section": " > ".join(doc.section_path(section.id)),
            "pages": format_pages(start, end),
            "parts": self._budget(parts),
        }
        if not parts:
            result["note"] = "This section has no body text."
        if any(p.get("truncated") for p in parts):
            result["note"] = "Some parts were truncated; use read_pages or search_documents for the rest."
        return result, f"'{section.title}' ({len(parts)} part(s))"

    def _read_pages(self, document_id: str, start_page: int, end_page: int) -> tuple[dict, str]:
        doc = self._resolve(document_id)
        start, end = int(start_page), int(end_page)
        if start > end:
            start, end = end, start
        if start < 1 or start > doc.num_pages:
            raise ToolError(f"{doc.filename} has pages 1-{doc.num_pages}.")
        end = min(end, doc.num_pages, start + MAX_PAGES_PER_READ - 1)
        pages = []
        for page in range(start, end + 1):
            titles, texts = [], []
            for s in doc.sections:
                blocks = [b.text for b in s.blocks if b.page == page]
                if blocks:
                    titles.append(s.title)
                    texts.append(f"[{s.title}]\n" + "\n\n".join(blocks))
            text = "\n\n".join(texts) or "(no extractable text on this page)"
            src = self.registry.register(doc, "; ".join(titles) or f"Page {page}", page, page, text)
            pages.append({"source_id": src.id, "page": page, "sections": titles, "text": text})
        return (
            {"document": doc.title, "document_id": doc.id, "pages": self._budget(pages)},
            f"{doc.filename} {format_pages(start, end)}",
        )
