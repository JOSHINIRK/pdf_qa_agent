"""Section-aware chunking.

Chunks never cross a section boundary, so every chunk has one section path and
an exact page range for its citation. Inside a section, paragraphs are packed
greedily up to `chunk_tokens`; an oversized paragraph is split at sentence
boundaries (and, for a single giant "sentence" such as a table, at token
boundaries). The last sentences of a chunk are repeated at the start of the next
one (`overlap_tokens`) so a fact that straddles the boundary stays retrievable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Chunk, Document
from .tokens import count_tokens, truncate_tokens

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[\"'(\[A-Z0-9])|\n+")


@dataclass
class _Unit:
    page: int
    text: str
    tokens: int
    overlap: bool = False  # repeated from the previous chunk


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


def _hard_split(text: str, max_tokens: int) -> list[str]:
    parts = []
    rest = text
    while rest:
        head, cut = truncate_tokens(rest, max_tokens)
        parts.append(head.strip())
        rest = rest[len(head):].strip() if cut else ""
    return [p for p in parts if p]


def _units(doc: Document, section_id: str, chunk_tokens: int) -> list[_Unit]:
    section = doc.section(section_id)
    assert section is not None
    units: list[_Unit] = []
    for block in section.blocks:
        n = count_tokens(block.text)
        if n <= chunk_tokens:
            units.append(_Unit(block.page, block.text, n))
            continue
        pieces: list[str] = []
        for sentence in split_sentences(block.text):
            if count_tokens(sentence) > chunk_tokens:
                pieces.extend(_hard_split(sentence, chunk_tokens))
            else:
                pieces.append(sentence)
        # Re-pack sentences into paragraph-sized units.
        buf: list[str] = []
        for piece in pieces:
            if buf and count_tokens(" ".join(buf + [piece])) > chunk_tokens:
                text = " ".join(buf)
                units.append(_Unit(block.page, text, count_tokens(text)))
                buf = []
            buf.append(piece)
        if buf:
            text = " ".join(buf)
            units.append(_Unit(block.page, text, count_tokens(text)))
    return units


def _overlap(units: list[_Unit], overlap_tokens: int) -> list[_Unit]:
    """Trailing sentences of the last unit, up to overlap_tokens."""
    if overlap_tokens <= 0 or not units:
        return []
    last = units[-1]
    taken: list[str] = []
    total = 0
    for sentence in reversed(split_sentences(last.text)):
        n = count_tokens(sentence)
        if total + n > overlap_tokens:
            break
        taken.insert(0, sentence)
        total += n
    return [_Unit(last.page, " ".join(taken), total, overlap=True)] if taken else []


def chunk_document(doc: Document, chunk_tokens: int, overlap_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(units: list[_Unit], section_id: str, path: str) -> None:
        chunks.append(
            Chunk(
                id=f"{doc.id}-{len(chunks):05d}",
                doc_id=doc.id,
                section_id=section_id,
                section_path=path,
                page_start=min(u.page for u in units),
                page_end=max(u.page for u in units),
                index=len(chunks),
                text="\n\n".join(u.text for u in units),
            )
        )

    for section in doc.sections:
        if not section.blocks:
            continue
        path = " > ".join(doc.section_path(section.id))
        buf: list[_Unit] = []
        size = 0
        for unit in _units(doc, section.id, chunk_tokens):
            if buf and size + unit.tokens > chunk_tokens:
                emit(buf, section.id, path)
                buf = _overlap(buf, overlap_tokens)
                size = sum(u.tokens for u in buf)
                if size + unit.tokens > chunk_tokens:
                    buf, size = [], 0
            buf.append(unit)
            size += unit.tokens
        if any(not u.overlap for u in buf):
            emit(buf, section.id, path)
    return chunks
