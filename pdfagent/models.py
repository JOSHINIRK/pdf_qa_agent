"""Plain data classes shared by every module. All are JSON-serialisable via to_dict/from_dict."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Block:
    """A paragraph-like run of text on one page (1-based page number)."""

    page: int
    text: str


@dataclass
class Section:
    """One node of a document's outline. Its own body text is in `blocks`;
    subsections are separate Section objects that point back via `parent_id`."""

    id: str
    title: str
    level: int
    parent_id: str | None
    page_start: int
    page_end: int
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


@dataclass
class Document:
    id: str
    filename: str
    title: str
    num_pages: int
    sha256: str
    sections: list[Section]
    structure_source: str  # "outline" (PDF bookmarks), "layout" (fonts) or "pages" (fallback)
    ingested_at: str
    warnings: list[str] = field(default_factory=list)
    num_chunks: int = 0

    def section(self, section_id: str) -> Section | None:
        return next((s for s in self.sections if s.id == section_id), None)

    def section_path(self, section_id: str) -> list[str]:
        """Titles from the top-level ancestor down to this section."""
        by_id = {s.id: s for s in self.sections}
        path: list[str] = []
        node = by_id.get(section_id)
        while node is not None:
            path.append(node.title)
            node = by_id.get(node.parent_id) if node.parent_id else None
        return list(reversed(path))

    def descendants(self, section_id: str) -> list[Section]:
        """The section itself followed by all its subsections, in document order."""
        wanted = {section_id}
        out: list[Section] = []
        for s in self.sections:
            if s.id in wanted or s.parent_id in wanted:
                wanted.add(s.id)
                out.append(s)
        return out

    def page_range(self, section_id: str) -> tuple[int, int]:
        """Page span of a section including its subsections."""
        nodes = self.descendants(section_id)
        return min(s.page_start for s in nodes), max(s.page_end for s in nodes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        sections = [
            Section(**{**s, "blocks": [Block(**b) for b in s.get("blocks", [])]})
            for s in data.get("sections", [])
        ]
        return cls(**{**data, "sections": sections})


@dataclass
class Chunk:
    """The unit that is embedded and searched."""

    id: str
    doc_id: str
    section_id: str
    section_path: str
    page_start: int
    page_end: int
    index: int
    text: str

    def embedding_text(self, doc_title: str) -> str:
        # Prefixing title and section path lets a passage match questions that
        # name the section ("what does the warranty say...") even if the body doesn't.
        return f"{doc_title}\nSection: {self.section_path}\n\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(**data)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None


def format_pages(start: int, end: int) -> str:
    return f"p. {start}" if start == end else f"pp. {start}-{end}"
