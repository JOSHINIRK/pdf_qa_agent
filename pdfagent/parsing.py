"""PDF parsing: text extraction, header/footer removal and section detection.

Section structure comes from the best available source, in this order:

1. "outline": the PDF's bookmarks (table of contents), when it has at least two.
2. "layout":  headings detected from typography: lines set in a larger font than
              the body text, or short stand-alone bold lines (numbered clauses
              such as "8. TERM AND TERMINATION"). Numbering ("3.1") sets the depth.
3. "pages":   one section per page, when neither of the above finds headings.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from .models import Block, Section

log = logging.getLogger(__name__)

MARGIN = 0.08  # top/bottom fraction of the page where running headers and footers live
_NUMBERED = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\.?\s+\S")
_KEYWORD = re.compile(r"^(article|section|schedule|appendix|annex|chapter|part)\s+[0-9IVXLCA-Z]", re.I)
_PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.I)
_LIST_ITEM = re.compile(r"^([•●▪\-*]|\(?[a-z0-9]{1,3}[.)])\s")
_BAD_META_TITLE = re.compile(r"(^microsoft (word|powerpoint)|\.(pdf|docx?|pptx?)$|^untitled|^document\d*$)", re.I)


class PDFParseError(ValueError):
    """The file cannot be used: not a PDF, encrypted, or without extractable text."""


@dataclass
class ParsedPDF:
    title: str
    num_pages: int
    sections: list[Section]
    structure_source: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Line:
    page: int
    block: int  # global block number; lines in the same block form one paragraph
    text: str
    size: float
    bold: bool
    top: float  # vertical position as a fraction of page height
    bottom: float
    heading_level: int | None = None
    is_title: bool = False


def parse_pdf(data: bytes, filename: str = "document.pdf") -> ParsedPDF:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PDFParseError(f"Not a readable PDF ({exc}).") from exc

    with doc:
        if doc.needs_pass and not doc.authenticate(""):
            raise PDFParseError("The PDF is password-protected. Remove the password and upload it again.")
        if doc.page_count == 0:
            raise PDFParseError("The PDF has no pages.")

        lines, empty_pages = _extract_lines(doc)
        if not lines:
            raise PDFParseError(
                "The PDF contains no extractable text. It is probably a scanned image; "
                "run it through OCR (for example ocrmypdf) first."
            )
        warnings: list[str] = []
        if empty_pages:
            warnings.append(
                f"{len(empty_pages)} page(s) have no extractable text (scanned images?): "
                f"{_compact_ranges(empty_pages)}. Their content is not searchable."
            )

        lines = _drop_headers_footers(lines, doc.page_count)
        body_size = _body_font_size(lines)
        title_text = _mark_title(lines, body_size)
        title = _choose_title(doc.metadata or {}, title_text, filename)

        toc = [(lvl, t.strip(), p) for lvl, t, p in doc.get_toc(simple=True) if t.strip() and p >= 1]
        if len(toc) >= 2:
            lines = _apply_outline(lines, toc)
            source = "outline"
        else:
            n = _detect_headings(lines, body_size)
            content_lines = sum(1 for ln in lines if not ln.is_title)
            if n == 0 or n > 0.4 * content_lines:
                for ln in lines:
                    ln.heading_level = None
                source = "pages"
            else:
                source = "layout"

        sections = _sections_by_page(lines) if source == "pages" else _build_sections(lines)
        log.info("Parsed %s: %d pages, %d sections (%s)", filename, doc.page_count, len(sections), source)
        return ParsedPDF(title, doc.page_count, sections, source, warnings)


def parse_pdf_file(path: str | Path) -> ParsedPDF:
    p = Path(path)
    return parse_pdf(p.read_bytes(), p.name)


# --------------------------------------------------------------------------- extraction


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or bool(re.search(r"bold|black|heavy|semibold", span["font"], re.I))


def _extract_lines(doc: pymupdf.Document) -> tuple[list[_Line], list[int]]:
    # Expand ligatures ("ﬁ" -> "fi") so keyword search matches, and join hyphenated line breaks.
    flags = (pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES) | pymupdf.TEXT_DEHYPHENATE
    lines: list[_Line] = []
    empty_pages: list[int] = []
    block_no = 0
    for pno, page in enumerate(doc, start=1):
        height = page.rect.height or 1.0
        before = len(lines)
        for block in page.get_text("dict", flags=flags)["blocks"]:
            if block.get("type") != 0:
                continue
            block_no += 1
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                raw = unicodedata.normalize("NFKC", "".join(s["text"] for s in line["spans"]))
                text = " ".join(raw.split())
                weights: Counter[float] = Counter()
                for s in spans:
                    weights[round(s["size"] * 2) / 2] += len(s["text"].strip())
                y0, y1 = line["bbox"][1], line["bbox"][3]
                lines.append(
                    _Line(
                        page=pno,
                        block=block_no,
                        text=text,
                        size=weights.most_common(1)[0][0],
                        bold=all(_is_bold(s) for s in spans),
                        top=y0 / height,
                        bottom=y1 / height,
                    )
                )
        if len(lines) == before:
            empty_pages.append(pno)
    return lines, empty_pages


def _drop_headers_footers(lines: list[_Line], num_pages: int) -> list[_Line]:
    """Remove page numbers and running headers/footers repeated on many pages."""

    def in_margin(ln: _Line) -> bool:
        return ln.top < MARGIN or ln.bottom > 1 - MARGIN

    def norm(text: str) -> str:
        return re.sub(r"\d+", "#", text.lower()).strip()

    pages_by_text: dict[str, set[int]] = defaultdict(set)
    for ln in lines:
        if in_margin(ln):
            pages_by_text[norm(ln.text)].add(ln.page)
    threshold = max(2, round(0.5 * num_pages))

    kept = []
    for ln in lines:
        if in_margin(ln) and (
            _PAGE_NUMBER.match(ln.text) or (num_pages >= 2 and len(pages_by_text[norm(ln.text)]) >= threshold)
        ):
            continue
        kept.append(ln)
    return kept


def _body_font_size(lines: list[_Line]) -> float:
    weights: Counter[float] = Counter()
    for ln in lines:
        weights[ln.size] += len(ln.text)
    return weights.most_common(1)[0][0]


def _mark_title(lines: list[_Line], body_size: float) -> str:
    """Mark the largest-font text in the top part of page 1 as the title."""
    first = [ln for ln in lines if ln.page == 1 and ln.top < 0.4]
    if not first:
        return ""
    biggest = max(ln.size for ln in first)
    if biggest < body_size * 1.15:
        return ""
    parts = []
    for ln in first:
        if ln.size == biggest:
            ln.is_title = True
            parts.append(ln.text)
        elif parts:
            break  # title lines are consecutive
    return " ".join(parts)


def _choose_title(meta: dict, title_text: str, filename: str) -> str:
    meta_title = (meta.get("title") or "").strip()
    if len(meta_title) >= 3 and not _BAD_META_TITLE.search(meta_title):
        return meta_title
    if title_text:
        return title_text
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip() or "Untitled"


# --------------------------------------------------------------------------- headings


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _matches(line_key: str, key: str) -> bool:
    return line_key == key or (len(line_key) >= 0.5 * len(key) and key.startswith(line_key))


def _apply_outline(lines: list[_Line], toc: list[tuple[int, str, int]]) -> list[_Line]:
    """Mark the lines that the PDF bookmarks point to as headings.

    A bookmark is looked up on its target page first, then (for bookmarks with a
    wrong page number, which are common) anywhere after the previous heading. If
    its text is not found at all it becomes a synthetic heading at the top of its page.
    """
    result = list(lines)
    pos = 0
    for level, title, page in toc:
        key = _key(title)
        if not key:
            continue
        candidates = [i for i in range(pos, len(result)) if result[i].heading_level is None]
        found = next(
            (i for i in candidates if result[i].page == page and _matches(_key(result[i].text), key)), None
        )
        if found is None:
            found = next((i for i in candidates if _key(result[i].text) == key), None)
        if found is not None:
            # A heading wrapped over several lines: absorb the continuation lines.
            acc = _key(result[found].text)
            while (
                acc != key
                and found + 1 < len(result)
                and result[found + 1].block == result[found].block
                and key.startswith(acc + _key(result[found + 1].text))
            ):
                acc += _key(result.pop(found + 1).text)
            result[found].heading_level = level
            result[found].text = title
            pos = found + 1
        else:
            idx = next((i for i in range(pos, len(result)) if result[i].page >= page), len(result))
            at_page = max(page, result[idx - 1].page if idx > 0 else 1)
            result.insert(idx, _Line(at_page, -1, title, 0.0, True, 0.0, 0.0, heading_level=level))
            pos = idx + 1
    return result


def _detect_headings(lines: list[_Line], body_size: float) -> int:
    """Mark heading lines from typography. Returns the number of headings found."""
    block_lines: Counter[int] = Counter(ln.block for ln in lines)
    block_bold: dict[int, bool] = defaultdict(lambda: True)
    for ln in lines:
        block_bold[ln.block] = block_bold[ln.block] and ln.bold

    # Lines sharing a baseline with other lines are table cells or columns, not headings.
    row_size: Counter[tuple[int, float]] = Counter((ln.page, round(ln.top, 3)) for ln in lines)

    kinds: list[tuple[int, str]] = []  # (line index, "size" | "bold")
    for i, ln in enumerate(lines):
        text = ln.text
        if ln.is_title or not 2 <= len(text) <= 150 or not re.search(r"[A-Za-z]", text):
            continue
        if _PAGE_NUMBER.match(text) or text.endswith((",", ";")):
            continue
        if ln.size >= body_size + 1.0 and ln.size >= body_size * 1.12:
            kinds.append((i, "size"))
        elif (
            ln.bold
            and block_bold[ln.block]
            and block_lines[ln.block] <= 3
            and row_size[(ln.page, round(ln.top, 3))] == 1
            and len(text) <= 100
            and not text.endswith((".", ":"))
        ):
            kinds.append((i, "bold"))

    # Merge a heading that wraps onto the next line of the same block.
    merged: list[tuple[int, str]] = []
    last_line = -2  # index of the last line absorbed into the previous heading
    for i, kind in kinds:
        if merged:
            pi, pkind = merged[-1]
            prev = lines[pi]
            if pkind == kind and i == last_line + 1 and lines[i].block == prev.block and lines[i].size == prev.size:
                prev.text = f"{prev.text} {lines[i].text}"
                lines[i].heading_level = -1  # marker: absorbed into the previous heading
                last_line = i
                continue
        merged.append((i, kind))
        last_line = i

    size_rank = {s: r for r, s in enumerate(sorted({lines[i].size for i, k in merged if k == "size"}, reverse=True), 1)}
    bold_level = len(size_rank) + 1
    for i, kind in merged:
        ln = lines[i]
        m = _NUMBERED.match(ln.text)
        if m:
            level = m.group("num").count(".") + 1
        else:
            level = size_rank[ln.size] if kind == "size" else bold_level
        ln.heading_level = min(level, 6)

    # Drop absorbed continuation lines.
    lines[:] = [ln for ln in lines if ln.heading_level != -1]
    return len(merged)


# --------------------------------------------------------------------------- sections


@dataclass
class _Cursor:
    """Where the previous line went: lets the next line join the same paragraph or table row."""

    last: _Line | None = None
    band: tuple[float, float] = (0.0, 0.0)  # vertical extent of the current line or table row


def _append_line(section: Section, ln: _Line, cur: _Cursor) -> None:
    """Add a line to the section. Lines of the same block join into one paragraph;
    lines that overlap vertically (table cells) join with " | ", and consecutive
    table rows stay together in one block, one row per line."""
    prev = section.blocks[-1] if section.blocks else None
    last = cur.last
    overlap = min(ln.bottom, cur.band[1]) - max(ln.top, cur.band[0])
    overlaps = overlap > 0.5 * (ln.bottom - ln.top)
    if prev is None or last is None or last.page != ln.page or prev.page != ln.page:
        section.blocks.append(Block(ln.page, ln.text))
        cur.band = (ln.top, ln.bottom)
    elif overlaps:
        prev.text += " | " + ln.text  # next cell of the same row
        cur.band = (min(cur.band[0], ln.top), max(cur.band[1], ln.bottom))
    elif ln.block == last.block:
        prev.text += ("\n" if _LIST_ITEM.match(ln.text) else " ") + ln.text  # wrapped line
        in_row = " | " in prev.text.rsplit("\n", 1)[-1]
        cur.band = (min(cur.band[0], ln.top), max(cur.band[1], ln.bottom)) if in_row else (ln.top, ln.bottom)
    elif " | " in prev.text.rsplit("\n", 1)[-1] and ln.size == last.size and -0.002 <= ln.top - cur.band[1] < 0.02:
        prev.text += "\n" + ln.text  # next table row
        cur.band = (ln.top, ln.bottom)
    else:
        section.blocks.append(Block(ln.page, ln.text))
        cur.band = (ln.top, ln.bottom)
    cur.last = ln
    section.page_end = max(section.page_end, ln.page)


def _build_sections(lines: list[_Line]) -> list[Section]:
    sections: list[Section] = []
    stack: list[Section] = []

    def open_section(title: str, level: int, page: int) -> Section:
        while stack and stack[-1].level >= level:
            stack.pop()
        sec = Section(
            id=f"s{len(sections) + 1}",
            title=title,
            level=level,
            parent_id=stack[-1].id if stack else None,
            page_start=page,
            page_end=page,
        )
        sections.append(sec)
        stack.append(sec)
        return sec

    current: Section | None = None
    cur = _Cursor()
    for ln in lines:
        if ln.heading_level is not None:
            current = open_section(ln.text, ln.heading_level, ln.page)
            cur = _Cursor()
            continue
        if current is None:
            current = open_section("Front matter", 1, ln.page)
        _append_line(current, ln, cur)
    return sections


def _sections_by_page(lines: list[_Line]) -> list[Section]:
    sections: dict[int, Section] = {}
    cur = _Cursor()
    for ln in lines:
        sec = sections.get(ln.page)
        if sec is None:
            sec = sections[ln.page] = Section(f"s{len(sections) + 1}", f"Page {ln.page}", 1, None, ln.page, ln.page)
            cur = _Cursor()
        _append_line(sec, ln, cur)
    return list(sections.values())


def _compact_ranges(pages: list[int]) -> str:
    out, start = [], None
    for i, p in enumerate(pages):
        if start is None:
            start = p
        if i + 1 == len(pages) or pages[i + 1] != p + 1:
            out.append(str(start) if start == p else f"{start}-{p}")
            start = None
    return ", ".join(out)
