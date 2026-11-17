"""Render a PDF page to PNG with a cited passage highlighted (in memory only)."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import pymupdf

MIN_RUN = 4  # shortest run of consecutive matching words that gets highlighted


def _norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


def highlight_rects(page: pymupdf.Page, passage: str) -> list[pymupdf.Rect]:
    """Rectangles of the words on the page that belong to runs of the passage text."""
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
    page_tokens = [_norm(w[4]) for w in words]
    passage_tokens = [t for t in (_norm(w) for w in passage.split()) if t]
    if not page_tokens or not passage_tokens:
        return []
    matcher = SequenceMatcher(None, page_tokens, passage_tokens, autojunk=False)
    hit: list[int] = []
    for block in matcher.get_matching_blocks():
        if block.size >= MIN_RUN:
            hit.extend(range(block.a, block.a + block.size))
    # The alignment occasionally skips a word between two runs; close gaps of up to 2 words.
    filled: list[int] = []
    for i in sorted(set(hit)):
        if filled and 1 < i - filled[-1] <= 3:
            filled.extend(range(filled[-1] + 1, i))
        filled.append(i)
    return [pymupdf.Rect(words[i][:4]) for i in filled]


def render_page_png(pdf_path: str | Path, page: int, passage: str = "", zoom: float = 1.5) -> bytes:
    with pymupdf.open(pdf_path) as doc:
        pg = doc[max(1, min(page, doc.page_count)) - 1]
        rects = highlight_rects(pg, passage) if passage else []
        if rects:
            pg.add_highlight_annot(rects)
        return pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), annots=True).tobytes("png")
