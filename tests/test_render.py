import pymupdf

from pdfagent.parsing import parse_pdf
from pdfagent.render import highlight_rects, render_page_png


def test_highlight_covers_the_cited_clause(sample_pdfs, tmp_path):
    data = sample_pdfs["service_contract.pdf"]
    section = next(s for s in parse_pdf(data).sections if s.title.startswith("8."))
    doc = pymupdf.open(stream=data, filetype="pdf")
    page = doc[section.page_start - 1]
    rects = highlight_rects(page, section.text)
    n_words = len(section.text.split())
    assert len(rects) >= 0.95 * n_words  # (almost) every word of the clause
    assert len(rects) <= n_words + 5  # and not the rest of the page

    path = tmp_path / "c.pdf"
    path.write_bytes(data)
    png = render_page_png(path, section.page_start, section.text)
    assert png.startswith(b"\x89PNG")
    assert path.read_bytes() == data  # highlighting never modifies the file


def test_unrelated_text_is_not_highlighted(sample_pdfs):
    doc = pymupdf.open(stream=sample_pdfs["product_manual.pdf"], filetype="pdf")
    assert highlight_rects(doc[0], "completely unrelated words about astronomy and galaxies far away") == []
