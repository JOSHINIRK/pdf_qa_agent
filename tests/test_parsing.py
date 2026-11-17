import pymupdf
import pytest

from pdfagent.parsing import PDFParseError, parse_pdf
from pdfagent.samples import CONTRACT, MANUAL, PAPER, headings


def titles(parsed):
    return [s.title for s in parsed.sections]


@pytest.mark.parametrize(
    "name, spec, source",
    [
        ("research_paper.pdf", PAPER, "layout"),     # font-size headings, no bookmarks
        ("product_manual.pdf", MANUAL, "outline"),   # PDF bookmarks
        ("service_contract.pdf", CONTRACT, "layout"),  # bold numbered clauses in body font size
    ],
)
def test_every_heading_becomes_a_section(sample_pdfs, name, spec, source):
    parsed = parse_pdf(sample_pdfs[name], name)
    assert parsed.structure_source == source
    expected = [t for _, t in headings(spec)]
    assert titles(parsed) == ["Front matter"] + expected
    assert parsed.warnings == []


def test_heading_levels_follow_numbering(sample_pdfs):
    parsed = parse_pdf(sample_pdfs["research_paper.pdf"])
    by_title = {s.title: s for s in parsed.sections}
    assert by_title["3 Method"].level == 1
    assert by_title["3.2 Budget Controller"].level == 2
    assert by_title["3.2 Budget Controller"].parent_id == by_title["3 Method"].id


def test_title_from_metadata(sample_pdfs):
    assert parse_pdf(sample_pdfs["service_contract.pdf"]).title == "Master Software Services Agreement"


def test_running_header_and_page_numbers_removed(sample_pdfs):
    parsed = parse_pdf(sample_pdfs["product_manual.pdf"])
    text = "\n".join(s.text for s in parsed.sections)
    assert "Page 1 of" not in text
    assert "User Manual" not in text.replace("Installation and User Manual", "")


def test_tables_become_rows(sample_pdfs):
    parsed = parse_pdf(sample_pdfs["product_manual.pdf"])
    schedule = next(s for s in parsed.sections if s.title == "5.1 Filter Replacement Schedule")
    assert "1 | PF-1 sediment pre-filter | 6 months" in schedule.text


def test_pages_are_tracked(sample_pdfs):
    parsed = parse_pdf(sample_pdfs["service_contract.pdf"])
    law = next(s for s in parsed.sections if s.title.startswith("12."))
    assert law.page_start == law.page_end == parsed.num_pages


def test_document_without_headings_falls_back_to_pages():
    doc = pymupdf.open()
    for i in range(2):
        page = doc.new_page()
        page.insert_text((72, 100), f"Plain body text on page {i + 1} with no headings at all.", fontsize=11)
    parsed = parse_pdf(doc.tobytes())
    assert parsed.structure_source == "pages"
    assert titles(parsed) == ["Page 1", "Page 2"]


def test_scanned_pdf_is_rejected():
    doc = pymupdf.open()
    doc.new_page()
    with pytest.raises(PDFParseError, match="no extractable text"):
        parse_pdf(doc.tobytes())


def test_empty_page_produces_warning():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "Some text.", fontsize=11)
    doc.new_page()
    parsed = parse_pdf(doc.tobytes())
    assert parsed.warnings and "2" in parsed.warnings[0]


def test_not_a_pdf():
    with pytest.raises(PDFParseError):
        parse_pdf(b"%PDF-1.7 this is garbage")


def test_encrypted_pdf_is_rejected():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "Secret", fontsize=11)
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="owner")
    with pytest.raises(PDFParseError, match="password"):
        parse_pdf(data)
