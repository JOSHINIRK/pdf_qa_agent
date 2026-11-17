from pdfagent.chunking import chunk_document, split_sentences
from pdfagent.models import Block, Document, Section
from pdfagent.tokens import count_tokens


def make_doc(blocks_per_section):
    sections = []
    for i, blocks in enumerate(blocks_per_section, 1):
        pages = [b.page for b in blocks] or [1]
        sections.append(Section(f"s{i}", f"Section {i}", 1, None, min(pages), max(pages), blocks))
    return Document("doc1", "d.pdf", "Doc", 5, "x", sections, "layout", "now")


SENT = "The quick brown fox jumps over the lazy dog near the river bank today."


def test_chunks_never_cross_sections():
    doc = make_doc([[Block(1, SENT)], [Block(2, SENT)]])
    chunks = chunk_document(doc, 200, 20)
    assert [c.section_id for c in chunks] == ["s1", "s2"]
    assert chunks[1].section_path == "Section 2"


def test_long_section_is_split_with_overlap_and_page_ranges():
    blocks = [Block(p, " ".join([SENT] * 6)) for p in (1, 1, 2, 2, 3)]
    chunks = chunk_document(make_doc([blocks]), 120, 30)
    assert len(chunks) > 2
    for c in chunks:
        assert count_tokens(c.text) <= 120 + 40  # body + overlap tolerance
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 3
    # the start of each chunk repeats the end of the previous one
    assert chunks[1].text.startswith(split_sentences(chunks[0].text)[-1])


def test_giant_paragraph_is_split():
    text = " ".join(f"word{i}" for i in range(2000))  # no sentence boundaries at all
    chunks = chunk_document(make_doc([[Block(4, text)]]), 150, 0)
    assert len(chunks) >= 5
    assert all(c.page_start == 4 for c in chunks)


def test_empty_sections_are_skipped():
    doc = make_doc([[], [Block(1, SENT)]])
    assert [c.section_id for c in chunk_document(doc, 200, 20)] == ["s2"]


def test_ids_are_unique_and_ordered():
    blocks = [Block(1, " ".join([SENT] * 10)) for _ in range(4)]
    chunks = chunk_document(make_doc([blocks, blocks]), 100, 20)
    assert len({c.id for c in chunks}) == len(chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
