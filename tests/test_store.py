from pdfagent.ingestion import Ingestor
from pdfagent.store import DocumentStore

from .conftest import FakeEmbedder


def doc_by_name(store, name):
    return next(d for d in store.list_documents() if d.filename == name)


def test_ingest_three_document_types(library):
    docs = {d.filename: d for d in library.list_documents()}
    assert set(docs) == {"research_paper.pdf", "product_manual.pdf", "service_contract.pdf"}
    for d in docs.values():
        assert d.num_chunks > 0
        assert library.pdf_path(d.id).exists()


def test_duplicate_upload_is_detected(library, settings, sample_pdfs):
    result = Ingestor(library, settings).ingest_bytes(sample_pdfs["product_manual.pdf"], "copy.pdf")
    assert result.status == "duplicate"
    assert len(library.list_documents()) == 3


def test_rejects_non_pdf_and_oversized(store, settings):
    ing = Ingestor(store, settings)
    assert ing.ingest_bytes(b"hello", "x.pdf").status == "failed"
    big = b"%PDF-1.7" + b"0" * (settings.max_upload_mb * 1024 * 1024)
    assert "larger" in ing.ingest_bytes(big, "big.pdf").message


def test_keyword_search_finds_exact_codes(library):
    hits = library.search("UF-3 ultrafiltration membrane replacement interval", top_k=3)
    assert hits[0].chunk.section_path.endswith("5.1 Filter Replacement Schedule")


def test_search_can_be_scoped(library):
    contract = doc_by_name(library, "service_contract.pdf")
    hits = library.search("warranty period years", top_k=5, doc_ids=[contract.id])
    assert hits and all(h.chunk.doc_id == contract.id for h in hits)


def test_hits_carry_pages_and_sections(library):
    hit = library.search("governing law England and Wales courts of London", top_k=1)[0]
    assert "GOVERNING LAW" in hit.chunk.section_path
    assert hit.chunk.page_start >= 1


def test_library_persists_across_restarts(library, settings):
    reopened = DocumentStore(settings.data_dir, FakeEmbedder(), "test_chunks")
    assert len(reopened.list_documents()) == 3
    assert reopened.search("liability cap fees paid", top_k=1)


def test_delete_removes_everything(library, settings):
    manual = doc_by_name(library, "product_manual.pdf")
    assert library.delete(manual.id)
    assert library.get_document(manual.id) is None
    assert not library.pdf_path(manual.id).exists()
    assert all(h.chunk.doc_id != manual.id for h in library.search("filter cartridge", top_k=10))
    reopened = DocumentStore(settings.data_dir, FakeEmbedder(), "test_chunks")
    assert len(reopened.list_documents()) == 2


def test_failed_embedding_leaves_no_partial_document(store, settings, sample_pdfs):
    class Broken(FakeEmbedder):
        def embed(self, texts):
            raise RuntimeError("network down")

    store.embedder = Broken()
    result = Ingestor(store, settings).ingest_bytes(sample_pdfs["research_paper.pdf"], "paper.pdf")
    assert result.status == "failed"
    assert store.list_documents() == []
    assert list(store.pdf_dir.iterdir()) == []


def test_interrupted_ingestion_is_cleaned_up_on_start(library, settings):
    paper = doc_by_name(library, "research_paper.pdf")
    (library.docs_dir / f"{paper.id}.json").unlink()  # simulate a crash before the commit marker
    reopened = DocumentStore(settings.data_dir, FakeEmbedder(), "test_chunks")
    assert reopened.get_document(paper.id) is None
    assert not reopened.pdf_path(paper.id).exists()
    assert reopened._collection.get(where={"doc_id": paper.id})["ids"] == []
