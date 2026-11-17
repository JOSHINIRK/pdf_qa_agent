"""Persistent document library: parsed documents, chunks, vectors and hybrid search.

Layout of data_dir:

    chroma/                 ChromaDB vector index (one vector per chunk)
    documents/<id>.json     parsed Document (outline + full text by section)
    documents/<id>.chunks.json
    pdfs/<id>.pdf           original file, used to render cited pages in the UI

Writes are ordered so that documents/<id>.json is written last: it is the
"commit marker". On start-up, vectors or files without it are treated as the
leftovers of an interrupted ingestion and removed.

Search is hybrid: vector similarity (meaning) and BM25 (exact terms such as
clause numbers, part codes or names) are fused with reciprocal-rank fusion.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from .llm import Embedder
from .models import Chunk, Document, SearchHit

log = logging.getLogger(__name__)

RRF_K = 60  # standard reciprocal-rank-fusion constant
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was were what when "
    "where which who why will with does do did can should shall may i you we they our their there".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text.lower()) if t not in _STOPWORDS]


class DocumentStore:
    def __init__(self, data_dir: Path, embedder: Embedder | None, collection_name: str = "pdf_chunks"):
        self.data_dir = Path(data_dir)
        self.docs_dir = self.data_dir / "documents"
        self.pdf_dir = self.data_dir / "pdfs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._lock = threading.RLock()

        self._client = chromadb.PersistentClient(
            path=str(self.data_dir / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

        self._docs: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._load()

    # ------------------------------------------------------------------ loading

    def _load(self) -> None:
        for path in sorted(self.docs_dir.glob("*.json")):
            if path.name.endswith(".chunks.json"):
                continue
            try:
                doc = Document.from_dict(json.loads(path.read_text(encoding="utf-8")))
                chunks_path = self.docs_dir / f"{doc.id}.chunks.json"
                chunks = [Chunk.from_dict(c) for c in json.loads(chunks_path.read_text(encoding="utf-8"))]
            except Exception as exc:
                log.error("Skipping unreadable document record %s: %s", path.name, exc)
                continue
            self._docs[doc.id] = doc
            for c in chunks:
                self._chunks[c.id] = c
        self._remove_orphans()
        log.info("Library loaded: %d documents, %d chunks", len(self._docs), len(self._chunks))

    def _remove_orphans(self) -> None:
        """Delete vectors and files that belong to no committed document."""
        try:
            meta = self._collection.get(include=["metadatas"])
        except Exception as exc:  # pragma: no cover - only on a corrupt index
            log.error("Could not read the vector index: %s", exc)
            return
        orphan_ids = [
            cid for cid, m in zip(meta["ids"], meta["metadatas"] or []) if (m or {}).get("doc_id") not in self._docs
        ]
        if orphan_ids:
            log.warning("Removing %d orphaned vectors from an interrupted ingestion", len(orphan_ids))
            self._collection.delete(ids=orphan_ids)
        for path in list(self.docs_dir.glob("*.chunks.json")) + list(self.pdf_dir.glob("*.pdf")):
            doc_id = path.name.split(".")[0]
            if doc_id not in self._docs:
                path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ queries

    def list_documents(self) -> list[Document]:
        with self._lock:
            return sorted(self._docs.values(), key=lambda d: d.ingested_at)

    def get_document(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def has(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def chunks_for(self, doc_id: str) -> list[Chunk]:
        return sorted((c for c in self._chunks.values() if c.doc_id == doc_id), key=lambda c: c.index)

    def pdf_path(self, doc_id: str) -> Path:
        return self.pdf_dir / f"{doc_id}.pdf"

    # ------------------------------------------------------------------ writes

    def add(self, doc: Document, chunks: list[Chunk], pdf_bytes: bytes) -> None:
        if self.embedder is None:
            raise RuntimeError("An embedder is required to add documents.")
        texts = [c.embedding_text(doc.title) for c in chunks]
        vectors = self.embedder.embed(texts) if texts else []
        doc.num_chunks = len(chunks)
        with self._lock:
            try:
                for i in range(0, len(chunks), 500):
                    part = chunks[i : i + 500]
                    self._collection.upsert(
                        ids=[c.id for c in part],
                        embeddings=vectors[i : i + 500],
                        documents=texts[i : i + 500],
                        metadatas=[
                            {
                                "doc_id": c.doc_id,
                                "section_id": c.section_id,
                                "page_start": c.page_start,
                                "page_end": c.page_end,
                            }
                            for c in part
                        ],
                    )
                self.pdf_path(doc.id).write_bytes(pdf_bytes)
                _atomic_write(self.docs_dir / f"{doc.id}.chunks.json", [c.to_dict() for c in chunks])
                _atomic_write(self.docs_dir / f"{doc.id}.json", doc.to_dict())  # commit marker
            except Exception:
                self._purge(doc.id)
                raise
            self._docs[doc.id] = doc
            for c in chunks:
                self._chunks[c.id] = c
            self._bm25 = None
        log.info("Stored %s (%s): %d chunks", doc.filename, doc.id, len(chunks))

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            existed = self._docs.pop(doc_id, None) is not None
            self._purge(doc_id)
            for cid in [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]:
                del self._chunks[cid]
            self._bm25 = None
        return existed

    def reset(self) -> None:
        """Delete every document (used when the embedding model or chunking changes)."""
        with self._lock:
            for doc_id in list(self._docs):
                self.delete(doc_id)
            name = self._collection.name
            self._client.delete_collection(name)
            self._collection = self._client.get_or_create_collection(
                name, metadata={"hnsw:space": "cosine"}, embedding_function=None
            )
            shutil.rmtree(self.pdf_dir, ignore_errors=True)
            self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def _purge(self, doc_id: str) -> None:
        try:
            self._collection.delete(where={"doc_id": doc_id})
        except Exception as exc:  # pragma: no cover
            log.error("Could not delete vectors of %s: %s", doc_id, exc)
        for path in (self.docs_dir / f"{doc_id}.json", self.docs_dir / f"{doc_id}.chunks.json", self.pdf_path(doc_id)):
            path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ search

    def search(
        self, query: str, top_k: int = 6, doc_ids: list[str] | None = None, candidates: int = 30
    ) -> list[SearchHit]:
        if not self._chunks or not query.strip():
            return []
        allowed = set(doc_ids) if doc_ids else None
        vector_ranked = self._vector_search(query, candidates, allowed)
        keyword_ranked = self._keyword_search(query, candidates, allowed)

        scores: dict[str, float] = {}
        vrank = {cid: r for r, cid in enumerate(vector_ranked, 1)}
        krank = {cid: r for r, cid in enumerate(keyword_ranked, 1)}
        for cid in set(vrank) | set(krank):
            scores[cid] = sum(1.0 / (RRF_K + r[cid]) for r in (vrank, krank) if cid in r)
        best = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [
            SearchHit(self._chunks[cid], scores[cid], vrank.get(cid), krank.get(cid)) for cid in best if cid in self._chunks
        ]

    def _vector_search(self, query: str, n: int, allowed: set[str] | None) -> list[str]:
        if self.embedder is None:
            return []
        count = self._collection.count()
        if count == 0:
            return []
        vector = self.embedder.embed([query])[0]
        where = None
        if allowed:
            where = {"doc_id": next(iter(allowed))} if len(allowed) == 1 else {"doc_id": {"$in": sorted(allowed)}}
        res = self._collection.query(query_embeddings=[vector], n_results=min(n, count), where=where, include=[])
        return [cid for cid in res["ids"][0] if cid in self._chunks]

    def _keyword_search(self, query: str, n: int, allowed: set[str] | None) -> list[str]:
        terms = tokenize(query)
        if not terms:
            return []
        with self._lock:
            if self._bm25 is None:
                self._bm25_ids = list(self._chunks)
                corpus = [
                    tokenize(self._chunks[cid].embedding_text(self._docs[self._chunks[cid].doc_id].title))
                    for cid in self._bm25_ids
                ]
                self._bm25 = BM25Okapi(corpus) if corpus else None
            bm25, ids = self._bm25, self._bm25_ids
        if bm25 is None:
            return []
        scores = bm25.get_scores(terms)
        ranked = sorted(
            (i for i in range(len(ids)) if scores[i] > 0 and (allowed is None or self._chunks[ids[i]].doc_id in allowed)),
            key=lambda i: scores[i],
            reverse=True,
        )
        return [ids[i] for i in ranked[:n]]


def _atomic_write(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
