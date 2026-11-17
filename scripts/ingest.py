"""Ingest PDFs into the library from the command line.

    python scripts/ingest.py samples/*.pdf          add files (duplicates are skipped)
    python scripts/ingest.py --list                 show the library
    python scripts/ingest.py --delete <doc_id>      remove a document
    python scripts/ingest.py --reset samples/*.pdf  wipe the library, then add files
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdfagent import build_services, get_settings  # noqa: E402
from pdfagent.config import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="PDF files, folders or glob patterns")
    parser.add_argument("--list", action="store_true", help="list the documents in the library")
    parser.add_argument("--delete", metavar="DOC_ID", action="append", default=[], help="remove a document")
    parser.add_argument("--reset", action="store_true", help="delete every document first")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    services = build_services(settings)
    store = services.store

    if args.reset:
        store.reset()
        print("Library cleared.")
    for doc_id in args.delete:
        print(f"{'Deleted' if store.delete(doc_id) else 'Not found'}: {doc_id}")

    files: list[Path] = []
    for pattern in args.paths:
        p = Path(pattern)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.pdf")))
        else:
            files.extend(Path(m) for m in sorted(glob.glob(pattern)) or [pattern])

    failed = 0
    for path in files:
        result = services.ingestor.ingest_path(path)
        print(f"[{result.status.upper():9}] {path.name}: {result.message}")
        for w in result.warnings:
            print(f"            warning: {w}")
        failed += not result.ok

    if args.list or not (files or args.delete or args.reset):
        docs = store.list_documents()
        print(f"\n{len(docs)} document(s) in {settings.data_dir}:")
        for d in docs:
            print(f"  {d.id}  {d.filename:28} {d.num_pages:4} pages {len(d.sections):4} sections "
                  f"{d.num_chunks:5} chunks  [{d.structure_source}]  {d.title}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
