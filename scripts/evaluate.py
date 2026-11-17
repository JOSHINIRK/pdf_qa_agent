"""Evaluate retrieval and answers against eval/golden.jsonl.

Ingest the samples first:  python scripts/ingest.py samples
Then:                      python scripts/evaluate.py                 (full: agent answers, ~1-2 US cents)
                           python scripts/evaluate.py --retrieval-only (search only, embeddings cost)

Metrics
  retrieval hit   the expected document and page are among the top-k search results for the question
  answer correct  every must_include string appears in the answer
  citation ok     at least one citation points to the expected document and page
  refusal ok      for unanswerable questions: the answer says the documents don't contain it
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfagent import build_services, get_settings  # noqa: E402
from pdfagent.config import setup_logging  # noqa: E402

REFUSAL = re.compile(
    r"(do(es)? not|don't|doesn't|no (information|mention|details?)|not (specified|mentioned|stated|covered|"
    r"provided|included|found|available)|cannot find|couldn't find|could not find|unable to find)",
    re.I,
)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'")).lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", default=str(ROOT / "eval" / "golden.jsonl"))
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--only", help="run only cases whose id contains this text")
    parser.add_argument("--out", help="write per-case results as JSON lines to this file")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging("WARNING")
    services = build_services(settings)
    by_name = {d.filename: d for d in services.store.list_documents()}

    cases = [json.loads(line) for line in Path(args.golden).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only:
        cases = [c for c in cases if args.only in c["id"]]
    needed = {c["doc"] for c in cases if c.get("doc")} | {d for c in cases for d in c.get("multi_doc", [])}
    missing = needed - set(by_name)
    if missing:
        print(f"Missing documents in the library: {sorted(missing)}. Run: python scripts/ingest.py samples")
        return 2

    totals = {"retrieval": [0, 0], "answer": [0, 0], "citation": [0, 0], "refusal": [0, 0]}
    rows = []
    for case in cases:
        answerable = case.get("answerable", True)
        row = {"id": case["id"]}
        expected_doc = by_name.get(case["doc"]) if case.get("doc") else None

        if answerable and expected_doc:
            hits = services.store.search(case["question"], top_k=settings.search_top_k, candidates=settings.search_candidates)
            ok = any(h.chunk.doc_id == expected_doc.id and
                     any(h.chunk.page_start <= p <= h.chunk.page_end for p in case["pages"]) for h in hits)
            row["retrieval_hit"] = ok
            totals["retrieval"][0] += ok
            totals["retrieval"][1] += 1

        if not args.retrieval_only:
            ans = services.agent.ask(case["question"])
            row.update(answer=ans.text, error=ans.error, tools=[t.name for t in ans.trace],
                       citations=[f"{c.source.filename} {c.source.pages}" for c in ans.citations],
                       seconds=round(ans.seconds, 1), tokens=ans.prompt_tokens + ans.completion_tokens)
            text = norm(ans.text)
            if answerable:
                ok = not ans.error and all(norm(s) in text for s in case["must_include"])
                row["answer_correct"] = ok
                totals["answer"][0] += ok
                totals["answer"][1] += 1
                docs = [by_name[d].id for d in case.get("multi_doc", [])] or [expected_doc.id]
                if case.get("multi_doc"):
                    cited = {c.source.doc_id for c in ans.citations}
                    ok = set(docs) <= cited
                else:
                    ok = any(c.source.doc_id == docs[0] and
                             any(c.source.page_start <= p <= c.source.page_end for p in case["pages"])
                             for c in ans.citations)
                row["citation_ok"] = ok
                totals["citation"][0] += ok
                totals["citation"][1] += 1
            else:
                ok = not ans.error and bool(REFUSAL.search(ans.text))
                row["refusal_ok"] = ok
                totals["refusal"][0] += ok
                totals["refusal"][1] += 1

        flags = " ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in row.items() if isinstance(v, bool))
        print(f"{case['id']:24} {flags}")
        if not args.retrieval_only and not all(v for v in row.values() if isinstance(v, bool)):
            print(f"    answer: {row.get('answer', '')[:300]!r}\n    citations: {row.get('citations')}  error: {row.get('error')}")
        rows.append(row)

    print("\nSummary")
    for name, (ok, n) in totals.items():
        if n:
            print(f"  {name:10} {ok}/{n}  ({100 * ok / n:.0f}%)")
    if not args.retrieval_only:
        tokens = sum(r.get("tokens", 0) for r in rows)
        secs = sum(r.get("seconds", 0) for r in rows)
        print(f"  total tokens {tokens:,}, mean latency {secs / max(1, len(rows)):.1f}s")
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return 0 if all(ok == n for ok, n in totals.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
