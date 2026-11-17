"""Ask the agent a question from the command line.

    python scripts/ask.py "How often should the PF-1 filter be replaced?"
    python scripts/ask.py --doc product_manual.pdf "What does a flashing red light mean?"
    python scripts/ask.py --trace "Compare the warranty and the contract's liability cap"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdfagent import build_services, get_settings  # noqa: E402
from pdfagent.config import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--doc", action="append", default=[], help="restrict to a document (id, filename or title)")
    parser.add_argument("--trace", action="store_true", help="print the agent's tool calls")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging("WARNING")
    services = build_services(settings)

    scope = None
    if args.doc:
        docs = services.store.list_documents()
        scope = [d.id for d in docs for ref in args.doc if ref in (d.id, d.filename, d.title)]
        if not scope:
            print(f"No document matches {args.doc}.", file=sys.stderr)
            return 2

    answer = services.agent.ask(args.question, scope=scope)
    if args.trace:
        for t in answer.trace:
            print(f"  -> {t.name}({t.arguments})  =>  {t.summary}  [{t.seconds:.2f}s]")
        print()
    if answer.error:
        print(f"Error: {answer.error}", file=sys.stderr)
        return 1
    print(answer.text)
    if answer.citations:
        print("\nSources:")
        for c in answer.citations:
            s = c.source
            print(f"  [{c.number}] {s.doc_title} ({s.filename}), {s.section}, {s.pages}")
    print(f"\n({answer.seconds:.1f}s, {answer.model_calls} model calls, "
          f"{answer.prompt_tokens + answer.completion_tokens} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
