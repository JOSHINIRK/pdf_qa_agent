"""Generate the three fictional sample PDFs (research paper, product manual, contract).

    python scripts/make_samples.py            writes samples/*.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfagent.samples import write_samples  # noqa: E402

if __name__ == "__main__":
    for path in write_samples(ROOT / "samples"):
        print(f"wrote {path.relative_to(ROOT)}")
