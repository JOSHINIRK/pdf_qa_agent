"""Token counting. Uses tiktoken when available, else a ~4 characters/token estimate
(tiktoken downloads its vocabulary on first use, which fails on offline machines)."""

from __future__ import annotations

import logging
from functools import lru_cache

log = logging.getLogger(__name__)
_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=1)
def _encoding():
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception as exc:  # network error, missing package, ...
        log.warning("tiktoken unavailable (%s); using approximate token counts", exc)
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _encoding()
    if enc is None:
        return max(1, len(text) // _CHARS_PER_TOKEN)
    return len(enc.encode(text, disallowed_special=()))


def truncate_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Cut text to at most max_tokens. Returns (text, was_truncated)."""
    enc = _encoding()
    if enc is None:
        limit = max_tokens * _CHARS_PER_TOKEN
        return (text, False) if len(text) <= limit else (text[:limit], True)
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        return text, False
    return enc.decode(ids[:max_tokens]), True
