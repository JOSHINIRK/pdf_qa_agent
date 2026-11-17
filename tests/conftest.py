"""Offline fixtures: a deterministic bag-of-words embedder and a scripted chat model."""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdfagent.config import load_settings  # noqa: E402
from pdfagent.ingestion import Ingestor  # noqa: E402
from pdfagent.llm import AssistantTurn, ToolCall  # noqa: E402
from pdfagent.samples import SAMPLES  # noqa: E402
from pdfagent.store import DocumentStore, tokenize  # noqa: E402


class FakeEmbedder:
    """Hashes words into a 256-dim vector: texts sharing words get similar vectors."""

    DIM = 256

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for text in texts:
            vec = [0.0] * self.DIM
            for tok in tokenize(text):
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class FakeChatModel:
    """Replays a script. Each entry is an AssistantTurn or a function(messages) -> AssistantTurn."""

    def __init__(self, script: list[AssistantTurn | Callable[[list[dict[str, Any]]], AssistantTurn]]):
        self.script = list(script)
        self.requests: list[list[dict[str, Any]]] = []
        self.tool_choices: list[str] = []

    def stream(self, messages, tools, tool_choice="auto") -> Iterator[str | AssistantTurn]:
        self.requests.append([dict(m) for m in messages])
        self.tool_choices.append(tool_choice)
        step = self.script.pop(0)
        turn = step(messages) if callable(step) else step
        for word in turn.content.split(" "):
            if word:
                yield word + " "
        yield turn


def call(name: str, arguments: str, cid: str = "call_1") -> ToolCall:
    return ToolCall(cid, name, arguments)


@pytest.fixture(scope="session")
def sample_pdfs() -> dict[str, bytes]:
    return {name: build() for name, build in SAMPLES.items()}


@pytest.fixture
def settings(tmp_path):
    return load_settings(data_dir=str(tmp_path / "data"), chunk_tokens=200, chunk_overlap_tokens=30)


@pytest.fixture
def store(settings):
    return DocumentStore(settings.data_dir, FakeEmbedder(), "test_chunks")


@pytest.fixture
def library(store, settings, sample_pdfs):
    """A store with the three sample documents ingested."""
    ingestor = Ingestor(store, settings)
    for name, data in sample_pdfs.items():
        result = ingestor.ingest_bytes(data, name)
        assert result.status == "added", result.message
    return store
