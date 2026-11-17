"""Runs the real Streamlit script headlessly with fake OpenAI services."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import pdfagent
from pdfagent.agent import PDFAgent
from pdfagent.ingestion import Ingestor
from pdfagent.llm import AssistantTurn
from pdfagent.store import DocumentStore

from .conftest import FakeChatModel, FakeEmbedder, call

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def cite_first(messages):
    tool = [m for m in messages if m["role"] == "tool"][-1]
    sid = json.loads(tool["content"])["passages"][0]["source_id"]
    return AssistantTurn(f"Replace the PF-1 pre-filter every 6 months [{sid}].")


@pytest.fixture
def fake_services(monkeypatch, settings, tmp_path):
    class FakeServices:
        def __init__(self):
            self.settings = settings
            self.store = DocumentStore(settings.data_dir, FakeEmbedder(), "app_test")
            self.ingestor = Ingestor(self.store, settings)
            self.llm = FakeChatModel([])
            self.agent = PDFAgent(self.store, self.llm, settings)

    svc = FakeServices()
    monkeypatch.setattr(pdfagent, "build_services", lambda settings=None: svc)
    # The "Load sample documents" button writes PDFs; send them to tmp instead of the project folder.
    import pdfagent.samples as samples

    original = samples.write_samples
    monkeypatch.setattr(samples, "write_samples", lambda out: original(tmp_path / "samples"))
    import streamlit as st

    st.cache_resource.clear()
    st.cache_data.clear()
    return svc


def test_load_samples_ask_and_browse(fake_services):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    load = next(b for b in at.sidebar.button if b.label == "Load sample documents")
    load.click().run()
    assert not at.exception
    assert len(fake_services.store.list_documents()) == 3

    fake_services.llm.script.extend([
        AssistantTurn("", [call("search_documents", '{"query": "PF-1 pre-filter replace months"}')]),
        cite_first,
    ])
    at.chat_input[0].set_value("How often is the pre-filter replaced?").run()
    assert not at.exception
    rendered = " ".join(m.value for m in at.markdown)
    assert "every 6 months [1]" in rendered
    assert any("[1] AquaPure" in e.label for e in at.expander)

    at.segmented_control[0].set_value("Documents").run()
    assert not at.exception
    first_doc = fake_services.store.list_documents()[0]
    assert at.metric[0].value == str(first_doc.num_pages)
    assert at.metric[1].value == str(len(first_doc.sections))
    assert any(e.label.startswith("Front matter") for e in at.expander)


def test_missing_key_shows_instructions(monkeypatch):
    import streamlit as st

    def boom(settings=None):
        raise pdfagent.MissingAPIKeyError("No OpenAI API key found.")

    monkeypatch.setattr(pdfagent, "build_services", boom)
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert at.error and "No OpenAI API key" in at.error[0].value
