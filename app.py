"""Streamlit UI for the PDF Q&A agent.

Start it with:  .\\.venv\\Scripts\\python.exe -m streamlit run app.py
"""

from __future__ import annotations

import json
import re

import streamlit as st

from pdfagent import build_services, get_settings
from pdfagent.agent import AgentAnswer
from pdfagent.config import PROJECT_ROOT, setup_logging
from pdfagent.models import format_pages
from pdfagent.render import render_page_png
from pdfagent.samples import write_samples

st.set_page_config(page_title="PDF Q&A Agent", page_icon="📄", layout="wide")

EXAMPLES = [
    "How often should each filter cartridge in the AquaPure 300 be replaced?",
    "Can the customer terminate the services agreement early, and what does it cost?",
    "Summarise the limitations of the Adaptive Sparse Attention paper.",
    "Compare the AquaPure warranty with the contract's liability cap.",
]


# ----------------------------------------------------------------------------- services


@st.cache_resource(show_spinner="Opening the document library...")
def services():
    settings = get_settings()
    setup_logging(settings.log_level)
    return build_services(settings)


try:
    svc = services()
except Exception as exc:  # missing key, unreadable config, ...
    st.title("📄 PDF Q&A Agent")
    st.error(str(exc))
    st.info("Copy `.env.example` to `.env`, paste your OpenAI API key, then restart the app. See README.md.")
    st.stop()

store, ingestor, agent, settings = svc.store, svc.ingestor, svc.agent, svc.settings

ss = st.session_state
ss.setdefault("messages", [])  # {"role": "user"|"assistant", "content": str, "answer": AgentAnswer | None}
ss.setdefault("uploader_key", 0)
ss.setdefault("pending", None)


# ----------------------------------------------------------------------------- helpers


def esc(text: str) -> str:
    """Escape markdown so document text renders literally ($ would otherwise start LaTeX)."""
    return re.sub(r"([\\`*_{}\[\]<>#|$~])", r"\\\1", text)


def esc_dollars(text: str) -> str:
    return text.replace("$", "\\$")


@st.cache_data(max_entries=64, show_spinner=False)
def render_page(pdf_path: str, page: int, highlight: str = "") -> bytes:
    """PNG of one page, with the cited passage highlighted (in memory; the file is not changed)."""
    return render_page_png(pdf_path, page, highlight)


def answer_markdown(question: str, ans: AgentAnswer) -> str:
    lines = [f"# Question\n\n{question}\n\n# Answer\n\n{ans.text}\n"]
    if ans.citations:
        lines.append("# Sources\n")
        for c in ans.citations:
            s = c.source
            lines.append(f"[{c.number}] {s.doc_title} ({s.filename}), {s.section}, {s.pages}\n\n> "
                         + s.text[:800].replace("\n", "\n> ") + "\n")
    return "\n".join(lines)


def show_answer(ans: AgentAnswer, key: str, question: str) -> None:
    if ans.error:
        st.error(ans.error)
    if ans.text:
        st.markdown(esc_dollars(ans.text))
    if ans.text and not ans.citations and not ans.error:
        st.caption("⚠️ This answer cites no document passages; treat it with care.")

    if ans.citations:
        st.markdown("**Sources**")
        for c in ans.citations:
            s = c.source
            with st.expander(f"[{c.number}] {s.doc_title} — {s.section} · {s.pages}"):
                st.caption(f"{s.filename}")
                st.markdown("> " + esc(s.text[:1500]).replace("\n", "\n> "))
                pdf = store.pdf_path(s.doc_id)
                if pdf.exists() and st.toggle("Show page", key=f"pg-{key}-{c.number}"):
                    page = s.page_start
                    if s.page_end > s.page_start:
                        page = st.select_slider("Page", list(range(s.page_start, s.page_end + 1)),
                                                key=f"ps-{key}-{c.number}")
                    st.image(render_page(str(pdf), page, s.text), caption=f"{s.filename}, page {page}")

    calls = len(ans.trace)
    with st.expander(f"Agent steps · {calls} tool call{'s' * (calls != 1)} · {ans.seconds:.1f}s · "
                     f"{ans.prompt_tokens + ans.completion_tokens:,} tokens"):
        if not ans.trace:
            st.write("The model answered without calling tools.")
        for i, t in enumerate(ans.trace, 1):
            try:
                args = json.dumps(json.loads(t.arguments or "{}"), ensure_ascii=False)
            except json.JSONDecodeError:
                args = t.arguments
            st.markdown(f"{i}. `{t.name}` `{esc(args)}` → {esc(t.summary)}")
        if ans.unknown_citations:
            st.caption(f"Dropped invalid citations: {', '.join(ans.unknown_citations)}")
    if ans.text:
        st.download_button("Download answer (.md)", answer_markdown(question, ans), file_name="answer.md",
                           mime="text/markdown", key=f"dl-{key}")


def ingest_files(files: list[tuple[str, bytes]]) -> None:
    with st.status(f"Ingesting {len(files)} file(s)...", expanded=True) as status:
        failed = 0
        for name, data in files:
            st.write(f"Reading **{name}**...")
            r = ingestor.ingest_bytes(data, name)
            icon = {"added": "✅", "duplicate": "↩️"}.get(r.status, "❌")
            st.write(f"{icon} {name}: {r.message}")
            for w in r.warnings:
                st.warning(w)
            failed += not r.ok
        status.update(label=f"Ingestion finished ({failed} failed)" if failed else "Ingestion finished",
                      state="error" if failed else "complete", expanded=bool(failed))


# ----------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("📚 Library")
    uploads = st.file_uploader("Add PDFs", type=["pdf"], accept_multiple_files=True, key=f"up-{ss.uploader_key}",
                               help=f"Up to {settings.max_upload_mb} MB and {settings.max_pages} pages per file.")
    if uploads and st.button("Ingest", type="primary", use_container_width=True):
        ingest_files([(u.name, u.getvalue()) for u in uploads])
        ss.uploader_key += 1
        st.rerun()

    docs = store.list_documents()
    if not docs:
        st.info("The library is empty. Upload PDFs, or load the three sample documents "
                "(a research paper, a product manual and a contract).")
    if st.button("Load sample documents", use_container_width=True,
                 disabled={"research_paper.pdf", "product_manual.pdf", "service_contract.pdf"}
                 <= {d.filename for d in docs}):
        paths = write_samples(PROJECT_ROOT / "samples")
        ingest_files([(p.name, p.read_bytes()) for p in paths])
        st.rerun()

    for d in docs:
        with st.container(border=True):
            st.markdown(f"**{esc(d.title)}**")
            st.caption(f"{d.filename} · {d.num_pages} pages · {len(d.sections)} sections · {d.num_chunks} chunks")
            for w in d.warnings:
                st.caption(f"⚠️ {w}")
            with st.popover("Remove", use_container_width=False):
                st.write(f"Remove **{esc(d.filename)}** from the library?")
                if st.button("Yes, remove", key=f"del-{d.id}", type="primary"):
                    store.delete(d.id)
                    st.rerun()

    if docs:
        st.divider()
        options = {d.id: d.filename for d in docs}
        scope = st.multiselect("Search in", options, format_func=options.get, placeholder="All documents",
                               help="Restrict questions to some documents. Empty = all.")
        if st.button("New conversation", use_container_width=True):
            ss.messages = []
            st.rerun()
    else:
        scope = []
    st.caption(f"Model: `{settings.chat_model}` · Embeddings: `{settings.embedding_model}`")

# ----------------------------------------------------------------------------- main

st.title("📄 PDF Q&A Agent")
view = st.segmented_control("View", ["Ask", "Documents"], default="Ask", label_visibility="collapsed") or "Ask"

if view == "Documents":
    if not docs:
        st.info("No documents yet.")
        st.stop()
    by_id = {d.id: d for d in docs}
    doc = by_id[st.selectbox("Document", by_id, format_func=lambda i: f"{by_id[i].title} ({by_id[i].filename})")]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pages", doc.num_pages)
    c2.metric("Sections", len(doc.sections))
    c3.metric("Chunks", doc.num_chunks)
    c4.metric("Structure from", {"outline": "Bookmarks", "layout": "Headings", "pages": "Pages"}[doc.structure_source])
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Section breakdown")
        for s in doc.sections:
            start, end = doc.page_range(s.id)
            indent = " " * (s.level - 1)
            with st.expander(f"{indent}{s.title} · {format_pages(start, end)}"):
                if s.text:
                    st.markdown(esc(s.text).replace("\n", "  \n"))
                else:
                    st.caption("No body text; see the subsections.")
    with right:
        st.subheader("Page viewer")
        page = st.number_input("Page", 1, doc.num_pages, 1)
        pdf = store.pdf_path(doc.id)
        if pdf.exists():
            st.image(render_page(str(pdf), int(page)))
    st.stop()

# Ask view --------------------------------------------------------------------------

if not ss.messages:
    if docs:
        st.markdown("Ask a question about your documents. Each answer cites the passages it is based on, "
                    "with document, section and page.")
        sample_names = {"research_paper.pdf", "product_manual.pdf", "service_contract.pdf"}
        if sample_names <= {d.filename for d in docs}:
            cols = st.columns(2)
            for i, q in enumerate(EXAMPLES):
                if cols[i % 2].button(q, key=f"ex-{i}", use_container_width=True):
                    ss.pending = q
                    st.rerun()
    else:
        st.markdown("Add PDFs in the sidebar to get started.")

for i, msg in enumerate(ss.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(esc_dollars(msg["content"]))
        else:
            show_answer(msg["answer"], f"m{i}", ss.messages[i - 1]["content"])

question = st.chat_input("Ask about your PDFs...", disabled=not docs) or ss.pending
ss.pending = None

if question:
    history = [
        (ss.messages[j]["content"], ss.messages[j + 1]["answer"].text)
        for j in range(0, len(ss.messages) - 1, 2)
        if ss.messages[j + 1]["answer"] and ss.messages[j + 1]["answer"].text
    ]
    ss.messages.append({"role": "user", "content": question, "answer": None})
    with st.chat_message("user"):
        st.markdown(esc_dollars(question))

    with st.chat_message("assistant"):
        status = st.status("Working...", expanded=False)
        placeholder = st.empty()
        streamed = ""
        answer: AgentAnswer | None = None
        labels = {
            "search_documents": "Searching", "read_section": "Reading section", "read_pages": "Reading pages",
            "get_document_outline": "Reading outline", "list_documents": "Listing documents",
        }
        for event in agent.run(question, history, scope or None):
            if event.kind == "text":
                streamed += event.text
                placeholder.markdown(esc_dollars(streamed) + "▌")
            elif event.kind == "reset":
                streamed = ""
                placeholder.empty()
            elif event.kind == "tool_start":
                status.update(label=f"{labels.get(event.tool, event.tool)}...")
                try:
                    args = json.loads(event.text or "{}")
                    detail = args.get("query") or args.get("section_id") or args.get("document_id") or ""
                except json.JSONDecodeError:
                    detail = ""
                status.write(f"**{labels.get(event.tool, event.tool)}** {esc(str(detail))}")
            elif event.kind == "tool_end":
                status.write(f"↳ {esc(event.text)}")
            elif event.kind == "final":
                answer = event.answer
        assert answer is not None
        status.update(label=f"Done · {len(answer.trace)} tool calls · {answer.seconds:.1f}s",
                      state="error" if answer.error else "complete")
        placeholder.empty()
        show_answer(answer, f"m{len(ss.messages)}", question)
    ss.messages.append({"role": "assistant", "content": answer.text, "answer": answer})
