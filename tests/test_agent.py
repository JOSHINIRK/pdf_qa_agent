import json

from pdfagent.agent import PDFAgent, render_citations
from pdfagent.llm import AssistantTurn
from pdfagent.tools import TOOL_SPECS, SourceRegistry, ToolExecutor

from .conftest import FakeChatModel, call


def doc_id(store, name):
    return next(d.id for d in store.list_documents() if d.filename == name)


def first_source_id(messages, tool_index=-1):
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    data = json.loads(tool_msgs[tool_index]["content"])
    items = data.get("passages") or data.get("parts") or data.get("pages")
    return items[0]["source_id"], items[0]


# ---------------------------------------------------------------- tools


def test_tool_specs_are_valid_openai_functions():
    names = {t["function"]["name"] for t in TOOL_SPECS}
    assert names == {"list_documents", "search_documents", "get_document_outline", "read_section", "read_pages"}
    for t in TOOL_SPECS:
        assert t["type"] == "function"
        assert t["function"]["parameters"]["type"] == "object"


def test_outline_and_read_section(library, settings):
    ex = ToolExecutor(library, settings)
    outline, _ = ex.execute(call("get_document_outline", json.dumps({"document_id": "product_manual.pdf"})))
    sections = json.loads(outline)["sections"]
    maint = next(s for s in sections if s["title"] == "5 Maintenance")
    result, summary = ex.execute(call("read_section", json.dumps({"document_id": "product_manual.pdf",
                                                                   "section_id": maint["section_id"]})))
    parts = json.loads(result)["parts"]
    assert [p["section"].split(" > ")[-1] for p in parts] == [
        "5.1 Filter Replacement Schedule", "5.2 Replacing a Cartridge", "5.3 Sanitizing"]
    assert all(p["source_id"] in ex.registry for p in parts)


def test_read_section_accepts_title(library, settings):
    ex = ToolExecutor(library, settings)
    result, _ = ex.execute(call("read_section", json.dumps({"document_id": "service_contract.pdf",
                                                             "section_id": "limitation of liability"})))
    assert "12 months preceding the claim" in result


def test_read_pages_is_capped(library, settings):
    ex = ToolExecutor(library, settings)
    result, _ = ex.execute(call("read_pages", json.dumps({"document_id": "research_paper.pdf",
                                                           "start_page": 1, "end_page": 99})))
    pages = json.loads(result)["pages"]
    assert [p["page"] for p in pages] == list(range(1, len(pages) + 1))


def test_tool_errors_are_returned_not_raised(library, settings):
    ex = ToolExecutor(library, settings)
    for c in (call("nope", "{}"), call("read_section", "{bad json"),
              call("get_document_outline", '{"document_id": "missing.pdf"}'),
              call("search_documents", '{"query": ""}'), call("read_pages", '{"document_id": "x"}')):
        result, _ = ex.execute(c)
        assert "error" in json.loads(result)


def test_scope_limits_tools(library, settings):
    manual = doc_id(library, "product_manual.pdf")
    ex = ToolExecutor(library, settings, scope=[manual])
    docs = json.loads(ex.execute(call("list_documents", "{}"))[0])["documents"]
    assert [d["document_id"] for d in docs] == [manual]
    passages = json.loads(ex.execute(call("search_documents", '{"query": "liability"}'))[0])["passages"]
    assert all(p["document_id"] == manual for p in passages)


def test_results_respect_token_budget(library, settings):
    from dataclasses import replace

    ex = ToolExecutor(library, replace(settings, max_tool_result_tokens=200))
    result, _ = ex.execute(call("read_pages", '{"document_id": "service_contract.pdf", "start_page": 1, "end_page": 2}'))
    assert any(p.get("truncated") for p in json.loads(result)["pages"])


# ---------------------------------------------------------------- citations


def test_render_citations_renumbers_and_drops_unknown(library):
    reg = SourceRegistry()
    doc = library.list_documents()[0]
    a = reg.register(doc, "A", 1, 1, "alpha")
    b = reg.register(doc, "B", 2, 2, "beta")
    text = f"First [{b.id}]. Second [{a.id}, {b.id}]. Bogus [S99]."
    rendered, cites, unknown = render_citations(text, reg)
    assert rendered == "First [1]. Second [2][1]. Bogus."
    assert [c.source.id for c in cites] == [b.id, a.id]
    assert unknown == ["S99"]


def test_registry_deduplicates(library):
    reg = SourceRegistry()
    doc = library.list_documents()[0]
    assert reg.register(doc, "A", 1, 1, "x").id == reg.register(doc, "A", 1, 1, "x").id


# ---------------------------------------------------------------- agent loop


def test_agent_searches_then_answers_with_citation(library, settings):
    def answer(messages):
        sid, passage = first_source_id(messages)
        assert "PF-1" in passage["text"]
        return AssistantTurn(f"Replace the PF-1 pre-filter every 6 months [{sid}].")

    llm = FakeChatModel([
        AssistantTurn("", [call("search_documents", '{"query": "PF-1 sediment pre-filter replacement interval"}')]),
        answer,
    ])
    events = list(PDFAgent(library, llm, settings).run("How often do I replace the pre-filter?"))
    kinds = [e.kind for e in events]
    assert kinds[0] == "tool_start" and kinds[-1] == "final"
    result = events[-1].answer
    assert result.error is None
    assert result.text == "Replace the PF-1 pre-filter every 6 months [1]."
    cite = result.citations[0].source
    assert cite.filename == "product_manual.pdf" and cite.page_start == 2
    assert result.trace[0].name == "search_documents" and result.model_calls == 2


def test_parallel_tool_calls_and_multi_document_answer(library, settings):
    def answer(messages):
        s1, _ = first_source_id(messages, -2)
        s2, _ = first_source_id(messages, -1)
        return AssistantTurn(f"Warranty: 2 years [{s1}]. Initial term: 36 months [{s2}].")

    llm = FakeChatModel([
        AssistantTurn("", [
            call("search_documents", '{"query": "AquaPure warranty years", "document_ids": ["product_manual.pdf"]}', "c1"),
            call("search_documents", '{"query": "initial term months", "document_ids": ["service_contract.pdf"]}', "c2"),
        ]),
        answer,
    ])
    result = PDFAgent(library, llm, settings).ask("Compare warranty and contract term")
    assert {c.source.filename for c in result.citations} == {"product_manual.pdf", "service_contract.pdf"}
    tool_msgs = [m for m in llm.requests[1] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]


def test_agent_stops_at_step_limit(library, settings):
    from dataclasses import replace

    looping = [AssistantTurn("", [call("list_documents", "{}", f"c{i}")]) for i in range(2)]
    llm = FakeChatModel(looping + [AssistantTurn("I could not verify this.")])
    result = PDFAgent(library, llm, replace(settings, agent_max_steps=3)).ask("loop forever")
    assert llm.tool_choices == ["auto", "auto", "none"]
    assert result.text == "I could not verify this." and not result.grounded


def test_text_streamed_before_tool_call_is_reset(library, settings):
    llm = FakeChatModel([
        AssistantTurn("Let me check.", [call("list_documents", "{}")]),
        AssistantTurn("Three documents."),
    ])
    kinds = [e.kind for e in PDFAgent(library, llm, settings).run("what is in the library?")]
    assert kinds.index("reset") < kinds.index("tool_start")


def test_history_is_sent_without_citation_markers(library, settings):
    llm = FakeChatModel([AssistantTurn("Sure.")])
    PDFAgent(library, llm, settings).ask("And the second one?", history=[("First?", "It is 6 months [1].")])
    msgs = llm.requests[0]
    assert msgs[0]["role"] == "system" and "product_manual.pdf" in msgs[0]["content"]
    assert msgs[2] == {"role": "assistant", "content": "It is 6 months ."}
    assert msgs[-1] == {"role": "user", "content": "And the second one?"}


def test_llm_errors_are_reported(library, settings):
    from pdfagent.llm import LLMError

    class Failing:
        def stream(self, *a, **k):
            raise LLMError("quota reached")
            yield  # pragma: no cover

    result = PDFAgent(library, Failing(), settings).ask("anything")
    assert result.error == "quota reached"
