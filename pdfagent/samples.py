"""Generates three fictional sample PDFs used for demos, tests and evaluation.

Each one exercises a different part of the parser:

- research_paper.pdf  - no bookmarks; headings found from font size + numbering ("3.1")
- product_manual.pdf  - PDF bookmarks, running header/footer on every page, tables
- service_contract.pdf - headings are bold numbered clauses in the body font size

All names, products, figures and organisations are invented.
"""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf

NOTICE = "Sample document with fictional content, generated for testing the PDF Q&amp;A agent."

# Each spec is a list of ("h", level, text) headings and ("p", html) body items.

PAPER = [
    ("title", "Adaptive Sparse Attention for Long-Document Question Answering"),
    ("p", "<i>R. Alvarez, N. Okonkwo-Hale and T. Lindqvist</i><br/>Halden Institute for Language Systems (fictional)"),
    ("p", f"<i>{NOTICE}</i>"),
    ("h", 1, "Abstract"),
    ("p", "Transformer readers struggle with documents longer than a few thousand tokens because full self-attention "
          "grows quadratically with input length. We introduce <b>Adaptive Sparse Attention (ASA)</b>, which scores "
          "fixed-size segments of a document for relevance to the question and lets only the most relevant segments "
          "attend to each other in full. On the LongDocQA benchmark, ASA improves exact match by 2.4 points over a "
          "full-attention baseline while reducing peak GPU memory by 63% on 32k-token inputs."),
    ("h", 1, "1 Introduction"),
    ("p", "Question answering over long documents such as contracts, manuals and research articles requires a model to "
          "locate a few relevant passages among tens of thousands of tokens. Existing approaches either truncate the "
          "input, use a sliding window that loses long-range context, or apply a fixed sparsity pattern that ignores "
          "the question. Our key observation is that relevance to the question is highly concentrated: in LongDocQA, "
          "91% of gold answers fall within the 8 most question-relevant segments of 256 tokens."),
    ("p", "This paper makes three contributions: (1) a lightweight segment scorer that adds less than 2% parameters, "
          "(2) a budget controller that adapts the number of attended segments per layer, and (3) an evaluation on "
          "three long-document benchmarks with an ablation of each component."),
    ("h", 1, "2 Related Work"),
    ("p", "Efficient attention methods reduce the cost of self-attention with fixed local windows, global tokens or "
          "low-rank projections. Retrieval-augmented readers instead select passages before reading them, but the "
          "selection is not trained jointly with the reader. ASA sits between the two families: segment selection is "
          "learned end-to-end and conditioned on the question."),
    ("h", 1, "3 Method"),
    ("p", "ASA splits the input into segments and replaces each full attention layer with a two-stage procedure: "
          "segment scoring followed by sparse attention over the selected segments."),
    ("h", 2, "3.1 Segment Scoring"),
    ("p", "The document is split into segments of 256 tokens. A two-layer scorer reads the mean-pooled segment "
          "representation together with the question representation and outputs a relevance score. The top-r "
          "segments, with r = 8 by default, receive full attention; all other segments attend only to their own "
          "tokens and to the question."),
    ("h", 2, "3.2 Budget Controller"),
    ("p", "A fixed budget wastes computation on easy questions and starves hard ones. The budget controller measures the "
          "entropy of the score distribution at each layer. When the entropy exceeds the threshold tau = 0.35, the "
          "budget r is doubled for that layer, up to a maximum of 32 segments."),
    ("h", 2, "3.3 Training Objective"),
    ("p", "The reader is trained with the standard span-extraction loss plus an auxiliary distillation loss that aligns "
          "segment scores with the attention mass of a full-attention teacher. The auxiliary loss weight is "
          "lambda = 0.1. Training runs for 40,000 steps with a batch size of 64 on 8 GPUs."),
    ("h", 1, "4 Experiments"),
    ("h", 2, "4.1 Datasets"),
    ("p", "We evaluate on three benchmarks. <b>LongDocQA</b> contains 12,400 questions over 3,100 documents with an "
          "average length of 21,000 tokens. <b>PolicyQA-Long</b> contains 4,800 questions about insurance and privacy "
          "policies. <b>ManualBench</b> contains 2,950 questions about technical product manuals. All three datasets "
          "are described in detail in Appendix A of the full report."),
    ("h", 2, "4.2 Baselines"),
    ("p", "We compare ASA with (a) full attention over the whole document, (b) a sliding window of 4,096 tokens and "
          "(c) a fixed block-sparse pattern with global tokens. All models share the same 350M-parameter backbone."),
    ("h", 2, "4.3 Results"),
    ("p", "Table 1 reports results on LongDocQA with 32k-token inputs. ASA achieves the best exact match (EM) and F1 "
          "while using 14.3 GB of peak memory, compared with 38.5 GB for full attention."),
    ("table", ["Model", "EM", "F1", "Memory (GB)", "Latency (ms)"],
              [["Full attention", "41.2", "55.8", "38.5", "912"],
               ["Sliding window", "36.9", "50.1", "9.7", "301"],
               ["Fixed sparse", "39.0", "52.6", "14.2", "388"],
               ["ASA (ours)", "43.6", "58.3", "14.3", "402"]]),
    ("p", "<i>Table 1: LongDocQA test results with 32k-token inputs.</i>"),
    ("p", "On PolicyQA-Long and ManualBench, ASA improves EM by 1.8 and 3.1 points respectively over full attention."),
    ("h", 2, "4.4 Ablation Study"),
    ("p", "Removing the budget controller reduces EM on LongDocQA by 1.7 points, and removing the distillation loss "
          "reduces it by 0.9 points. Using 128-token segments instead of 256 gives similar accuracy but increases "
          "latency by 18%."),
    ("h", 1, "5 Limitations"),
    ("p", "The segment scorer was trained only on English text, and its behaviour on other languages is untested. Fixed "
          "segment boundaries can split an answer across two segments. ASA ignores tables and figures, which are "
          "removed during preprocessing. Finally, we evaluated inputs of up to 64k tokens only."),
    ("h", 1, "6 Conclusion"),
    ("p", "ASA shows that question-conditioned sparsity can match or exceed full attention on long documents at a "
          "fraction of the memory cost. Future work will extend the scorer to multilingual inputs and to tables."),
    ("h", 1, "References"),
    ("p", "[1] P. Marsh and K. Ito. Windowed attention for long sequences. Fictional Journal of NLP, 2024.<br/>"
          "[2] D. Feld. Learning to retrieve for reading. Proceedings of the Imaginary Conference, 2025."),
]

MANUAL_HEADER = "AquaPure 300 Under-Sink Filtration System - User Manual"
MANUAL = [
    ("title", "AquaPure 300 Under-Sink Water Filtration System"),
    ("p", "<b>Installation and User Manual - Model AP-300</b><br/>Clearbrook Home Systems (fictional company)"),
    ("p", f"<i>{NOTICE}</i>"),
    ("h", 1, "1 Safety Information"),
    ("p", "Read all instructions before installing the system. Connect the AquaPure 300 to the <b>cold water supply "
          "only</b>; water hotter than 38 °C damages the filter membranes. The system operates at a water pressure "
          "between 1.4 and 6.2 bar (20-90 psi). If your supply pressure exceeds 6.2 bar, install a pressure-reducing "
          "valve before the unit. Do not use the system with water that is microbiologically unsafe or of unknown "
          "quality without adequate disinfection before or after the system."),
    ("h", 1, "2 Package Contents"),
    ("p", "The box contains: the filter head assembly with three housings; one PF-1 sediment pre-filter, one CB-2 "
          "carbon block filter and one UF-3 ultrafiltration membrane; a chrome faucet; 3 m of 1/4-inch tubing; a "
          "feed-water adapter; a leak sensor; a housing wrench; and mounting screws."),
    ("h", 1, "3 Installation"),
    ("h", 2, "3.1 Tools Required"),
    ("p", "You need an adjustable wrench, a drill with a 12 mm bit (only if the sink has no spare faucet hole), a "
          "screwdriver, a tape measure and a bucket."),
    ("h", 2, "3.2 Mounting the Unit"),
    ("p", "Mount the filter head on the cabinet wall with the supplied screws. Leave at least 5 cm of clearance below "
          "the housings so that cartridges can be removed. Place the leak sensor on the cabinet floor directly "
          "beneath the unit."),
    ("h", 2, "3.3 Connecting the Water Supply"),
    ("p", "Turn off the cold water supply. Install the feed-water adapter on the cold water valve, connect the tubing "
          "to the IN port of the filter head and push it in until it seats firmly (about 15 mm). Connect the OUT port "
          "to the faucet. Turn the water supply back on slowly and check every fitting for leaks."),
    ("h", 2, "3.4 Flushing New Filters"),
    ("p", "New cartridges contain carbon fines. Open the faucet and let water run for <b>10 minutes</b> (about 20 "
          "litres) before drinking. Some cloudiness during the first day is normal and is caused by trapped air."),
    ("h", 1, "4 Operation"),
    ("h", 2, "4.1 Indicator Lights"),
    ("table", ["Light", "Meaning", "Action"],
              [["Green, steady", "Normal operation", "None"],
               ["Amber, steady", "A filter is due within 30 days", "Order a replacement cartridge"],
               ["Red, flashing", "Filter expired or leak detected", "Close the supply valve and check for leaks"],
               ["Red, steady", "Sensor fault", "Contact customer support"]]),
    ("h", 2, "4.2 Daily Use"),
    ("p", "Open the faucet lever to draw filtered water. If the system has not been used for more than 7 days, run "
          "water for 2 minutes before drinking."),
    ("h", 1, "5 Maintenance"),
    ("h", 2, "5.1 Filter Replacement Schedule"),
    ("table", ["Stage", "Cartridge", "Replace every"],
              [["1", "PF-1 sediment pre-filter", "6 months"],
               ["2", "CB-2 carbon block", "12 months"],
               ["3", "UF-3 ultrafiltration membrane", "24 months"]]),
    ("p", "Replace cartridges sooner if the flow rate drops noticeably or the water tastes different."),
    ("h", 2, "5.2 Replacing a Cartridge"),
    ("p", "Close the supply valve and open the faucet to release pressure. Unscrew the housing with the housing wrench, "
          "remove the old cartridge, insert the new one and hand-tighten the housing. Reopen the supply valve, flush "
          "as described in section 3.4, then reset the indicator by holding the RESET button for <b>5 seconds</b> "
          "until the light turns green."),
    ("h", 2, "5.3 Sanitizing"),
    ("p", "Sanitize the housings every 12 months, when you replace the CB-2 cartridge, using a food-grade sanitizer."),
    ("h", 1, "6 Troubleshooting"),
    ("table", ["Problem", "Likely cause", "Solution"],
              [["Low flow", "Clogged PF-1 pre-filter", "Replace the PF-1 cartridge"],
               ["Leak at a fitting", "Tubing not fully inserted", "Push tubing in about 15 mm and retest"],
               ["Bad taste or odour", "New or expired carbon filter", "Flush for 10 minutes or replace CB-2"],
               ["Light flashes red", "Leak detected or filter expired", "Dry the sensor, fix the leak, reset"]]),
    ("h", 1, "7 Technical Specifications"),
    ("table", ["Specification", "Value"],
              [["Flow rate", "2.8 L/min at 4 bar"],
               ["Operating pressure", "1.4-6.2 bar (20-90 psi)"],
               ["Water temperature", "4-38 °C"],
               ["Dimensions (W x D x H)", "38 x 12 x 40 cm"],
               ["Weight (empty)", "4.6 kg"]]),
    ("h", 1, "8 Warranty"),
    ("p", "Clearbrook Home Systems warrants the AquaPure 300 against defects in materials and workmanship for "
          "<b>2 years</b> from the date of purchase. Registering the product within 60 days of purchase extends the "
          "warranty to 3 years. Filter cartridges are consumables and are not covered. The warranty is void if the "
          "system was connected to hot water or operated above 6.2 bar."),
]

CONTRACT = [
    ("title", "Master Software Services Agreement"),
    ("p", "This Agreement is made on <b>1 March 2026</b> (the \"Effective Date\") between <b>Kestrel Ridge Software "
          "Inc.</b> (the \"Supplier\") and <b>Bramblecote Analytics Ltd</b> (the \"Customer\"). Both parties are "
          "fictional."),
    ("p", f"<i>{NOTICE}</i>"),
    ("clause", "1. DEFINITIONS"),
    ("p", "1.1 \"Confidential Information\" means any information disclosed by one party to the other that is marked "
          "confidential or would reasonably be considered confidential."),
    ("p", "1.2 \"Services\" means the hosted analytics platform and the support services described in clause 2 and "
          "clause 4."),
    ("p", "1.3 \"Severity 1 Incident\" means a complete outage of the Services or a defect that prevents the Customer "
          "from using any material function of the Services."),
    ("clause", "2. SERVICES"),
    ("p", "2.1 The Supplier shall provide the Services in accordance with this Agreement and with good industry "
          "practice."),
    ("p", "2.2 The Supplier shall ensure that the Services are available at least 99.9% of each calendar month, "
          "excluding scheduled maintenance announced at least 5 business days in advance."),
    ("clause", "3. FEES AND PAYMENT"),
    ("p", "3.1 The Customer shall pay an annual subscription fee of <b>GBP 84,000</b>, invoiced quarterly in advance."),
    ("p", "3.2 Invoices are payable within 30 days of receipt. Late payments bear interest at 4% per year above the "
          "Bank of England base rate."),
    ("p", "3.3 The Supplier may increase the fees on each renewal by no more than 5%, by giving at least 90 days' "
          "written notice."),
    ("clause", "4. SERVICE LEVELS AND SUPPORT"),
    ("p", "4.1 The Supplier shall respond to incidents within the following times: Severity 1 Incidents within "
          "<b>1 hour</b>, 24 hours a day, 7 days a week; Severity 2 incidents within 4 business hours; and Severity "
          "3 incidents within 2 business days."),
    ("p", "4.2 If availability in a month falls below the level in clause 2.2, the Customer is entitled to a service "
          "credit of 5% of the monthly fee for each 0.5% (or part) below that level, up to a maximum of 25% of the "
          "monthly fee. Service credits are the Customer's sole remedy for availability failures."),
    ("clause", "5. DATA PROTECTION"),
    ("p", "5.1 The Supplier processes personal data only as a processor on the documented instructions of the "
          "Customer, and stores all Customer data in data centres located in the United Kingdom or the European "
          "Union."),
    ("p", "5.2 The Supplier shall notify the Customer of any personal data breach without undue delay and in any "
          "case within <b>48 hours</b> of becoming aware of it."),
    ("clause", "6. CONFIDENTIALITY"),
    ("p", "6.1 Each party shall keep the other party's Confidential Information confidential and use it only to "
          "perform this Agreement. This obligation survives for 5 years after termination of this Agreement."),
    ("clause", "7. INTELLECTUAL PROPERTY"),
    ("p", "7.1 The Supplier retains all intellectual property rights in the Services. The Customer retains all rights "
          "in its data and in reports it creates using the Services."),
    ("clause", "8. TERM AND TERMINATION"),
    ("p", "8.1 This Agreement starts on the Effective Date and continues for an initial term of <b>36 months</b>. It "
          "then renews automatically for successive periods of 12 months unless either party gives at least 90 "
          "days' written notice before the end of the current term."),
    ("p", "8.2 After the first 12 months, the Customer may terminate this Agreement for convenience by giving "
          "<b>60 days' written notice</b> and paying an early termination fee equal to 3 months of fees."),
    ("p", "8.3 Either party may terminate this Agreement immediately by written notice if the other party commits a "
          "material breach that is not remedied within 30 days of being notified of it."),
    ("p", "8.4 For 30 days after termination the Supplier shall allow the Customer to export its data in a common "
          "machine-readable format, after which the Supplier shall delete it."),
    ("clause", "9. LIMITATION OF LIABILITY"),
    ("p", "9.1 Nothing in this Agreement limits liability for death or personal injury caused by negligence, or for "
          "fraud."),
    ("p", "9.2 Subject to clause 9.1, each party's total liability under this Agreement in any 12-month period is "
          "limited to the total fees paid by the Customer in the <b>12 months preceding the claim</b>. Liability for "
          "breach of clause 5 (Data Protection) is limited to twice the annual subscription fee."),
    ("p", "9.3 Neither party is liable for indirect or consequential loss, or for loss of profit, revenue or "
          "goodwill."),
    ("clause", "10. INSURANCE"),
    ("p", "10.1 The Supplier shall maintain professional indemnity insurance of at least GBP 5 million for the term "
          "of this Agreement and for 2 years afterwards."),
    ("clause", "11. GENERAL"),
    ("p", "11.1 Neither party may assign this Agreement without the prior written consent of the other party, which "
          "shall not be unreasonably withheld."),
    ("p", "11.2 Notices must be in writing and delivered by hand, by recorded post or by email to the addresses set "
          "out in the order form."),
    ("p", "11.3 This Agreement is the entire agreement between the parties and supersedes all earlier discussions."),
    ("clause", "12. GOVERNING LAW AND JURISDICTION"),
    ("p", "12.1 This Agreement is governed by the laws of <b>England and Wales</b>. The courts of London have "
          "exclusive jurisdiction over any dispute arising out of it."),
    ("p", "Signed for and on behalf of the Supplier and the Customer by their authorised representatives."),
]

_CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.3; }
p { margin-top: 0; margin-bottom: 6pt; }
.title { font-size: 20pt; font-weight: bold; margin-bottom: 8pt; }
h1 { font-size: 14pt; font-weight: bold; margin-top: 10pt; margin-bottom: 4pt; }
h2 { font-size: 12pt; font-weight: bold; margin-top: 8pt; margin-bottom: 3pt; }
table { border-collapse: collapse; margin-bottom: 8pt; }
td, th { border: 1px solid #888; padding: 2pt 4pt; font-size: 9pt; }
th { font-weight: bold; text-align: left; }
"""


def _html(spec: list) -> str:
    out = []
    for item in spec:
        kind = item[0]
        if kind == "title":
            out.append(f'<p class="title">{item[1]}</p>')
        elif kind == "h":
            out.append(f"<h{item[1]}>{item[2]}</h{item[1]}>")
        elif kind == "clause":
            # Contract clause headings: bold, same size as the body text.
            out.append(f"<p><b>{item[1]}</b></p>")
        elif kind == "p":
            out.append(f"<p>{item[1]}</p>")
        elif kind == "table":
            head = "".join(f"<th>{c}</th>" for c in item[1])
            rows = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in item[2])
            out.append(f"<table><tr>{head}</tr>{rows}</table>")
    return "\n".join(out)


def headings(spec: list) -> list[tuple[int, str]]:
    """Expected (level, title) headings of a spec, used by tests and for bookmarks."""
    out = []
    for item in spec:
        if item[0] == "h":
            out.append((item[1], item[2]))
        elif item[0] == "clause":
            out.append((1, item[1]))
    return out


def _render(spec: list, title: str, header: str | None = None, bookmarks: bool = False) -> bytes:
    page_rect = pymupdf.paper_rect("a4")
    content = pymupdf.Rect(64, 72, page_rect.width - 64, page_rect.height - 72)
    story = pymupdf.Story(_html(spec), user_css=_CSS)
    buf = io.BytesIO()
    writer = pymupdf.DocumentWriter(buf)
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content)
        story.draw(device)
        writer.end_page()
    writer.close()

    doc = pymupdf.open(stream=buf.getvalue(), filetype="pdf")
    n = doc.page_count
    for i, page in enumerate(doc, start=1):
        if header:
            page.insert_text((64, 40), header, fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))
        page.insert_text((page_rect.width / 2 - 20, page_rect.height - 36), f"Page {i} of {n}", fontsize=8,
                         fontname="helv", color=(0.4, 0.4, 0.4))
    if bookmarks:
        toc = []
        page_lines = [(p.number + 1, p.get_text("text").splitlines()) for p in doc]
        for level, text in headings(spec):
            page_no = next((no for no, lines in page_lines if text in (ln.strip() for ln in lines)), 1)
            toc.append([level, text, page_no])
        doc.set_toc(toc)
    doc.set_metadata({"title": title, "author": "PDF Q&A agent sample generator", "subject": "Fictional sample"})
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return data


SAMPLES = {
    "research_paper.pdf": lambda: _render(PAPER, "Adaptive Sparse Attention for Long-Document Question Answering"),
    "product_manual.pdf": lambda: _render(MANUAL, "AquaPure 300 User Manual", header=MANUAL_HEADER, bookmarks=True),
    "service_contract.pdf": lambda: _render(CONTRACT, "Master Software Services Agreement"),
}


def write_samples(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, build in SAMPLES.items():
        path = out / name
        path.write_bytes(build())
        paths.append(path)
    return paths
