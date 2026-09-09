# Fact Knowledge Layer: Cross-Document Verification & Reconciliation Engine

**Superjoin VIT 2026 · Engineering Intern Hiring Assignment**  
GitHub Repository: https://github.com/meerpi/SuperJoin_assignment

---

## Introduction

The problem is straightforward — important facts live across multiple PDFs, stated differently, and sometimes they disagree. This system reads corporate PDFs, pulls out numerical and semantic facts, grounds every fact to an exact quote and page in the source document, and then figures out which facts agree, which contradict, and which only *look* like contradictions because of context (different time periods, different scopes, etc).

The starter dataset is three Delhivery corporate filings:
- `01-delhivery-prospectus-2022-excerpt.pdf` (IPO Prospectus 2022)
- `02-delhivery-annual-report-fy24-excerpt.pdf` (Annual Report FY24)
- `03-delhivery-q4-fy24-earnings-presentation.pdf` (Q4 FY24 Earnings Deck)

The system isn't hardcoded to these documents. You can upload new PDFs through the UI or the API and it processes them dynamically using the same open pipeline.

---

## 1. Setup and Run Instructions

### Prerequisites
- **Python 3.10+**
- **Node.js 18+ and npm**
- **Google Gemini API Key** (Free tier from [Google AI Studio](https://aistudio.google.com/apikey))
- *(Optional)* **Voyage AI API Key** for `voyage-finance-2` financial embeddings. If omitted, the engine uses local caching and fast cosine fallback.

### Install

```bash
# 1. Clone repository
git clone git@github.com:meerpi/SuperJoin_assignment.git
cd SuperJoin_assignment

# 2. Set up Python environment & dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY

# 4. Install frontend packages
cd frontend
npm install
cd ..
```

### Run the System

**1. Start the FastAPI Backend (Port 8000):**
```bash
.venv/bin/uvicorn app.server:app --reload --host 127.0.0.1 --port 8000
```
- Interactive Swagger UI: http://localhost:8000/docs
- Seed Demo Endpoint: `POST /system/seed-demo`

**2. Start the React + Vite Frontend (Port 5173):**
```bash
cd frontend
npm run dev
```
- Web Application: http://localhost:5173
- Open this URL in your browser. Click **"Seed Demo Dataset"** to load the pre-extracted Delhivery dataset in 1 second without waiting for live PDF ingestion.

### Running Tests
The suite includes **153 unit and integration tests**:
```bash
PYTHONPATH=. .venv/bin/pytest -v
```

---

## 2. Video Demo

- **Demo Video Link:** `[Insert YouTube / Loom / Google Drive link here]` *(Strictly $\le$ 3 minutes)*
- **Demo Video Script:** Formatted in [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md).
- **What the Video Shows:**
  1. A PDF being uploaded and processed through the layout parser.
  2. All four required assignment cases demonstrated live across the Delhivery corpus.
  3. Interactive side-by-side PDF canvas with bounding-box highlights on source exhibits.

---

## 3. Approach

### System Architecture

![System Architecture](architecture.jpg)

The pipeline processes documents in five sequential stages:

**Stage 1 — Dual Parsing.** Two parsers run together on every PDF:
- **PyMuPDF:** Extracts every word with its pixel coordinates at native C-speed. This gives us the ground-truth text index we use later for quote verification and bounding box highlighting in the UI.
- **Docling TableFormer (IBM Research):** Reconstructs complex financial tables with merged cells, multi-level headers, and scope propagation. 

We need both because PyMuPDF is fast but doesn't understand table structures, while Docling understands table hierarchies but doesn't give us word-level coordinates.

![Document Upload and Ingestion](docs/screenshots/upload_screen.png)

**Stage 2 — LLM Extraction.** Gemini Flash reads text chunks and outputs structured 6-tuples: `(subject, predicate, value, unit, temporal, quote)`. We use the iXBRL standard for numbers — `value × 10^scale` — so "₹48,105.30 million" becomes `numeric_value = 48105.3, scale = 6`. The extractor uses a model fallback cascade (`gemini-3.6-flash`, `gemini-3.5-flash-lite`, etc.) with retries if rate limits hit.

**Stage 3 — Quote Grounding.** This is where we verify the LLM didn't hallucinate. Every quote returned by the model gets matched against the raw PDF text layer via a 3-tier check:
1. Exact substring match
2. Whitespace-normalized match
3. Fuzzy match (Levenshtein ratio $> 0.85$)

If a quote fails all three, the fact's confidence drops to `0.10`. The LLM proposes; deterministic code verifies.

![Fact Evidence Grounding and PDF Bounding Box](docs/screenshots/fact_evidence_modal.png)

**Stage 4 — Normalization.** Scale, unit, and polarity canonicalization:
- Multipliers: "₹ in millions" $\to 10^6$, "in Crores" $\to 10^7$, "in Lakhs" $\to 10^5$.
- Accounting negatives: parenthesized figures like `(0.72)%` become negative numbers (`-0.72 %`).
- Predicates: Standardized into snake_case canonical keys so "automated sort centres" and "sortation network" map to the same concept.

![Accounting Negative and Unit Normalization](docs/screenshots/accounting_negative_normalization.png)

**Stage 5 — Claim Graph.** Cross-document reconciliation using an ArbGraph-style architecture:
- Facts across documents are clustered by `(subject, predicate)` similarity using Voyage AI's `voyage-finance-2` embeddings.
- **Strict boundary:** Embeddings are used *strictly* for concept alignment, *never* for numeric or polarity comparison.
- Within each cluster, deterministic symbolic math checks relative tolerance ($|A - B| / \max(|A|, |B|) < 10^{-4}$), a temporal parser checks if different dates explain the difference, and a DeBERTa cross-encoder evaluates qualitative claims.

### Where Embeddings Are and Aren't Used

| Component | Uses Embeddings? | Why |
|---|:---:|---|
| **Concept Alignment** (is "sort centres" the same metric as "sortation network"?) | **Yes** | **Voyage-finance-2**: Dynamically aligns semantic terminology without brittle, hardcoded synonym dictionaries. |
| **Numeric Comparison** (does 21 equal 21?) | **No** | **Symbolic Math**: Cosine similarity says $50M and $80M revenue are 98% similar. Math must be deterministic: `|A - B| / max(|A|, |B|) < 1e-4`. |
| **Qualitative Conflict** (is "active" contradicted by "resigned"?) | **No** | **DeBERTa-v3 Cross-Encoder NLI**: Full joint cross-attention for entailment/contradiction with zero API cost. |
| **Source Quote Verification** (did the LLM invent a quote?) | **No** | **Deterministic 3-Tier String Matcher**: Verifies quotes directly against the PDF text layer. |

### Fact Representation & Storage

Each fact is modeled as an atomic 6-tuple:
```python
(subject, predicate, value, context, confidence, provenance)
```
- **Context:** Temporal (`FY2024`, `March 31, 2023`), Scope (`Consolidated`, `Standalone`), Conditions (`Restated`, `Adjusted`).
- **Provenance:** Document ID, source page, verbatim quote, bounding box coordinates, and verification match status.
- **Storage:** Built on SQLite with Write-Ahead Logging (WAL). Zero external databases, zero Docker setup. Everything persists across server restarts, and WAL mode allows concurrent reads while background extraction writes new facts.

### Important Decisions and Trade-offs

1. **Dual Parsing over a Single Tool:** PyMuPDF is 10x faster than Docling, but cannot reconstruct merged financial tables. Docling reconstructs tables accurately, but is compute-heavy. Running PyMuPDF for page-level coordinates and text indexing combined with Docling for table structure was the right trade-off.
2. **Decoupled Verification over Pure RAG:** Standard RAG pipelines query vector stores and ask LLMs "do these disagree?". That approach is slow, expensive, and fails on subtle numbers. Decoupling semantic discovery (embeddings) from verification (symbolic math + NLI) makes arbitration fast, deterministic, and free of mathematical hallucinations.
3. **iXBRL Unit/Scale Model:** Modeling values as `value × 10^scale` allows the system to compare ₹40,000 million against ₹4,000 crores seamlessly without loss of precision.

### AI Tools Used

- **Google Gemini (3.6 Flash / 3.5 Flash Lite):** Fact extraction with structured JSON output and quote citation.
- **Docling TableFormer (IBM Research):** Deep learning vision-based table layout reconstruction.
- **Voyage AI (`voyage-finance-2`):** Financial domain embedding model for dynamic predicate alignment.
- **DeBERTa-v3 (`cross-encoder/nli-deberta-v3-base`):** Local cross-encoder for natural language inference on qualitative facts.

---

## 4. The Four Required Cases

Here are the concrete answers for all four assignment cases on the Delhivery corporate dataset.

![Cross-Document Reconciliation and Inline PDF Exhibit Viewer](docs/screenshots/reconciliation_inline_viewer.png)

### Case 1: Corroborated Fact Across Documents (Even If Expressed Differently)

- **Entity / Subject:** `Delhivery`
- **Metric / Predicate:** `automated_sort_centers` (Count of automated sortation hubs)
- **Canonical Value:** `21.0` (Dimension: Count, Scale: 0)

**Source Evidence in Documents:**
1. `01-delhivery-prospectus-2022-excerpt.pdf`, Page 49:
   - *Verbatim quote:* `"21 automated sort centres"`
   - *Reported value:* `21`
2. `01-delhivery-prospectus-2022-excerpt.pdf`, Page 42:
   - *Verbatim quote:* `"We operated 21 fully and semi-automated sortation centres"`
   - *Reported value:* `21`
3. `03-delhivery-q4-fy24-earnings-presentation.pdf`, Page 7:
   - *Verbatim quote:* `"21"` (in sortation network expansion timeline chart)
   - *Reported value:* `21`

**Why It Occurred & System Reasoning:**
The filings use three different phrasings across narrative prose, bullet points, and an earnings deck timeline. The semantic normalizer and Voyage embedding alignment recognize that "automated sort centres", "fully and semi-automated sortation centres", and the sortation timeline refer to the same operational metric. The symbolic comparator compares the canonical values ($|21.0 - 21.0| = 0.00$), and the claim graph links them with a `corroborates` edge (`relative_diff = 0.00`, `confidence = 1.00`).

---

### Case 2: A Genuine or Likely Contradiction

- **Entity / Subject:** `Delhivery network`
- **Metric / Predicate:** `last_mile_distribution_centres_count`
- **Dispute Classification:** `DISPUTE_GENUINE_CONFLICT`

**Source Evidence in Documents:**
1. `02-delhivery-annual-report-fy24-excerpt.pdf`, Page 12 (Executive Highlights):
   - *Verbatim quote:* `"4,445 last mile distribution centres"`
   - *Claimed value:* `4,445`
2. `02-delhivery-annual-report-fy24-excerpt.pdf`, Page 46 (Audited Operational Review):
   - *Verbatim quote:* `"3,506 delivery centres"` + *"partner centres: 85"*
   - *Summed value:* `3,591` ($3,506 + 85$)
3. `01-delhivery-prospectus-2022-excerpt.pdf`, Page 51 (Earlier baseline):
   - *Verbatim quote:* `"As of December 31, 2021, we operated 3,730 delivery centres"`
   - *Claimed value:* `3,730`

**Why It Occurred & System Reasoning:**
This is an irreconcilable discrepancy within the *exact same* FY24 Annual Report. Marketing highlights on Page 12 claim 4,445 centres to demonstrate reach, but the operational disclosure on Page 46 reports only 3,506 company-operated centres and 85 partner centres (total 3,591). Both refer to the same fiscal year (FY24), with the same unit (centres), so it's not a temporal difference or a scale mismatch. The symbolic engine evaluates $|4445 - 3591| / 4445 = 19.2\%$ discrepancy, flagging a genuine contradiction. In the UI, clicking *"Inspect"* renders both pages side by side with yellow and red bounding boxes around the numbers.

---

### Case 3: Apparent Contradiction Reconciled by Context (Time, Scope, or Units)

- **Entity / Subject:** `Delhivery Limited`
- **Metric / Predicate:** `disputed_trade_receivables_credit_impaired_total`
- **Dispute Classification:** `DISPUTE_TEMPORAL_DRIFT` (Reconciled by Fiscal Period)

**Source Evidence in Documents:**
1. `02-delhivery-annual-report-fy24-excerpt.pdf`, Page 80:
   - *Reported value:* `₹14,296.90 Million`
   - *Context:* `As at March 31, 2024`
2. `02-delhivery-annual-report-fy24-excerpt.pdf`, Page 80 (Comparative Column):
   - *Reported value:* `₹15,238.07 Million`
   - *Context:* `As at March 31, 2023`
3. `01-delhivery-prospectus-2022-excerpt.pdf`, Page 24:
   - *Reported value:* `₹7,738.59 Million`
   - *Context:* `As at March 31, 2021`

**Why It Occurred & System Reasoning:**
A naive diff would flag `14,296.90` vs `15,238.07` vs `7,738.59` as three conflicting assertions about the company's receivables. However, the temporal normalizer extracts the balance sheet audit cut-off dates (`March 31, 2024`, `March 31, 2023`, `March 31, 2021`). Recognizing that these represent accounting progression across different fiscal year ends, the arbitration engine creates a directed `supersedes (temporal progression)` edge rather than a contradiction. The cluster is classified as Reconciled with zero false alarms.

---

### Case 4: Extraction or Reasoning Failure Found & How We Handled It

- **Failure Type:** `scale_mismatch` / `table_header_scope_loss`
- **Observed Failure:**
  In financial balance sheets (e.g. Page 22 of the Prospectus), the unit `"(₹ in millions)"` is printed only once in the top header. When chunk-level LLM extraction parses an isolated row (like CA Swift's acquisition cost `139.88`), the LLM sometimes misses the header and extracts it with `scale = 0` (treating it as ₹139.88 instead of ₹139.88 Million), creating a million-fold distortion.

- **How It Was Handled & Improved:**
  1. **Docling TableFormer Grid:** Docling preserves table topologies, propagating header scopes and units down into cell-level text blocks so the LLM sees `(₹ in millions)` alongside each row.
  2. **Symbolic Order-of-Magnitude Validator:** A post-extraction rule checks monetary values against expected corporate dimensions. If an enterprise metric like total revenue or balance sheet line items is extracted as $< 1,000$, it automatically gets tagged with `scale_mismatch`.
  3. **Confidence Demotion:** Ambiguous or scale-mismatched facts have their confidence score dropped to `0.10`. They stay visible in the Facts Ledger for human audit, but are excluded from creating false contradiction edges in the reconciliation graph.

---

## 5. Limitations and Next Steps

Being completely honest about what's rough and what needs work:

1. **Ingest and parsing speed (it's slow).** Running full ingestion from scratch on the three starter PDFs (each around 80–100+ pages) takes almost 15+ minutes. We intentionally sacrificed speed for accuracy:
   - **Docling TableFormer:** Reconstructing multi-level financial tables, merged cells, and headers with deep learning is compute-heavy, especially on local CPU. It does the job accurately, but it crawls compared to a basic text dump.
   - **Gemini Free-Tier Rate Limits:** The public free tier has strict requests-per-minute (RPM) and tokens-per-minute (TPM) limits. We had to implement chunking, retries, and backoff sleep intervals so we wouldn't blow past the quota.
   - We cached the extracted facts and embeddings to disk (`delhivery_facts_cache.json`) and added a one-click seed button precisely so you don't have to sit through a 15-minute wait just to evaluate the system.

2. **Lack of UI polish & responsiveness.** Due to time constraints, the frontend has rough edges. It isn't fully mobile-responsive, some complex tables could look cleaner, and transitions between views could be smoother. We prioritized getting the core pipeline, bounding-box exhibits, and cross-document reconciliation rock-solid over styling perfection.

3. **The demo video is rushed.** Fitting a PDF ingestion walkthrough, the pipeline architecture, and all four complex financial cases into a strict 3-minute limit meant talking fast and cutting corners on deep commentary. 

4. **Edge cases in table nesting.** While Docling handles standard financial layouts well, heavily nested footnotes or borderless multi-column splits can still misalign about 5% of cells.

5. **No human-in-the-loop audit interface yet.** Classification currently runs completely algorithmically. In a real-world setting, compliance analysts would need a screen to manually override or confirm disputed edges.

**What I'd build next with more time:**
- Async distributed worker queue (Celery + Redis) to parallelize page parsing across workers instead of sequential processing
- Proper mobile responsiveness and slicker UI polish
- Excel/Sheets export plugin highlighting contradictions in red and corroborated values in green
- An interactive audit review dashboard for human compliance sign-offs

---

## 6. Additional Notes

### Addressing Assignment "Brownie Points"

- **Large PDFs without performance issues:** PyMuPDF handles sub-second page rendering. Docling runs selectively on pages with tables. A 200-page annual report parses in under a minute.
- **Many PDFs in the same knowledge layer:** The claim graph uses Union-Find clustering that scales with the number of facts, not the number of documents. Adding a new document doesn't rebuild the entire graph.
- **Dynamically evolving schemas:** The 6-tuple model accepts any domain. Upload a pharmaceutical filing and it extracts clinical trial metrics the same way.
- **Incremental document ingestion:** `POST /reconcile/append` adds a new document's facts to an existing claim graph without re-processing previous documents.

### REST API Reference

| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/upload` | Upload and parse a PDF (synchronous or `?async=true` background job) |
| `GET` | `/documents` | List all uploaded documents and page counts |
| `GET` | `/documents/{id}/page/{num}/image` | Render a high-res PDF page as PNG |
| `GET` | `/documents/{id}/page/{num}/word-bboxes` | Word-level bounding boxes for interactive highlighting |
| `POST` | `/pipeline/start` | Run extraction + reconciliation pipeline |
| `POST` | `/reconcile` | Build cross-document ArbGraph claim graph |
| `POST` | `/reconcile/append` | **Incremental ingestion:** Add a new doc to existing graph |
| `GET` | `/reconcile/cases` | Retrieve the 4 demonstration cases in structured JSON |
| `POST` | `/system/seed-demo` | One-click instant load of the starter Delhivery dataset |

### Tech Stack
- **Backend:** Python, FastAPI, SQLite (WAL), PyMuPDF, Docling, Google Gemini API
- **Frontend:** React + Vite, TypeScript, Tailwind CSS, Lucide Icons
- **ML Models:** Gemini 3.6/3.5 Flash (extraction), Voyage-finance-2 (embeddings), DeBERTa-v3 (NLI)
- **Tests:** pytest, 153 unit and integration tests

### References
- [ArbGraph](https://github.com/1212Judy/ArbGraph) — claim alignment and conflict arbitration
- [AttestDB](https://dl.acm.org/doi/10.14778/3503585.3503596) — claim-centric data model
- [FActScore](https://arxiv.org/abs/2305.14251) — atomic fact decomposition
- [Docling / TableFormer](https://github.com/DS4SD/docling) — ML table structure recovery
