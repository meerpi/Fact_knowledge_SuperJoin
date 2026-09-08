# Fact Knowledge Layer: Cross-Document Verification & Reconciliation Engine

> **Superjoin Engineering Intern Assignment (VIT 2026)**  
> An industrial-grade Fact Knowledge Layer that ingests financial and economic PDFs, extracts atomic 6-tuple facts anchored to deterministic source evidence, and arbitrates **Corroborations**, **Genuine Contradictions**, and **Contextual Reconciliations** across multi-document corpora.

---

## 1. System Architecture

Standard RAG systems fail on fact verification because cosine similarity treats opposing values (e.g. *$50M revenue* vs *$80M revenue*) as semantically identical. This system decouples semantic discovery from mathematical and natural language verification using a dual-channel architecture:

```
                            ┌──────────────────────────────────────────────┐
                            │               Uploaded PDFs                  │
                            └──────────────────────┬───────────────────────┘
                                                   │
                   ┌───────────────────────────────┴───────────────────────────────┐
                   ▼                                                               ▼
        [Layer 1: PyMuPDF]                                              [Layer 2: Docling]
   • Word-level bounding boxes                                      • DocLayNet layout recovery
   • Ground-truth text index                                        • TableFormer ML tables
   • Ultra-fast quote verification                                  • Multi-page table continuity
                   │                                                               │
                   └───────────────────────────────┬───────────────────────────────┘
                                                   ▼
                                     [LLM Extraction Cascade]
                                    • Gemini 3.8 / 3.7 / 3.6 Flash
                                    • Structured JSON via Pydantic
                                    • iXBRL Dimension & Scale Enums
                                                   │
                                                   ▼
                                   [Deterministic Quote Grounding]
                               • 3-Tier Match: Exact → Norm → Fuzzy
                               • Hallucinated quotes flagged (conf=0.10)
                                                   │
                                                   ▼
                                    [Symbolic Canonicalization]
                                 • Scale Multipliers (10⁶, 10⁷, 10⁹)
                                 • Indian & Intl units (Cr, Lakh, Mn, Bn)
                                 • Accounting negatives: (x) → -x
                                                   │
                                                   ▼
                                    [ArbGraph Claim Graph Engine]
                                 1. Union-Find Semantic Alignment
                                 2. Intra-Cluster Edge Classification
                                 3. Intensity-Driven Credibility Propagation
                                 4. Cluster Arbitration & Dispute Taxonomy
                                                   │
                                                   ▼
                         ┌──────────────────────────────────────────────────┐
                         │           4 Required Assignment Cases            │
                         │  • Case 1: Corroborated Fact                     │
                         │  • Case 2: Genuine Contradiction                 │
                         │  • Case 3: Apparent Contradiction Reconciled     │
                         │  • Case 4: Extraction Failure & Mitigation       │
                         └──────────────────────────────────────────────────┘
```

---

## 2. Core Innovations & Differentiators

1. **Deterministic 3-Tier Quote Verifier**: Every claim must contain a verbatim excerpt. The engine verifies quotes against the raw PDF text layer via Exact $\to$ Whitespace-Normalized $\to$ Fuzzy Sequence matching. Hallucinated citations are penalized with low confidence (0.10) and surfaced in Case 4.
2. **iXBRL-Pattern Unit & Scale Canonicalization**: Following the SEC/XBRL financial standard, numeric facts are represented as `canonical_value = numeric_value × 10^scale`. An amount reported as `₹4,810.5` with table header `(₹ in million)` is canonically resolved to `4,810,500,000.0 INR`, allowing deterministic mathematical comparisons across documents with disparate scales.
3. **Discrepancy Taxonomy Engine**: Contradictions are tagged with machine-readable dispute reason codes (`DISPUTE_GENUINE_CONFLICT`, `DISPUTE_SIGN_MISMATCH`, `DISPUTE_ORDER_OF_MAGNITUDE`, `DISPUTE_TEMPORAL_DRIFT`, `DISPUTE_UNIT_MISMATCH`, `DISPUTE_SCOPE_DIFFERENCE`, `DISPUTE_ACCOUNTING_BASIS`).
4. **ArbGraph Credibility Propagation**: Implements intensity-driven authority propagation. Official filings (Annual Reports, RBI, IMF, Prospectus) carry prior authority weights to determine a consensus value when genuine contradictions occur.
5. **Hybrid Verification**: Symbolic math comparison for numbers; DeBERTa-v3 cross-encoder NLI (`cross-encoder/nli-deberta-v3-base`) for qualitative claims.

---

## 3. The 4 Required Assignment Cases (Demonstrated on India Macro Dataset)

The engine was run across the 3 starter documents:
1. `01-india-economic-survey-2024-25-excerpt.pdf` (Economic Survey 2024-25)
2. `02-rbi-annual-report-2024-25-excerpt.pdf` (Reserve Bank of India Annual Report)
3. `03-imf-india-2025-article-iv-excerpt.pdf` (IMF India 2025 Article IV Consultation)

Output from `e2e_india_macro_cases.json`:

### Case 1: Corroborated Fact
* **Subject**: `India` | **Predicate**: `projected_real_gdp_growth`
* **Evidence**:
  * **RBI Annual Report (p. 23)**: `6.5-7.0 per cent` (canonical: `6.75%`)
  * **Economic Survey (p. 35)**: `6.5-7 per cent` (canonical: `6.75%`)
* **System Reasoning**: Both independent regulatory bodies project matching real GDP growth intervals for FY25 within numeric tolerance ($\text{relative diff} < 10^{-4}$). **Verdict: Corroborated (Confidence: 1.00)**.

### Case 2: Genuine Contradiction
* **Subject**: `India` | **Predicate**: `merchandise_exports`
* **Evidence**:
  * **IMF Article IV (p. 4)**: `$429.2 billion` (canonical: `$429,200,000,000.00`)
  * **Economic Survey (p. 27)**: `$437.1 billion` (canonical: `$437,100,000,000.00`)
* **System Reasoning**: Both reports evaluate FY24 merchandise exports under the same temporal period and USD currency basis, but state irreconcilable figures ($\Delta = \$7.9\text{B}$). **Verdict: Genuine Contradiction (`DISPUTE_GENUINE_CONFLICT`)**.

### Case 3: Apparent Contradiction Reconciled by Context
* **Subject**: `India` | **Predicate**: `headline_inflation`
* **Evidence**:
  * **Economic Survey (p. 46)**: `5.4%` (Annual average FY24)
  * **RBI Report (p. 10)**: `3.6%` (Q2 FY25 projection)
* **System Reasoning**: Superficial text matching flags a conflict between `5.4%` and `3.6%`. The reconciliation engine analyzes temporal context: Doc 1 specifies historical FY24 average while Doc 2 specifies Q2 FY25 forecast. **Verdict: Apparent Contradiction Reconciled (`DISPUTE_TEMPORAL_DRIFT`)**.

### Case 4: Extraction & Reasoning Failure Analysis
* **Failure Type**: `unverified_quote`
* **Description**: LLM generated a non-verbatim quote (`"The decline is attributed to a 0.9 percentage point reduction in core..."`) from the Economic Survey that failed exact, normalized, and fuzzy sequence matching against the raw PDF text layer.
* **Mitigation**: The deterministic verification guardrail detected the citation hallucination, automatically demoted fact confidence to `0.10`, and flagged the cluster for analyst review rather than allowing corrupted evidence into the knowledge graph.

---

## 4. Setup and Run Instructions

### Prerequisites
* Python 3.10+ (tested on Python 3.14)
* Free Gemini API Key from [Google AI Studio](https://aistudio.google.com/apikey)

### Installation
```bash
# Clone the repository
git clone git@github.com:meerpi/SuperJoin_assignment.git
cd SuperJoin_assignment

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key
```

### Running the Test Suite
The codebase includes **106 comprehensive unit and integration tests**:
```bash
PYTHONPATH=. pytest -v
```

### Running the End-to-End Validation Pipelines
Validate the full pipeline and output the 4 assignment cases on the starter datasets:

```bash
# 1. India Macroeconomy Dataset (3 PDFs: Economic Survey, RBI, IMF)
python e2e_india_macro.py

# 2. Delhivery Corporate Dataset (3 PDFs: Prospectus, Annual Report, Earnings Call)
python e2e_claim_graph.py
```

### Starting the FastAPI Server
```bash
uvicorn app.server:app --reload --port 8000
```
Interactive Swagger API documentation will be available at: `http://localhost:8000/docs`.

---

## 5. REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/upload` | Upload a PDF; parses raw text index and Docling table structure |
| `GET` | `/documents` | List all uploaded documents and page counts |
| `GET` | `/documents/{id}/page/{num}` | Get structured layout, text blocks, and tables for a page |
| `POST` | `/documents/{id}/extract` | Extract atomic facts with quote grounding and canonicalization |
| `POST` | `/documents/{id}/contradictions` | Run 3-stage contradiction detection against another document |
| `POST` | `/reconcile` | Build cross-document ArbGraph claim graph across multiple PDFs |
| `GET` | `/reconcile/cases` | Retrieve the 4 required demonstration cases in structured JSON |

---

## 6. Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── models.py           # Pydantic v2 schemas: Fact, Provenance, ClaimGraph, DisputeCode
│   ├── pdf_parser.py       # Dual-layer parser: PyMuPDF + Docling + 3-tier quote verifier
│   ├── extractor.py        # Gemini extraction cascade with rate limiting and prompt caching
│   ├── normalizer.py       # Symbolic canonicalization (scales, currencies, accounting negatives)
│   ├── embeddings.py       # Dynamic semantic alignment via Voyage AI (voyage-finance-2)
│   ├── contradiction.py    # 3-stage contradiction detection (Symbolic + DeBERTa-v3 NLI)
│   ├── claim_graph.py      # ArbGraph claim alignment, credibility propagation, 4 cases
│   └── server.py           # FastAPI REST API endpoints
├── tests/
│   ├── test_normalizer.py  # Unit tests for scale & currency parsing
│   ├── test_extractor.py   # Unit tests for cascade & quote verification
│   ├── test_contradiction.py # Unit tests for contradiction classification
│   ├── test_claim_graph.py # Unit tests for Union-Find clustering & cases
│   └── test_server.py      # Integration tests for FastAPI endpoints
├── starter-datasets/
│   ├── delhivery/          # 3 curated corporate PDFs (Prospectus, Annual Report, Earnings)
│   └── india-macroeconomy/ # 3 institutional PDFs (Economic Survey, RBI, IMF)
├── e2e_india_macro.py      # End-to-end evaluation script on India Macro dataset
├── e2e_claim_graph.py      # End-to-end evaluation script on Delhivery dataset
├── requirements.txt        # Pinned project dependencies
└── README.md
```

---

## 7. Limitations & Next Steps

1. **Visual Bounding Box Viewer**: The backend extracts pixel-accurate word-level bounding boxes (`app/pdf_parser.py::get_word_bboxes`), but does not yet include a frontend PDF canvas viewer to highlight coordinates interactively.
2. **Spreadsheet Sync**: Add a native 1-click export to Excel/CSV with conditional red/green discrepancy formatting to integrate directly with spreadsheet due diligence workflows.
3. **Human-in-the-Loop Arbitration**: Add audit endpoints allowing financial analysts to override or approve machine-classified reconciliations.
