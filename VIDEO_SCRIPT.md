# Video Script — Fact Knowledge Layer Demo
**Duration:** Under 3 minutes  
**Target word count:** ~400 words (~2:50 at natural pace)

---

## Pipeline Overview (0:00 – 0:40)

**Screen:** Open UI, click "Seed Demo Dataset", show documents loading.

> Upload a PDF, five things happen.
>
> Parsing — PyMuPDF grabs words and positions, Docling reconstructs tables. Extraction — Gemini reads the text, pulls out structured facts with quotes. Grounding — every quote gets verified against the raw PDF. If it's not there, confidence drops. Normalization — units and scales get standardized so millions, crores, lakhs all compare correctly. Reconciliation — facts get clustered by meaning using embeddings, then compared using math and NLI. Embeddings find that two facts are about the same thing. They never decide if the values agree — that's deterministic.

---

## Case 1: Corroborated (0:40 – 1:05)

**Screen:** Click "Corroborated" filter, open `automated_sort_centers`.

> Delhivery reports 21 automated sort centres. Shows up three times — Prospectus page 49, page 42, and the Earnings deck page 7. Different phrasing each time. System matched all three to the same predicate, confirmed the values are identical. Corroborated, zero difference.

---

## Case 2: Contradiction (1:05 – 1:30)

**Screen:** Click "Contradicted" filter, open distribution centres cluster, show PDF side-by-side.

> Same Annual Report, two pages disagree. Page 12 says 4,445 distribution centres. Page 46 says 3,506 plus 85 partner centres — that's 3,591. Not 4,445. Same document, same year, no rounding explanation. Flagged as genuine conflict. You can see both pages side by side with the numbers highlighted.

---

## Case 3: Reconciled (1:30 – 1:55)

**Screen:** Click "Reconciled" filter, open temporal drift cluster.

> Trade receivables show three different numbers — 14,297 million, 15,238 million, 7,739 million. Looks like a contradiction until you check the dates. March 2024, March 2023, March 2021. Same metric, different years. System creates a temporal progression link instead of flagging a conflict.

---

## Case 4: Extraction Failure (1:55 – 2:20)

**Screen:** Show Facts Ledger, point to low-confidence row.

> Financial tables put "₹ in millions" once at the top. When the LLM misses that header, it records 139 instead of 139 million. Docling's table parser preserves header scope to prevent this. A post-extraction validator catches what slips through — if a revenue figure comes out under a thousand rupees, confidence drops to 0.10 and it gets flagged for review.

---

## Wrap-up (2:20 – 2:30)

**Screen:** Pan across document list and cluster summary.

> Parse, extract, ground, normalize, reconcile. LLMs handle language, deterministic logic handles math. Everything traced to source quotes and page coordinates. Thanks for watching.

---

## Recording Notes
- Pre-seed the demo dataset.
- Natural pace, ~140 wpm.
- Target: 2:30.
