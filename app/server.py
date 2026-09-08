"""FastAPI server — PDF upload and document inspection endpoints."""

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.claim_graph import build_claim_graph, get_assignment_cases
from app.contradiction import detect_contradictions
from app.extractor import GeminiFactExtractor
from app.pdf_parser import chunk_document, parse_pdf

app = FastAPI(title="Fact Knowledge Layer", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory stores — swap for sqlite later if needed
documents: dict = {}  # doc_id -> DocumentData
extracted_facts_db: dict = {}  # doc_id -> ExtractedFacts
contradiction_reports_db: dict = {}  # doc_id -> ContradictionReport
reconciliation_db: dict = {}  # "latest" -> ClaimGraph


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, parse it, and store the structured result."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    # Save to disk
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Parse
    try:
        doc = parse_pdf(dest)
    except Exception as e:
        raise HTTPException(422, f"Failed to parse PDF: {e}")

    documents[doc.doc_id] = doc

    return {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "page_count": doc.page_count,
        "text_blocks": sum(len(p.text_blocks) for p in doc.pages),
        "tables": sum(len(p.tables) for p in doc.pages),
    }


@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    return [
        {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "page_count": doc.page_count,
        }
        for doc in documents.values()
    ]


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get full parsed data for a document."""
    doc = documents.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


@app.get("/documents/{doc_id}/chunks")
async def get_chunks(doc_id: str, max_tokens: int = 2500):
    """Get LLM-ready chunks for a document."""
    doc = documents.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return chunk_document(doc, max_tokens=max_tokens)


@app.get("/documents/{doc_id}/page/{page_num}")
async def get_page(doc_id: str, page_num: int):
    """Get extracted data for a single page."""
    doc = documents.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    if page_num < 0 or page_num >= doc.page_count:
        raise HTTPException(400, f"Page {page_num} out of range (0-{doc.page_count - 1})")
    return doc.pages[page_num]


@app.post("/documents/{doc_id}/extract")
async def extract_facts(
    doc_id: str,
    page: int | None = None,
    api_key: str | None = None,
):
    """Extract semantic & numerical facts from a document with fallback cascade."""
    doc = documents.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    try:
        extractor = GeminiFactExtractor(api_key=api_key)
        result = extractor.extract_and_verify(doc, page_num=page)
        extracted_facts_db[doc_id] = result
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Extraction failed: {str(e)}")


@app.get("/documents/{doc_id}/facts")
async def get_facts(doc_id: str):
    """Retrieve extracted facts for a document."""
    if doc_id not in extracted_facts_db:
        raise HTTPException(404, "No facts extracted for this document yet. Run POST /documents/{doc_id}/extract first.")
    return extracted_facts_db[doc_id]


@app.post("/documents/{doc_id}/contradictions")
async def run_contradiction_detection(
    doc_id: str,
    cross_doc_id: str | None = None,
    nli_threshold: float = 0.7,
):
    """Run 3-stage contradiction detection on extracted facts.

    Stages:
    1. Symbolic alignment — group by (subject, predicate)
    2. Deterministic numeric comparison on canonical_value
    3. DeBERTa-v3 NLI for remaining qualitative claim pairs

    Optional: pass cross_doc_id to compare facts across two documents.
    """
    if doc_id not in extracted_facts_db:
        raise HTTPException(
            404,
            "No facts extracted for this document yet. "
            "Run POST /documents/{doc_id}/extract first.",
        )

    facts = extracted_facts_db[doc_id].facts

    cross_doc_facts = None
    if cross_doc_id:
        if cross_doc_id not in extracted_facts_db:
            raise HTTPException(
                404,
                f"No facts extracted for cross-document '{cross_doc_id}'. "
                "Run POST /documents/{cross_doc_id}/extract first.",
            )
        cross_doc_facts = extracted_facts_db[cross_doc_id].facts

    try:
        report = detect_contradictions(
            facts=facts,
            doc_id=doc_id,
            cross_doc_facts=cross_doc_facts,
            cross_doc_id=cross_doc_id,
            nli_threshold=nli_threshold,
        )
        contradiction_reports_db[doc_id] = report
        return report
    except Exception as e:
        raise HTTPException(502, f"Contradiction detection failed: {str(e)}")


@app.get("/documents/{doc_id}/claim-graph")
async def get_claim_graph(doc_id: str):
    """Get the full claim graph with typed edges for a document."""
    if doc_id not in contradiction_reports_db:
        raise HTTPException(
            404,
            "No contradiction report for this document yet. "
            "Run POST /documents/{doc_id}/contradictions first.",
        )

    report = contradiction_reports_db[doc_id]
    facts = extracted_facts_db[doc_id].facts

    # Build a graph representation with nodes (facts) and edges
    nodes = []
    for idx, fact in enumerate(facts):
        nodes.append({
            "id": idx,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "value": fact.value,
            "canonical_value": fact.canonical_value,
            "canonical_unit": fact.canonical_unit,
            "temporal": fact.context.temporal,
            "scope": fact.context.scope,
            "conditions": fact.context.conditions,
            "page": fact.provenance.page,
        })

    graph_edges = []
    for edge in report.edges:
        graph_edges.append({
            "source": edge.source_fact_idx,
            "target": edge.target_fact_idx,
            "type": edge.edge_type.value,
            "method": edge.detection_method,
            "confidence": edge.confidence,
            "explanation": edge.explanation,
        })

    return {
        "doc_id": doc_id,
        "nodes": nodes,
        "edges": graph_edges,
        "summary": report.summary,
    }


# ---------------------------------------------------------------------------
# Cross-Document Reconciliation — ArbGraph Claim Graph pipeline
# ---------------------------------------------------------------------------

@app.post("/reconcile")
async def reconcile_documents(
    doc_ids: list[str] | None = None,
    nli_threshold: float = 0.7,
):
    """Build a cross-document claim graph using the ArbGraph pipeline.

    Runs the full 4-stage pipeline:
        1. Claim Alignment (Union-Find clustering by subject + predicate)
        2. Evidence Graph (intra-cluster edge classification)
        3. Credibility Propagation (ArbGraph intensity-driven scoring)
        4. Cluster Arbitration (classify into the 4 assignment cases)

    If doc_ids is None, reconciles ALL documents with extracted facts.
    """
    # Determine which documents to reconcile
    if doc_ids:
        missing = [d for d in doc_ids if d not in extracted_facts_db]
        if missing:
            raise HTTPException(
                404,
                f"No extracted facts for document(s): {', '.join(missing)}. "
                "Run POST /documents/{{doc_id}}/extract first.",
            )
        target_ids = doc_ids
    else:
        target_ids = list(extracted_facts_db.keys())

    if len(target_ids) == 0:
        raise HTTPException(
            400,
            "No documents with extracted facts available. "
            "Upload PDFs and extract facts first.",
        )

    # Build inputs
    doc_facts = {
        doc_id: extracted_facts_db[doc_id].facts
        for doc_id in target_ids
    }
    doc_filenames = {}
    for doc_id in target_ids:
        if doc_id in documents:
            doc_filenames[doc_id] = documents[doc_id].filename
        elif doc_id in extracted_facts_db:
            doc_filenames[doc_id] = doc_id  # fallback

    try:
        graph = build_claim_graph(
            doc_facts=doc_facts,
            doc_filenames=doc_filenames,
            nli_threshold=nli_threshold,
        )
        reconciliation_db["latest"] = graph
        return graph
    except Exception as e:
        raise HTTPException(502, f"Reconciliation failed: {str(e)}")


@app.get("/reconcile/cases")
async def get_reconciliation_cases():
    """Get the 4 required assignment cases from the latest reconciliation.

    Returns one best example per case type:
    - Case 1: Corroborated fact across documents
    - Case 2: Genuine contradiction
    - Case 3: Apparent contradiction reconciled by context
    - Case 4: Extraction or reasoning failure
    """
    if "latest" not in reconciliation_db:
        raise HTTPException(
            404,
            "No reconciliation results available. Run POST /reconcile first.",
        )

    graph = reconciliation_db["latest"]
    return get_assignment_cases(graph)

