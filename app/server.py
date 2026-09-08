import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.claim_graph import (
    append_document_to_claim_graph,
    build_claim_graph,
    get_assignment_cases,
)
from app.contradiction import detect_contradictions
from app.extractor import GeminiFactExtractor
from app.pdf_parser import chunk_document, parse_pdf
from app.storage import SQLiteStorage

app = FastAPI(title="Fact Knowledge Layer", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Persistent SQLite storage engine
storage = SQLiteStorage()

# In-memory stores — backed by SQLite for fast retrieval and persistence across restarts
documents: dict = {}  # doc_id -> DocumentData
extracted_facts_db: dict = {}  # doc_id -> ExtractedFacts
contradiction_reports_db: dict = {}  # doc_id -> ContradictionReport
reconciliation_db: dict = {}  # "latest" -> ClaimGraph


def _sync_from_storage():
    """Hydrate memory state from SQLite database on startup."""
    try:
        for item in storage.list_documents():
            doc = storage.get_document(item["doc_id"])
            if doc:
                documents[doc.doc_id] = doc
        for doc_id in storage.list_extracted_doc_ids():
            ef = storage.get_extracted_facts(doc_id)
            if ef:
                extracted_facts_db[doc_id] = ef
        latest_graph = storage.get_claim_graph("latest")
        if latest_graph:
            reconciliation_db["latest"] = latest_graph
    except Exception as e:
        pass


_sync_from_storage()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, parse it, and store the structured result in SQLite."""
    if not file.filename:
        raise HTTPException(400, "Filename is missing")

    # Sanitize filename against directory traversal attacks
    safe_filename = Path(file.filename).name
    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    # Save to disk
    dest = UPLOAD_DIR / safe_filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Parse
    try:
        doc = parse_pdf(dest)
    except Exception as e:
        raise HTTPException(422, f"Failed to parse PDF: {e}")

    # Persist in storage and update in-memory map
    storage.save_document(doc)
    documents[doc.doc_id] = doc

    return {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "page_count": doc.page_count,
        "text_blocks": sum(len(p.text_blocks) for p in doc.pages),
        "tables": sum(len(p.tables) for p in doc.pages),
        "scanned_pages": doc.scanned_pages,
        "warnings": doc.warnings,
    }


@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    return storage.list_documents()


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get full parsed data for a document."""
    doc = storage.get_document(doc_id) or documents.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


@app.get("/documents/{doc_id}/chunks")
async def get_chunks(doc_id: str, max_tokens: int = 2500):
    """Get LLM-ready chunks for a document."""
    doc = storage.get_document(doc_id) or documents.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return chunk_document(doc, max_tokens=max_tokens)


@app.get("/documents/{doc_id}/page/{page_num}")
async def get_page(doc_id: str, page_num: int):
    """Get extracted data for a single page."""
    doc = storage.get_document(doc_id) or documents.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    if page_num < 0 or page_num >= doc.page_count:
        raise HTTPException(400, f"Page {page_num} out of range (0-{doc.page_count - 1})")
    return doc.pages[page_num]


@app.post("/documents/{doc_id}/extract")
async def extract_facts(
    doc_id: str,
    page: int | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
    api_key: str | None = None,
):
    """Extract semantic & numerical facts from a document with fallback cascade.

    Supports:
    - Single page via ?page=N
    - Page range via ?start_page=N&end_page=M
    - Entire document (default)
    """
    doc = documents.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    target_pages = None
    page_num = page
    if page is None and (start_page is not None or end_page is not None):
        s = start_page if start_page is not None else 0
        e = end_page if end_page is not None else doc.page_count - 1
        if s < 0 or e >= doc.page_count or s > e:
            raise HTTPException(400, f"Invalid page range [{s}, {e}] for document with {doc.page_count} pages")
        target_pages = list(range(s, e + 1))

    try:
        from starlette.concurrency import run_in_threadpool

        extractor = GeminiFactExtractor(api_key=api_key)
        result = await run_in_threadpool(
            extractor.extract_and_verify,
            doc,
            page_num=page_num,
            pages=target_pages,
        )
        extracted_facts_db[doc_id] = result
        storage.save_extracted_facts(doc_id, result)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Extraction failed: {str(e)}")


@app.get("/documents/{doc_id}/facts")
async def get_facts(doc_id: str):
    """Retrieve extracted facts for a document."""
    if doc_id in extracted_facts_db:
        return extracted_facts_db[doc_id]
    ef = storage.get_extracted_facts(doc_id)
    if ef:
        extracted_facts_db[doc_id] = ef
        return ef
    raise HTTPException(404, "No facts extracted for this document yet. Run POST /documents/{doc_id}/extract first.")


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
    facts_container = extracted_facts_db.get(doc_id) or storage.get_extracted_facts(doc_id)
    if not facts_container:
        raise HTTPException(
            404,
            "No facts extracted for this document yet. "
            "Run POST /documents/{doc_id}/extract first.",
        )

    facts = facts_container.facts

    cross_doc_facts = None
    if cross_doc_id:
        cross_container = extracted_facts_db.get(cross_doc_id) or storage.get_extracted_facts(cross_doc_id)
        if not cross_container:
            raise HTTPException(
                404,
                f"No facts extracted for cross-document '{cross_doc_id}'. "
                "Run POST /documents/{cross_doc_id}/extract first.",
            )
        cross_doc_facts = cross_container.facts

    try:
        report = detect_contradictions(
            facts=facts,
            doc_id=doc_id,
            cross_doc_facts=cross_doc_facts,
            cross_doc_id=cross_doc_id,
            nli_threshold=nli_threshold,
        )
        contradiction_reports_db[doc_id] = report
        storage.save_contradiction_report(report)
        return report
    except Exception as e:
        raise HTTPException(502, f"Contradiction detection failed: {str(e)}")


@app.get("/documents/{doc_id}/claim-graph")
async def get_claim_graph(doc_id: str):
    """Get the full claim graph with typed edges for a document."""
    report = contradiction_reports_db.get(doc_id) or storage.get_contradiction_report(doc_id)
    if not report:
        raise HTTPException(
            404,
            "No contradiction report for this document yet. "
            "Run POST /documents/{doc_id}/contradictions first.",
        )

    facts_container = extracted_facts_db.get(doc_id) or storage.get_extracted_facts(doc_id)
    facts = facts_container.facts if facts_container else []

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
    doc_authority: dict[str, float] | None = None,
    max_facts: int | None = None,
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
    all_stored_extracted = storage.get_all_extracted_facts()
    for d_id, f_list in all_stored_extracted.items():
        if d_id not in extracted_facts_db:
            ef = storage.get_extracted_facts(d_id)
            if ef:
                extracted_facts_db[d_id] = ef

    if doc_ids:
        missing = [d for d in doc_ids if d not in extracted_facts_db]
        if missing:
            raise HTTPException(
                404,
                f"No extracted facts for document(s): {', '.join(missing)}. "
                "Run POST /documents/{doc_id}/extract first.",
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

    total_facts = sum(len(f) for f in doc_facts.values())
    if max_facts is not None and total_facts > max_facts:
        raise HTTPException(
            413,
            f"Total facts ({total_facts}) exceeds maximum allowed limit ({max_facts}). "
            "Please filter documents or increase max_facts.",
        )

    doc_filenames = {}
    for doc_id in target_ids:
        doc = documents.get(doc_id) or storage.get_document(doc_id)
        if doc:
            doc_filenames[doc_id] = doc.filename
        elif doc_id in extracted_facts_db:
            doc_filenames[doc_id] = doc_id  # fallback

    try:
        graph = build_claim_graph(
            doc_facts=doc_facts,
            doc_filenames=doc_filenames,
            nli_threshold=nli_threshold,
            doc_authority_overrides=doc_authority,
        )
        reconciliation_db["latest"] = graph
        storage.save_claim_graph(graph, "latest")
        return graph
    except Exception as e:
        raise HTTPException(502, f"Reconciliation failed: {str(e)}")


@app.post("/reconcile/append")
async def reconcile_append(
    doc_id: str,
    nli_threshold: float = 0.7,
    doc_authority: dict[str, float] | None = None,
):
    """Incrementally append a new document's facts into the latest claim graph.

    Avoids full re-clustering by:
    - Routing new facts only to matching clusters or singletons
    - Evaluating edges only for new pairs in affected clusters
    - Re-arbitrating only dirty clusters, preserving unaffected clusters untouched
    """
    facts_container = extracted_facts_db.get(doc_id) or storage.get_extracted_facts(doc_id)
    if not facts_container:
        raise HTTPException(
            404,
            f"No extracted facts found for document '{doc_id}'. Run POST /documents/{doc_id}/extract first.",
        )

    new_facts = facts_container.facts

    doc_meta = documents.get(doc_id) or storage.get_document(doc_id)
    new_filename = doc_meta.filename if doc_meta else doc_id

    existing_graph = reconciliation_db.get("latest") or storage.get_claim_graph("latest")

    if existing_graph is None:
        # Fall back to initial reconciliation
        graph = await reconcile_documents(
            doc_ids=[doc_id],
            nli_threshold=nli_threshold,
            doc_authority=doc_authority,
        )
        return {
            "graph": graph,
            "incremental_stats": {
                "new_facts_count": len(new_facts),
                "clusters_updated": 0,
                "clusters_created": len(graph.clusters),
                "new_singletons": len(graph.unmatched_facts),
                "unaffected_clusters": 0,
                "duration_seconds": 0.0,
            },
        }

    # Build existing facts mapping
    existing_doc_facts = {}
    for d_id in existing_graph.documents:
        fc = extracted_facts_db.get(d_id) or storage.get_extracted_facts(d_id)
        if fc:
            existing_doc_facts[d_id] = fc.facts

    try:
        updated_graph, stats = append_document_to_claim_graph(
            existing_graph=existing_graph,
            existing_doc_facts=existing_doc_facts,
            new_doc_id=doc_id,
            new_doc_facts=new_facts,
            new_doc_filename=new_filename,
            nli_threshold=nli_threshold,
            doc_authority_overrides=doc_authority,
        )
        reconciliation_db["latest"] = updated_graph
        storage.save_claim_graph(updated_graph, "latest")
        return {
            "graph": updated_graph,
            "incremental_stats": stats,
        }
    except Exception as e:
        raise HTTPException(502, f"Incremental reconciliation failed: {str(e)}")


@app.get("/reconcile/cases")
async def get_reconciliation_cases():
    """Get the 4 required assignment cases from the latest reconciliation.

    Returns one best example per case type:
    - Case 1: Corroborated fact across documents
    - Case 2: Genuine contradiction
    - Case 3: Apparent contradiction reconciled by context
    - Case 4: Extraction or reasoning failure
    """
    graph = reconciliation_db.get("latest") or storage.get_claim_graph("latest")
    if not graph:
        raise HTTPException(
            404,
            "No reconciliation results available. Run POST /reconcile first.",
        )

    return get_assignment_cases(graph)

