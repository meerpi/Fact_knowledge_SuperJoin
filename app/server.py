import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.claim_graph import (
    append_document_to_claim_graph,
    build_claim_graph,
    get_assignment_cases,
)
from app.contradiction import detect_contradictions
from app.extractor import GeminiFactExtractor
from app.models import ClaimGraph, ExtractedFacts, Fact
from app.pdf_parser import chunk_document, get_word_bboxes, parse_pdf
from app.storage import SQLiteStorage

logger = logging.getLogger(__name__)

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


# In-memory parse job registry
parse_jobs: dict[str, dict] = {}


def _run_parse_job(job_id: str, filepath: Path):
    job = parse_jobs[job_id]
    t0 = time.time()

    def on_progress(cur: int, total: int, msg: str):
        job["current_page"] = cur
        job["total_pages"] = total
        job["progress_percent"] = min(100, int((cur / max(total, 1)) * 100))
        job["current_step"] = msg
        job["elapsed_seconds"] = round(time.time() - t0, 1)

    try:
        doc = parse_pdf(filepath, progress_callback=on_progress)
        storage.save_document(doc)
        documents[doc.doc_id] = doc

        job["document"] = {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "page_count": doc.page_count,
            "text_blocks": sum(len(p.text_blocks) for p in doc.pages),
            "tables": sum(len(p.tables) for p in doc.pages),
            "scanned_pages": doc.scanned_pages,
            "warnings": doc.warnings,
        }
        job["status"] = "completed"
        job["progress_percent"] = 100
        job["current_step"] = f"Parsed {doc.page_count} pages successfully"
        job["elapsed_seconds"] = round(time.time() - t0, 1)
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["elapsed_seconds"] = round(time.time() - t0, 1)


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    background: bool = Query(False, alias="async"),
):
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

    if background:
        import pymupdf
        try:
            with pymupdf.open(str(dest)) as p_doc:
                est_pages = len(p_doc)
        except Exception:
            est_pages = 1

        job_id = str(uuid.uuid4())[:8]
        t_start = time.time()
        parse_jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "start_time": t_start,
            "filename": safe_filename,
            "current_page": 0,
            "total_pages": est_pages,
            "progress_percent": 0,
            "current_step": f"Uploaded {safe_filename}. Initializing layout parser...",
            "elapsed_seconds": 0.0,
            "error": None,
            "document": None,
        }

        asyncio.get_running_loop().run_in_executor(None, _run_parse_job, job_id, dest)

        return {
            "job_id": job_id,
            "status": "running",
            "filename": safe_filename,
            "total_pages": est_pages,
        }

    # Synchronous parse
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


@app.get("/upload/status/{job_id}")
async def get_upload_status(job_id: str):
    """Poll PDF parsing progress and status."""
    job = parse_jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Upload job '{job_id}' not found")
    if job.get("status") == "running":
        start_t = job.get("start_time")
        if start_t is None:
            start_t = time.time() - job.get("elapsed_seconds", 0.0)
            job["start_time"] = start_t
        job["elapsed_seconds"] = round(time.time() - start_t, 1)
    return job


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document, cascade all extracted facts, and remove physical file."""
    doc = storage.get_document(doc_id) or documents.get(doc_id)
    filename = doc.filename if doc else None

    deleted = storage.delete_document(doc_id)
    documents.pop(doc_id, None)
    extracted_facts_db.pop(doc_id, None)
    contradiction_reports_db.pop(doc_id, None)

    if filename:
        filepath = UPLOAD_DIR / filename
        if filepath.exists():
            try:
                filepath.unlink()
            except Exception as e:
                logger.warning("Failed to delete file %s: %s", filepath, e)

    # Clean up latest graph if no docs remain
    remaining = storage.list_documents()
    if not remaining:
        reconciliation_db.clear()
        conn = storage._get_connection()
        with conn:
            conn.execute("DELETE FROM claim_graphs;")

    if not deleted and not doc:
        raise HTTPException(404, f"Document '{doc_id}' not found")

    return {"deleted": True, "doc_id": doc_id}


@app.delete("/documents")
async def clear_all_documents():
    """Clear all documents, extracted facts, and reconciliation results."""
    count = storage.clear_all_documents()
    documents.clear()
    extracted_facts_db.clear()
    contradiction_reports_db.clear()
    reconciliation_db.clear()

    # Clean uploads directory
    for f in UPLOAD_DIR.glob("*.pdf"):
        try:
            f.unlink()
        except Exception as e:
            logger.warning("Failed to remove %s: %s", f, e)

    return {"deleted_count": count, "message": "All documents and analysis records cleared"}


@app.delete("/documents/{doc_id}/facts")
async def delete_document_facts(doc_id: str):
    """Delete extracted facts for a document from SQLite and in-memory cache,
    allowing re-extraction without deleting the uploaded PDF document itself."""
    deleted = storage.delete_extracted_facts(doc_id)
    extracted_facts_db.pop(doc_id, None)
    return {"deleted": deleted, "doc_id": doc_id, "message": f"Extracted facts for '{doc_id}' cleared"}


@app.delete("/facts")
async def clear_all_facts():
    """Clear all extracted facts across all documents from SQLite and in-memory cache."""
    count = storage.clear_all_extracted_facts()
    extracted_facts_db.clear()
    return {"deleted_count": count, "message": f"Cleared {count} extracted fact record(s)"}


@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    docs = storage.list_documents()
    for d in docs:
        if not d.get("has_extracted_facts"):
            d["has_extracted_facts"] = (
                d["doc_id"] in extracted_facts_db and len(extracted_facts_db[d["doc_id"]].facts) > 0
            )
    return docs


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


@app.get("/reconcile/latest")
async def get_latest_reconciliation():
    """Get the latest full reconciliation ClaimGraph."""
    graph = reconciliation_db.get("latest") or storage.get_claim_graph("latest")
    if not graph:
        raise HTTPException(
            404,
            "No reconciliation graph available. Run POST /reconcile first.",
        )
    return graph


# ---------------------------------------------------------------------------
# Companion endpoints for UI & Real Pipeline Progress
# ---------------------------------------------------------------------------

def _find_document_file(filename: str) -> Path | None:
    """Resolve a document's PDF file path across uploads and starter datasets."""
    candidates = [
        UPLOAD_DIR / filename,
        Path("starter-datasets/india-macroeconomy") / filename,
        Path("starter-datasets/delhivery") / filename,
        Path(filename),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def _check_docling_available() -> bool:
    try:
        from docling.document_converter import DocumentConverter
        return True
    except Exception:
        return False


def _check_nli_available() -> bool:
    try:
        from sentence_transformers import CrossEncoder
        return True
    except Exception:
        return False


@app.get("/system/capabilities")
async def get_capabilities():
    """Return runtime status of optional model and service capabilities."""
    latest_graph = storage.get_claim_graph("latest") or reconciliation_db.get("latest")
    return {
        "gemini_api_key_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "voyage_api_key_configured": bool(os.environ.get("VOYAGE_API_KEY")),
        "docling_available": _check_docling_available(),
        "nli_model_available": _check_nli_available(),
        "loaded_documents": len(storage.list_documents()),
        "extracted_documents": len(storage.list_extracted_doc_ids()),
        "has_reconciliation": latest_graph is not None,
    }


@app.get("/documents/{doc_id}/file")
async def get_document_file(doc_id: str):
    """Serve the raw PDF file for in-browser viewing."""
    doc = storage.get_document(doc_id) or documents.get(doc_id)
    if not doc:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    filepath = _find_document_file(doc.filename)
    if not filepath or not filepath.exists():
        raise HTTPException(404, f"PDF file '{doc.filename}' not found on disk")
    return FileResponse(
        str(filepath),
        media_type="application/pdf",
        content_disposition_type="inline",
        filename=doc.filename,
    )


@app.get("/documents/{doc_id}/page/{page_num}/image")
async def get_page_image(doc_id: str, page_num: int, dpi: int = 150):
    """Render a high-resolution PNG image of a PDF page for reliable in-browser exhibit display."""
    doc = storage.get_document(doc_id) or documents.get(doc_id)
    if not doc:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    if page_num < 0 or page_num >= doc.page_count:
        raise HTTPException(400, f"Page {page_num} out of range (0-{doc.page_count - 1})")
    filepath = _find_document_file(doc.filename)
    if not filepath or not filepath.exists():
        raise HTTPException(404, f"PDF file '{doc.filename}' not found on disk")
    try:
        import pymupdf
        from starlette.responses import Response
        doc_fitz = pymupdf.open(str(filepath))
        page = doc_fitz[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        pw, ph = float(page.rect.width), float(page.rect.height)
        doc_fitz.close()
        return Response(
            content=img_bytes,
            media_type="image/png",
            headers={
                "X-Page-Width": str(pw),
                "X-Page-Height": str(ph),
                "Cache-Control": "public, max-age=3600",
            },
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to render page image: {e}")


@app.get("/documents/{doc_id}/page/{page_num}/word-bboxes")
async def get_page_word_bboxes(doc_id: str, page_num: int):
    """Retrieve word bounding boxes for in-place quote highlighting on a PDF page."""
    doc = storage.get_document(doc_id) or documents.get(doc_id)
    if not doc:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    if page_num < 0 or page_num >= doc.page_count:
        raise HTTPException(400, f"Page {page_num} out of range (0-{doc.page_count - 1})")
    filepath = _find_document_file(doc.filename)
    if not filepath or not filepath.exists():
        raise HTTPException(404, f"PDF file '{doc.filename}' not found on disk")
    try:
        import pymupdf
        doc_fitz = pymupdf.open(str(filepath))
        page = doc_fitz[page_num]
        pw, ph = float(page.rect.width), float(page.rect.height)
        doc_fitz.close()
        bboxes = get_word_bboxes(filepath, page_num)
        return {
            "doc_id": doc_id,
            "page_num": page_num,
            "page_width": pw,
            "page_height": ph,
            "words": bboxes,
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to compute word bounding boxes: {e}")


# In-memory pipeline job registry
pipeline_jobs: dict[str, dict] = {}


class PipelineStartRequest(BaseModel):
    doc_ids: list[str] | None = None
    target_pages: dict[str, list[int]] | None = None
    nli_threshold: float = 0.7
    force_reextract: bool = False


async def _run_pipeline_job(
    job_id: str,
    doc_ids: list[str],
    target_pages_map: dict[str, list[int]] | None,
    nli_threshold: float,
    force_reextract: bool = False,
):
    job = pipeline_jobs[job_id]
    t0 = job.get("start_time") or time.time()
    job["start_time"] = t0
    try:
        extractor = GeminiFactExtractor()
        total_facts = 0

        # Calculate exact total pages across all target documents
        total_pages_all_docs = 0
        doc_page_counts: dict[str, int] = {}
        for d_id in doc_ids:
            d = documents.get(d_id) or storage.get_document(d_id)
            if d:
                pgs = len(target_pages_map[d_id]) if (target_pages_map and d_id in target_pages_map) else d.page_count
                doc_page_counts[d_id] = pgs
                total_pages_all_docs += pgs
        if total_pages_all_docs == 0:
            total_pages_all_docs = 1

        job["total_pages"] = total_pages_all_docs
        job["current_page"] = 0
        job["progress_percent"] = 0

        pages_completed_so_far = 0

        for idx, d_id in enumerate(doc_ids):
            doc = documents.get(d_id) or storage.get_document(d_id)
            if not doc:
                continue
            job["current_doc_name"] = doc.filename
            job["current_doc_index"] = idx + 1
            doc_pgs = doc_page_counts.get(d_id, doc.page_count)
            pages = target_pages_map.get(d_id) if target_pages_map else None

            # SQLite Caching: check if facts already exist in SQLite
            cached_ef = None if force_reextract else (extracted_facts_db.get(d_id) or storage.get_extracted_facts(d_id))
            if cached_ef and len(cached_ef.facts) > 0 and pages is None:
                extracted_facts_db[d_id] = cached_ef
                total_facts += len(cached_ef.facts)
                pages_completed_so_far += doc_pgs
                job["current_page"] = pages_completed_so_far
                job["progress_percent"] = min(85, int((pages_completed_so_far / total_pages_all_docs) * 85))
                job["facts_extracted_so_far"] = total_facts
                job["elapsed_seconds"] = round(time.time() - t0, 1)
                job["stages_completed"].append(
                    f"Loaded {len(cached_ef.facts)} cached facts from SQLite for {doc.filename} (0.0s)"
                )
                continue

            def _on_batch_progress(pages_in_doc_done, total_in_doc, msg):
                cur_total = pages_completed_so_far + pages_in_doc_done
                job["current_page"] = cur_total
                job["progress_percent"] = min(85, max(0, int((cur_total / total_pages_all_docs) * 85)))
                job["current_step"] = (
                    f"Extracting facts from {doc.filename}: page {pages_in_doc_done} of {total_in_doc} "
                    f"({job['progress_percent']}%)"
                )
                job["elapsed_seconds"] = round(time.time() - t0, 1)

            job["current_step"] = (
                f"Extracting and verifying facts from {doc.filename} "
                f"(Document {idx + 1} of {len(doc_ids)}, {doc_pgs} pages)..."
            )
            job["elapsed_seconds"] = round(time.time() - t0, 1)

            ef = await extractor.aextract_and_verify(doc, pages=pages, on_progress=_on_batch_progress)
            pages_completed_so_far += doc_pgs
            job["current_page"] = pages_completed_so_far
            job["progress_percent"] = min(85, int((pages_completed_so_far / total_pages_all_docs) * 85))

            extracted_facts_db[d_id] = ef
            storage.save_extracted_facts(d_id, ef)
            total_facts += len(ef.facts)
            job["facts_extracted_so_far"] = total_facts
            job["elapsed_seconds"] = round(time.time() - t0, 1)
            job["stages_completed"].append(
                f"Extracted and grounded {len(ef.facts)} facts from {doc.filename}"
            )

        job["progress_percent"] = 88
        job["current_step"] = (
            f"Running ArbGraph claim alignment across {len(doc_ids)} document(s)..."
        )
        job["elapsed_seconds"] = round(time.time() - t0, 1)

        doc_facts = {
            d_id: extracted_facts_db[d_id].facts
            for d_id in doc_ids
            if d_id in extracted_facts_db
        }
        doc_filenames = {}
        for d_id in doc_ids:
            d = documents.get(d_id) or storage.get_document(d_id)
            if d:
                doc_filenames[d_id] = d.filename

        job["progress_percent"] = 92
        job["current_step"] = "Arbitrating cross-document relationships & contradiction verdicts..."

        graph = build_claim_graph(
            doc_facts=doc_facts,
            doc_filenames=doc_filenames,
            nli_threshold=nli_threshold,
        )
        reconciliation_db["latest"] = graph
        storage.save_claim_graph(graph, "latest")

        job["stages_completed"].append(
            f"Arbitrated {len(graph.clusters)} claim clusters into cross-document ledger"
        )
        job["progress_percent"] = 100
        job["current_step"] = "Analysis complete. Verification exhibit ready."
        job["status"] = "completed"
        job["elapsed_seconds"] = round(time.time() - t0, 1)
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["elapsed_seconds"] = round(time.time() - t0, 1)


@app.post("/pipeline/start")
async def start_pipeline(req: PipelineStartRequest):
    """Start asynchronous multi-document extraction and reconciliation pipeline."""
    target_ids = req.doc_ids
    if not target_ids:
        target_ids = [d["doc_id"] for d in storage.list_documents()]

    if not target_ids:
        raise HTTPException(
            400,
            "No documents available for analysis. Please upload at least one PDF first.",
        )

    # Pre-calculate estimated pages for immediate progress display
    est_total_pages = 0
    for d_id in target_ids:
        d = documents.get(d_id) or storage.get_document(d_id)
        if d:
            pgs = len(req.target_pages[d_id]) if (req.target_pages and d_id in req.target_pages) else d.page_count
            est_total_pages += pgs
    if est_total_pages == 0:
        est_total_pages = 1

    job_id = str(uuid.uuid4())[:8]
    t_start = time.time()
    pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "start_time": t_start,
        "current_doc_name": "",
        "current_doc_index": 0,
        "total_docs": len(target_ids),
        "current_step": f"Queued {len(target_ids)} document(s) ({est_total_pages} pages) for extraction...",
        "facts_extracted_so_far": 0,
        "elapsed_seconds": 0.0,
        "stages_completed": [],
        "progress_percent": 0,
        "current_page": 0,
        "total_pages": est_total_pages,
        "error": None,
    }

    asyncio.create_task(
        _run_pipeline_job(
            job_id=job_id,
            doc_ids=target_ids,
            target_pages_map=req.target_pages,
            nli_threshold=req.nli_threshold,
            force_reextract=req.force_reextract,
        )
    )

    return {
        "job_id": job_id,
        "status": "running",
        "total_docs": len(target_ids),
        "total_pages": est_total_pages,
    }


@app.get("/pipeline/status/{job_id}")
async def get_pipeline_status(job_id: str):
    """Poll pipeline execution status and progress logs."""
    job = pipeline_jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    if job.get("status") == "running":
        start_t = job.get("start_time")
        if start_t is None:
            start_t = time.time() - job.get("elapsed_seconds", 0.0)
            job["start_time"] = start_t
        job["elapsed_seconds"] = round(time.time() - start_t, 1)
    return job


@app.post("/system/seed-demo")
async def seed_demo_dataset():
    """Seed storage with the starter Delhivery dataset for instant demonstration."""
    results_path = Path("delhivery_claim_graph.json")
    if not results_path.exists():
        raise HTTPException(404, "delhivery_claim_graph.json not found")

    with open(results_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    graph = ClaimGraph.model_validate(graph_data)
    storage.save_claim_graph(graph, "latest")
    reconciliation_db["latest"] = graph

    # Load facts cache
    cache_path = Path("delhivery_facts_cache.json")
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            raw_cache = json.load(f)

        doc_facts_map: dict[str, list[Fact]] = {}
        for key, val in raw_cache.items():
            doc_id = key.split(":")[0]
            if doc_id not in doc_facts_map:
                doc_facts_map[doc_id] = []
            facts_list = val.get("facts", []) if isinstance(val, dict) else (val if isinstance(val, list) else [])
            for item in facts_list:
                try:
                    doc_facts_map[doc_id].append(Fact.model_validate(item))
                except Exception:
                    pass

        for doc_id, facts in doc_facts_map.items():
            ef = ExtractedFacts(
                doc_id=doc_id,
                facts=facts,
                model_used="gemini-3.8-flash",
                fallback_attempts=0,
            )
            storage.save_extracted_facts(doc_id, ef)
            extracted_facts_db[doc_id] = ef

    # Register the 3 documents in SQLite if not already present
    from app.models import DocumentData
    starter_dir = Path("starter-datasets/delhivery")
    for doc_id, filename in graph.documents.items():
        if not storage.get_document(doc_id):
            pdf_path = starter_dir / filename
            if pdf_path.exists():
                try:
                    import pymupdf
                    doc_fitz = pymupdf.open(str(pdf_path))
                    page_count = len(doc_fitz)
                    doc_fitz.close()
                    doc_data = DocumentData(
                        doc_id=doc_id,
                        filename=filename,
                        page_count=page_count,
                        pages=[],
                    )
                    storage.save_document(doc_data)
                    documents[doc_id] = doc_data
                except Exception as e:
                    logger.warning("Could not register starter doc %s: %s", filename, e)

    return {
        "status": "seeded",
        "documents": graph.documents,
        "total_facts": graph.total_facts,
        "clusters_count": len(graph.clusters),
        "case_summary": graph.case_summary,
    }


