"""End-to-end validation: Multi-doc PDF → Parse → Extract → Claim Graph → 4 Cases.

Run directly: .venv/bin/python e2e_claim_graph.py

Tests the full ArbGraph-inspired pipeline across all 3 Delhivery PDFs:
1. Parse all PDFs
2. Extract facts (pages with financial tables)
3. Build the cross-document claim graph
4. Output the 4 required assignment cases
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from app.pdf_parser import parse_pdf
from app.extractor import GeminiFactExtractor
from app.claim_graph import build_claim_graph, get_assignment_cases

# ---------------------------------------------------------------------------
# Configuration — which PDFs and pages to test
# ---------------------------------------------------------------------------

DELHIVERY_DOCS = [
    {
        "path": "starter-datasets/delhivery/01-delhivery-prospectus-2022-excerpt.pdf",
        "pages": [16, 44],  # Financial statement pages
        "label": "Prospectus 2022",
    },
    {
        "path": "starter-datasets/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf",
        "pages": [26, 70],  # Financial pages
        "label": "Annual Report FY24",
    },
    {
        "path": "starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf",
        "pages": [6, 12],   # Earnings summary pages
        "label": "Q4 FY24 Earnings",
    },
]


def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_cluster(cluster, facts_pool=None):
    """Print a cluster in a human-readable format."""
    print(f"\n  📌 Cluster: {cluster.cluster_id}")
    print(f"     Subject:   {cluster.subject}")
    print(f"     Predicate: {cluster.predicate}")
    print(f"     Case Type: {cluster.case_type.value}")
    print(f"     Documents: {cluster.doc_count}")
    print(f"     Edges:     {len(cluster.edges)}")
    if cluster.consensus_value is not None:
        print(f"     Consensus:  {cluster.consensus_value:,.2f} {cluster.consensus_unit or ''}")
    print(f"\n     Evidence:")
    for ev in cluster.evidence:
        print(f"       • {ev.doc_filename[:40]} (p.{ev.page})")
        print(f"         Value: {ev.value}")
        print(f"         Quote: \"{ev.evidence_quote[:80]}...\"")
        print(f"         Temporal: {ev.temporal or 'N/A'}, Scope: {ev.scope or 'N/A'}")
        print(f"         Confidence: {ev.confidence:.2f} ({ev.match_type})")
    if cluster.explanation:
        print(f"\n     Explanation: {cluster.explanation}")


def main():
    t_start = time.time()

    # Override with CLI args if provided
    docs_to_test = DELHIVERY_DOCS
    if len(sys.argv) > 1:
        # Single PDF mode: python e2e_claim_graph.py <pdf1> <pdf2> ...
        docs_to_test = [{"path": p, "pages": None, "label": os.path.basename(p)} for p in sys.argv[1:]]

    # -----------------------------------------------------------------------
    # Step 1: Parse all PDFs
    # -----------------------------------------------------------------------
    print_header("Step 1: Parsing PDFs")
    parsed_docs = {}
    for doc_cfg in docs_to_test:
        path = doc_cfg["path"]
        if not os.path.exists(path):
            print(f"  ⚠ Skipping: {path} (not found)")
            continue
        doc = parse_pdf(path)
        parsed_docs[doc.doc_id] = {"doc": doc, "config": doc_cfg}
        print(f"  ✅ {doc_cfg['label']}: {doc.page_count} pages (doc_id={doc.doc_id})")

    if not parsed_docs:
        print("\n❌ No PDFs found. Exiting.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2: Extract facts from each document
    # -----------------------------------------------------------------------
    print_header("Step 2: Extracting Facts")
    extractor = GeminiFactExtractor()
    doc_facts = {}
    doc_filenames = {}

    for doc_id, info in parsed_docs.items():
        doc = info["doc"]
        cfg = info["config"]
        test_pages = cfg.get("pages")

        doc_filenames[doc_id] = doc.filename
        all_doc_facts = []

        if test_pages:
            for page_num in test_pages:
                try:
                    result = extractor.extract_and_verify(doc, page_num=page_num)
                    all_doc_facts.extend(result.facts)
                    print(f"  ✅ {cfg['label']} p.{page_num}: {len(result.facts)} facts")
                except Exception as e:
                    print(f"  ⚠ {cfg['label']} p.{page_num}: extraction failed — {e}")
        else:
            # Extract from all pages (for CLI mode)
            try:
                result = extractor.extract_and_verify(doc)
                all_doc_facts.extend(result.facts)
                print(f"  ✅ {cfg['label']}: {len(result.facts)} facts (all pages)")
            except Exception as e:
                print(f"  ⚠ {cfg['label']}: extraction failed — {e}")

        doc_facts[doc_id] = all_doc_facts

    total_facts = sum(len(f) for f in doc_facts.values())
    print(f"\n  📊 Total facts across {len(doc_facts)} docs: {total_facts}")

    if total_facts == 0:
        print("\n❌ No facts extracted. Cannot build claim graph.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3: Build Claim Graph
    # -----------------------------------------------------------------------
    print_header("Step 3: Building Claim Graph (ArbGraph Pipeline)")
    graph = build_claim_graph(
        doc_facts=doc_facts,
        doc_filenames=doc_filenames,
        nli_threshold=0.7,
    )

    print(f"\n  📊 Claim Graph Summary:")
    print(f"     Total facts:      {graph.total_facts}")
    print(f"     Multi-fact clusters: {len(graph.clusters)}")
    print(f"     Singletons:       {len(graph.unmatched_facts)}")
    print(f"     Extraction failures: {len(graph.extraction_failures)}")
    print(f"     Duration:         {graph.pipeline_metadata.get('duration_seconds', 0):.2f}s")
    print(f"\n     Case Distribution:")
    for case, count in sorted(graph.case_summary.items()):
        print(f"       {case}: {count}")

    # -----------------------------------------------------------------------
    # Step 4: Display top clusters by type
    # -----------------------------------------------------------------------
    print_header("Step 4: Top Clusters")

    # Show up to 3 of each type
    from collections import defaultdict
    by_type = defaultdict(list)
    for c in graph.clusters:
        by_type[c.case_type.value].append(c)

    for case_type, clusters in sorted(by_type.items()):
        print(f"\n  ── {case_type.upper()} ({len(clusters)} clusters) ──")
        for c in clusters[:2]:  # show 2 per type
            print_cluster(c)

    # -----------------------------------------------------------------------
    # Step 5: The 4 Assignment Cases
    # -----------------------------------------------------------------------
    print_header("Step 5: The 4 Required Assignment Cases")
    cases = get_assignment_cases(graph)

    for case_key, case_data in cases.items():
        print(f"\n  {'─'*60}")
        print(f"  {case_key.upper()}")
        print(f"  {'─'*60}")
        if case_data is None:
            print("    (No example found for this case type)")
        else:
            print(f"    {json.dumps(case_data, indent=4, default=str)[:800]}")

    # -----------------------------------------------------------------------
    # Step 6: Extraction Failures (Case 4 details)
    # -----------------------------------------------------------------------
    if graph.extraction_failures:
        print_header("Step 6: Extraction Failures (Case 4)")
        for i, fail in enumerate(graph.extraction_failures[:5]):
            print(f"\n  [{i}] Type: {fail.failure_type}")
            print(f"      Doc: {fail.doc_id}, Page: {fail.page}")
            print(f"      Description: {fail.description[:120]}")
            print(f"      Mitigation: {fail.mitigation[:120]}")

    # -----------------------------------------------------------------------
    # Save full results
    # -----------------------------------------------------------------------
    duration = time.time() - t_start
    output = {
        "pipeline": "ArbGraph Claim Graph",
        "documents": doc_filenames,
        "total_facts": graph.total_facts,
        "cluster_count": len(graph.clusters),
        "singleton_count": len(graph.unmatched_facts),
        "extraction_failures": len(graph.extraction_failures),
        "case_summary": graph.case_summary,
        "duration_seconds": round(duration, 2),
        "claim_graph": graph.model_dump(mode="json"),
        "assignment_cases": cases,
    }

    output_path = "e2e_claim_graph_results.json"
    with open(output_path, "w") as fp:
        json.dump(output, fp, indent=2, default=str)

    print_header("FINAL SUMMARY")
    print(f"  Documents:         {len(doc_filenames)}")
    print(f"  Total facts:       {graph.total_facts}")
    print(f"  Clusters:          {len(graph.clusters)}")
    print(f"  Contradictions:    {graph.case_summary.get('contradicted', 0)}")
    print(f"  Corroborations:    {graph.case_summary.get('corroborated', 0)}")
    print(f"  Reconciled:        {sum(v for k, v in graph.case_summary.items() if 'reconciled' in k)}")
    print(f"  Failures:          {graph.case_summary.get('extraction_failure', 0)}")
    print(f"  Total time:        {duration:.1f}s")
    print(f"\n  📄 Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
