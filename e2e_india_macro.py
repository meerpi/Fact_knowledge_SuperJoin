"""End-to-end test and validation for India Macroeconomy Dataset.

Documents:
1. 01-india-economic-survey-2024-25-excerpt.pdf (Economic Survey 2024-25)
2. 02-rbi-annual-report-2024-25-excerpt.pdf (RBI Annual Report 2024-25)
3. 03-imf-india-2025-article-iv-excerpt.pdf (IMF India 2025 Article IV)

Runs full 4-stage pipeline:
  Parse -> Extract & Verify Quotes -> Claim Graph (UnionFind + Credibility + NLI) -> 4 Assignment Cases
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from app.models import Fact
from app.pdf_parser import parse_pdf
from app.extractor import GeminiFactExtractor
from app.claim_graph import build_claim_graph, get_assignment_cases
from app.normalizer import standardize_predicate

CACHE_FILE = "india_macro_facts_cache.json"

INDIA_MACRO_DOCS = [
    {
        "path": "starter-datasets/india-macroeconomy/01-india-economic-survey-2024-25-excerpt.pdf",
        "pages": [19, 20, 27, 29, 35],
        "label": "India Economic Survey 2024-25",
    },
    {
        "path": "starter-datasets/india-macroeconomy/02-rbi-annual-report-2024-25-excerpt.pdf",
        "pages": [10, 16, 22, 23, 24, 37],
        "label": "RBI Annual Report 2024-25",
    },
    {
        "path": "starter-datasets/india-macroeconomy/03-imf-india-2025-article-iv-excerpt.pdf",
        "pages": [4, 9, 12],
        "label": "IMF India 2025 Article IV",
    },
]


def load_page_cache() -> dict[str, list[dict]]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_page_cache(cache: dict[str, list[dict]]):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def print_header(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_cluster(cluster):
    print(f"\n  📌 Cluster: {cluster.cluster_id}")
    print(f"     Subject:     {cluster.subject}")
    print(f"     Predicate:   {cluster.predicate}")
    print(f"     Case Type:   {cluster.case_type.value}")
    print(f"     Documents:   {cluster.doc_count}")
    print(f"     Edges:       {len(cluster.edges)}")
    if cluster.consensus_value is not None:
        print(f"     Consensus:   {cluster.consensus_value:,.2f} {cluster.consensus_unit or ''}")
    print(f"\n     Evidence ({len(cluster.evidence)} facts):")
    for i, ev in enumerate(cluster.evidence):
        fact_idx = cluster.fact_indices[i] if i < len(cluster.fact_indices) else None
        cred = cluster.credibility_scores.get(fact_idx, 0.0) if fact_idx is not None else 0.0
        print(f"       • [{ev.doc_filename}] p.{ev.page}: {ev.value} (canonical={ev.canonical_value} {ev.canonical_unit or ''})")
        print(f"         Temporal: {ev.temporal or 'N/A'}, Scope: {ev.scope or 'N/A'}, Conditions: {ev.conditions or 'N/A'}")
        print(f"         Quote: \"{ev.evidence_quote[:100]}...\"")
        print(f"         Confidence: {ev.confidence:.2f} ({ev.match_type}) | Credibility: {cred:.3f}")
    if cluster.edges:
        print(f"\n     Graph Edges:")
        for edge in cluster.edges:
            print(f"       -> {edge.edge_type.value.upper()} ({edge.detection_method}, conf={edge.confidence:.2f}): {edge.explanation}")
    if cluster.explanation:
        print(f"\n     Explanation: {cluster.explanation}")


def main():
    t_start = time.time()

    print_header("Step 1: Parsing India Macroeconomy PDFs")
    parsed_docs = {}
    for doc_cfg in INDIA_MACRO_DOCS:
        path = doc_cfg["path"]
        if not os.path.exists(path):
            print(f"  ⚠ Missing: {path}")
            continue
        doc = parse_pdf(path)
        parsed_docs[doc.doc_id] = {"doc": doc, "config": doc_cfg}
        print(f"  ✅ {doc_cfg['label']}: {doc.page_count} pages (doc_id={doc.doc_id})")

    if not parsed_docs:
        print("\n❌ No documents found. Exiting.")
        sys.exit(1)

    print_header("Step 2: Extracting Facts via Gemini Cascade with Quote Grounding")
    extractor = GeminiFactExtractor()
    page_cache = load_page_cache()
    doc_facts = {}
    doc_filenames = {}

    for doc_id, info in parsed_docs.items():
        doc = info["doc"]
        cfg = info["config"]
        test_pages = cfg.get("pages", [])
        doc_filenames[doc_id] = doc.filename
        all_facts = []

        print(f"\n  Processing {cfg['label']} (pages: {test_pages})...")
        for p_idx in test_pages:
            cache_key = f"{doc_id}:{p_idx}"
            if cache_key in page_cache:
                facts = [Fact.model_validate(f) for f in page_cache[cache_key]]
                for f in facts:
                    std_pred, cond = standardize_predicate(f.predicate)
                    f.predicate = std_pred
                    if cond and not f.context.conditions:
                        f.context.conditions = cond
                all_facts.extend(facts)
                print(f"    p.{p_idx}: {len(facts)} facts (from cache)")
                continue

            try:
                # Slight pause to respect API rate limits
                time.sleep(2.0)
                res = extractor.extract_and_verify(doc, page_num=p_idx)
                all_facts.extend(res.facts)
                page_cache[cache_key] = [f.model_dump(mode="json") for f in res.facts]
                save_page_cache(page_cache)
                print(f"    p.{p_idx}: {len(res.facts)} facts (model={res.model_used})")
            except Exception as e:
                print(f"    ⚠ p.{p_idx} failed: {e}")

        doc_facts[doc_id] = all_facts
        print(f"  Total for {cfg['label']}: {len(all_facts)} facts")

    total_facts = sum(len(f) for f in doc_facts.values())
    print(f"\n  📊 Total facts extracted across 3 documents: {total_facts}")

    if total_facts == 0:
        print("\n❌ No facts extracted. Exiting.")
        sys.exit(1)

    print_header("Step 3: Building Cross-Document Claim Graph (ArbGraph Pipeline)")
    graph = build_claim_graph(
        doc_facts=doc_facts,
        doc_filenames=doc_filenames,
        nli_threshold=0.7,
    )

    print(f"\n  📊 Claim Graph Summary:")
    print(f"     Total facts:            {graph.total_facts}")
    print(f"     Multi-fact clusters:    {len(graph.clusters)}")
    print(f"     Singletons (unmatched): {len(graph.unmatched_facts)}")
    print(f"     Extraction failures:    {len(graph.extraction_failures)}")
    print(f"     Pipeline duration:      {graph.pipeline_metadata.get('duration_seconds', 0):.2f}s")
    print(f"\n     Case Distribution:")
    for case, count in sorted(graph.case_summary.items()):
        print(f"       {case:25}: {count}")

    print_header("Step 4: Fact Clusters Detailed Inspection")
    for cluster in graph.clusters:
        print_cluster(cluster)

    print_header("Step 5: The 4 Required Assignment Cases")
    cases = get_assignment_cases(graph)

    for case_key, case_data in cases.items():
        print(f"\n  {'─'*70}")
        print(f"  CASE: {case_key.upper()}")
        print(f"  {'─'*70}")
        if case_data is None:
            print("    (No cluster matched this case type)")
        elif case_key == "case_4_extraction_failure":
            print(f"    Failure Type: {case_data.get('failure_type')}")
            print(f"    Doc ID:       {case_data.get('doc_id')}, Page: {case_data.get('page')}")
            print(f"    Description:  {case_data.get('description')}")
            print(f"    Mitigation:   {case_data.get('mitigation')}")
        else:
            print(f"    Cluster ID:   {case_data.get('cluster_id')}")
            print(f"    Subject:      {case_data.get('subject')}")
            print(f"    Predicate:    {case_data.get('predicate')}")
            print(f"    Case Type:    {case_data.get('case_type')}")
            print(f"    Explanation:  {case_data.get('explanation')}")
            print(f"    Evidence Count: {len(case_data.get('evidence', []))}")
            for ev in case_data.get("evidence", []):
                print(f"      • [{ev.get('doc_filename')}] p.{ev.get('page')}: {ev.get('value')} (temporal={ev.get('temporal')}, scope={ev.get('scope')}, conditions={ev.get('conditions')})")

    if graph.extraction_failures:
        print_header("Step 6: Extraction Failures (Case 4 Details)")
        for i, fail in enumerate(graph.extraction_failures[:5]):
            print(f"\n  [{i+1}] Type: {fail.failure_type}")
            print(f"      Doc: {fail.doc_id}, Page: {fail.page}")
            print(f"      Description: {fail.description}")
            print(f"      Mitigation:  {fail.mitigation}")

    # Persist JSON results
    output_path = "e2e_india_macro_results.json"
    with open(output_path, "w") as fp:
        json.dump(graph.model_dump(mode="json"), fp, indent=2, default=str)
    print(f"\n  💾 Complete JSON results saved to: {output_path}")

    # Also save assignment cases
    cases_output_path = "e2e_india_macro_cases.json"
    with open(cases_output_path, "w") as fp:
        json.dump(cases, fp, indent=2, default=str)
    print(f"  💾 Assignment cases saved to: {cases_output_path}")

    duration = time.time() - t_start
    print_header("FINAL VERIFICATION SUMMARY")
    print(f"  Duration:          {duration:.1f}s")
    print(f"  Corroborated:      {graph.case_summary.get('corroborated', 0)}")
    print(f"  Contradicted:      {graph.case_summary.get('contradicted', 0)}")
    print(f"  Reconciled:        {sum(v for k, v in graph.case_summary.items() if 'reconciled' in k)}")
    print(f"  Failures (Case 4): {len(graph.extraction_failures)}")


if __name__ == "__main__":
    main()
