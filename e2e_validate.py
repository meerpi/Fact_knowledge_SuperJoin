"""End-to-end validation: Delhivery PDF → Parse → Extract → Canonicalize → Contradictions.

Run directly: .venv/bin/python e2e_validate.py
No pytest dependency. Uses the actual starter dataset.
"""

import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from app.pdf_parser import parse_pdf
from app.extractor import GeminiFactExtractor
from app.contradiction import detect_contradictions

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else "starter-datasets/delhivery/01-delhivery-prospectus-2022-excerpt.pdf"

# Pages with rich financial data (default or from CLI: python e2e_validate.py <pdf> 26,70)
TEST_PAGES = [int(p.strip()) for p in sys.argv[2].split(",")] if len(sys.argv) > 2 else [16, 44]



def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_fact_table(facts, title="Extracted Facts"):
    """Print facts in a human-readable table for manual verification."""
    print_header(title)
    print(f"{'#':>3} | {'Subject':<25} | {'Predicate':<25} | {'Value':<20} | {'BUnit':<10} | {'Scl':>3} | {'Dim':<10} | {'Canon.Value':>18} | {'Temporal':<20} | {'Conf':>5} | {'Match':<10}")
    print("-" * 200)
    for i, f in enumerate(facts):
        canon_v = f"{f.canonical_value:>18,.2f}" if f.canonical_value is not None else f"{'N/A':>18}"
        temp = (f.context.temporal or "")[:20]
        subj = f.subject[:25]
        pred = f.predicate[:25]
        val = f.value[:20]
        bunit = (f.base_unit or "")[:10]
        scl = f"{f.scale:>3}"
        dim = (f.dimension.value if f.dimension else "")[:10]
        conf = f"{f.confidence:.2f}"
        match = f.provenance.match_type[:10]
        print(f"{i:>3} | {subj:<25} | {pred:<25} | {val:<20} | {bunit:<10} | {scl} | {dim:<10} | {canon_v} | {temp:<20} | {conf:>5} | {match:<10}")


def verify_canonicalization(facts):
    """Check that iXBRL-pattern canonicalization is working correctly."""
    print_header("Canonicalization Verification (iXBRL-pattern)")

    issues = []
    verified = 0
    scale_stats = {"correct": 0, "zero_when_expected": 0, "total_monetary": 0}

    for i, f in enumerate(facts):
        # Check 1: Numeric facts should have canonical_value
        if f.numeric_value is not None and f.canonical_value is None:
            issues.append(f"  ISSUE #{i}: Has numeric_value={f.numeric_value} but canonical_value is None")

        # Check 2: Monetary facts should have scale > 0 when table says 'in million/crore'
        if f.dimension and f.dimension.value == "monetary":
            scale_stats["total_monetary"] += 1
            if f.scale > 0:
                scale_stats["correct"] += 1
            else:
                scale_stats["zero_when_expected"] += 1
                issues.append(
                    f"  ISSUE #{i}: Monetary fact with scale=0 — LLM may have missed table header scale. "
                    f"base_unit='{f.base_unit}', value='{f.value}'"
                )

        # Check 3: base_unit should NOT contain scale words
        if f.base_unit:
            scale_words = ["million", "mn", "billion", "bn", "crore", "cr", "lakh", "thousand"]
            found_scale_in_unit = [sw for sw in scale_words if sw in f.base_unit.lower()]
            if found_scale_in_unit:
                issues.append(
                    f"  ISSUE #{i}: base_unit '{f.base_unit}' contains scale word(s) {found_scale_in_unit} "
                    f"— LLM should have put this in scale field instead"
                )

        # Check 4: Dimension should be set for numeric facts
        if f.numeric_value is not None and f.dimension is None:
            issues.append(f"  ISSUE #{i}: Numeric fact missing dimension classification")

        # Check 5: Currency should be in base_unit for monetary facts
        if f.canonical_value is not None:
            has_currency_in_value = any(sym in f.value for sym in ["₹", "$", "€", "£"])
            has_currency_in_unit = f.base_unit in ("INR", "USD", "EUR", "GBP")
            if has_currency_in_value and not has_currency_in_unit:
                issues.append(
                    f"  ISSUE #{i}: Value '{f.value}' has currency symbol but "
                    f"base_unit='{f.base_unit}' — currency not extracted?"
                )

        if f.canonical_value is not None:
            verified += 1

    if issues:
        print(f"\n⚠ {len(issues)} potential issue(s) found:")
        for issue in issues:
            print(issue)
    else:
        print("\n✅ No canonicalization issues detected.")

    print(f"\n📊 Stats:")
    print(f"   Facts with canonical_value:  {verified}/{len(facts)}")
    print(f"   Monetary facts:              {scale_stats['total_monetary']}")
    print(f"   Scale correctly classified:  {scale_stats['correct']}")
    if scale_stats['total_monetary'] > 0:
        acc = scale_stats['correct'] / scale_stats['total_monetary'] * 100
        print(f"   Scale accuracy:              {acc:.0f}%")

    return len(issues) == 0


def verify_contradiction_detection(facts, doc_id):
    """Run contradiction detection and display results."""
    print_header("Contradiction Detection (Stages 1 + 2)")

    report = detect_contradictions(facts=facts, doc_id=doc_id, nli_threshold=0.7)

    print(f"\n📊 Pipeline Summary:")
    print(f"   Total facts:              {report.total_facts}")
    print(f"   Candidate pairs evaluated: {report.candidate_pairs_evaluated}")
    print(f"   Edges found:              {len(report.edges)}")
    print(f"   Summary:                  {report.summary}")

    if report.edges:
        print(f"\n{'#':>3} | {'Type':<14} | {'Method':<18} | {'Conf':>5} | {'Explanation'}")
        print("-" * 120)
        for i, edge in enumerate(report.edges):
            src = facts[edge.source_fact_idx]
            tgt = facts[edge.target_fact_idx]
            print(f"{i:>3} | {edge.edge_type.value:<14} | {edge.detection_method:<18} | {edge.confidence:.2f}  | {edge.explanation[:80]}")
            print(f"    |   Fact A: [{edge.source_fact_idx}] {src.subject} / {src.predicate} = {src.value} ({src.context.temporal})")
            print(f"    |   Fact B: [{edge.target_fact_idx}] {tgt.subject} / {tgt.predicate} = {tgt.value} ({tgt.context.temporal})")
            print()
    else:
        print("\n   No edges (contradictions/corroborations/supersedes) detected.")

    # Verify no false positives: operating_profit vs net_profit should never be paired
    for edge in report.edges:
        src_pred = facts[edge.source_fact_idx].predicate
        tgt_pred = facts[edge.target_fact_idx].predicate
        if src_pred != tgt_pred:
            print(f"  ⚠ CROSS-PREDICATE EDGE: {src_pred} vs {tgt_pred} — possible false positive!")

    return report


def main():
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found at {PDF_PATH}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 1: Parse PDF
    # -----------------------------------------------------------------------
    print_header("Step 1: Parsing PDF")
    doc = parse_pdf(PDF_PATH)
    print(f"✅ Parsed '{doc.filename}': {doc.page_count} pages, doc_id={doc.doc_id}")

    # -----------------------------------------------------------------------
    # Step 2: Extract facts from specific pages
    # -----------------------------------------------------------------------
    extractor = GeminiFactExtractor()
    all_facts = []

    for page_num in TEST_PAGES:
        print_header(f"Step 2: Extracting Facts from Page {page_num}")

        try:
            result = extractor.extract_and_verify(doc, page_num=page_num)
            print(f"✅ Extracted {len(result.facts)} facts using model: {result.model_used}")
            print(f"   Fallback attempts: {result.fallback_attempts}")

            # Print the full fact table
            print_fact_table(result.facts, title=f"Page {page_num}: Fact Table")

            all_facts.extend(result.facts)

        except Exception as e:
            print(f"❌ Extraction failed for page {page_num}: {e}")
            continue

    if not all_facts:
        print("\n❌ No facts extracted. Cannot continue.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3: Verify canonicalization
    # -----------------------------------------------------------------------
    canon_ok = verify_canonicalization(all_facts)

    # -----------------------------------------------------------------------
    # Step 4: Run contradiction detection
    # -----------------------------------------------------------------------
    report = verify_contradiction_detection(all_facts, doc.doc_id)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_header("FINAL SUMMARY")
    print(f"  PDF:                    {doc.filename}")
    print(f"  Pages tested:           {TEST_PAGES}")
    print(f"  Total facts extracted:  {len(all_facts)}")
    print(f"  Canonicalization:       {'✅ PASS' if canon_ok else '⚠ ISSUES FOUND'}")
    print(f"  Contradiction edges:    {len(report.edges)}")
    print(f"    CONTRADICTS:          {report.summary.get('contradicts', 0)}")
    print(f"    CORROBORATES:         {report.summary.get('corroborates', 0)}")
    print(f"    SUPERSEDES:           {report.summary.get('supersedes', 0)}")
    print()

    # Dump full JSON for inspection
    output_path = "e2e_results.json"
    output = {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "pages_tested": TEST_PAGES,
        "total_facts": len(all_facts),
        "facts": [f.model_dump(mode="json") for f in all_facts],
        "contradiction_report": report.model_dump(mode="json"),
    }
    with open(output_path, "w") as fp:
        json.dump(output, fp, indent=2, default=str)
    print(f"📄 Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
