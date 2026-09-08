"""Tests for the ArbGraph-inspired claim graph engine."""

import pytest

from app.models import (
    CaseType,
    ClaimEdge,
    Context,
    Dimension,
    EdgeType,
    Fact,
    Provenance,
)
from app.claim_graph import (
    DEFAULT_DOC_AUTHORITY,
    _UnionFind,
    _align_facts,
    _classify_cluster,
    _detect_extraction_failures,
    _estimate_doc_authority,
    _normalize_for_lookup,
    _normalize_subject,
    _propagate_credibility,
    _subjects_match,
    append_document_to_claim_graph,
    build_claim_graph,
    get_assignment_cases,
)


# ---------------------------------------------------------------------------
# Helpers — factory functions for test facts
# ---------------------------------------------------------------------------

def _fact(
    subject: str,
    predicate: str,
    value: str,
    numeric: float | None = None,
    canonical: float | None = None,
    unit: str | None = None,
    temporal: str | None = None,
    scope: str | None = None,
    conditions: str | None = None,
    doc_id: str = "doc1",
    page: int = 1,
    quote: str = "test quote",
    verified: bool = True,
    confidence: float = 0.90,
    dimension: Dimension | None = None,
    scale: int = 0,
    base_unit: str | None = None,
) -> Fact:
    return Fact(
        subject=subject,
        predicate=predicate,
        value=value,
        numeric_value=numeric,
        canonical_value=canonical,
        canonical_unit=unit,
        base_unit=base_unit,
        scale=scale,
        dimension=dimension,
        context=Context(temporal=temporal, scope=scope, conditions=conditions),
        confidence=confidence,
        provenance=Provenance(
            doc_id=doc_id,
            page=page,
            evidence_quote=quote,
            verified=verified,
            match_type="exact" if verified else "unverified",
        ),
    )


# ---------------------------------------------------------------------------
# Union-Find tests
# ---------------------------------------------------------------------------

class TestUnionFind:
    def test_single_elements(self):
        uf = _UnionFind()
        uf.find(0)
        uf.find(1)
        uf.find(2)
        clusters = uf.clusters()
        assert len(clusters) == 3

    def test_union_creates_cluster(self):
        uf = _UnionFind()
        uf.find(0)
        uf.find(1)
        uf.find(2)
        uf.union(0, 1)
        clusters = uf.clusters()
        assert len(clusters) == 2
        # One cluster has 2 elements, the other has 1
        sizes = sorted(len(v) for v in clusters.values())
        assert sizes == [1, 2]

    def test_transitive_union(self):
        uf = _UnionFind()
        for i in range(5):
            uf.find(i)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(3, 4)
        clusters = uf.clusters()
        assert len(clusters) == 2
        sizes = sorted(len(v) for v in clusters.values())
        assert sizes == [2, 3]


# ---------------------------------------------------------------------------
# Subject normalization tests
# ---------------------------------------------------------------------------

class TestSubjectNormalization:
    def test_strip_legal_suffixes(self):
        assert _normalize_subject("Delhivery Limited") == "delhivery"
        assert _normalize_subject("Apple Inc.") == "apple"
        assert _normalize_subject("Tesla Corp") == "tesla"

    def test_case_insensitive(self):
        assert _normalize_subject("DELHIVERY") == "delhivery"

    def test_possessive(self):
        assert _normalize_subject("Delhivery's") == "delhivery"

    def test_match_with_suffix(self):
        assert _subjects_match("Delhivery", "Delhivery Limited")
        assert _subjects_match("Apple Inc.", "Apple")

    def test_no_match_different_entities(self):
        assert not _subjects_match("Delhivery", "Spoton")


# ---------------------------------------------------------------------------
# Claim alignment tests
# ---------------------------------------------------------------------------

class TestClaimAlignment:
    def test_exact_match_groups(self):
        facts = [
            _fact("Delhivery", "revenue", "1000", doc_id="doc1"),
            _fact("Delhivery", "revenue", "1000", doc_id="doc2"),
            _fact("Spoton", "revenue", "500", doc_id="doc1"),
        ]
        clusters = _align_facts(facts)
        # Delhivery-revenue cluster + Spoton-revenue cluster
        assert len(clusters) == 2

    def test_fuzzy_subject_alignment(self):
        facts = [
            _fact("Delhivery", "revenue", "1000", doc_id="doc1"),
            _fact("Delhivery Limited", "revenue", "1050", doc_id="doc2"),
        ]
        clusters = _align_facts(facts)
        # Should merge into 1 cluster
        assert len(clusters) == 1

    def test_different_predicates_separate(self):
        facts = [
            _fact("Delhivery", "revenue", "1000"),
            _fact("Delhivery", "net_profit", "-500"),
        ]
        clusters = _align_facts(facts)
        assert len(clusters) == 2


# ---------------------------------------------------------------------------
# Cluster classification tests
# ---------------------------------------------------------------------------

class TestClusterClassification:
    def test_single_fact_corroborated(self):
        facts = [_fact("Delhivery", "revenue", "1000")]
        case, dispute = _classify_cluster(facts, [0], [])
        assert case == CaseType.CORROBORATED
        assert dispute == "AGREEMENT_EXACT"

    def test_corroborating_edges(self):
        facts = [
            _fact("Delhivery", "revenue", "1000", doc_id="doc1"),
            _fact("Delhivery", "revenue", "1000", doc_id="doc2"),
        ]
        edges = [ClaimEdge(
            source_fact_idx=0, target_fact_idx=1,
            edge_type=EdgeType.CORROBORATES,
            detection_method="symbolic_numeric",
            confidence=1.0,
            explanation="Values match",
        )]
        case, dispute = _classify_cluster(facts, [0, 1], edges)
        assert case == CaseType.CORROBORATED
        assert dispute == "AGREEMENT_EXACT"

    def test_contradiction_different_temporal(self):
        facts = [
            _fact("Delhivery", "revenue", "1000", temporal="FY2021", doc_id="doc1"),
            _fact("Delhivery", "revenue", "1500", temporal="FY2022", doc_id="doc2"),
        ]
        edges = [ClaimEdge(
            source_fact_idx=0, target_fact_idx=1,
            edge_type=EdgeType.CONTRADICTS,
            detection_method="symbolic_numeric",
            confidence=1.0,
            explanation="Values differ",
        )]
        case, dispute = _classify_cluster(facts, [0, 1], edges)
        assert case == CaseType.RECONCILED_TEMPORAL
        assert dispute == "DISPUTE_TEMPORAL_DRIFT"

    def test_contradiction_different_scope(self):
        facts = [
            _fact("Delhivery", "revenue", "1000", scope="Consolidated", doc_id="doc1"),
            _fact("Delhivery", "revenue", "800", scope="Standalone", doc_id="doc2"),
        ]
        edges = [ClaimEdge(
            source_fact_idx=0, target_fact_idx=1,
            edge_type=EdgeType.CONTRADICTS,
            detection_method="symbolic_numeric",
            confidence=1.0,
            explanation="Values differ",
        )]
        case, dispute = _classify_cluster(facts, [0, 1], edges)
        assert case == CaseType.RECONCILED_SCOPE
        assert dispute == "DISPUTE_SCOPE_DIFFERENCE"

    def test_genuine_contradiction(self):
        facts = [
            _fact("Delhivery", "revenue", "1000", temporal="FY2021", scope="Consolidated", doc_id="doc1"),
            _fact("Delhivery", "revenue", "1500", temporal="FY2021", scope="Consolidated", doc_id="doc2"),
        ]
        edges = [ClaimEdge(
            source_fact_idx=0, target_fact_idx=1,
            edge_type=EdgeType.CONTRADICTS,
            detection_method="symbolic_numeric",
            confidence=1.0,
            explanation="Values differ",
        )]
        case, dispute = _classify_cluster(facts, [0, 1], edges)
        assert case == CaseType.CONTRADICTED
        assert dispute == "DISPUTE_GENUINE_CONFLICT"

    def test_supersedes_is_temporal_reconciliation(self):
        facts = [
            _fact("Delhivery", "revenue", "1000", temporal="FY2021"),
            _fact("Delhivery", "revenue", "1500", temporal="FY2022"),
        ]
        edges = [ClaimEdge(
            source_fact_idx=0, target_fact_idx=1,
            edge_type=EdgeType.SUPERSEDES,
            detection_method="temporal",
            confidence=1.0,
            explanation="FY2022 supersedes FY2021",
        )]
        case, dispute = _classify_cluster(facts, [0, 1], edges)
        assert case == CaseType.RECONCILED_TEMPORAL
        assert dispute == "DISPUTE_TEMPORAL_DRIFT"


# ---------------------------------------------------------------------------
# Credibility propagation tests
# ---------------------------------------------------------------------------

class TestCredibilityPropagation:
    def test_corroboration_boosts_scores(self):
        facts = [
            _fact("X", "rev", "100", confidence=0.8, doc_id="doc1"),
            _fact("X", "rev", "100", confidence=0.8, doc_id="doc2"),
        ]
        edges = [ClaimEdge(
            source_fact_idx=0, target_fact_idx=1,
            edge_type=EdgeType.CORROBORATES,
            detection_method="test", confidence=1.0, explanation="",
        )]
        scores = _propagate_credibility(
            facts, [0, 1], edges, {"doc1": "report.pdf", "doc2": "report2.pdf"}
        )
        # Both should be ≥ initial (0.8 × 0.75 default authority = 0.60)
        assert scores[0] >= 0.60
        assert scores[1] >= 0.60

    def test_contradiction_suppresses_less_credible(self):
        facts = [
            _fact("X", "rev", "100", confidence=0.95, doc_id="doc1"),  # high conf
            _fact("X", "rev", "200", confidence=0.30, doc_id="doc2"),  # low conf
        ]
        edges = [ClaimEdge(
            source_fact_idx=0, target_fact_idx=1,
            edge_type=EdgeType.CONTRADICTS,
            detection_method="test", confidence=1.0, explanation="",
        )]
        scores = _propagate_credibility(
            facts, [0, 1], edges, {"doc1": "annual report.pdf", "doc2": "notes.pdf"}
        )
        # High-confidence fact from authoritative doc should be more credible
        assert scores[0] > scores[1]


# ---------------------------------------------------------------------------
# Document authority estimation & override tests
# ---------------------------------------------------------------------------

class TestDocAuthority:
    """Tests for the document authority estimation and override system."""

    # --- Normalized keyword matching ---

    def test_normalize_hyphens_underscores_spaces(self):
        """All separator variants normalize to the same form."""
        assert _normalize_for_lookup("annual-report") == "annual report"
        assert _normalize_for_lookup("annual_report") == "annual report"
        assert _normalize_for_lookup("annual report") == "annual report"
        assert _normalize_for_lookup("annual--report") == "annual report"
        assert _normalize_for_lookup("ANNUAL_REPORT") == "annual report"

    def test_estimate_matches_hyphenated_filenames(self):
        """Filenames with hyphens match space-keyed authority entries."""
        # This was the bug: "economic-survey" didn't match "economic survey"
        score = _estimate_doc_authority("01-india-economic-survey-2024.pdf")
        assert score == 0.90
        score = _estimate_doc_authority("02-rbi-annual-report-2024.pdf")
        assert score == 0.95

    def test_estimate_matches_underscored_filenames(self):
        """Filenames with underscores match space-keyed authority entries."""
        score = _estimate_doc_authority("delhivery_annual_report_fy24.pdf")
        assert score == 0.95
        score = _estimate_doc_authority("tesla_10_k_2024.pdf")
        assert score == 0.95

    def test_estimate_returns_default_for_unknown(self):
        """Non-matching filenames get the DEFAULT_DOC_AUTHORITY."""
        score = _estimate_doc_authority("random-document.pdf")
        assert score == DEFAULT_DOC_AUTHORITY
        score = _estimate_doc_authority("clinical-trial-results.pdf")
        assert score == DEFAULT_DOC_AUTHORITY

    def test_estimate_generic_financial_types(self):
        """Generic financial document types are recognized regardless of domain."""
        # These should work for any company, not just demo datasets
        assert _estimate_doc_authority("walmart-10-k-2024.pdf") == 0.95
        assert _estimate_doc_authority("apple-quarterly-report-q3.pdf") == 0.85
        assert _estimate_doc_authority("company-press-release.pdf") == 0.70
        assert _estimate_doc_authority("acme-prospectus-2024.pdf") == 0.90

    # --- Caller-supplied overrides ---

    def test_override_changes_propagation_scores(self):
        """doc_authority_overrides take priority over filename heuristic."""
        facts = [
            _fact("X", "rev", "100", confidence=0.95, doc_id="doc1"),
            _fact("X", "rev", "100", confidence=0.95, doc_id="doc2"),
        ]
        # No edges: test pure initial score assignment (no boost/penalty)
        edges = []
        # Without overrides: both "unknown.pdf" → DEFAULT_DOC_AUTHORITY
        scores_no_override = _propagate_credibility(
            facts, [0, 1], edges,
            {"doc1": "unknown.pdf", "doc2": "unknown.pdf"},
        )
        # With overrides: doc1 is an SEC filing (0.95), doc2 is a blog (0.60)
        scores_with_override = _propagate_credibility(
            facts, [0, 1], edges,
            {"doc1": "unknown.pdf", "doc2": "unknown.pdf"},
            doc_authority_overrides={"doc1": 0.95, "doc2": 0.60},
        )
        # Override should produce different starting scores
        assert scores_with_override[0] != scores_with_override[1]
        # doc1 (0.95 authority) should be higher than doc2 (0.60 authority)
        assert scores_with_override[0] > scores_with_override[1]
        # Without override, both start identical (same confidence × same authority)
        assert abs(scores_no_override[0] - scores_no_override[1]) < 0.01

    def test_override_flips_consensus_winner(self):
        """Overrides can change which fact wins a contradiction arbitration."""
        facts = [
            _fact("X", "rev", "100", confidence=0.95, doc_id="doc1"),
            _fact("X", "rev", "200", confidence=0.95, doc_id="doc2"),
        ]
        edges = [ClaimEdge(
            source_fact_idx=0, target_fact_idx=1,
            edge_type=EdgeType.CONTRADICTS,
            detection_method="test", confidence=1.0, explanation="",
        )]
        # Override: doc2 is highly authoritative → should win
        scores = _propagate_credibility(
            facts, [0, 1], edges,
            {"doc1": "unknown.pdf", "doc2": "unknown.pdf"},
            doc_authority_overrides={"doc1": 0.60, "doc2": 0.95},
        )
        assert scores[1] > scores[0], "Higher-authority doc should win the contradiction"

    def test_uniform_authority_produces_zero_penalty_on_equal_confidence(self):
        """When all docs have identical authority and confidence, contradiction penalty is zero."""
        facts = [
            _fact("X", "rev", "100", confidence=0.95, doc_id="doc1"),
            _fact("X", "rev", "200", confidence=0.95, doc_id="doc2"),
        ]
        edges = [ClaimEdge(
            source_fact_idx=0, target_fact_idx=1,
            edge_type=EdgeType.CONTRADICTS,
            detection_method="test", confidence=0.85, explanation="",
        )]
        # Both get same authority (default)
        scores = _propagate_credibility(
            facts, [0, 1], edges,
            {"doc1": "unknown.pdf", "doc2": "unknown.pdf"},
        )
        # Scores should remain identical — propagation can't differentiate
        assert scores[0] == scores[1], (
            f"Expected identical scores but got {scores[0]} vs {scores[1]}. "
            "Uniform authority should produce zero penalty on equal-confidence facts."
        )


# ---------------------------------------------------------------------------
# Extraction failure detection tests
# ---------------------------------------------------------------------------

class TestExtractionFailures:
    def test_unverified_quote_detected(self):
        facts = [_fact("X", "rev", "100", verified=False, confidence=0.1)]
        failures = _detect_extraction_failures(facts, {"doc1": "test.pdf"})
        types = [f.failure_type for f in failures]
        assert "unverified_quote" in types

    def test_scale_in_unit_detected(self):
        facts = [_fact("X", "rev", "100 million", base_unit="INR million")]
        failures = _detect_extraction_failures(facts, {"doc1": "test.pdf"})
        types = [f.failure_type for f in failures]
        assert "scale_in_unit" in types

    def test_clean_fact_no_failures(self):
        facts = [_fact(
            "X", "rev", "100", numeric=100.0, canonical=100.0,
            unit="INR", base_unit="INR", dimension=Dimension.MONETARY,
        )]
        failures = _detect_extraction_failures(facts, {"doc1": "test.pdf"})
        # Should have no failures (verified=True, clean unit, has dimension)
        assert len(failures) == 0


# ---------------------------------------------------------------------------
# Integration: build_claim_graph
# ---------------------------------------------------------------------------

class TestBuildClaimGraph:
    def test_empty_input(self):
        graph = build_claim_graph({}, {})
        assert graph.total_facts == 0
        assert len(graph.clusters) == 0

    def test_single_doc_produces_clusters(self):
        facts = [
            _fact("Delhivery", "revenue", "₹1,000 Mn", numeric=1000, canonical=1e9,
                  unit="INR", temporal="FY2021", doc_id="doc1"),
            _fact("Delhivery", "revenue", "₹1,500 Mn", numeric=1500, canonical=1.5e9,
                  unit="INR", temporal="FY2022", doc_id="doc1"),
        ]
        graph = build_claim_graph(
            {"doc1": facts},
            {"doc1": "delhivery_prospectus.pdf"},
        )
        assert graph.total_facts == 2
        # Should form 1 cluster with 2 facts (same subject + predicate)
        assert len(graph.clusters) >= 1

    def test_cross_doc_corroboration(self):
        f1 = _fact("Delhivery", "revenue", "₹1,000 Mn", numeric=1000, canonical=1e9,
                    unit="INR", temporal="FY2021", doc_id="doc1", confidence=0.95)
        f2 = _fact("Delhivery", "revenue", "₹1,000 Mn", numeric=1000, canonical=1e9,
                    unit="INR", temporal="FY2021", doc_id="doc2", confidence=0.90)
        graph = build_claim_graph(
            {"doc1": [f1], "doc2": [f2]},
            {"doc1": "prospectus.pdf", "doc2": "annual_report.pdf"},
        )
        assert graph.total_facts == 2
        # Should form exactly 1 multi-doc cluster
        multi_doc = [c for c in graph.clusters if c.doc_count > 1]
        assert len(multi_doc) >= 1

    def test_get_assignment_cases_structure(self):
        f1 = _fact("X", "rev", "100", numeric=100, canonical=100, doc_id="doc1")
        f2 = _fact("X", "rev", "100", numeric=100, canonical=100, doc_id="doc2")
        graph = build_claim_graph(
            {"doc1": [f1], "doc2": [f2]},
            {"doc1": "a.pdf", "doc2": "b.pdf"},
        )
        cases = get_assignment_cases(graph)
        assert "case_1_corroborated" in cases
        assert "case_2_contradicted" in cases
        assert "case_3_reconciled" in cases
        assert "case_4_extraction_failure" in cases


# ---------------------------------------------------------------------------
# Incremental Claim Graph tests (Streaming Entity Resolution & Re-arbitration)
# ---------------------------------------------------------------------------

class TestIncrementalClaimGraph:
    def test_append_corroborating_fact_updates_cluster(self):
        # Initial graph with 1 cluster
        f1 = _fact("Delhivery", "revenue", "₹1,000 Mn", numeric=1000, canonical=1e9,
                   unit="INR", temporal="FY2021", doc_id="doc1", confidence=0.95)
        f2 = _fact("Delhivery", "pin_codes", "10,000", numeric=10000, canonical=10000,
                   unit="PIN codes", doc_id="doc1")
        initial_facts = {"doc1": [f1, f2]}
        initial_names = {"doc1": "prospectus.pdf"}
        initial_graph = build_claim_graph(initial_facts, initial_names)

        # Initial state: 0 multi-fact clusters (f1 and f2 are distinct singletons)
        assert len(initial_graph.clusters) == 0
        assert len(initial_graph.unmatched_facts) == 2

        # New document arrives with a matching revenue fact
        f3 = _fact("Delhivery", "revenue", "₹1,000 Mn", numeric=1000, canonical=1e9,
                   unit="INR", temporal="FY2021", doc_id="doc2", confidence=0.90)
        
        updated_graph, stats = append_document_to_claim_graph(
            existing_graph=initial_graph,
            existing_doc_facts=initial_facts,
            new_doc_id="doc2",
            new_doc_facts=[f3],
            new_doc_filename="annual_report.pdf",
        )

        assert updated_graph.total_facts == 3
        assert len(updated_graph.documents) == 2
        # f1 and f3 should have merged into 1 multi-doc cluster!
        assert len(updated_graph.clusters) == 1
        c = updated_graph.clusters[0]
        assert c.subject.lower() == "delhivery"
        assert c.predicate == "revenue"
        assert c.case_type == CaseType.CORROBORATED
        assert c.doc_count == 2
        assert stats["clusters_created"] == 1
        # pin_codes remains unmatched singleton
        assert len(updated_graph.unmatched_facts) == 1

    def test_append_fact_promotes_singleton_and_preserves_unaffected(self):
        # Doc 1 has revenue and ebitda
        f1 = _fact("RBI", "repo_rate", "6.5%", numeric=6.5, canonical=6.5,
                   unit="%", doc_id="doc1")
        f2 = _fact("India", "inflation", "5.4%", numeric=5.4, canonical=5.4,
                   unit="%", doc_id="doc1")
        # Doc 2 already corroborates repo_rate
        f3 = _fact("RBI", "repo_rate", "6.5%", numeric=6.5, canonical=6.5,
                   unit="%", doc_id="doc2")

        doc_facts = {"doc1": [f1, f2], "doc2": [f3]}
        graph = build_claim_graph(doc_facts, {"doc1": "rbi.pdf", "doc2": "survey.pdf"})
        assert len(graph.clusters) == 1  # repo_rate cluster
        assert len(graph.unmatched_facts) == 1  # inflation is singleton

        # Doc 3 arrives with inflation = 3.6% (different temporal/context)
        f4 = _fact("India", "inflation", "3.6%", numeric=3.6, canonical=3.6,
                   unit="%", temporal="Q2 FY25", doc_id="doc3")
        f5 = _fact("India", "exports", "$400B", numeric=400, canonical=400e9,
                   unit="USD", doc_id="doc3")

        updated_graph, stats = append_document_to_claim_graph(
            existing_graph=graph,
            existing_doc_facts=doc_facts,
            new_doc_id="doc3",
            new_doc_facts=[f4, f5],
            new_doc_filename="imf.pdf",
        )

        assert updated_graph.total_facts == 5
        # We now have 2 clusters: repo_rate (unaffected) and inflation (newly formed)
        assert len(updated_graph.clusters) == 2
        # stats should show exactly 1 unaffected cluster
        assert stats["unaffected_clusters"] == 1
        assert stats["clusters_created"] == 1
        # exports is a new singleton
        assert len(updated_graph.unmatched_facts) == 1

    def test_append_to_empty_graph_acts_as_initial_build(self):
        f = _fact("Delhivery", "revenue", "₹500 Cr", numeric=500, canonical=5e9, doc_id="doc1")
        graph, stats = append_document_to_claim_graph(
            existing_graph=None,
            existing_doc_facts={},
            new_doc_id="doc1",
            new_doc_facts=[f],
            new_doc_filename="report.pdf",
        )
    def test_large_cluster_batched_in_sets_of_50_checks_all_facts(self):
        """Verify clusters >50 facts are processed in iterative sets of 50 and do not stop at 50."""
        from app.claim_graph import _build_cluster_edges

        # Create 70 facts (exceeding 50 cap)
        facts = []
        for i in range(70):
            val = 100.0 if i != 65 else 200.0  # Fact 65 contradicts the others!
            facts.append(_fact(
                subject="Acme",
                predicate="revenue",
                value=f"${val}M",
                numeric=val,
                canonical=val * 1e6,
                unit="USD",
                doc_id=f"doc_{i % 5}",
                confidence=0.95 - (i * 0.001),  # decreasing confidence
            ))

        indices = list(range(70))
        edges = _build_cluster_edges(facts, indices)

        # Ensure we have edges
        assert len(edges) > 0

        # Verify that facts past index 50 (such as fact 65) were NOT ignored and have edges!
        edges_with_fact_65 = [
            e for e in edges
            if e.source_fact_idx == 65 or e.target_fact_idx == 65
        ]
        assert len(edges_with_fact_65) > 0, "Fact 65 must not be ignored by the 50-cap batcher"

        # Fact 65 ($200M) vs Anchor facts ($100M) should detect a contradiction!
        contradictions_for_65 = [
            e for e in edges_with_fact_65
            if e.edge_type == EdgeType.CONTRADICTS
        ]
        assert len(contradictions_for_65) > 0, "Fact 65 must detect contradiction against anchors"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


