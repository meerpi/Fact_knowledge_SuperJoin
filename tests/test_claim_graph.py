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
    _UnionFind,
    _align_facts,
    _classify_cluster,
    _detect_extraction_failures,
    _normalize_subject,
    _propagate_credibility,
    _subjects_match,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
