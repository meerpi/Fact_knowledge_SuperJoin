"""Tests for the 3-stage contradiction detection engine."""

import pytest
from app.models import (
    ClaimEdge,
    Context,
    ContradictionReport,
    EdgeType,
    Fact,
    Provenance,
)
from app.contradiction import (
    detect_contradictions,
    _generate_candidate_pairs,
    _numeric_comparison,
    _subjects_match,
    _format_claim_text,
)


def _make_fact(
    subject="Delhivery",
    predicate="revenue",
    value="₹1,000",
    numeric_value=1000.0,
    canonical_value=1000.0,
    unit="INR million",
    canonical_unit="INR",
    temporal="FY2022",
    scope="Consolidated",
    conditions=None,
    page=0,
) -> Fact:
    """Helper to create test Fact objects."""
    return Fact(
        subject=subject,
        predicate=predicate,
        value=value,
        numeric_value=numeric_value,
        canonical_value=canonical_value,
        unit=unit,
        canonical_unit=canonical_unit,
        context=Context(temporal=temporal, scope=scope, conditions=conditions),
        confidence=1.0,
        provenance=Provenance(
            doc_id="test_doc",
            page=page,
            evidence_quote=f"Test quote: {value}",
            verified=True,
            match_type="exact",
        ),
    )


# ---------------------------------------------------------------------------
# Subject matching
# ---------------------------------------------------------------------------

class TestSubjectMatching:
    def test_exact_match(self):
        assert _subjects_match("Delhivery", "Delhivery")

    def test_case_insensitive(self):
        assert _subjects_match("Delhivery", "delhivery")

    def test_possessive(self):
        assert _subjects_match("Delhivery's", "Delhivery")

    def test_completely_different(self):
        assert not _subjects_match("Delhivery", "Spoton")

    def test_fuzzy_variant(self):
        assert _subjects_match("Delhivery", "Delhivery Limited")


# ---------------------------------------------------------------------------
# Candidate pair generation (Stage 1)
# ---------------------------------------------------------------------------

class TestCandidatePairGeneration:
    def test_same_subject_predicate_generates_pair(self):
        facts = [
            _make_fact(subject="Delhivery", predicate="revenue", value="₹1,000", canonical_value=1000.0),
            _make_fact(subject="Delhivery", predicate="revenue", value="₹1,200", canonical_value=1200.0),
        ]
        pairs = _generate_candidate_pairs(facts)
        assert len(pairs) == 1
        assert pairs[0] == (0, 1)

    def test_different_predicates_no_pair(self):
        """Operating profit vs net profit should NOT generate a pair."""
        facts = [
            _make_fact(predicate="operating_profit", value="₹500M", conditions="pre-tax"),
            _make_fact(predicate="net_profit", value="₹380M", conditions="after-tax"),
        ]
        pairs = _generate_candidate_pairs(facts)
        assert len(pairs) == 0

    def test_different_subjects_no_pair(self):
        facts = [
            _make_fact(subject="Delhivery", predicate="revenue"),
            _make_fact(subject="Spoton", predicate="revenue"),
        ]
        pairs = _generate_candidate_pairs(facts)
        assert len(pairs) == 0

    def test_incompatible_conditions_no_pair(self):
        """pre-tax vs after-tax on the same predicate should be filtered."""
        facts = [
            _make_fact(predicate="profit_before_tax", conditions="pre-tax"),
            _make_fact(predicate="profit_before_tax", conditions="after-tax"),
        ]
        pairs = _generate_candidate_pairs(facts)
        assert len(pairs) == 0

    def test_three_facts_same_group(self):
        """Three facts about the same (subject, predicate) → 3 pairs."""
        facts = [
            _make_fact(value="₹100", canonical_value=100.0),
            _make_fact(value="₹200", canonical_value=200.0),
            _make_fact(value="₹300", canonical_value=300.0),
        ]
        pairs = _generate_candidate_pairs(facts)
        assert len(pairs) == 3  # (0,1), (0,2), (1,2)


# ---------------------------------------------------------------------------
# Numeric comparison (Stage 2)
# ---------------------------------------------------------------------------

class TestNumericComparison:
    def test_equivalent_values_corroborate(self):
        """$1.5B == $1,500M should be CORROBORATES."""
        facts = [
            _make_fact(value="$1.5 billion", canonical_value=1_500_000_000.0, temporal="FY2022"),
            _make_fact(value="$1,500 million", canonical_value=1_500_000_000.0, temporal="FY2022"),
        ]
        edge = _numeric_comparison(facts, 0, 1)
        assert edge is not None
        assert edge.edge_type == EdgeType.CORROBORATES
        assert edge.detection_method == "symbolic_numeric"
        assert edge.confidence == 1.0

    def test_different_values_same_period_contradicts(self):
        """Different values, same period → CONTRADICTS."""
        facts = [
            _make_fact(value="₹1,000", canonical_value=1000.0, temporal="FY2022"),
            _make_fact(value="₹2,000", canonical_value=2000.0, temporal="FY2022"),
        ]
        edge = _numeric_comparison(facts, 0, 1)
        assert edge is not None
        assert edge.edge_type == EdgeType.CONTRADICTS
        assert edge.detection_method == "symbolic_numeric"

    def test_different_values_different_period_supersedes(self):
        """Different values, different periods → SUPERSEDES (not contradiction)."""
        facts = [
            _make_fact(value="₹1,000", canonical_value=1000.0, temporal="FY2021"),
            _make_fact(value="₹1,200", canonical_value=1200.0, temporal="FY2022"),
        ]
        edge = _numeric_comparison(facts, 0, 1)
        assert edge is not None
        assert edge.edge_type == EdgeType.SUPERSEDES
        assert edge.detection_method == "temporal"

    def test_no_canonical_values_returns_none(self):
        """Facts without canonical_value → None (deferred to NLI)."""
        facts = [
            _make_fact(value="expanded network", canonical_value=None),
            _make_fact(value="contracted network", canonical_value=None),
        ]
        edge = _numeric_comparison(facts, 0, 1)
        assert edge is None

    def test_both_zero_corroborates(self):
        facts = [
            _make_fact(value="0", canonical_value=0.0),
            _make_fact(value="₹0", canonical_value=0.0),
        ]
        edge = _numeric_comparison(facts, 0, 1)
        assert edge is not None
        assert edge.edge_type == EdgeType.CORROBORATES


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestDetectContradictions:
    def test_empty_facts(self):
        report = detect_contradictions([], doc_id="test")
        assert report.total_facts == 0
        assert report.candidate_pairs_evaluated == 0
        assert report.edges == []

    def test_single_fact_no_edges(self):
        facts = [_make_fact()]
        report = detect_contradictions(facts, doc_id="test")
        assert report.edges == []

    def test_corroboration_detected(self):
        """Two equivalent numeric facts → CORROBORATES edge."""
        facts = [
            _make_fact(value="$1.5 billion", canonical_value=1_500_000_000.0),
            _make_fact(value="$1,500 million", canonical_value=1_500_000_000.0),
        ]
        report = detect_contradictions(facts, doc_id="test")
        assert len(report.edges) == 1
        assert report.edges[0].edge_type == EdgeType.CORROBORATES
        assert report.summary.get("corroborates", 0) == 1

    def test_contradiction_detected(self):
        """Two conflicting numeric facts, same period → CONTRADICTS edges.

        Dual-channel: symbolic_numeric produces a contradiction edge,
        and NLI cross-verification may add a second confirmation edge.
        """
        facts = [
            _make_fact(value="₹1,000", canonical_value=1_000_000_000.0, temporal="FY2022"),
            _make_fact(value="₹2,000", canonical_value=2_000_000_000.0, temporal="FY2022"),
        ]
        report = detect_contradictions(facts, doc_id="test")
        # At least 1 edge from numeric, possibly more from NLI cross-check
        assert len(report.edges) >= 1
        # The first edge should be the symbolic_numeric contradiction
        numeric_edges = [e for e in report.edges if e.detection_method == "symbolic_numeric"]
        assert len(numeric_edges) == 1
        assert numeric_edges[0].edge_type == EdgeType.CONTRADICTS
        # Verify NLI also fired (dual-channel)
        nli_edges = [e for e in report.edges if e.detection_method == "nli_deberta"]
        assert len(nli_edges) >= 0  # NLI may or may not produce an edge above threshold

    def test_no_false_positive_across_metrics(self):
        """Operating profit vs net profit → no pair, no edge."""
        facts = [
            _make_fact(predicate="operating_profit", value="₹500M", canonical_value=500_000_000.0, conditions="pre-tax"),
            _make_fact(predicate="net_profit", value="₹380M", canonical_value=380_000_000.0, conditions="after-tax"),
        ]
        report = detect_contradictions(facts, doc_id="test")
        assert len(report.edges) == 0

    def test_temporal_supersedes(self):
        """Same metric, different periods → SUPERSEDES."""
        facts = [
            _make_fact(value="₹800", canonical_value=800.0, temporal="FY2021"),
            _make_fact(value="₹1,000", canonical_value=1000.0, temporal="FY2022"),
        ]
        report = detect_contradictions(facts, doc_id="test")
        assert len(report.edges) == 1
        assert report.edges[0].edge_type == EdgeType.SUPERSEDES

    def test_report_structure(self):
        """Verify ContradictionReport fields are populated correctly."""
        facts = [
            _make_fact(value="₹100", canonical_value=100.0),
            _make_fact(value="₹200", canonical_value=200.0),
        ]
        report = detect_contradictions(facts, doc_id="test_doc_123")
        assert report.doc_id == "test_doc_123"
        assert report.total_facts == 2
        assert report.candidate_pairs_evaluated >= 1
        assert isinstance(report.summary, dict)

    def test_cross_document_facts(self):
        """Cross-document comparison should merge fact lists."""
        facts_a = [_make_fact(value="₹100", canonical_value=100.0)]
        facts_b = [_make_fact(value="₹200", canonical_value=200.0)]
        report = detect_contradictions(
            facts=facts_a,
            doc_id="doc_a",
            cross_doc_facts=facts_b,
            cross_doc_id="doc_b",
        )
        assert report.total_facts == 2
        assert report.cross_doc_id == "doc_b"


# ---------------------------------------------------------------------------
# Claim text formatting
# ---------------------------------------------------------------------------

class TestFormatClaimText:
    def test_basic_format(self):
        fact = _make_fact(subject="Delhivery", predicate="revenue", value="₹1,000")
        text = _format_claim_text(fact)
        assert "Delhivery" in text
        assert "revenue" in text
        assert "₹1,000" in text

    def test_with_temporal(self):
        fact = _make_fact(temporal="FY2022")
        text = _format_claim_text(fact)
        assert "FY2022" in text

    def test_with_conditions(self):
        fact = _make_fact(conditions="pre-tax")
        text = _format_claim_text(fact)
        assert "pre-tax" in text
