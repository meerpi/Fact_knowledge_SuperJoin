"""Regression tests for verdict roll-up and source attribution fixes.

Tests:
1. CONTRADICTED cannot be the rolled-up status if all edges are SUPERSEDES.
2. Mixed SUPERSEDES + CONTRADICTS edges: SUPERSEDES pairs take priority.
3. Source attribution: same-document contradiction doesn't repeat filename.
"""

import pytest
from unittest.mock import MagicMock

from app.models import (
    CaseType,
    ClaimEdge,
    Context,
    DisputeCode,
    EdgeType,
    Fact,
    Provenance,
)
from app.claim_graph import _classify_cluster, _generate_explanation


def _make_fact(
    subject: str = "India",
    predicate: str = "merchandise_exports",
    value: str = "429.2",
    canonical_value: float = 429.2,
    canonical_unit: str = "USD billion",
    temporal: str | None = "FY2022",
    scope: str | None = None,
    doc_id: str = "doc1",
    page: int = 4,
    confidence: float = 0.9,
) -> Fact:
    return Fact(
        subject=subject,
        predicate=predicate,
        value=value,
        canonical_value=canonical_value,
        canonical_unit=canonical_unit,
        context=Context(temporal=temporal, scope=scope),
        confidence=confidence,
        provenance=Provenance(
            doc_id=doc_id,
            page=page,
            evidence_quote=f"The {predicate} was {value}.",
            verified=True,
            match_type="exact",
        ),
    )


class TestVerdictRollUp:
    """Regression: CONTRADICTED must not be the verdict when all edges are SUPERSEDES."""

    def test_all_supersedes_returns_reconciled_temporal(self):
        """A cluster with only SUPERSEDES edges must be RECONCILED_TEMPORAL."""
        facts = [
            _make_fact(value="429.2", canonical_value=429.2, temporal="FY2022", doc_id="doc1"),
            _make_fact(value="456.1", canonical_value=456.1, temporal="FY2023", doc_id="doc1"),
            _make_fact(value="441.4", canonical_value=441.4, temporal="FY2024", doc_id="doc2"),
        ]
        edges = [
            ClaimEdge(
                source_fact_idx=0, target_fact_idx=1,
                edge_type=EdgeType.SUPERSEDES,
                detection_method="temporal",
                confidence=1.0,
                explanation="Different periods (FY2022 vs FY2023): temporal progression",
            ),
            ClaimEdge(
                source_fact_idx=1, target_fact_idx=2,
                edge_type=EdgeType.SUPERSEDES,
                detection_method="temporal",
                confidence=1.0,
                explanation="Different periods (FY2023 vs FY2024): temporal progression",
            ),
        ]

        case_type, dispute_code = _classify_cluster(facts, [0, 1, 2], edges)
        assert case_type == CaseType.RECONCILED_TEMPORAL, (
            f"Expected RECONCILED_TEMPORAL but got {case_type.value}. "
            f"SUPERSEDES-only clusters must not be CONTRADICTED."
        )
        assert dispute_code == DisputeCode.DISPUTE_TEMPORAL_DRIFT.value

    def test_supersedes_overrides_contradicts_same_pair(self):
        """If a SUPERSEDES edge covers the same fact pair as a CONTRADICTS edge,
        SUPERSEDES takes priority for that pair."""
        facts = [
            _make_fact(value="429.2", canonical_value=429.2, temporal="FY2022", doc_id="doc1"),
            _make_fact(value="456.1", canonical_value=456.1, temporal="FY2023", doc_id="doc2"),
        ]
        # Both edge types on the same pair
        edges = [
            ClaimEdge(
                source_fact_idx=0, target_fact_idx=1,
                edge_type=EdgeType.SUPERSEDES,
                detection_method="temporal",
                confidence=1.0,
                explanation="Different periods (FY2022 vs FY2023): temporal progression",
            ),
            ClaimEdge(
                source_fact_idx=0, target_fact_idx=1,
                edge_type=EdgeType.CONTRADICTS,
                detection_method="nli_deberta",
                confidence=0.8,
                explanation="NLI contradiction: values differ",
            ),
        ]

        case_type, _ = _classify_cluster(facts, [0, 1], edges)
        assert case_type != CaseType.CONTRADICTED, (
            f"Expected RECONCILED (SUPERSEDES overrides CONTRADICTS for same pair) "
            f"but got {case_type.value}"
        )
        assert case_type == CaseType.RECONCILED_TEMPORAL

    def test_genuine_contradiction_still_detected(self):
        """A cluster with only CONTRADICTS edges and same temporal context
        must still be CONTRADICTED."""
        facts = [
            _make_fact(value="429.2", canonical_value=429.2, temporal="FY2023", doc_id="doc1"),
            _make_fact(value="500.0", canonical_value=500.0, temporal="FY2023", doc_id="doc2"),
        ]
        edges = [
            ClaimEdge(
                source_fact_idx=0, target_fact_idx=1,
                edge_type=EdgeType.CONTRADICTS,
                detection_method="symbolic_numeric",
                confidence=1.0,
                explanation="Numeric conflict: 429.2 != 500.0, same period: FY2023",
            ),
        ]

        case_type, _ = _classify_cluster(facts, [0, 1], edges)
        assert case_type == CaseType.CONTRADICTED

    def test_supersedes_dominant_overrides_minority_contradicts(self):
        """When SUPERSEDES edges outnumber genuine contradictions, 
        verdict should be RECONCILED_TEMPORAL."""
        facts = [
            _make_fact(value="429.2", canonical_value=429.2, temporal="FY2022", doc_id="doc1"),
            _make_fact(value="456.1", canonical_value=456.1, temporal="FY2023", doc_id="doc1"),
            _make_fact(value="441.4", canonical_value=441.4, temporal="FY2024", doc_id="doc2"),
            _make_fact(value="500.0", canonical_value=500.0, temporal="FY2024", doc_id="doc3"),
        ]
        edges = [
            ClaimEdge(
                source_fact_idx=0, target_fact_idx=1,
                edge_type=EdgeType.SUPERSEDES,
                detection_method="temporal",
                confidence=1.0,
                explanation="FY2022 vs FY2023",
            ),
            ClaimEdge(
                source_fact_idx=1, target_fact_idx=2,
                edge_type=EdgeType.SUPERSEDES,
                detection_method="temporal",
                confidence=1.0,
                explanation="FY2023 vs FY2024",
            ),
            ClaimEdge(
                source_fact_idx=2, target_fact_idx=3,
                edge_type=EdgeType.CONTRADICTS,
                detection_method="symbolic_numeric",
                confidence=1.0,
                explanation="Same period FY2024: 441.4 != 500.0",
            ),
        ]
        # 2 SUPERSEDES > 1 genuine CONTRADICTS
        case_type, _ = _classify_cluster(facts, [0, 1, 2, 3], edges)
        assert case_type == CaseType.RECONCILED_TEMPORAL


class TestSourceAttribution:
    """Source attribution: contradiction sentences must name distinct documents."""

    def test_same_doc_contradiction_explicit(self):
        """When both facts come from the same document, the explanation must
        say 'same document' instead of repeating the filename."""
        facts = [
            _make_fact(value="429.2", canonical_value=429.2, doc_id="doc1", page=4, temporal="FY2023"),
            _make_fact(value="44.8", canonical_value=44.8, doc_id="doc1", page=4, temporal="FY2023"),
        ]
        edges = [
            ClaimEdge(
                source_fact_idx=0, target_fact_idx=1,
                edge_type=EdgeType.CONTRADICTS,
                detection_method="symbolic_numeric",
                confidence=1.0,
                explanation="Numeric conflict",
            ),
        ]
        doc_filenames = {"doc1": "03-imf-india-2025-article-iv-excerpt.pdf"}
        credibility = {0: 0.9, 1: 0.5}

        explanation = _generate_explanation(
            facts, [0, 1], edges, CaseType.CONTRADICTED, credibility, doc_filenames
        )

        # Must say "same document" when both facts come from same doc
        assert "same document" in explanation.lower(), (
            f"Same-doc contradiction must say 'same document', got: {explanation}"
        )

    def test_cross_doc_contradiction_names_both(self):
        """When facts come from different documents, both filenames must appear."""
        facts = [
            _make_fact(value="429.2", canonical_value=429.2, doc_id="doc1", page=4, temporal="FY2023"),
            _make_fact(value="44.8", canonical_value=44.8, doc_id="doc2", page=7, temporal="FY2023"),
        ]
        edges = [
            ClaimEdge(
                source_fact_idx=0, target_fact_idx=1,
                edge_type=EdgeType.CONTRADICTS,
                detection_method="symbolic_numeric",
                confidence=1.0,
                explanation="Numeric conflict",
            ),
        ]
        doc_filenames = {
            "doc1": "03-imf-india-2025-article-iv-excerpt.pdf",
            "doc2": "02-rbi-annual-report-2024-25-excerpt.pdf",
        }
        credibility = {0: 0.9, 1: 0.5}

        explanation = _generate_explanation(
            facts, [0, 1], edges, CaseType.CONTRADICTED, credibility, doc_filenames
        )

        # Both filenames must appear
        assert "imf" in explanation.lower()
        assert "rbi" in explanation.lower()


class TestDimensionAndAlignment:
    """Tests for unit/dimension disambiguation and cluster alignment."""

    def test_growth_rate_and_level_do_not_cluster(self):
        """Growth rate percentage and dollar level must never be clustered together."""
        from app.claim_graph import _align_facts

        facts = [
            _make_fact(
                predicate="merchandise_exports",
                value="429.2",
                canonical_value=429.2,
                canonical_unit="USD billion",
                doc_id="doc1",
            ),
            _make_fact(
                predicate="merchandise_exports_annual_growth",
                value="0.1 per cent",
                canonical_value=0.1,
                canonical_unit="%",
                doc_id="doc2",
            ),
        ]
        clusters = _align_facts(facts)
        # Must be separated into 2 distinct clusters
        assert len(clusters) == 2, (
            f"Expected 2 separate clusters for export level vs growth rate, got {len(clusters)}"
        )

    def test_percentage_and_monetary_do_not_cluster_same_predicate(self):
        """Even with identical predicate, monetary and percentage dimensions must separate."""
        from app.claim_graph import _align_facts

        facts = [
            _make_fact(
                predicate="merchandise_exports",
                value="429.2",
                canonical_value=429.2,
                canonical_unit="USD billion",
                doc_id="doc1",
            ),
            _make_fact(
                predicate="merchandise_exports",
                value="44.8%",
                canonical_value=44.8,
                canonical_unit="%",
                doc_id="doc1",
            ),
        ]
        clusters = _align_facts(facts)
        assert len(clusters) == 2

    def test_most_credible_value_is_dimensionally_consistent(self):
        """The most credible value in explanation must match the cluster's dominant dimension."""
        facts = [
            _make_fact(value="429.2", canonical_value=429.2, canonical_unit="USD billion", doc_id="doc1"),
            _make_fact(value="456.1", canonical_value=456.1, canonical_unit="USD billion", doc_id="doc2"),
            _make_fact(value="0.1 per cent", canonical_value=0.1, canonical_unit="%", doc_id="doc2"),
        ]
        edges = [
            ClaimEdge(
                source_fact_idx=0, target_fact_idx=1,
                edge_type=EdgeType.CONTRADICTS,
                detection_method="symbolic_numeric",
                confidence=1.0,
                explanation="Conflict between 429.2 and 456.1",
            ),
        ]
        doc_filenames = {"doc1": "IMF.pdf", "doc2": "RBI.pdf"}
        # Even if the percentage fact has higher credibility, the monetary fact must be chosen for a monetary cluster
        credibility = {0: 0.8, 1: 0.7, 2: 0.99}

        explanation = _generate_explanation(
            facts, [0, 1, 2], edges, CaseType.CONTRADICTED, credibility, doc_filenames
        )
        assert "429.2" in explanation or "456.1" in explanation, (
            f"Expected dollar amount as credible value, got: {explanation}"
        )
        assert "0.1 per cent" not in explanation.split("Most credible value:")[1]
