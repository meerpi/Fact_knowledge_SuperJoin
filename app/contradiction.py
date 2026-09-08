"""3-stage contradiction detection engine.

Architecture inspired by:
- ArbGraph (1212Judy/ArbGraph): claim_alignment → evidence_graph → conflict_arbitration
- TabFact: symbolic + linguistic dual-channel reasoning
- SemEval-2024 NumEval: pre-normalization before NLI is mandatory
- DeBERTa-v3 cross-encoder: local GPU NLI at $0.00

Stage 1: Symbolic alignment — group by (subject, predicate), filter by context
Stage 2: Deterministic numeric comparison — canonical_value math
Stage 3: DeBERTa-v3 NLI — qualitative claims only
"""

import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher

from app.models import (
    ClaimEdge,
    ContradictionReport,
    EdgeType,
    Fact,
)
from app.normalizer import normalize_temporal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded NLI model singleton
# ---------------------------------------------------------------------------

_nli_model = None
_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
_NLI_LABELS = ["contradiction", "entailment", "neutral"]


def _get_nli_model():
    """Lazy-load the DeBERTa-v3 cross-encoder NLI model."""
    global _nli_model
    if _nli_model is not None:
        return _nli_model

    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading NLI model '%s' (first call, may take a few seconds)...", _NLI_MODEL_NAME)
        _nli_model = CrossEncoder(_NLI_MODEL_NAME)
        logger.info("NLI model loaded successfully.")
        return _nli_model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. NLI stage will be skipped. "
            "Install with: pip install sentence-transformers"
        )
        return None
    except Exception as e:
        logger.warning("Failed to load NLI model: %s. NLI stage will be skipped.", e)
        return None


# ---------------------------------------------------------------------------
# Stage 1: Candidate pair generation (semantic & symbolic alignment)
# ---------------------------------------------------------------------------


def _normalize_subject(subject: str) -> str:
    """Normalize subject for matching (strip possessives, hyphens, company suffixes)."""
    s = subject.strip().lower().replace("'s", "").replace("'", "").replace("-", " ")
    for suffix in ("limited", "ltd", "inc", "corp", "corporation", "plc", "llc", "pvt", "private"):
        s = re.sub(r'\b' + suffix + r'\.?\b', '', s).strip()
    return s.strip('. ')


def _subjects_match(s1: str, s2: str) -> bool:
    """Check if two subjects refer to the same entity using normalization + Voyage embeddings."""
    n1, n2 = _normalize_subject(s1), _normalize_subject(s2)
    if n1 == n2:
        return True
    # Check if one contains the other (e.g. "Delhivery" in "Delhivery Logistics")
    if (n1 in n2 or n2 in n1) and min(len(n1), len(n2)) >= 4:
        return True
    # Fuzzy match for minor spelling typos
    if SequenceMatcher(None, n1, n2).ratio() > 0.80:
        return True

    # Dynamic semantic match using Voyage embeddings (no hardcoded synonyms)
    try:
        from app.embeddings import is_semantic_match
        return is_semantic_match(s1, s2, kind="entity", threshold=0.55)
    except Exception:
        return False



def _temporals_match(t1: str | None, t2: str | None) -> bool:
    """Check if two temporal contexts refer to the same period."""
    if t1 is None and t2 is None:
        return True
    if t1 is None or t2 is None:
        return False  # One has temporal, one doesn't — ambiguous, treat as match
    n1 = normalize_temporal(t1)
    n2 = normalize_temporal(t2)
    return n1 == n2


_MUTUALLY_EXCLUSIVE_MODIFIERS = [
    ({"pre-tax", "pre tax"}, {"after-tax", "after tax", "post-tax"}),
    ({"standalone"}, {"consolidated"}),
    ({"diluted"}, {"basic"}),
    ({"real", "constant prices", "constant"}, {"nominal", "current prices", "current"}),
    ({"gross"}, {"net"}),
    ({"adjusted"}, {"unadjusted", "standard", "as reported"}),
]


def _conditions_compatible(c1: str | None, c2: str | None) -> bool:
    """Check if two conditions are compatible.
    Different conditions (e.g. pre-tax vs after-tax, adjusted vs standard/unspecified)
    mean different metrics that should not be flagged as numeric contradictions.
    """
    if c1 is None and c2 is None:
        return True
    if c1 is None or c2 is None:
        # If one has an explicit modifier condition, it does not conflict with an unconditioned metric
        active_cond = (c1 or c2).strip().lower()
        modifiers = {"adjusted", "pre-tax", "after-tax", "core", "diluted", "basic", "constant currency", "pro forma", "standalone", "consolidated"}
        if any(m in active_cond for m in modifiers):
            return False
        return True

    s1, s2 = c1.strip().lower(), c2.strip().lower()
    if s1 == s2:
        return True

    # Check for mutually exclusive modifier pairs
    for group_a, group_b in _MUTUALLY_EXCLUSIVE_MODIFIERS:
        in_a1 = any(term in s1 for term in group_a)
        in_b1 = any(term in s1 for term in group_b)
        in_a2 = any(term in s2 for term in group_a)
        in_b2 = any(term in s2 for term in group_b)
        if (in_a1 and in_b2) or (in_b1 and in_a2):
            return False

    return True



def _generate_candidate_pairs(facts: list[Fact]) -> list[tuple[int, int]]:
    """Group facts by (subject, predicate) and generate candidate pairs.

    Only facts sharing the same entity and metric can contradict.
    Pairs with different temporal contexts or incompatible conditions are filtered out.
    """
    # Group by normalized (subject, predicate)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, fact in enumerate(facts):
        key = f"{_normalize_subject(fact.subject)}||{fact.predicate.lower()}"
        groups[key].append(idx)

    sorted_keys = sorted(groups.keys())
    unique_preds = list({k.split("||", 1)[1] for k in sorted_keys})
    unique_subjs = list({k.split("||", 1)[0] for k in sorted_keys})

    try:
        from app.embeddings import precompute_term_matches
        pred_matches = precompute_term_matches(unique_preds, kind="metric", threshold=0.55)
        subj_matches = precompute_term_matches(unique_subjs, kind="entity", threshold=0.55)
    except Exception as e:
        logger.warning("Embedding precomputation failed in contradiction engine: %s", e)
        pred_matches = {}
        subj_matches = {}


    # Cross-group fuzzy/semantic subject and predicate matching
    merged_groups: list[list[int]] = []
    used_keys: set[str] = set()

    for i, key_i in enumerate(sorted_keys):
        if key_i in used_keys:
            continue
        cluster = list(groups[key_i])
        subj_i, pred_i = key_i.split("||", 1)

        for j in range(i + 1, len(sorted_keys)):
            key_j = sorted_keys[j]
            if key_j in used_keys:
                continue
            subj_j, pred_j = key_j.split("||", 1)

            # Predicates must match (exact or semantic embedding)
            is_pred_match = pred_matches.get((pred_i, pred_j), pred_i == pred_j)
            if not is_pred_match:
                continue

            # Subjects must match (exact, substring, or semantic embedding)
            is_subj_match = subj_matches.get((subj_i, subj_j), _subjects_match(subj_i, subj_j))
            if is_subj_match:
                cluster.extend(groups[key_j])
                used_keys.add(key_j)

        used_keys.add(key_i)
        if len(cluster) >= 2:
            merged_groups.append(cluster)


    # Generate pairs within each group, filtering by context
    candidate_pairs: list[tuple[int, int]] = []
    for group in merged_groups:
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                i, j = group[a], group[b]
                fi, fj = facts[i], facts[j]

                # Skip if conditions are incompatible (different metrics)
                if not _conditions_compatible(
                    fi.context.conditions, fj.context.conditions
                ):
                    continue

                candidate_pairs.append((i, j))

    return candidate_pairs


# ---------------------------------------------------------------------------
# Stage 2: Deterministic numeric comparison
# ---------------------------------------------------------------------------

_NUMERIC_EPSILON = 1e-4  # Relative tolerance for "equal" values


def _numeric_comparison(
    facts: list[Fact], i: int, j: int
) -> ClaimEdge | None:
    """Compare two facts with canonical_value. Returns edge or None if not applicable."""
    fi, fj = facts[i], facts[j]

    # Both must have canonical values
    if fi.canonical_value is None or fj.canonical_value is None:
        return None

    v1, v2 = fi.canonical_value, fj.canonical_value
    max_abs = max(abs(v1), abs(v2))

    if max_abs == 0:
        # Both zero
        return ClaimEdge(
            source_fact_idx=i,
            target_fact_idx=j,
            edge_type=EdgeType.CORROBORATES,
            detection_method="symbolic_numeric",
            confidence=1.0,
            explanation=f"Both values are zero: {fi.value} == {fj.value}",
        )

    relative_diff = abs(v1 - v2) / max_abs

    # Check if temporal contexts differ → SUPERSEDES instead of CONTRADICTS
    temporals_same = _temporals_match(fi.context.temporal, fj.context.temporal)

    if relative_diff < _NUMERIC_EPSILON:
        return ClaimEdge(
            source_fact_idx=i,
            target_fact_idx=j,
            edge_type=EdgeType.CORROBORATES,
            detection_method="symbolic_numeric",
            confidence=1.0,
            explanation=(
                f"Numerically equivalent: {fi.value} ({fi.canonical_value:,.2f}) "
                f"≈ {fj.value} ({fj.canonical_value:,.2f}), "
                f"relative diff = {relative_diff:.2e}"
            ),
        )
    else:
        if not temporals_same:
            # Different time periods — this is temporal progression, not contradiction
            return ClaimEdge(
                source_fact_idx=i,
                target_fact_idx=j,
                edge_type=EdgeType.SUPERSEDES,
                detection_method="temporal",
                confidence=1.0,
                explanation=(
                    f"Different periods ({fi.context.temporal} vs {fj.context.temporal}): "
                    f"{fi.value} vs {fj.value} — temporal progression, not contradiction"
                ),
            )
        else:
            return ClaimEdge(
                source_fact_idx=i,
                target_fact_idx=j,
                edge_type=EdgeType.CONTRADICTS,
                detection_method="symbolic_numeric",
                confidence=1.0,
                explanation=(
                    f"Numeric conflict: {fi.value} ({fi.canonical_value:,.2f}) "
                    f"≠ {fj.value} ({fj.canonical_value:,.2f}), "
                    f"relative diff = {relative_diff:.4f}, "
                    f"same period: {fi.context.temporal}"
                ),
            )


# ---------------------------------------------------------------------------
# Stage 3: DeBERTa-v3 NLI cross-encoder
# ---------------------------------------------------------------------------

def _format_claim_text(fact: Fact) -> str:
    """Format a Fact into a natural language claim string for NLI input."""
    parts = [fact.subject]

    # Add temporal context
    if fact.context.temporal:
        parts.append(f"({fact.context.temporal})")

    parts.append(f"{fact.predicate.replace('_', ' ')} is {fact.value}")

    if fact.context.conditions:
        parts.append(f"({fact.context.conditions})")
    if fact.context.scope:
        parts.append(f"[{fact.context.scope}]")

    return " ".join(parts)


def _nli_classify(
    facts: list[Fact],
    pairs: list[tuple[int, int]],
    threshold: float = 0.7,
) -> list[ClaimEdge]:
    """Run DeBERTa-v3 NLI on qualitative claim pairs."""
    if not pairs:
        return []

    model = _get_nli_model()
    if model is None:
        logger.warning("NLI model unavailable. Skipping %d qualitative pairs.", len(pairs))
        return []

    # Build input pairs
    text_pairs = []
    for i, j in pairs:
        premise = _format_claim_text(facts[i])
        hypothesis = _format_claim_text(facts[j])
        text_pairs.append((premise, hypothesis))

    # Batch predict
    try:
        scores = model.predict(text_pairs, show_progress_bar=False)
    except Exception as e:
        logger.error("NLI prediction failed: %s", e)
        return []

    edges: list[ClaimEdge] = []
    for pair_idx, (i, j) in enumerate(pairs):
        score_vec = scores[pair_idx]
        # scores order: [contradiction, entailment, neutral]
        max_idx = score_vec.argmax()
        max_score = float(score_vec[max_idx])
        label = _NLI_LABELS[max_idx]

        if max_score < threshold:
            continue  # Below confidence threshold

        if label == "contradiction":
            edges.append(ClaimEdge(
                source_fact_idx=i,
                target_fact_idx=j,
                edge_type=EdgeType.CONTRADICTS,
                detection_method="nli_deberta",
                confidence=max_score,
                explanation=(
                    f"NLI contradiction ({max_score:.3f}): "
                    f"'{_format_claim_text(facts[i])}' vs '{_format_claim_text(facts[j])}'"
                ),
            ))
        elif label == "entailment":
            edges.append(ClaimEdge(
                source_fact_idx=i,
                target_fact_idx=j,
                edge_type=EdgeType.CORROBORATES,
                detection_method="nli_deberta",
                confidence=max_score,
                explanation=(
                    f"NLI entailment ({max_score:.3f}): "
                    f"'{_format_claim_text(facts[i])}' entails '{_format_claim_text(facts[j])}'"
                ),
            ))
        # neutral → no edge

    return edges


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_contradictions(
    facts: list[Fact],
    doc_id: str,
    cross_doc_facts: list[Fact] | None = None,
    cross_doc_id: str | None = None,
    nli_threshold: float = 0.7,
) -> ContradictionReport:
    """Run the dual-channel contradiction detection pipeline.

    Stage 1: Symbolic alignment — group by (subject, predicate)
    Stage 2: Deterministic numeric comparison
    Stage 3: DeBERTa-v3 NLI — dual-channel verification

    NLI runs on:
      (a) Qualitative pairs that Stage 2 could not resolve (no canonical_value)
      (b) Numeric contradiction pairs — as a cross-check / second opinion
      (c) Numeric corroboration pairs from different documents — to verify
          semantic entailment of the evidence quotes

    This ensures the NLI stage is never entirely skipped, even when all facts
    are numeric (the common case for financial documents).

    Args:
        facts: List of Fact objects from a single document.
        doc_id: Document ID.
        cross_doc_facts: Optional facts from another document for cross-doc comparison.
        cross_doc_id: ID of the cross-document.
        nli_threshold: Minimum confidence for NLI edges.

    Returns:
        ContradictionReport with typed edges.
    """
    # If cross-document, merge fact lists with offset tracking
    all_facts = list(facts)
    if cross_doc_facts:
        all_facts.extend(cross_doc_facts)

    # Stage 1: Generate candidate pairs
    candidate_pairs = _generate_candidate_pairs(all_facts)
    total_candidates = len(candidate_pairs)

    logger.info(
        "Stage 1: %d candidate pairs from %d facts (groups by subject+predicate)",
        total_candidates, len(all_facts),
    )

    # Stage 2: Deterministic numeric comparison
    numeric_edges: list[ClaimEdge] = []
    remaining_pairs: list[tuple[int, int]] = []
    numeric_contradiction_pairs: list[tuple[int, int]] = []
    numeric_corroboration_cross_doc_pairs: list[tuple[int, int]] = []

    for i, j in candidate_pairs:
        edge = _numeric_comparison(all_facts, i, j)
        if edge is not None:
            numeric_edges.append(edge)
            # Track interesting pairs for NLI dual-channel verification
            if edge.edge_type == EdgeType.CONTRADICTS:
                numeric_contradiction_pairs.append((i, j))
            elif (edge.edge_type == EdgeType.CORROBORATES
                  and all_facts[i].provenance.doc_id != all_facts[j].provenance.doc_id):
                numeric_corroboration_cross_doc_pairs.append((i, j))
        else:
            remaining_pairs.append((i, j))

    logger.info(
        "Stage 2: %d edges from numeric comparison (%d contradictions, %d cross-doc corroborations), "
        "%d pairs remaining for NLI",
        len(numeric_edges), len(numeric_contradiction_pairs),
        len(numeric_corroboration_cross_doc_pairs), len(remaining_pairs),
    )

    # Stage 3: DeBERTa-v3 NLI — dual-channel
    # Channel A: Qualitative pairs (no canonical_value on either side)
    nli_qualitative_edges: list[ClaimEdge] = []
    if remaining_pairs:
        nli_qualitative_edges = _nli_classify(all_facts, remaining_pairs, threshold=nli_threshold)
        logger.info("Stage 3a: %d edges from NLI on qualitative pairs", len(nli_qualitative_edges))

    # Channel B: Cross-verify numeric contradictions with NLI
    nli_crosscheck_edges: list[ClaimEdge] = []
    nli_verify_pairs = numeric_contradiction_pairs + numeric_corroboration_cross_doc_pairs[:10]
    if nli_verify_pairs:
        raw_nli_edges = _nli_classify(all_facts, nli_verify_pairs, threshold=nli_threshold)
        for nli_edge in raw_nli_edges:
            # Find the corresponding numeric edge
            matching_numeric = None
            for ne in numeric_edges:
                if (ne.source_fact_idx == nli_edge.source_fact_idx
                        and ne.target_fact_idx == nli_edge.target_fact_idx):
                    matching_numeric = ne
                    break

            if matching_numeric is None:
                # No matching numeric edge — add standalone
                nli_crosscheck_edges.append(nli_edge)
                continue

            if (matching_numeric.edge_type == EdgeType.CONTRADICTS
                    and nli_edge.edge_type == EdgeType.CONTRADICTS):
                # Both channels agree: contradiction. Add NLI edge as reinforcement.
                nli_crosscheck_edges.append(ClaimEdge(
                    source_fact_idx=nli_edge.source_fact_idx,
                    target_fact_idx=nli_edge.target_fact_idx,
                    edge_type=EdgeType.CONTRADICTS,
                    detection_method="nli_deberta",
                    confidence=nli_edge.confidence,
                    explanation=(
                        f"NLI CONFIRMS numeric contradiction ({nli_edge.confidence:.3f}): "
                        f"'{_format_claim_text(all_facts[nli_edge.source_fact_idx])}' vs "
                        f"'{_format_claim_text(all_facts[nli_edge.target_fact_idx])}'"
                    ),
                ))
            elif (matching_numeric.edge_type == EdgeType.CONTRADICTS
                  and nli_edge.edge_type == EdgeType.CORROBORATES):
                # Numeric says contradiction, NLI says entailment — flag as needing review
                nli_crosscheck_edges.append(ClaimEdge(
                    source_fact_idx=nli_edge.source_fact_idx,
                    target_fact_idx=nli_edge.target_fact_idx,
                    edge_type=EdgeType.CORROBORATES,
                    detection_method="nli_deberta",
                    confidence=nli_edge.confidence,
                    explanation=(
                        f"NLI DISAGREES with numeric contradiction ({nli_edge.confidence:.3f}): "
                        f"NLI sees entailment — possible contextual reconciliation. "
                        f"'{_format_claim_text(all_facts[nli_edge.source_fact_idx])}' vs "
                        f"'{_format_claim_text(all_facts[nli_edge.target_fact_idx])}'"
                    ),
                ))
            elif (matching_numeric.edge_type == EdgeType.CORROBORATES
                  and nli_edge.edge_type == EdgeType.CORROBORATES):
                # Both agree: corroboration confirmed by NLI
                nli_crosscheck_edges.append(ClaimEdge(
                    source_fact_idx=nli_edge.source_fact_idx,
                    target_fact_idx=nli_edge.target_fact_idx,
                    edge_type=EdgeType.CORROBORATES,
                    detection_method="nli_deberta",
                    confidence=nli_edge.confidence,
                    explanation=(
                        f"NLI CONFIRMS cross-doc corroboration ({nli_edge.confidence:.3f}): "
                        f"'{_format_claim_text(all_facts[nli_edge.source_fact_idx])}' entails "
                        f"'{_format_claim_text(all_facts[nli_edge.target_fact_idx])}'"
                    ),
                ))
            else:
                # Any other NLI verdict — add as supplementary signal
                nli_crosscheck_edges.append(nli_edge)

        logger.info(
            "Stage 3b: %d edges from NLI cross-verification of %d numeric pairs",
            len(nli_crosscheck_edges), len(nli_verify_pairs),
        )

    # Merge all edges: numeric + NLI qualitative + NLI cross-check
    edges = numeric_edges + nli_qualitative_edges + nli_crosscheck_edges

    logger.info(
        "Final: %d total edges (numeric=%d, nli_qualitative=%d, nli_crosscheck=%d)",
        len(edges), len(numeric_edges), len(nli_qualitative_edges), len(nli_crosscheck_edges),
    )

    # Build summary
    summary: dict[str, int] = {}
    for edge in edges:
        key = edge.edge_type.value
        summary[key] = summary.get(key, 0) + 1

    return ContradictionReport(
        doc_id=doc_id,
        cross_doc_id=cross_doc_id,
        total_facts=len(all_facts),
        candidate_pairs_evaluated=total_candidates,
        edges=edges,
        summary=summary,
    )
