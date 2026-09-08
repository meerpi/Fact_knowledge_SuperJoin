"""ArbGraph-inspired claim graph engine.

Architecture (following ArbGraph: Niu et al., 2026):
    1. Claim Alignment   — Union-Find clustering by (subject, predicate) across documents
    2. Evidence Graph     — Intra-cluster edge classification (numeric, temporal, NLI)
    3. Credibility Prop.  — Intensity-driven credibility scores per fact node
    4. Cluster Arbitration — Classify each cluster into the 4 assignment cases

References:
    - ArbGraph (1212Judy/ArbGraph): claim_alignment → evidence_graph → conflict_arbitration
    - AttestDB: claim-centric data model with provenance and contradiction coexistence
    - FActScore (Min et al., EMNLP 2023): atomic fact decomposition + verification
    - TabFact (Chen et al., ICLR 2020): dual-channel symbolic + linguistic verification
"""

import hashlib
import logging
import time
from collections import defaultdict
from difflib import SequenceMatcher

from app.contradiction import detect_contradictions
from app.models import (
    CaseType,
    ClaimEdge,
    ClaimGraph,
    DisputeCode,
    EdgeType,
    EvidenceEntry,
    ExtractionFailure,
    Fact,
    FactCluster,
)
from app.normalizer import normalize_temporal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document authority weights (AttestDB / ArbGraph credibility prior)
# ---------------------------------------------------------------------------

# Higher = more authoritative. Used as prior credibility before propagation.
_DOC_AUTHORITY: dict[str, float] = {
    "prospectus": 0.90,
    "annual report": 0.95,
    "annual-report": 0.95,
    "10-k": 0.95,
    "10-q": 0.85,
    "earnings": 0.80,
    "presentation": 0.75,
    "press release": 0.70,
    "economic survey": 0.90,
    "rbi": 0.92,
    "imf": 0.90,
    "article iv": 0.90,
}


def _estimate_doc_authority(filename: str) -> float:
    """Estimate document authority from filename keywords."""
    fn_lower = filename.lower()
    for keyword, score in _DOC_AUTHORITY.items():
        if keyword in fn_lower:
            return score
    return 0.75  # default


# ---------------------------------------------------------------------------
# Stage 1: Union-Find Claim Alignment
# ---------------------------------------------------------------------------

class _UnionFind:
    """Disjoint Set Union with path compression and union-by-rank."""

    def __init__(self):
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def clusters(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for x in self.parent:
            groups[self.find(x)].append(x)
        return dict(groups)


def _normalize_subject(subject: str) -> str:
    """Normalize subject for entity alignment."""
    import re
    s = subject.strip().lower().replace("'s", "").replace("'", "").replace("-", " ")
    for suffix in ("limited", "ltd", "inc", "corp", "corporation", "plc", "llc", "pvt", "private"):
        s = re.sub(r'\b' + suffix + r'\.?\b', '', s).strip()
    return s.strip('. ')


def _subjects_match(s1: str, s2: str) -> bool:
    """Check if two subjects refer to the same entity."""
    n1, n2 = _normalize_subject(s1), _normalize_subject(s2)
    if n1 == n2:
        return True
    if (n1 in n2 or n2 in n1) and min(len(n1), len(n2)) >= 4:
        return True
    if SequenceMatcher(None, n1, n2).ratio() > 0.80:
        return True

    try:
        from app.embeddings import is_semantic_match
        return is_semantic_match(s1, s2, kind="entity", threshold=0.55)
    except Exception:
        return False


def _predicates_match(p1: str, p2: str) -> bool:
    """Check if two predicates refer to the same metric."""
    if p1.lower().strip() == p2.lower().strip():
        return True
    try:
        from app.embeddings import is_semantic_match
        return is_semantic_match(p1, p2, kind="metric", threshold=0.55)
    except Exception:
        return False


def _align_facts(facts: list[Fact]) -> dict[int, list[int]]:
    """Stage 1: Group facts by (subject, predicate) using complete-linkage semantic alignment.

    Prevents transitive smearing (where A~B and B~C links unrelated metrics into a giant megacluster).
    Returns mapping from cluster root -> list of fact indices.
    """
    # Group by normalized (subject, predicate) — fast exact-match pass
    key_groups: dict[str, list[int]] = defaultdict(list)
    for i, fact in enumerate(facts):
        key = f"{_normalize_subject(fact.subject)}||{fact.predicate.lower()}"
        key_groups[key].append(i)

    # Cross-group semantic alignment pass using Voyage AI embeddings
    sorted_keys = sorted(key_groups.keys())
    unique_preds = list({k.split("||", 1)[1] for k in sorted_keys})
    unique_subjs = list({k.split("||", 1)[0] for k in sorted_keys})

    try:
        from app.embeddings import precompute_term_matches
        pred_matches = precompute_term_matches(unique_preds, kind="metric", threshold=0.65)
        subj_matches = precompute_term_matches(unique_subjs, kind="entity", threshold=0.65)
    except Exception as e:
        logger.warning("Embedding precomputation failed in claim_graph: %s", e)
        pred_matches = {}
        subj_matches = {}

    # Complete-linkage / Clique clustering on keys:
    # A key k can join an existing cluster only if it is a semantic match to ALL keys in that cluster.
    clusters_of_keys: list[list[str]] = []
    for k in sorted_keys:
        subj_k, pred_k = k.split("||", 1)
        matched_cluster_idx = -1

        for c_idx, clus in enumerate(clusters_of_keys):
            can_join = True
            for existing_k in clus:
                subj_e, pred_e = existing_k.split("||", 1)
                is_subj_match = subj_matches.get((subj_k, subj_e), _subjects_match(subj_k, subj_e))
                if not is_subj_match:
                    can_join = False
                    break
                is_pred_match = pred_matches.get((pred_k, pred_e), pred_k == pred_e)
                if not is_pred_match:
                    can_join = False
                    break
            if can_join:
                matched_cluster_idx = c_idx
                break

        if matched_cluster_idx >= 0:
            clusters_of_keys[matched_cluster_idx].append(k)
        else:
            clusters_of_keys.append([k])

    result: dict[int, list[int]] = {}
    for clus in clusters_of_keys:
        all_indices = []
        for k in clus:
            all_indices.extend(key_groups[k])
        if all_indices:
            result[all_indices[0]] = all_indices

    return result




# ---------------------------------------------------------------------------
# Stage 2: Evidence Graph — intra-cluster edge classification
# ---------------------------------------------------------------------------

def _build_cluster_edges(
    facts: list[Fact],
    cluster_indices: list[int],
    nli_threshold: float = 0.7,
) -> list[ClaimEdge]:
    """Run the 3-stage contradiction engine on facts within a cluster.

    Reuses the existing detect_contradictions() pipeline, but scoped to
    only the facts in this cluster.
    """
    if len(cluster_indices) < 2:
        return []

    # Extract the subset of facts
    cluster_facts = [facts[i] for i in cluster_indices]

    # Run contradiction detection on this subset
    report = detect_contradictions(
        facts=cluster_facts,
        doc_id="cluster",
        nli_threshold=nli_threshold,
    )

    # Remap indices back to global fact pool
    remapped_edges: list[ClaimEdge] = []
    for edge in report.edges:
        remapped_edges.append(ClaimEdge(
            source_fact_idx=cluster_indices[edge.source_fact_idx],
            target_fact_idx=cluster_indices[edge.target_fact_idx],
            edge_type=edge.edge_type,
            detection_method=edge.detection_method,
            confidence=edge.confidence,
            explanation=edge.explanation,
        ))

    return remapped_edges


# ---------------------------------------------------------------------------
# Stage 3: ArbGraph-style credibility propagation
# ---------------------------------------------------------------------------

def _propagate_credibility(
    facts: list[Fact],
    cluster_indices: list[int],
    edges: list[ClaimEdge],
    doc_filenames: dict[str, str],
    iterations: int = 3,
    damping: float = 0.15,
) -> dict[int, float]:
    """Intensity-driven credibility propagation (ArbGraph Section 3.3).

    Initial credibility = (quote_verification_confidence × doc_authority).
    Each iteration: nodes supported by high-credibility neighbors gain score;
    nodes contradicted by high-credibility neighbors lose score.
    """
    # Initialize credibility scores
    scores: dict[int, float] = {}
    for idx in cluster_indices:
        fact = facts[idx]
        doc_auth = _estimate_doc_authority(
            doc_filenames.get(fact.provenance.doc_id, "")
        )
        # Initial credibility: verification confidence × document authority
        scores[idx] = fact.confidence * doc_auth

    # Build adjacency
    support_edges: list[tuple[int, int, float]] = []
    contra_edges: list[tuple[int, int, float]] = []
    for edge in edges:
        if edge.edge_type == EdgeType.CORROBORATES:
            support_edges.append((edge.source_fact_idx, edge.target_fact_idx, edge.confidence))
        elif edge.edge_type == EdgeType.CONTRADICTS:
            contra_edges.append((edge.source_fact_idx, edge.target_fact_idx, edge.confidence))
        # SUPERSEDES edges don't affect credibility — they're informational

    # Iterative propagation
    for _ in range(iterations):
        new_scores = dict(scores)
        for src, tgt, conf in support_edges:
            # Mutual reinforcement
            boost = damping * scores[src] * conf
            new_scores[tgt] = min(1.0, new_scores[tgt] + boost)
            boost = damping * scores[tgt] * conf
            new_scores[src] = min(1.0, new_scores[src] + boost)

        for src, tgt, conf in contra_edges:
            # The less credible one gets suppressed
            if scores[src] > scores[tgt]:
                penalty = damping * conf * (scores[src] - scores[tgt])
                new_scores[tgt] = max(0.0, new_scores[tgt] - penalty)
            else:
                penalty = damping * conf * (scores[tgt] - scores[src])
                new_scores[src] = max(0.0, new_scores[src] - penalty)

        scores = new_scores

    return scores


# ---------------------------------------------------------------------------
# Stage 4: Cluster arbitration — case classification
# ---------------------------------------------------------------------------

def _scopes_differ(facts: list[Fact], indices: list[int]) -> bool:
    """Check if facts have meaningfully different scopes."""
    scopes = set()
    for idx in indices:
        s = (facts[idx].context.scope or "").strip().lower()
        if s:
            scopes.add(s)
    return len(scopes) > 1


def _temporals_differ(facts: list[Fact], indices: list[int]) -> bool:
    """Check if facts have meaningfully different temporal contexts."""
    temporals = set()
    for idx in indices:
        t = normalize_temporal(facts[idx].context.temporal)
        if t:
            temporals.add(t)
    return len(temporals) > 1


def _conditions_differ(facts: list[Fact], indices: list[int]) -> bool:
    """Check if facts have meaningfully different conditions."""
    conditions = set()
    for idx in indices:
        c = (facts[idx].context.conditions or "").strip().lower()
        if c:
            conditions.add(c)
    return len(conditions) > 1


def _units_differ(facts: list[Fact], indices: list[int]) -> bool:
    """Check if facts have meaningfully different units/scales."""
    units = set()
    for idx in indices:
        u = (facts[idx].canonical_unit or "").strip().lower()
        s = facts[idx].scale
        units.add(f"{u}|{s}")
    return len(units) > 1


def _classify_cluster(
    facts: list[Fact],
    cluster_indices: list[int],
    edges: list[ClaimEdge],
) -> tuple[CaseType, str]:
    """Determine which of the 4 assignment cases a cluster represents.

    Returns (CaseType, DisputeCode) — the case classification and
    the fine-grained dispute reason code.
    """
    if len(cluster_indices) < 2:
        return CaseType.CORROBORATED, DisputeCode.AGREEMENT_EXACT.value

    edge_types = {e.edge_type for e in edges}

    has_contradiction = EdgeType.CONTRADICTS in edge_types
    has_corroboration = EdgeType.CORROBORATES in edge_types
    has_supersedes = EdgeType.SUPERSEDES in edge_types

    if has_contradiction:
        contradicting_edges = [e for e in edges if e.edge_type == EdgeType.CONTRADICTS]
        all_reconciled = True
        reconciled_type = CaseType.CONTRADICTED
        dispute_code = DisputeCode.DISPUTE_GENUINE_CONFLICT.value

        for e in contradicting_edges:
            f1, f2 = facts[e.source_fact_idx], facts[e.target_fact_idx]
            t1 = normalize_temporal(f1.context.temporal)
            t2 = normalize_temporal(f2.context.temporal)
            if t1 and t2 and t1 != t2:
                reconciled_type = CaseType.RECONCILED_TEMPORAL
                dispute_code = DisputeCode.DISPUTE_TEMPORAL_DRIFT.value
                continue
            s1 = (f1.context.scope or "").strip().lower()
            s2 = (f2.context.scope or "").strip().lower()
            if s1 and s2 and s1 != s2:
                reconciled_type = CaseType.RECONCILED_SCOPE
                dispute_code = DisputeCode.DISPUTE_SCOPE_DIFFERENCE.value
                continue
            c1 = (f1.context.conditions or "").strip().lower()
            c2 = (f2.context.conditions or "").strip().lower()
            if c1 and c2 and c1 != c2:
                reconciled_type = CaseType.RECONCILED_CONDITIONS
                dispute_code = DisputeCode.DISPUTE_ACCOUNTING_BASIS.value
                continue
            u1 = (f1.canonical_unit or "").strip().lower()
            u2 = (f2.canonical_unit or "").strip().lower()
            if u1 and u2 and u1 != u2:
                reconciled_type = CaseType.RECONCILED_UNIT
                dispute_code = DisputeCode.DISPUTE_UNIT_MISMATCH.value
                continue
            # Un-reconciled contradiction — sub-classify the dispute
            all_reconciled = False
            dispute_code = _sub_classify_dispute(f1, f2)
            break

        if all_reconciled and reconciled_type != CaseType.CONTRADICTED:
            return reconciled_type, dispute_code
        return CaseType.CONTRADICTED, dispute_code

    if has_supersedes:
        return CaseType.RECONCILED_TEMPORAL, DisputeCode.DISPUTE_TEMPORAL_DRIFT.value

    # All corroborated — check if exact or approximate
    corr_edges = [e for e in edges if e.edge_type == EdgeType.CORROBORATES]
    if corr_edges:
        # Check if any corroboration edge has relative diff > 0 (approximate match)
        for e in corr_edges:
            if e.detection_method == "symbolic_numeric" and "≈" in e.explanation:
                return CaseType.CORROBORATED, DisputeCode.AGREEMENT_APPROXIMATE.value
    return CaseType.CORROBORATED, DisputeCode.AGREEMENT_EXACT.value


def _sub_classify_dispute(f1: Fact, f2: Fact) -> str:
    """Sub-classify a genuine contradiction into a fine-grained dispute code."""
    v1, v2 = f1.canonical_value, f2.canonical_value

    if v1 is not None and v2 is not None:
        # Sign mismatch: one positive, one negative (profit vs loss)
        if (v1 > 0 and v2 < 0) or (v1 < 0 and v2 > 0):
            return DisputeCode.DISPUTE_SIGN_MISMATCH.value

        # Order of magnitude: >10x difference (likely a scale extraction error)
        max_abs = max(abs(v1), abs(v2))
        min_abs = min(abs(v1), abs(v2))
        if max_abs > 0 and min_abs > 0:
            ratio = max_abs / min_abs
            if ratio >= 10.0:
                return DisputeCode.DISPUTE_ORDER_OF_MAGNITUDE.value
            if ratio < 1.01:
                return DisputeCode.DISPUTE_ROUNDING.value

    return DisputeCode.DISPUTE_GENUINE_CONFLICT.value


def _generate_explanation(
    facts: list[Fact],
    cluster_indices: list[int],
    edges: list[ClaimEdge],
    case_type: CaseType,
    credibility: dict[int, float],
    doc_filenames: dict[str, str],
) -> str:
    """Generate a human-readable explanation for the cluster's case classification."""
    subject = facts[cluster_indices[0]].subject
    predicate = facts[cluster_indices[0]].predicate.replace("_", " ")
    n_facts = len(cluster_indices)
    doc_ids = {facts[i].provenance.doc_id for i in cluster_indices}
    n_docs = len(doc_ids)
    doc_names = [doc_filenames.get(d, d)[:40] for d in doc_ids]

    if case_type == CaseType.CORROBORATED:
        values = [facts[i].value for i in cluster_indices]
        return (
            f"CORROBORATED: '{subject}' / '{predicate}' is confirmed across "
            f"{n_docs} document(s) ({', '.join(doc_names)}). "
            f"Values reported: {', '.join(values)}. "
            f"All sources agree within numeric tolerance."
        )

    elif case_type == CaseType.CONTRADICTED:
        # Find the contradiction edge
        contra_edges = [e for e in edges if e.edge_type == EdgeType.CONTRADICTS]
        parts = []
        for e in contra_edges[:3]:  # limit to 3 examples
            fi, fj = facts[e.source_fact_idx], facts[e.target_fact_idx]
            fi_doc = doc_filenames.get(fi.provenance.doc_id, fi.provenance.doc_id)[:30]
            fj_doc = doc_filenames.get(fj.provenance.doc_id, fj.provenance.doc_id)[:30]
            parts.append(
                f"{fi_doc} (p.{fi.provenance.page}) states '{fi.value}' "
                f"but {fj_doc} (p.{fj.provenance.page}) states '{fj.value}'"
            )
        # Identify most credible
        best_idx = max(cluster_indices, key=lambda i: credibility.get(i, 0))
        best_fact = facts[best_idx]
        return (
            f"GENUINE CONTRADICTION: '{subject}' / '{predicate}' has conflicting values "
            f"across {n_docs} document(s). {'; '.join(parts)}. "
            f"Most credible value: {best_fact.value} "
            f"(credibility {credibility.get(best_idx, 0):.2f}, "
            f"from {doc_filenames.get(best_fact.provenance.doc_id, '')[:30]})."
        )

    elif case_type in (CaseType.RECONCILED_TEMPORAL, CaseType.RECONCILED_SCOPE,
                       CaseType.RECONCILED_UNIT, CaseType.RECONCILED_CONDITIONS):
        # Determine reconciliation dimension
        dimension_map = {
            CaseType.RECONCILED_TEMPORAL: ("fiscal period/date", "temporal"),
            CaseType.RECONCILED_SCOPE: ("reporting scope", "scope"),
            CaseType.RECONCILED_UNIT: ("unit/scale", "unit"),
            CaseType.RECONCILED_CONDITIONS: ("accounting conditions", "conditions"),
        }
        dim_label, dim_field = dimension_map[case_type]

        # Collect the differing context values
        context_vals = []
        for idx in cluster_indices:
            f = facts[idx]
            if dim_field == "temporal":
                cv = f.context.temporal or "unspecified"
            elif dim_field == "scope":
                cv = f.context.scope or "unspecified"
            elif dim_field == "conditions":
                cv = f.context.conditions or "unspecified"
            else:
                cv = f.canonical_unit or "unspecified"
            doc = doc_filenames.get(f.provenance.doc_id, f.provenance.doc_id)[:30]
            context_vals.append(f"{doc}: {f.value} ({cv})")

        return (
            f"APPARENT CONTRADICTION RECONCILED BY {dim_label.upper()}: "
            f"'{subject}' / '{predicate}' appears to conflict across documents, "
            f"but the values refer to different {dim_label}s. "
            f"{'; '.join(context_vals)}."
        )

    return f"Cluster of {n_facts} facts for '{subject}' / '{predicate}'"


def _build_evidence_entries(
    facts: list[Fact],
    cluster_indices: list[int],
    doc_filenames: dict[str, str],
) -> list[EvidenceEntry]:
    """Build side-by-side evidence for a cluster."""
    entries = []
    for idx in cluster_indices:
        f = facts[idx]
        entries.append(EvidenceEntry(
            doc_id=f.provenance.doc_id,
            doc_filename=doc_filenames.get(f.provenance.doc_id, f.provenance.doc_id),
            page=f.provenance.page,
            value=f.value,
            canonical_value=f.canonical_value,
            canonical_unit=f.canonical_unit,
            temporal=f.context.temporal,
            scope=f.context.scope,
            conditions=f.context.conditions,
            evidence_quote=f.provenance.evidence_quote,
            confidence=f.confidence,
            match_type=f.provenance.match_type,
        ))
    return entries


# ---------------------------------------------------------------------------
# Extraction failure detection (Case 4)
# ---------------------------------------------------------------------------

def _detect_extraction_failures(
    facts: list[Fact],
    doc_filenames: dict[str, str],
) -> list[ExtractionFailure]:
    """Scan the fact pool for extraction or reasoning failures."""
    failures: list[ExtractionFailure] = []

    for idx, f in enumerate(facts):
        doc_name = doc_filenames.get(f.provenance.doc_id, f.provenance.doc_id)

        # 1. Unverified quote (LLM hallucinated a citation)
        if not f.provenance.verified and f.provenance.evidence_quote:
            failures.append(ExtractionFailure(
                failure_type="unverified_quote",
                fact_index=idx,
                doc_id=f.provenance.doc_id,
                page=f.provenance.page,
                description=(
                    f"Quote not found in source PDF: '{f.provenance.evidence_quote[:80]}...' "
                    f"(from {doc_name}, page {f.provenance.page}). "
                    f"The LLM generated a plausible but non-verbatim citation."
                ),
                mitigation=(
                    "3-tier verification (exact → normalized → fuzzy) assigned confidence=0.10. "
                    "Fact is retained but flagged as low-confidence. "
                    "Improvement: re-prompt the LLM with the exact page text and ask for a verbatim substring."
                ),
            ))

        # 2. Scale word leaked into base_unit
        if f.base_unit:
            scale_words = ["million", "mn", "billion", "bn", "crore", "cr", "lakh", "thousand"]
            found = [sw for sw in scale_words if sw in f.base_unit.lower()]
            if found:
                failures.append(ExtractionFailure(
                    failure_type="scale_in_unit",
                    fact_index=idx,
                    doc_id=f.provenance.doc_id,
                    page=f.provenance.page,
                    description=(
                        f"base_unit='{f.base_unit}' contains scale word(s) {found}. "
                        f"The LLM should have separated the scale multiplier into the 'scale' field. "
                        f"Value: '{f.value}', canonical_value: {f.canonical_value}."
                    ),
                    mitigation=(
                        "Post-extraction normalizer attempts to parse scale from unit string. "
                        "Improvement: add explicit examples of correct vs. incorrect unit/scale "
                        "separation in the system prompt."
                    ),
                ))

        # 3. Monetary fact with scale=0 (likely missed table header)
        if (f.dimension and f.dimension.value == "monetary"
                and f.scale == 0 and f.numeric_value is not None
                and abs(f.numeric_value) > 100):
            failures.append(ExtractionFailure(
                failure_type="scale_mismatch",
                fact_index=idx,
                doc_id=f.provenance.doc_id,
                page=f.provenance.page,
                description=(
                    f"Monetary fact with scale=0 but value={f.value} appears large enough "
                    f"to likely be in millions/crores. The LLM may have missed the table header "
                    f"indicating '(₹ in million)' or similar."
                ),
                mitigation=(
                    "Normalizer checks for scale clues in the value string. "
                    "Improvement: pre-extract table headers/footnotes and inject them into "
                    "the LLM prompt as explicit context."
                ),
            ))

        # 4. Numeric fact with missing dimension classification
        if f.numeric_value is not None and f.dimension is None:
            failures.append(ExtractionFailure(
                failure_type="missing_dimension",
                fact_index=idx,
                doc_id=f.provenance.doc_id,
                page=f.provenance.page,
                description=(
                    f"Numeric fact (value='{f.value}', numeric={f.numeric_value}) "
                    f"has no dimension classification. Without knowing if this is monetary, "
                    f"a count, or a percentage, cross-document comparison may produce false matches."
                ),
                mitigation=(
                    "The normalizer infers dimension from base_unit when possible. "
                    "Improvement: add dimension as a required (not optional) field in the "
                    "extraction schema."
                ),
            ))

    return failures


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_claim_graph(
    doc_facts: dict[str, list[Fact]],
    doc_filenames: dict[str, str],
    nli_threshold: float = 0.7,
) -> ClaimGraph:
    """Build the complete cross-document claim graph.

    This is the main entry point implementing the ArbGraph 4-stage pipeline:
        1. Claim Alignment (Union-Find)
        2. Evidence Graph (intra-cluster edge classification)
        3. Credibility Propagation
        4. Cluster Arbitration (case classification + explanation)

    Args:
        doc_facts: Mapping of doc_id → list of extracted Fact objects.
        doc_filenames: Mapping of doc_id → original filename.
        nli_threshold: Minimum confidence for DeBERTa NLI edges.

    Returns:
        A ClaimGraph with typed clusters, evidence, and explanations.
    """
    t_start = time.time()

    # --- Merge all facts into a global pool with stable indices ---
    all_facts: list[Fact] = []
    doc_id_order: list[str] = []
    for doc_id in sorted(doc_facts.keys()):
        facts = doc_facts[doc_id]
        all_facts.extend(facts)
        doc_id_order.append(doc_id)

    total_facts = len(all_facts)
    if total_facts == 0:
        return ClaimGraph(
            documents=doc_filenames,
            total_facts=0,
            pipeline_metadata={"duration_seconds": 0.0},
        )

    logger.info("Building claim graph: %d facts across %d documents", total_facts, len(doc_facts))

    # --- Stage 1: Claim Alignment ---
    t_align = time.time()
    raw_clusters = _align_facts(all_facts)
    logger.info("Stage 1 (Alignment): %d clusters from %d facts in %.2fs",
                len(raw_clusters), total_facts, time.time() - t_align)

    # --- Stage 2: Evidence Graph (edges) + Stage 3: Credibility ---
    t_edges = time.time()
    clusters: list[FactCluster] = []
    unmatched: list[int] = []

    for cluster_root, indices in raw_clusters.items():
        if len(indices) == 1:
            # Single fact — check if it has multi-doc potential
            unmatched.append(indices[0])
            continue

        # Build edges within this cluster
        edges = _build_cluster_edges(all_facts, indices, nli_threshold=nli_threshold)

        # Credibility propagation
        credibility = _propagate_credibility(
            all_facts, indices, edges, doc_filenames
        )

        # Classify the cluster and assign dispute code
        case_type, dispute_code = _classify_cluster(all_facts, indices, edges)

        # Generate explanation
        explanation = _generate_explanation(
            all_facts, indices, edges, case_type, credibility, doc_filenames
        )

        # Build side-by-side evidence
        evidence = _build_evidence_entries(all_facts, indices, doc_filenames)

        # Determine consensus value (highest credibility fact)
        consensus_val = None
        consensus_unit = None
        if credibility:
            best_idx = max(indices, key=lambda i: credibility.get(i, 0))
            consensus_val = all_facts[best_idx].canonical_value
            consensus_unit = all_facts[best_idx].canonical_unit

        # Count distinct documents
        doc_ids_in_cluster = {all_facts[i].provenance.doc_id for i in indices}

        cluster_id = hashlib.md5(
            f"{all_facts[indices[0]].subject}|{all_facts[indices[0]].predicate}|{cluster_root}".encode()
        ).hexdigest()[:8]

        clusters.append(FactCluster(
            cluster_id=cluster_id,
            subject=all_facts[indices[0]].subject,
            predicate=all_facts[indices[0]].predicate,
            fact_indices=indices,
            edges=edges,
            case_type=case_type,
            dispute_code=dispute_code,
            evidence=evidence,
            explanation=explanation,
            credibility_scores=credibility,
            consensus_value=consensus_val,
            consensus_unit=consensus_unit,
            doc_count=len(doc_ids_in_cluster),
        ))

    logger.info("Stage 2-3 (Edges + Credibility): %d multi-fact clusters, %d singletons in %.2fs",
                len(clusters), len(unmatched), time.time() - t_edges)

    # --- Extraction failure detection ---
    failures = _detect_extraction_failures(all_facts, doc_filenames)
    logger.info("Case 4 scan: %d extraction failures detected", len(failures))

    # --- Sort clusters: multi-doc first, then by case importance ---
    case_priority = {
        CaseType.CONTRADICTED: 0,
        CaseType.RECONCILED_TEMPORAL: 1,
        CaseType.RECONCILED_SCOPE: 2,
        CaseType.RECONCILED_CONDITIONS: 3,
        CaseType.RECONCILED_UNIT: 4,
        CaseType.CORROBORATED: 5,
    }
    clusters.sort(key=lambda c: (
        -c.doc_count,                          # Multi-doc clusters first
        case_priority.get(c.case_type, 99),    # Most interesting cases first
        -len(c.edges),                         # More edges = more interesting
    ))

    # --- Build case summary ---
    case_summary: dict[str, int] = defaultdict(int)
    for c in clusters:
        case_summary[c.case_type.value] += 1
    if failures:
        case_summary[CaseType.EXTRACTION_FAILURE.value] = len(failures)

    duration = time.time() - t_start
    logger.info("Claim graph complete in %.2fs: %d clusters, %d failures",
                duration, len(clusters), len(failures))

    return ClaimGraph(
        documents=doc_filenames,
        total_facts=total_facts,
        clusters=clusters,
        extraction_failures=failures,
        unmatched_facts=unmatched,
        case_summary=dict(case_summary),
        pipeline_metadata={
            "duration_seconds": round(duration, 3),
            "alignment_method": "union_find",
            "edge_classification": "symbolic_numeric + temporal + nli_deberta",
            "credibility_method": "arbgraph_intensity_propagation",
            "nli_threshold": nli_threshold,
            "document_count": len(doc_facts),
            "total_facts": total_facts,
            "cluster_count": len(clusters),
            "singleton_count": len(unmatched),
        },
    )


def get_assignment_cases(graph: ClaimGraph) -> dict:
    """Extract exactly the 4 required assignment cases from the claim graph.

    Returns a structured dict with one best example per case type,
    ready for API response or demo presentation.
    """
    cases: dict[str, dict | None] = {
        "case_1_corroborated": None,
        "case_2_contradicted": None,
        "case_3_reconciled": None,
        "case_4_extraction_failure": None,
    }

    # Case 1: Best corroboration
    # Find clusters with cross-document corroboration
    cross_doc_corroborated = []
    for c in graph.clusters:
        cross_corr = [
            e for e in c.edges
            if e.edge_type == EdgeType.CORROBORATES
            and c.evidence[c.fact_indices.index(e.source_fact_idx)].doc_id != c.evidence[c.fact_indices.index(e.target_fact_idx)].doc_id
        ]
        if cross_corr:
            cross_doc_corroborated.append((c, cross_corr))

    chosen_c1_id = None
    if cross_doc_corroborated:
        # Prefer clusters that are primarily focused / cohesive (e.g. 2 to 10 facts)
        cross_doc_corroborated.sort(key=lambda x: (x[0].doc_count, -abs(len(x[0].evidence) - 5), len(x[1])), reverse=True)
        best_c, corr_edges = cross_doc_corroborated[0]
        chosen_c1_id = best_c.cluster_id
        cases["case_1_corroborated"] = {
            "cluster_id": best_c.cluster_id,
            "subject": best_c.subject,
            "predicate": best_c.predicate,
            "case_type": CaseType.CORROBORATED.value,
            "documents_involved": best_c.doc_count,
            "evidence": [e.model_dump() for e in best_c.evidence],
            "corroborating_edges": [e.model_dump(mode="json") for e in corr_edges],
            "explanation": f"INDEPENDENT CORROBORATION: Independent sources confirm matching figures for '{best_c.subject}' / '{best_c.predicate}'. ({corr_edges[0].explanation})",
        }
    else:
        corroborated = [c for c in graph.clusters if c.case_type == CaseType.CORROBORATED]
        if corroborated:
            best = max(corroborated, key=lambda c: (c.doc_count, len(c.edges)))
            chosen_c1_id = best.cluster_id
            cases["case_1_corroborated"] = {
                "cluster_id": best.cluster_id,
                "subject": best.subject,
                "predicate": best.predicate,
                "case_type": CaseType.CORROBORATED.value,
                "documents_involved": best.doc_count,
                "evidence": [e.model_dump() for e in best.evidence],
                "corroborating_edges": [e.model_dump(mode="json") for e in best.edges if e.edge_type == EdgeType.CORROBORATES],
                "explanation": best.explanation,
            }

    # Case 2: Best genuine contradiction (must be distinct from Case 1)
    contradicted = [c for c in graph.clusters if c.case_type == CaseType.CONTRADICTED and c.cluster_id != chosen_c1_id]
    if not contradicted:
        contradicted = [c for c in graph.clusters if c.case_type == CaseType.CONTRADICTED]

    if contradicted:
        def _contra_score(c):
            same_period_cross = 0
            for e in c.edges:
                if e.edge_type == EdgeType.CONTRADICTS:
                    f1 = c.evidence[c.fact_indices.index(e.source_fact_idx)]
                    f2 = c.evidence[c.fact_indices.index(e.target_fact_idx)]
                    if f1.doc_id != f2.doc_id:
                        t1 = normalize_temporal(f1.temporal)
                        t2 = normalize_temporal(f2.temporal)
                        if t1 == t2 and t1 is not None:
                            same_period_cross += 10
                        else:
                            same_period_cross += 1
            size_penalty = -abs(len(c.evidence) - 4)
            return (same_period_cross, c.doc_count, size_penalty)

        contradicted.sort(key=_contra_score, reverse=True)
        best = contradicted[0]
        cases["case_2_contradicted"] = {
            "cluster_id": best.cluster_id,
            "subject": best.subject,
            "predicate": best.predicate,
            "case_type": best.case_type.value,
            "documents_involved": best.doc_count,
            "evidence": [e.model_dump() for e in best.evidence],
            "edges": [e.model_dump(mode="json") for e in best.edges],
            "explanation": best.explanation,
            "consensus_value": best.consensus_value,
            "credibility_scores": best.credibility_scores,
        }

    # Case 3: Best reconciled apparent contradiction (any reconciliation type)
    reconciled_types = {
        CaseType.RECONCILED_TEMPORAL, CaseType.RECONCILED_SCOPE,
        CaseType.RECONCILED_UNIT, CaseType.RECONCILED_CONDITIONS,
    }
    reconciled = [c for c in graph.clusters if c.case_type in reconciled_types]
    if reconciled:
        best = max(reconciled, key=lambda c: (c.doc_count, len(c.edges)))
        cases["case_3_reconciled"] = {
            "cluster_id": best.cluster_id,
            "subject": best.subject,
            "predicate": best.predicate,
            "case_type": best.case_type.value,
            "reconciliation_dimension": best.case_type.value.replace("reconciled_", ""),
            "documents_involved": best.doc_count,
            "evidence": [e.model_dump() for e in best.evidence],
            "edges": [e.model_dump(mode="json") for e in best.edges],
            "explanation": best.explanation,
        }

    # Case 4: Best extraction failure
    if graph.extraction_failures:
        best = graph.extraction_failures[0]
        cases["case_4_extraction_failure"] = best.model_dump()

    return cases
