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

# Generic document-type taxonomy — keywords are stored in normalized form
# (lowercase, spaces only). The lookup function normalizes filenames the
# same way, so "annual-report", "annual_report", and "annual report" all
# match the single key "annual report".
#
# Higher = more authoritative.  Used as prior credibility before propagation.
# Callers can override per-document via the doc_authority_overrides parameter.
_DOC_TYPE_AUTHORITY: dict[str, float] = {
    # ── Audited / regulatory filings (highest authority) ──
    "annual report": 0.95,
    "audited financial": 0.95,
    "10 k": 0.95,          # SEC annual filing
    "20 f": 0.95,          # SEC foreign private issuer annual
    "registration statement": 0.95,
    # ── Prospectus / offering documents ──
    "prospectus": 0.90,
    "offering memorandum": 0.90,
    "drhp": 0.90,          # Draft Red Herring Prospectus (India)
    "red herring": 0.90,
    # ── Government / institutional research ──
    "economic survey": 0.90,
    "central bank": 0.90,
    "monetary policy": 0.90,
    "article iv": 0.90,    # IMF Article IV consultation
    "world economic outlook": 0.90,
    "white paper": 0.85,
    # ── Quarterly / interim filings ──
    "10 q": 0.85,          # SEC quarterly filing
    "quarterly report": 0.85,
    "interim report": 0.85,
    # ── Earnings & investor communications ──
    "earnings": 0.80,
    "investor presentation": 0.80,
    "earnings call": 0.80,
    "shareholder letter": 0.80,
    # ── Research / analyst reports ──
    "research report": 0.78,
    "analyst report": 0.78,
    # ── Lower-authority communications ──
    "press release": 0.70,
    "news release": 0.70,
    "blog": 0.60,
    "presentation": 0.65,
}

DEFAULT_DOC_AUTHORITY: float = 0.75


def _normalize_for_lookup(text: str) -> str:
    """Normalize text for authority keyword matching.

    Replaces hyphens, underscores, dots, and extra whitespace with single
    spaces so that 'annual-report', 'annual_report', and 'annual report'
    all produce the same normalized form.
    """
    import re
    return re.sub(r'[\s_\-./]+', ' ', text.lower()).strip()


def _estimate_doc_authority(filename: str) -> float:
    """Estimate document authority from filename keywords.

    Uses normalized keyword matching: both the filename and all dictionary
    keys are normalized (hyphens/underscores → spaces) before comparison.
    Returns DEFAULT_DOC_AUTHORITY (0.75) if no keyword matches.
    """
    fn_norm = _normalize_for_lookup(filename)
    for keyword, score in _DOC_TYPE_AUTHORITY.items():
        if keyword in fn_norm:
            return score
    return DEFAULT_DOC_AUTHORITY


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




# Maximum batch size for pairwise NLI comparison.
# A batch of 50 facts = 1,225 pairs max — keeps DeBERTa inference under ~0.5s per batch.
# For clusters larger than 50, facts are processed in iterative sets of 50 using
# high-confidence cluster anchors, ensuring 100% of facts are checked without O(n²) explosion.
MAX_CLUSTER_NLI_SIZE: int = 50
CLUSTER_ANCHOR_SIZE: int = 10


def _build_cluster_edges(
    facts: list[Fact],
    cluster_indices: list[int],
    nli_threshold: float = 0.7,
) -> list[ClaimEdge]:
    """Run the 3-stage contradiction engine on facts within a cluster in sets of 50.

    Reuses the existing detect_contradictions() pipeline, but scoped to
    only the facts in this cluster.

    - Clusters <= 50 facts: evaluated in a single batch (all pairwise combinations).
    - Clusters > 50 facts: evaluated in iterative sets of 50:
      1. Top 10 highest-confidence facts are designated as cluster 'anchors'.
      2. Remaining facts are partitioned into chunks of 40.
      3. Each set combines [10 anchors + 40 chunk facts] = 50 facts max.
      4. Every fact is tested against the anchors and within its batch; all edges are
         merged and deduplicated so NO facts are skipped.
    """
    if len(cluster_indices) < 2:
        return []

    sorted_indices = sorted(
        cluster_indices,
        key=lambda idx: facts[idx].confidence,
        reverse=True,
    )

    # If small enough, run standard single batch
    if len(sorted_indices) <= MAX_CLUSTER_NLI_SIZE:
        batches = [sorted_indices]
    else:
        # Multi-batch: retain top anchors across all batches
        anchor_count = min(CLUSTER_ANCHOR_SIZE, len(sorted_indices) // 2)
        anchors = sorted_indices[:anchor_count]
        remaining = sorted_indices[anchor_count:]
        chunk_size = max(1, MAX_CLUSTER_NLI_SIZE - anchor_count)

        batches = []
        for i in range(0, len(remaining), chunk_size):
            chunk = remaining[i : i + chunk_size]
            batches.append(anchors + chunk)

    remapped_edges: list[ClaimEdge] = []
    seen_edge_keys: set[tuple[int, int, str]] = set()

    for batch in batches:
        cluster_facts = [facts[i] for i in batch]
        report = detect_contradictions(
            facts=cluster_facts,
            doc_id="cluster",
            nli_threshold=nli_threshold,
        )

        for edge in report.edges:
            src_global = batch[edge.source_fact_idx]
            tgt_global = batch[edge.target_fact_idx]
            key = (min(src_global, tgt_global), max(src_global, tgt_global), edge.edge_type.value)
            if key not in seen_edge_keys:
                seen_edge_keys.add(key)
                remapped_edges.append(ClaimEdge(
                    source_fact_idx=src_global,
                    target_fact_idx=tgt_global,
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
    doc_authority_overrides: dict[str, float] | None = None,
) -> dict[int, float]:
    """Intensity-driven credibility propagation (ArbGraph Section 3.3).

    Initial credibility = (quote_verification_confidence × doc_authority).
    Each iteration: nodes supported by high-credibility neighbors gain score;
    nodes contradicted by high-credibility neighbors lose score.

    Args:
        doc_authority_overrides: Optional mapping of doc_id → authority score
            (0.0–1.0). When provided, these take priority over the filename
            heuristic for the specified documents.
    """
    overrides = doc_authority_overrides or {}

    # Initialize credibility scores
    scores: dict[int, float] = {}
    for idx in cluster_indices:
        fact = facts[idx]
        doc_id = fact.provenance.doc_id
        if doc_id in overrides:
            doc_auth = overrides[doc_id]
        else:
            doc_auth = _estimate_doc_authority(
                doc_filenames.get(doc_id, "")
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
    doc_authority_overrides: dict[str, float] | None = None,
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
        doc_authority_overrides: Optional mapping of doc_id → authority
            score (0.0–1.0).  When provided, these take priority over
            the filename-based heuristic for credibility propagation.

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
            all_facts, indices, edges, doc_filenames,
            doc_authority_overrides=doc_authority_overrides,
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


def append_document_to_claim_graph(
    existing_graph: ClaimGraph | None,
    existing_doc_facts: dict[str, list[Fact]],
    new_doc_id: str,
    new_doc_facts: list[Fact],
    new_doc_filename: str,
    nli_threshold: float = 0.7,
    doc_authority_overrides: dict[str, float] | None = None,
) -> tuple[ClaimGraph, dict]:
    """Incrementally update a ClaimGraph with facts from a new document.

    Streaming entity resolution & neighborhood re-arbitration pattern:
    - New facts are routed to existing clusters by semantic alignment.
    - Matching an existing singleton upgrades it to a multi-fact cluster.
    - Only tainted/affected clusters have edges and credibility re-arbitrated.
    - Unaffected clusters are preserved with zero re-computation.
    """
    t_start = time.time()

    if existing_graph is None or existing_graph.total_facts == 0:
        all_docs = {new_doc_id: new_doc_facts}
        all_names = {new_doc_id: new_doc_filename}
        graph = build_claim_graph(
            all_docs, all_names,
            nli_threshold=nli_threshold,
            doc_authority_overrides=doc_authority_overrides,
        )
        stats = {
            "new_facts_count": len(new_doc_facts),
            "clusters_updated": 0,
            "clusters_created": len(graph.clusters),
            "new_singletons": len(graph.unmatched_facts),
            "unaffected_clusters": 0,
            "duration_seconds": round(time.time() - t_start, 3),
        }
        return graph, stats

    # Reconstruct stable all_facts order for existing facts
    all_facts: list[Fact] = []
    for doc_id in sorted(existing_doc_facts.keys()):
        all_facts.extend(existing_doc_facts[doc_id])

    base_offset = len(all_facts)
    all_facts.extend(new_doc_facts)
    total_facts = len(all_facts)

    # Document filenames map
    doc_filenames = dict(existing_graph.documents)
    doc_filenames[new_doc_id] = new_doc_filename

    # Track affected clusters and singletons
    existing_clusters: list[FactCluster] = [c.model_copy(deep=True) for c in existing_graph.clusters]
    cluster_by_id = {c.cluster_id: c for c in existing_clusters}
    dirty_cluster_ids: set[str] = set()
    initial_cluster_ids = set(cluster_by_id.keys())

    unmatched_set: set[int] = set(existing_graph.unmatched_facts)
    unassigned_new_indices: list[int] = []

    # Step 1: Route each new fact to an existing cluster or match an existing singleton
    for i, new_fact in enumerate(new_doc_facts):
        new_global_idx = base_offset + i
        matched_cluster = False

        # 1a. Check existing clusters
        for c in existing_clusters:
            if _subjects_match(new_fact.subject, c.subject) and _predicates_match(new_fact.predicate, c.predicate):
                c.fact_indices.append(new_global_idx)
                dirty_cluster_ids.add(c.cluster_id)
                matched_cluster = True
                break

        if matched_cluster:
            continue

        # 1b. Check existing singletons in unmatched_set
        matched_singleton_idx = None
        for s_idx in sorted(list(unmatched_set)):
            s_fact = all_facts[s_idx]
            if _subjects_match(new_fact.subject, s_fact.subject) and _predicates_match(new_fact.predicate, s_fact.predicate):
                matched_singleton_idx = s_idx
                break

        if matched_singleton_idx is not None:
            unmatched_set.remove(matched_singleton_idx)
            # Create a new cluster from the matched singleton and this new fact
            new_cluster_id = hashlib.md5(
                f"{all_facts[matched_singleton_idx].subject}|{all_facts[matched_singleton_idx].predicate}|{matched_singleton_idx}_{new_global_idx}".encode()
            ).hexdigest()[:8]
            new_cluster = FactCluster(
                cluster_id=new_cluster_id,
                subject=all_facts[matched_singleton_idx].subject,
                predicate=all_facts[matched_singleton_idx].predicate,
                fact_indices=[matched_singleton_idx, new_global_idx],
                case_type=CaseType.CORROBORATED,
                dispute_code=DisputeCode.AGREEMENT_EXACT.value,
                evidence=[],
                explanation="",
            )
            existing_clusters.append(new_cluster)
            cluster_by_id[new_cluster_id] = new_cluster
            dirty_cluster_ids.add(new_cluster_id)
            continue

        # 1c. Not matched to existing clusters or singletons
        unassigned_new_indices.append(new_global_idx)

    # Step 2: Check if unassigned new facts match each other
    clusters_created_from_new = 0
    if len(unassigned_new_indices) > 1:
        unassigned_facts = [all_facts[idx] for idx in unassigned_new_indices]
        local_clusters = _align_facts(unassigned_facts)
        for root_local, local_indices in local_clusters.items():
            global_indices = [unassigned_new_indices[li] for li in local_indices]
            if len(global_indices) >= 2:
                new_c_id = hashlib.md5(
                    f"{all_facts[global_indices[0]].subject}|{all_facts[global_indices[0]].predicate}|inc_{global_indices[0]}".encode()
                ).hexdigest()[:8]
                new_cluster = FactCluster(
                    cluster_id=new_c_id,
                    subject=all_facts[global_indices[0]].subject,
                    predicate=all_facts[global_indices[0]].predicate,
                    fact_indices=global_indices,
                    case_type=CaseType.CORROBORATED,
                    dispute_code=DisputeCode.AGREEMENT_EXACT.value,
                    evidence=[],
                    explanation="",
                )
                existing_clusters.append(new_cluster)
                cluster_by_id[new_c_id] = new_cluster
                dirty_cluster_ids.add(new_c_id)
                clusters_created_from_new += 1
            else:
                unmatched_set.add(global_indices[0])
    elif len(unassigned_new_indices) == 1:
        unmatched_set.add(unassigned_new_indices[0])

    # Step 3: Re-arbitrate ONLY dirty clusters
    clusters_updated_count = 0
    for c_id in dirty_cluster_ids:
        c = cluster_by_id[c_id]
        clusters_updated_count += 1

        # Build edges within this updated cluster
        edges = _build_cluster_edges(all_facts, c.fact_indices, nli_threshold=nli_threshold)
        c.edges = edges

        # Credibility propagation
        credibility = _propagate_credibility(
            all_facts, c.fact_indices, edges, doc_filenames,
            doc_authority_overrides=doc_authority_overrides,
        )
        c.credibility_scores = credibility

        # Case classification and dispute code
        case_type, dispute_code = _classify_cluster(all_facts, c.fact_indices, edges)
        c.case_type = case_type
        c.dispute_code = dispute_code

        # Explanation
        c.explanation = _generate_explanation(
            all_facts, c.fact_indices, edges, case_type, credibility, doc_filenames
        )

        # Evidence
        c.evidence = _build_evidence_entries(all_facts, c.fact_indices, doc_filenames)

        # Consensus
        if credibility:
            best_idx = max(c.fact_indices, key=lambda i: credibility.get(i, 0))
            c.consensus_value = all_facts[best_idx].canonical_value
            c.consensus_unit = all_facts[best_idx].canonical_unit

        c.doc_count = len({all_facts[i].provenance.doc_id for i in c.fact_indices})

    # Step 4: Scan new facts for extraction failures (Case 4)
    new_failures = _detect_extraction_failures(new_doc_facts, doc_filenames)
    all_failures = list(existing_graph.extraction_failures)
    for f in new_failures:
        if f.fact_index is not None:
            f.fact_index += base_offset
        all_failures.append(f)

    # Step 5: Sort clusters and assemble summary
    case_priority = {
        CaseType.CONTRADICTED: 0,
        CaseType.RECONCILED_TEMPORAL: 1,
        CaseType.RECONCILED_SCOPE: 2,
        CaseType.RECONCILED_CONDITIONS: 3,
        CaseType.RECONCILED_UNIT: 4,
        CaseType.CORROBORATED: 5,
    }
    existing_clusters.sort(key=lambda c: (
        -c.doc_count,
        case_priority.get(c.case_type, 99),
        -len(c.edges),
    ))

    case_summary: dict[str, int] = defaultdict(int)
    for c in existing_clusters:
        case_summary[c.case_type.value] += 1
    if all_failures:
        case_summary[CaseType.EXTRACTION_FAILURE.value] = len(all_failures)

    duration = time.time() - t_start
    unaffected_count = len(existing_clusters) - len(dirty_cluster_ids)

    updated_graph = ClaimGraph(
        documents=doc_filenames,
        total_facts=total_facts,
        clusters=existing_clusters,
        extraction_failures=all_failures,
        unmatched_facts=sorted(list(unmatched_set)),
        case_summary=dict(case_summary),
        pipeline_metadata={
            "duration_seconds": round(duration, 3),
            "alignment_method": "incremental_streaming_resolution",
            "edge_classification": "symbolic_numeric + temporal + nli_deberta",
            "credibility_method": "arbgraph_intensity_propagation",
            "nli_threshold": nli_threshold,
            "document_count": len(doc_filenames),
            "total_facts": total_facts,
            "cluster_count": len(existing_clusters),
            "singleton_count": len(unmatched_set),
            "incremental": True,
            "affected_clusters": len(dirty_cluster_ids),
            "unaffected_clusters": unaffected_count,
        },
    )

    new_clusters_created = len([c_id for c_id in dirty_cluster_ids if c_id not in initial_cluster_ids])
    existing_clusters_updated = len(dirty_cluster_ids) - new_clusters_created

    stats = {
        "new_facts_count": len(new_doc_facts),
        "clusters_updated": existing_clusters_updated,
        "clusters_created": new_clusters_created,
        "new_singletons": len(unmatched_set) - len(existing_graph.unmatched_facts),
        "unaffected_clusters": unaffected_count,
        "duration_seconds": round(duration, 3),
    }

    return updated_graph, stats


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
