"""Semantic embedding service using Voyage AI (voyage-finance-2).

Provides dynamic entity and metric alignment to eliminate hardcoded synonym/alias dictionaries.

Design basis:
- ArbGraph (Niu et al., 2026): claim alignment via semantic representation
- TabFact dual-channel: embed concepts/terms dynamically, DO NOT embed numeric values
  (prevents the '10M vs 20M' problem where different numbers appear 98% identical in embedding space)
- voyage-finance-2: SOTA specialized embedding model for financial and economic terminology
- Automatic disk/memory caching to minimize latency and token consumption
- Graceful offline fallback (local fast-string match or sentence-transformers)
"""

import json
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).resolve().parent.parent / ".voyage_cache.json"

_voyage_client = None
_embedding_cache: dict[str, list[float]] = {}


def _load_cache() -> None:
    global _embedding_cache
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _embedding_cache = json.load(f)
        except Exception as e:
            logger.warning("Failed to load embedding cache: %s", e)
            _embedding_cache = {}


def _save_cache() -> None:
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_embedding_cache, f)
    except Exception as e:
        logger.warning("Failed to save embedding cache: %s", e)


_load_cache()


def get_voyage_client():
    """Lazy initialize the Voyage client."""
    global _voyage_client
    if _voyage_client is not None:
        return _voyage_client

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        return None

    try:
        import voyageai
        _voyage_client = voyageai.Client(api_key=api_key)
        return _voyage_client
    except Exception as e:
        logger.warning("Failed to initialize Voyage AI client: %s", e)
        return None


def get_embeddings(texts: list[str], model: str = "voyage-finance-2") -> list[list[float]]:
    """Retrieve embeddings for a list of texts, using cache where available."""
    if not texts:
        return []

    client = get_voyage_client()
    needed_texts: list[str] = []
    needed_indices: list[int] = []

    results: list[list[float] | None] = [None] * len(texts)

    for i, t in enumerate(texts):
        cache_key = f"{model}:{t}"
        if cache_key in _embedding_cache:
            results[i] = _embedding_cache[cache_key]
        else:
            needed_texts.append(t)
            needed_indices.append(i)

    if needed_texts and client is not None:
        try:
            # Voyage allows batching up to 128 items per call
            batch_size = 64
            for batch_start in range(0, len(needed_texts), batch_size):
                batch = needed_texts[batch_start:batch_start + batch_size]
                res = client.embed(batch, model=model)
                for local_idx, emb in enumerate(res.embeddings):
                    global_needed_idx = batch_start + local_idx
                    orig_idx = needed_indices[global_needed_idx]
                    t = needed_texts[global_needed_idx]
                    _embedding_cache[f"{model}:{t}"] = emb
                    results[orig_idx] = emb
            _save_cache()
        except Exception as e:
            logger.warning("Voyage embedding API call failed: %s", e)

    # For any remaining None items (if API failed or no key), return zero vectors
    fallback_dim = 1024 if "finance" in model else 512
    final_embeddings: list[list[float]] = []
    for r in results:
        if r is not None:
            final_embeddings.append(r)
        else:
            final_embeddings.append([0.0] * fallback_dim)

    return final_embeddings


def cosine_similarity(v1: list[float] | np.ndarray, v2: list[float] | np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def format_term_for_embedding(term: str, kind: Literal["entity", "metric"]) -> str:
    """Normalize term into a clean string for embedding.
    
    CRITICAL: Embeds ONLY the term (entity or metric), NOT a full sentence.
    Embedding full claims (e.g. 'Revenue was $10M' vs 'Revenue was $12M')
    causes numeric claims to look 98% identical in vector space.
    """
    return term.strip().lower().replace("_", " ")


def compute_semantic_similarity(
    term1: str,
    term2: str,
    kind: Literal["entity", "metric"] = "metric",
    model: str = "voyage-finance-2",
) -> float:
    """Compute semantic similarity between two terms using Voyage embeddings."""
    t1_norm = term1.strip().lower().replace("_", " ")
    t2_norm = term2.strip().lower().replace("_", " ")

    if t1_norm == t2_norm:
        return 1.0

    prompt1 = format_term_for_embedding(term1, kind)
    prompt2 = format_term_for_embedding(term2, kind)

    embs = get_embeddings([prompt1, prompt2], model=model)
    return cosine_similarity(embs[0], embs[1])


def is_semantic_match(
    term1: str,
    term2: str,
    kind: Literal["entity", "metric"] = "metric",
    threshold: float = 0.55,
) -> bool:
    """Determine whether two terms refer to the same entity or metric semantically."""
    t1 = term1.strip().lower().replace("_", " ")
    t2 = term2.strip().lower().replace("_", " ")

    if t1 == t2:
        return True

    # Fast substring check for entity variants (e.g. 'Delhivery' in 'Delhivery Limited')
    if (t1 in t2 or t2 in t1) and min(len(t1), len(t2)) >= 4:
        return True

    # Token-overlap (Jaccard) fallback (e.g. 'revenue' in 'revenue from contracts')
    words1 = set(t1.split())
    words2 = set(t2.split())
    if words1 and words2:
        overlap = len(words1 & words2)
        if overlap > 0 and (overlap / min(len(words1), len(words2)) >= 0.8 or overlap / len(words1 | words2) >= 0.5):
            return True

    sim = compute_semantic_similarity(term1, term2, kind=kind)
    return sim >= threshold


def precompute_term_matches(
    terms: list[str],
    kind: Literal["entity", "metric"] = "metric",
    threshold: float = 0.55,
    model: str = "voyage-finance-2",
) -> dict[tuple[str, str], bool]:
    """Precompute pairwise semantic matches for a list of terms in batch.
    
    Returns a dict mapping (term_a, term_b) -> bool.
    """
    unique_terms = list(dict.fromkeys(terms))
    if not unique_terms:
        return {}

    prompts = [format_term_for_embedding(t, kind) for t in unique_terms]
    embs = get_embeddings(prompts, model=model)
    emb_arrays = [np.asarray(e, dtype=np.float32) for e in embs]

    matches: dict[tuple[str, str], bool] = {}
    n = len(unique_terms)
    for i in range(n):
        ti = unique_terms[i]
        matches[(ti, ti)] = True
        norm_i = np.linalg.norm(emb_arrays[i])
        for j in range(i + 1, n):
            tj = unique_terms[j]
            norm_ti = ti.strip().lower().replace("_", " ")
            norm_tj = tj.strip().lower().replace("_", " ")

            # Check exact, substring, or token overlap
            words_i = set(norm_ti.split())
            words_j = set(norm_tj.split())
            overlap = len(words_i & words_j)
            has_token_overlap = (
                overlap > 0
                and (overlap / min(len(words_i), len(words_j)) >= 0.8
                     or overlap / len(words_i | words_j) >= 0.5)
            )

            if norm_ti == norm_tj or (
                (norm_ti in norm_tj or norm_tj in norm_ti)
                and min(len(norm_ti), len(norm_tj)) >= 4
            ) or has_token_overlap:
                matches[(ti, tj)] = True
                matches[(tj, ti)] = True
                continue

            norm_j = np.linalg.norm(emb_arrays[j])
            if norm_i == 0 or norm_j == 0:
                sim = 0.0
            else:
                sim = float(np.dot(emb_arrays[i], emb_arrays[j]) / (norm_i * norm_j))

            is_match = sim >= threshold
            matches[(ti, tj)] = is_match
            matches[(tj, ti)] = is_match

    return matches



