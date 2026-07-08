"""Evaluation metrics and batched model scoring for recommendation quality.

Primary metrics:
    - Hit@5: Binary — does a relevant item appear in top 5?
    - NDCG@10: Ranking quality — position-sensitive relevance scoring

Batched scoring:
    score_test_cases() builds a SINGLE flat feature dict per batch and
    calls model(batch). The model routes keys to sub-towers internally.
    No knowledge of tower architecture needed here.
"""

import numpy as np
import pandas as pd
import torch

from src.data.dataset import (
    build_restaurant_arrays,
    FEATURE_DTYPES,
)
from src.data.features import PREFERENCE_FEATURES
from src.evaluation.sampling import TestCase
from src.utils.geo import haversine_distance_vectorized as haversine_np


def hit_at_k(ranked_items: list[str], relevant_items: set[str], k: int = 5) -> float:
    top_k = ranked_items[:k]
    return 1.0 if any(item in relevant_items for item in top_k) else 0.0


def ndcg_at_k(ranked_items: list[str], relevant_items: set[str], k: int = 10) -> float:
    top_k = ranked_items[:k]
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)
    num_relevant_in_k = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant_in_k))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_recommendations(
    all_ranked: list[list[str]],
    all_relevant: list[set[str]],
    hit_k: int = 5,
    ndcg_k: int = 10,
) -> dict[str, float]:
    hits = [hit_at_k(r, rel, hit_k) for r, rel in zip(all_ranked, all_relevant)]
    ndcgs = [ndcg_at_k(r, rel, ndcg_k) for r, rel in zip(all_ranked, all_relevant)]
    return {
        f"Hit@{hit_k}": float(np.mean(hits)),
        f"NDCG@{ndcg_k}": float(np.mean(ndcgs)),
        "n_cases": len(hits),
    }


# ── Batched model scoring ────────────────────────────────────────────────────


@torch.no_grad()
def score_test_cases(
    model,
    test_cases: list[TestCase],
    biz_features: dict,
    user_centroids: dict,
    category_vocab: dict,
    enabled_keys: set[str],
    feature_dtypes: dict[str, torch.dtype],
    user_preferences: pd.DataFrame | None = None,
    user_onboarding: pd.DataFrame | None = None,
    case_onboarding_overrides: np.ndarray | None = None,
    cold_restaurant_ids: set[str] | None = None,
    hit_k: int = 5,
    ndcg_k: int = 10,
    batch_size: int = 4096,
    # Backward-compatible kwargs (ignored)
    user_enabled_keys: set[str] | None = None,
    restaurant_enabled_keys: set[str] | None = None,
    user_feature_dtypes: dict | None = None,
    restaurant_feature_dtypes: dict | None = None,
    build_user_features_fn=None,
    build_restaurant_features_fn=None,
) -> dict[str, float]:
    """Score pre-built test cases with batched forward passes.

    Builds a SINGLE flat feature dict per batch; the model routes keys to
    the correct sub-towers internally.

    Vectorized: per-case features are built once per case and expanded to the
    (case, candidate) layout with ``np.repeat``; per-entity features (user
    preference/onboarding rows, restaurant content, restaurant lat/lon) are
    looked up on the compact set of unique ids and gathered via ``flat_idx``.
    This avoids per-case pandas ``.loc`` and per-(case, candidate) dict lookups.
    Ranking assumes a single relevant item per case (the positive) and reads
    its rank directly by counting higher-scoring candidates — no full sort.

    Args:
        cold_restaurant_ids: If provided, zero out temporal_vec for ALL
            candidates. Simulates a new restaurant with metadata but no
            checkins, and keeps the comparison fair (every candidate loses the
            temporal signal, not just the held-out one).
    """
    model.eval()
    device = next(model.parameters()).device

    if not test_cases:
        return {f"Hit@{hit_k}": 0.0, f"NDCG@{ndcg_k}": 0.0, "n_cases": 0}

    # ── Flatten (case, candidate) layout + compact unique-id index ───────
    n_cases = len(test_cases)
    case_sizes = np.array([len(c.candidate_ids) for c in test_cases], dtype=np.int64)
    n_total = int(case_sizes.sum())
    case_offsets = np.zeros(n_cases + 1, dtype=np.int64)
    np.cumsum(case_sizes, out=case_offsets[1:])

    all_bids: list[str] = []
    for case in test_cases:
        all_bids.extend(case.candidate_ids)
    unique_bids = sorted(set(all_bids))
    bid_to_idx = {bid: i for i, bid in enumerate(unique_bids)}
    flat_idx = np.fromiter((bid_to_idx[b] for b in all_bids), dtype=np.int64, count=n_total)

    arrays: dict[str, np.ndarray] = {}

    # User context scalars: one value per case, expanded by candidate count.
    for key in ("dow_sin", "dow_cos", "is_weekend",
                "checkin_hour_sin", "checkin_hour_cos",
                "checkin_dow_sin", "checkin_dow_cos"):
        if key in enabled_keys:
            vals = np.array([c.ctx.get(key, 0.0) for c in test_cases], dtype=np.float32)
            arrays[key] = np.repeat(vals, case_sizes)

    # Distance: per-case user centroid (repeat) vs per-candidate biz coords
    # (compact lookup on unique bids, then gather).
    if "distance" in enabled_keys:
        u_lat = np.array([user_centroids.get(c.user_id, (0.0, 0.0))[0] for c in test_cases], dtype=np.float64)
        u_lon = np.array([user_centroids.get(c.user_id, (0.0, 0.0))[1] for c in test_cases], dtype=np.float64)
        biz_lat_c = np.array([biz_features.get(b, {}).get("latitude", 0.0) for b in unique_bids], dtype=np.float64)
        biz_lon_c = np.array([biz_features.get(b, {}).get("longitude", 0.0) for b in unique_bids], dtype=np.float64)
        arrays["distance"] = np.clip(
            haversine_np(np.repeat(u_lat, case_sizes), np.repeat(u_lon, case_sizes),
                         biz_lat_c[flat_idx], biz_lon_c[flat_idx]),
            0, 200,
        ).astype(np.float32)

    pref_dim = len(PREFERENCE_FEATURES)

    # User preferences: precompute uid -> row once (no pandas .loc per case).
    if "preference_vec" in enabled_keys:
        per_case = np.zeros((n_cases, pref_dim), dtype=np.float32)
        if user_preferences is not None and len(user_preferences) > 0:
            pv = user_preferences.reindex(columns=list(PREFERENCE_FEATURES)).to_numpy(dtype=np.float32)
            ppos = {str(u): i for i, u in enumerate(user_preferences.index)}
            for ci, c in enumerate(test_cases):
                j = ppos.get(c.user_id)
                if j is not None:
                    per_case[ci] = pv[j]
        arrays["preference_vec"] = np.repeat(per_case, case_sizes, axis=0)

    # User onboarding: per-case LOO override, else precomputed uid -> row.
    if "onboarding_vec" in enabled_keys:
        per_case = np.zeros((n_cases, pref_dim), dtype=np.float32)
        ov = opos = None
        if user_onboarding is not None and len(user_onboarding) > 0:
            ov = user_onboarding.reindex(columns=list(PREFERENCE_FEATURES)).to_numpy(dtype=np.float32)
            opos = {str(u): i for i, u in enumerate(user_onboarding.index)}
        for ci, c in enumerate(test_cases):
            if case_onboarding_overrides is not None and case_onboarding_overrides[ci].any():
                per_case[ci] = case_onboarding_overrides[ci]
            elif opos is not None:
                j = opos.get(c.user_id)
                if j is not None:
                    per_case[ci] = ov[j]
        arrays["onboarding_vec"] = np.repeat(per_case, case_sizes, axis=0)

    # Restaurant content + temporal: compact build, expand via flat_idx.
    compact_rest = build_restaurant_arrays(unique_bids, biz_features, category_vocab, enabled_keys)
    if cold_restaurant_ids and "temporal_vec" in compact_rest:
        compact_rest["temporal_vec"][:] = 0.0
    for k, arr in compact_rest.items():
        arrays[k] = arr[flat_idx]

    # ── Batched forward passes ───────────────────────────────────────────
    all_scores = torch.empty(n_total)
    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        sl = slice(batch_start, batch_end)
        batch = {
            k: torch.from_numpy(arr[sl].copy()).to(dtype=feature_dtypes[k], device=device)
            for k, arr in arrays.items()
        }
        all_scores[batch_start:batch_end] = model(batch).cpu()

    # ── Metrics: rank of the single positive per case (no full sort) ─────
    scores_np = all_scores.numpy()
    hits = 0.0
    ndcgs = 0.0
    for ci, case in enumerate(test_cases):
        s = int(case_offsets[ci])
        e = int(case_offsets[ci + 1])
        seg = scores_np[s:e]
        pos_pos = case.candidate_ids.index(case.positive_id)
        rank = int(np.count_nonzero(seg > seg[pos_pos]))  # ties -> best rank for positive
        if rank < hit_k:
            hits += 1.0
        if rank < ndcg_k:
            ndcgs += 1.0 / np.log2(rank + 2)

    return {
        f"Hit@{hit_k}": hits / n_cases,
        f"NDCG@{ndcg_k}": ndcgs / n_cases,
        "n_cases": n_cases,
    }
