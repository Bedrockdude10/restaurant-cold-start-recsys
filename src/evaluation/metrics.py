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

    Builds a SINGLE flat feature dict per batch. The model routes keys
    to the correct sub-towers internally.

    Args:
        cold_restaurant_ids: If provided, zero out temporal_vec for these
            restaurants. Simulates the real cold-start scenario where a
            new restaurant has metadata (categories, price, attributes)
            but no user activity (checkins). Without this, cold restaurant
            evaluation leaks temporal signal the model wouldn't have in
            production.
    """
    model.eval()
    device = next(model.parameters()).device

    if not test_cases:
        return {f"Hit@{hit_k}": 0.0, f"NDCG@{ndcg_k}": 0.0, "n_cases": 0}

    # ── Phase 1a: Flatten all (case, candidate) pairs ────────────────────
    case_sizes = [len(c.candidate_ids) for c in test_cases]
    n_total = sum(case_sizes)

    case_offsets: list[tuple[int, int]] = []
    offset = 0
    for size in case_sizes:
        case_offsets.append((offset, offset + size))
        offset += size

    all_bids: list[str] = []
    for case in test_cases:
        all_bids.extend(case.candidate_ids)

    # ── Phase 1b: Build ALL features as flat arrays ──────────────────────
    arrays: dict[str, np.ndarray] = {}

    # User context scalars
    if "dow_sin" in enabled_keys:
        arr = np.empty(n_total, dtype=np.float32)
        for case, (s, e) in zip(test_cases, case_offsets):
            arr[s:e] = case.ctx["dow_sin"]
        arrays["dow_sin"] = arr

    if "dow_cos" in enabled_keys:
        arr = np.empty(n_total, dtype=np.float32)
        for case, (s, e) in zip(test_cases, case_offsets):
            arr[s:e] = case.ctx["dow_cos"]
        arrays["dow_cos"] = arr

    if "is_weekend" in enabled_keys:
        arr = np.empty(n_total, dtype=np.float32)
        for case, (s, e) in zip(test_cases, case_offsets):
            arr[s:e] = case.ctx["is_weekend"]
        arrays["is_weekend"] = arr
        
    # Checkin-matched time-of-visit context
    for ck in ("checkin_hour_sin", "checkin_hour_cos",
               "checkin_dow_sin", "checkin_dow_cos"):
        if ck in enabled_keys:
            arr = np.zeros(n_total, dtype=np.float32)
            for case, (s, e) in zip(test_cases, case_offsets):
                arr[s:e] = case.ctx.get(ck, 0.0)
            arrays[ck] = arr

    # Distance
    if "distance" in enabled_keys:
        user_lat = np.empty(n_total, dtype=np.float64)
        user_lon = np.empty(n_total, dtype=np.float64)
        for case, (s, e) in zip(test_cases, case_offsets):
            centroid = user_centroids.get(case.user_id, (0.0, 0.0))
            user_lat[s:e] = centroid[0]
            user_lon[s:e] = centroid[1]
        biz_lat = np.array([biz_features.get(bid, {}).get("latitude", 0.0) for bid in all_bids], dtype=np.float64)
        biz_lon = np.array([biz_features.get(bid, {}).get("longitude", 0.0) for bid in all_bids], dtype=np.float64)
        arrays["distance"] = np.clip(haversine_np(user_lat, user_lon, biz_lat, biz_lon), 0, 200).astype(np.float32)

    # User preferences
    if "preference_vec" in enabled_keys:
        pref_dim = len(PREFERENCE_FEATURES)
        arr = np.zeros((n_total, pref_dim), dtype=np.float32)
        for case, (s, e) in zip(test_cases, case_offsets):
            if user_preferences is not None and case.user_id in user_preferences.index:
                pref_row = user_preferences.loc[case.user_id]
                if isinstance(pref_row, pd.DataFrame):
                    pref_row = pref_row.iloc[0]
                arr[s:e] = [float(pref_row.get(f, 0.0)) for f in PREFERENCE_FEATURES]
        arrays["preference_vec"] = arr

    # User onboarding
    if "onboarding_vec" in enabled_keys:
        pref_dim = len(PREFERENCE_FEATURES)
        arr = np.zeros((n_total, pref_dim), dtype=np.float32)
        for case_i, (case, (s, e)) in enumerate(zip(test_cases, case_offsets)):
            if (case_onboarding_overrides is not None
                    and case_onboarding_overrides[case_i].any()):
                arr[s:e] = case_onboarding_overrides[case_i]
            elif user_onboarding is not None and case.user_id in user_onboarding.index:
                onb_row = user_onboarding.loc[case.user_id]
                if isinstance(onb_row, pd.DataFrame):
                    onb_row = onb_row.iloc[0]
                arr[s:e] = [float(onb_row.get(f, 0.0)) for f in PREFERENCE_FEATURES]
        arrays["onboarding_vec"] = arr

    # Restaurant content + temporal: build compact, expand via index
    unique_bids = sorted(set(all_bids))
    bid_to_idx = {bid: i for i, bid in enumerate(unique_bids)}
    flat_idx = np.array([bid_to_idx[bid] for bid in all_bids], dtype=np.int32)

    compact_rest = build_restaurant_arrays(unique_bids, biz_features, category_vocab, enabled_keys)

    # Zero out temporal features for ALL candidates in cold-restaurant
    # evaluation, not just the cold restaurant itself. This is the fair
    # cold-start question: "can the model rank restaurants by content
    # alone?" Zeroing only the cold restaurant while leaving temporal
    # for the 99 negatives creates an unfair comparison where the cold
    # restaurant is the only one missing a signal the model expects.
    if cold_restaurant_ids and "temporal_vec" in compact_rest:
        compact_rest["temporal_vec"][:] = 0.0

    for k, arr in compact_rest.items():
        arrays[k] = arr[flat_idx]

    # ── Phase 2: Batched forward passes ──────────────────────────────────
    all_scores = torch.empty(n_total)

    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        sl = slice(batch_start, batch_end)

        batch = {}
        for k, arr in arrays.items():
            batch[k] = torch.from_numpy(arr[sl].copy()).to(
                dtype=feature_dtypes[k], device=device,
            )

        scores = model(batch)
        all_scores[batch_start:batch_end] = scores.cpu()

    # ── Phase 3: Partition scores, compute metrics ───────────────────────
    all_hits = []
    all_ndcgs = []

    for case, (start, end) in zip(test_cases, case_offsets):
        case_scores = all_scores[start:end]
        ranked_idx = case_scores.argsort(descending=True).numpy()
        ranked_bids = [case.candidate_ids[i] for i in ranked_idx]
        all_hits.append(hit_at_k(ranked_bids, case.relevant_ids, k=hit_k))
        all_ndcgs.append(ndcg_at_k(ranked_bids, case.relevant_ids, k=ndcg_k))

    return {
        f"Hit@{hit_k}": float(np.mean(all_hits)),
        f"NDCG@{ndcg_k}": float(np.mean(all_ndcgs)),
        "n_cases": len(all_hits),
    }