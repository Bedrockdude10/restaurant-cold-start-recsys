"""Model-facing data layer: datasets, feature builders, and tensorization.

All features are built into a SINGLE flat dict of tensors. The model's
forward() receives this dict and routes keys to the correct sub-towers
internally. Dataset and scoring code never need to know about tower
architecture — they just provide all available features.

Handles:
    - Converting feature dicts → tensors (dtype specs, padding)
    - Vectorized negative sampling and dataset construction
    - Business feature aggregation (prepare_biz_features)
    - Shared array construction (build_restaurant_arrays)

Per-epoch resampling:
    Training negatives are reshuffled every epoch via resample_negatives().
    Onboarding vectors are resampled via resample_onboarding().
    See class docstring for details.

Used by train.py (TwoTowerDataset) and metrics.py (score_test_cases
uses build_restaurant_arrays for batched eval).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.features import (
    ATTR_DIM,
    ATTR_NAMES,
    PREFERENCE_FEATURES,
    encode_categories,
    extract_attribute_features,
    _sample_onboarding_array,
)
from src.models.two_tower import PREFERENCE_DIM, TEMPORAL_DIM
from src.utils.geo import haversine_distance_vectorized as haversine_np


# ── Dtype specs ──────────────────────────────────────────────────────────────
# Single flat dtype map for ALL features. Dataset and score_test_cases
# both tensorize from this — no separate user/restaurant dtype maps needed.

FEATURE_DTYPES: dict[str, torch.dtype] = {
    # User context
    "dow_sin": torch.float,
    "dow_cos": torch.float,
    "is_weekend": torch.float,
    "distance": torch.float,
    "checkin_hour_sin": torch.float,   
    "checkin_hour_cos": torch.float,   
    "checkin_dow_sin": torch.float,    
    "checkin_dow_cos": torch.float,    
    # User content
    "preference_vec": torch.float,
    "onboarding_vec": torch.float,
    # Restaurant content
    "category_ids": torch.long,
    "price_tier": torch.long,
    "attr_vec": torch.float,
    # Restaurant context
    "temporal_vec": torch.float,
}

# Backward-compatible aliases for code that still imports these
USER_FEATURE_DTYPES = FEATURE_DTYPES
RESTAURANT_FEATURE_DTYPES = FEATURE_DTYPES


# ── Feature builders ─────────────────────────────────────────────────────────
# Per-sample feature dict builders for non-batched use cases.
# Batched paths (TwoTowerDataset, score_test_cases) build arrays directly.


def build_user_features(
    ctx: dict,
    centroid: tuple[float, float],
    biz: dict,
    user_pref: pd.Series | None,
    enabled_keys: set[str],
    user_onboarding: pd.Series | None = None,
) -> dict:
    """Build user/context feature dict for one (user, restaurant) pair."""
    feats = {}
    if "dow_sin" in enabled_keys:
        feats["dow_sin"] = ctx["dow_sin"]
    if "dow_cos" in enabled_keys:
        feats["dow_cos"] = ctx["dow_cos"]
    if "is_weekend" in enabled_keys:
        feats["is_weekend"] = ctx["is_weekend"]
    if "distance" in enabled_keys:
        distance = haversine_np(
            centroid[0], centroid[1],
            biz.get("latitude", 0.0), biz.get("longitude", 0.0),
        )
        feats["distance"] = float(np.clip(distance, 0, 200))
    if "preference_vec" in enabled_keys:
        if user_pref is not None:
            feats["preference_vec"] = [float(user_pref.get(f, 0.0)) for f in PREFERENCE_FEATURES]
        else:
            feats["preference_vec"] = [0.0] * PREFERENCE_DIM
    if "onboarding_vec" in enabled_keys:
        if user_onboarding is not None:
            feats["onboarding_vec"] = [float(user_onboarding.get(f, 0.0)) for f in PREFERENCE_FEATURES]
        else:
            feats["onboarding_vec"] = [0.0] * PREFERENCE_DIM
    return feats


def build_restaurant_features(
    biz: dict,
    category_vocab: dict,
    enabled_keys: set[str],
    max_categories: int = 15,
) -> dict:
    """Build restaurant feature dict for one restaurant."""
    feats = {}
    if "category_ids" in enabled_keys:
        cat_ids = encode_categories(
            biz.get("categories", []), category_vocab, max_categories,
        )
        feats["category_ids"] = cat_ids + [0] * (max_categories - len(cat_ids))
    if "price_tier" in enabled_keys:
        feats["price_tier"] = biz.get("price_tier", 0)
    if "attr_vec" in enabled_keys:
        feats["attr_vec"] = biz.get("attr_vec", [0.0] * ATTR_DIM)
    if "temporal_vec" in enabled_keys:
        feats["temporal_vec"] = biz.get("temporal_vec", [0.0] * TEMPORAL_DIM)
    return feats


# ── Business feature preparation ─────────────────────────────────────────────


def prepare_biz_features(
    restaurants: pd.DataFrame,
    checkin_profiles: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Build business_id -> feature dict from restaurants DataFrame.

    Includes temporal_vec for the restaurant context tower.
    """
    temporal_lookup: dict[str, list[float]] = {}
    if checkin_profiles is not None:
        hour_cols = [f"hour_dist_{h}" for h in range(24)]
        dow_cols = [f"dow_dist_{d}" for d in range(7)]
        for _, row in checkin_profiles.iterrows():
            bid = row["business_id"]
            temporal_lookup[bid] = (
                [float(row[c]) for c in hour_cols]
                + [float(row[c]) for c in dow_cols]
            )

    zero_temporal = [0.0] * TEMPORAL_DIM

    biz_features = {}
    for _, row in restaurants.iterrows():
        bid = row["business_id"]
        attr_feats = extract_attribute_features(row.get("attributes"))
        attr_vec = [attr_feats[name] for name in ATTR_NAMES]
        biz_features[bid] = {
            "categories": row["categories"] if isinstance(row["categories"], (list, np.ndarray)) else [],
            "price_tier": int(row.get("price_tier", 0)),
            "attr_vec": attr_vec,
            "latitude": float(row.get("latitude", 0.0)),
            "longitude": float(row.get("longitude", 0.0)),
            "temporal_vec": temporal_lookup.get(bid, zero_temporal),
        }
    return biz_features


# ── Compact array builders ───────────────────────────────────────────────────
# Build numpy arrays indexed by compact restaurant ID. Used by both
# TwoTowerDataset (training) and score_test_cases (evaluation).


def build_restaurant_arrays(
    unique_bids: list[str],
    biz_features: dict[str, dict],
    category_vocab: dict[str, int],
    enabled_keys: set[str],
    max_categories: int = 15,
) -> dict[str, np.ndarray]:
    """Build compact arrays of restaurant features (content + context)."""
    n = len(unique_bids)
    arrays: dict[str, np.ndarray] = {}

    if "category_ids" in enabled_keys:
        cat_arr = np.zeros((n, max_categories), dtype=np.int64)
        for i, bid in enumerate(unique_bids):
            biz = biz_features.get(bid, {})
            cat_ids = encode_categories(
                biz.get("categories", []), category_vocab, max_categories,
            )
            cat_arr[i, :len(cat_ids)] = cat_ids
        arrays["category_ids"] = cat_arr

    if "price_tier" in enabled_keys:
        arrays["price_tier"] = np.array(
            [biz_features.get(bid, {}).get("price_tier", 0) for bid in unique_bids],
            dtype=np.int64,
        )

    if "attr_vec" in enabled_keys:
        arrays["attr_vec"] = np.array(
            [biz_features.get(bid, {}).get("attr_vec", [0.0] * ATTR_DIM)
             for bid in unique_bids],
            dtype=np.float32,
        )

    if "temporal_vec" in enabled_keys:
        arrays["temporal_vec"] = np.array(
            [biz_features.get(bid, {}).get("temporal_vec", [0.0] * TEMPORAL_DIM)
             for bid in unique_bids],
            dtype=np.float32,
        )

    return arrays


# ── Resampling state ─────────────────────────────────────────────────────────


@dataclass
class ResamplingState:
    """Pre-stored compact integer state for per-epoch negative resampling."""
    n_positives: int
    samples_per_pos: int
    num_negatives: int
    pos_compact_idx: np.ndarray
    pos_states: np.ndarray
    state_pool_compact: dict[str, np.ndarray]


# ── Dataset ──────────────────────────────────────────────────────────────────


class TwoTowerDataset(Dataset):
    """Dataset of (features, label) samples for two-tower training.

    All features are precomputed into a SINGLE flat dict of tensors.
    The model's forward() receives this dict and routes keys to sub-towers.

    Indexing scheme (two-level indirection for memory efficiency):

        sample_idx ──→ _rest_idx[sample_idx] ──→ compact restaurant arrays
                                                  (content + context/temporal)
        sample_idx ──→ _user_idx[sample_idx] ──→ _pref_compact, _onboarding_compact
                                              ──→ _user_lat/lon_compact

    Per-sample scalars (dow_sin, dow_cos, is_weekend, labels) are flat
    arrays indexed directly by sample_idx.

    Distance is NOT in the flat tensors — compact lat/lon arrays are
    stored and train.py computes distances per-batch on GPU.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        biz_features: dict,
        user_centroids: dict,
        category_vocab: dict,
        user_preferences: pd.DataFrame,
        enabled_keys: set[str],
        restaurant_state_map: dict[str, str],
        state_pools: dict[str, np.ndarray],
        num_negatives: int = 4,
        max_categories: int = 15,
        seed: int = 42,
    ):
        self.enabled_keys = enabled_keys
        self._n_samples: int = 0

        # Populated by _tensorize:
        self._labels: torch.Tensor = torch.empty(0)
        self._tensors: dict[str, torch.Tensor] = {}
        self._pref_compact: torch.Tensor | None = None
        self._user_idx: torch.Tensor | None = None
        self._onboarding_compact: torch.Tensor | None = None
        self._rest_idx: torch.Tensor = torch.empty(0, dtype=torch.long)

        # Compact arrays for per-batch distance computation (not in _tensors)
        self._user_lat_compact: torch.Tensor | None = None
        self._user_lon_compact: torch.Tensor | None = None
        self._rest_lat_compact: torch.Tensor | None = None
        self._rest_lon_compact: torch.Tensor | None = None

        self._onboarding_probs: np.ndarray | None = None
        self._resample_state: ResamplingState | None = None

        self._build_samples(
            interactions, biz_features, user_centroids, category_vocab,
            user_preferences, restaurant_state_map, state_pools,
            num_negatives, max_categories, seed,
        )

    def _build_samples(
        self,
        interactions: pd.DataFrame,
        biz_features: dict,
        user_centroids: dict,
        category_vocab: dict,
        user_preferences: pd.DataFrame,
        restaurant_state_map: dict[str, str],
        state_pools: dict[str, np.ndarray],
        num_negatives: int,
        max_categories: int,
        seed: int,
    ) -> None:
        import time
        t0 = time.time()

        rng = np.random.default_rng(seed)
        n = len(interactions)
        samples_per_pos = 1 + num_negatives
        total = n * samples_per_pos

        pos_uids = np.asarray(interactions["user_id"].values)
        pos_bids = np.asarray(interactions["business_id"].values)
        pos_dow_sin = np.asarray(interactions["dow_sin"].values, dtype=np.float32)
        pos_dow_cos = np.asarray(interactions["dow_cos"].values, dtype=np.float32)
        pos_is_wknd = np.asarray(interactions["is_weekend"].values, dtype=np.float32)
        
        # Checkin-matched time-of-visit scalars (zeros where unmatched)
        checkin_cols = {}
        for ck in ("checkin_hour_sin", "checkin_hour_cos",
                    "checkin_dow_sin", "checkin_dow_cos"):
            if ck in self.enabled_keys:
                checkin_cols[ck] = np.asarray(
                    interactions[ck].values, dtype=np.float32
                )

        t1 = time.time()
        labels, all_bids, pos_states = self._sample_negatives(
            pos_bids, restaurant_state_map, state_pools,
            samples_per_pos, num_negatives, rng,
        )
        self._n_samples = total
        print(f"    Negative sampling: {time.time() - t1:.1f}s")

        t2 = time.time()
        user_result = self._build_user_arrays(
            pos_uids, pos_dow_sin, pos_dow_cos, pos_is_wknd,
            user_centroids, user_preferences,
            samples_per_pos, rng,
            checkin_arrays=checkin_cols,
        )
        print(f"    User features:    {time.time() - t2:.1f}s")

        t3 = time.time()
        rest_result = self._build_restaurant_compact(
            all_bids, pos_bids, pos_states, biz_features,
            category_vocab, state_pools, samples_per_pos,
            num_negatives, max_categories,
        )
        print(f"    Restaurant feats: {time.time() - t3:.1f}s")

        t4 = time.time()
        self._tensorize(labels, user_result, rest_result)
        print(f"    Tensorization:    {time.time() - t4:.1f}s")

        n_unique = len(rest_result["unique_bids"])
        print(f"    Total precompute: {time.time() - t0:.1f}s "
              f"({total:,} samples, {n_unique:,} unique restaurants)")

    def _sample_negatives(
        self,
        pos_bids: np.ndarray,
        restaurant_state_map: dict[str, str],
        state_pools: dict[str, np.ndarray],
        samples_per_pos: int,
        num_negatives: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(pos_bids)
        total = n * samples_per_pos

        all_bids = np.empty(total, dtype=pos_bids.dtype)
        labels = np.zeros(total, dtype=np.float32)

        pos_indices = np.arange(0, total, samples_per_pos)
        all_bids[pos_indices] = pos_bids
        labels[pos_indices] = 1.0

        pos_states = np.array([
            restaurant_state_map.get(str(bid), "") for bid in pos_bids
        ])
        neg_bids = np.empty((n, num_negatives), dtype=pos_bids.dtype)

        for state, pool in state_pools.items():
            state_mask = pos_states == state
            n_state = state_mask.sum()
            if n_state == 0:
                continue
            rand_idx = rng.integers(0, len(pool), size=(n_state, num_negatives))
            neg_bids[state_mask] = pool[rand_idx]

        for k in range(num_negatives):
            neg_indices = pos_indices + 1 + k
            all_bids[neg_indices] = neg_bids[:, k]

        return labels, all_bids, pos_states

    def _build_user_arrays(
        self,
        pos_uids: np.ndarray,
        pos_dow_sin: np.ndarray,
        pos_dow_cos: np.ndarray,
        pos_is_wknd: np.ndarray,
        user_centroids: dict,
        user_preferences: pd.DataFrame,
        samples_per_pos: int,
        rng: np.random.Generator,
        checkin_arrays: dict[str, np.ndarray] | None = None, 

    ) -> dict:
        enabled = self.enabled_keys
        result: dict = {
            "scalars": {},
            "user_idx": None,
            "pref_matrix": None,
            "onboarding_probs": None,
            "onboarding_matrix": None,
            "user_lat_compact": None,
            "user_lon_compact": None,
        }

        if "dow_sin" in enabled:
            result["scalars"]["dow_sin"] = np.repeat(pos_dow_sin, samples_per_pos)
        if "dow_cos" in enabled:
            result["scalars"]["dow_cos"] = np.repeat(pos_dow_cos, samples_per_pos)
        if "is_weekend" in enabled:
            result["scalars"]["is_weekend"] = np.repeat(pos_is_wknd, samples_per_pos)

        # Checkin-matched time-of-visit scalars
        if checkin_arrays:
            for ck, arr in checkin_arrays.items():
                result["scalars"][ck] = np.repeat(arr, samples_per_pos)

        needs_user_index = bool({"preference_vec", "onboarding_vec", "distance"} & enabled)
        if not needs_user_index:
            return result

        pref_dim = len(PREFERENCE_FEATURES)
        unique_uids = sorted(set(str(u) for u in pos_uids))
        uid_to_idx = {uid: i for i, uid in enumerate(unique_uids)}

        pos_user_idx = np.array([uid_to_idx[str(uid)] for uid in pos_uids], dtype=np.int32)
        result["user_idx"] = np.repeat(pos_user_idx, samples_per_pos)

        if "distance" in enabled:
            result["user_lat_compact"] = np.array(
                [user_centroids.get(uid, (0.0, 0.0))[0] for uid in unique_uids],
                dtype=np.float32,
            )
            result["user_lon_compact"] = np.array(
                [user_centroids.get(uid, (0.0, 0.0))[1] for uid in unique_uids],
                dtype=np.float32,
            )

        if bool({"preference_vec", "onboarding_vec"} & enabled):
            aligned_prefs = np.zeros((len(unique_uids), pref_dim), dtype=np.float32)
            for i, uid in enumerate(unique_uids):
                if user_preferences is not None and uid in user_preferences.index:
                    pref_row = user_preferences.loc[uid]
                    if isinstance(pref_row, pd.DataFrame):
                        pref_row = pref_row.iloc[0]
                    aligned_prefs[i] = [float(pref_row.get(f, 0.0)) for f in PREFERENCE_FEATURES]

            if "preference_vec" in enabled:
                result["pref_matrix"] = aligned_prefs

            if "onboarding_vec" in enabled:
                result["onboarding_probs"] = aligned_prefs.astype(np.float64)
                result["onboarding_matrix"] = _sample_onboarding_array(
                    result["onboarding_probs"], max_k=5, rng=rng,
                )

        return result

    def _build_restaurant_compact(
        self,
        all_bids: np.ndarray,
        pos_bids: np.ndarray,
        pos_states: np.ndarray,
        biz_features: dict,
        category_vocab: dict,
        state_pools: dict[str, np.ndarray],
        samples_per_pos: int,
        num_negatives: int,
        max_categories: int,
    ) -> dict:
        unique_bids_set = set(str(b) for b in all_bids)
        for pool_arr in state_pools.values():
            unique_bids_set.update(str(b) for b in pool_arr)
        unique_bids_list = sorted(unique_bids_set)
        bid_to_idx = {bid: i for i, bid in enumerate(unique_bids_list)}

        rest_idx = np.array(
            [bid_to_idx[str(b)] for b in all_bids], dtype=np.int32,
        )

        rest_lat_compact = None
        rest_lon_compact = None
        if "distance" in self.enabled_keys:
            rest_lat_compact = np.array(
                [biz_features.get(bid, {}).get("latitude", 0.0) for bid in unique_bids_list],
                dtype=np.float32,
            )
            rest_lon_compact = np.array(
                [biz_features.get(bid, {}).get("longitude", 0.0) for bid in unique_bids_list],
                dtype=np.float32,
            )

        pos_compact_idx = np.array(
            [bid_to_idx[str(b)] for b in pos_bids], dtype=np.int32,
        )
        state_pool_compact = {
            state: np.array([bid_to_idx[str(b)] for b in pool_arr], dtype=np.int32)
            for state, pool_arr in state_pools.items()
        }
        resample_state = ResamplingState(
            n_positives=len(pos_bids),
            samples_per_pos=samples_per_pos,
            num_negatives=num_negatives,
            pos_compact_idx=pos_compact_idx,
            pos_states=pos_states.copy(),
            state_pool_compact=state_pool_compact,
        )

        # Restaurant arrays (content + context, all indexed by compact ID)
        rest_arrays = build_restaurant_arrays(
            unique_bids_list, biz_features, category_vocab,
            self.enabled_keys, max_categories,
        )

        return {
            "unique_bids": unique_bids_list,
            "bid_to_idx": bid_to_idx,
            "rest_idx": rest_idx,
            "rest_arrays": rest_arrays,
            "resample_state": resample_state,
            "rest_lat_compact": rest_lat_compact,
            "rest_lon_compact": rest_lon_compact,
        }

    def _tensorize(
        self,
        labels: np.ndarray,
        user_result: dict,
        rest_result: dict,
    ) -> None:
        self._labels = torch.from_numpy(labels)

        # All per-sample scalars go into one flat dict
        self._tensors = {
            k: torch.from_numpy(arr).to(dtype=FEATURE_DTYPES[k])
            for k, arr in user_result["scalars"].items()
        }

        # Restaurant content arrays (compact, indexed by rest_idx)
        for k, arr in rest_result["rest_arrays"].items():
            self._tensors[k] = torch.from_numpy(arr).to(dtype=FEATURE_DTYPES[k])

        if user_result["pref_matrix"] is not None:
            self._pref_compact = torch.from_numpy(user_result["pref_matrix"]).float()

        if user_result["onboarding_matrix"] is not None:
            self._onboarding_compact = torch.from_numpy(user_result["onboarding_matrix"]).float()

        self._onboarding_probs = user_result["onboarding_probs"]

        if user_result["user_idx"] is not None:
            self._user_idx = torch.from_numpy(user_result["user_idx"]).long()

        if user_result["user_lat_compact"] is not None:
            self._user_lat_compact = torch.from_numpy(user_result["user_lat_compact"])
            self._user_lon_compact = torch.from_numpy(user_result["user_lon_compact"])

        if rest_result["rest_lat_compact"] is not None:
            self._rest_lat_compact = torch.from_numpy(rest_result["rest_lat_compact"])
            self._rest_lon_compact = torch.from_numpy(rest_result["rest_lon_compact"])

        self._rest_idx = torch.from_numpy(rest_result["rest_idx"]).long()
        self._resample_state = rest_result["resample_state"]

    # ── Public interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._n_samples

    def resample_onboarding(self, rng: np.random.Generator) -> torch.Tensor | None:
        if self._onboarding_probs is None:
            return None
        arr = _sample_onboarding_array(self._onboarding_probs, max_k=5, rng=rng)
        return torch.from_numpy(arr).float()

    def resample_negatives(self, rng: np.random.Generator) -> torch.Tensor:
        rs = self._resample_state
        assert rs is not None

        n = rs.n_positives
        spp = rs.samples_per_pos
        num_neg = rs.num_negatives

        new_rest_idx = np.empty(n * spp, dtype=np.int32)
        pos_slots = np.arange(0, n * spp, spp)
        new_rest_idx[pos_slots] = rs.pos_compact_idx

        neg_compact = np.empty((n, num_neg), dtype=np.int32)
        for state_key, pool in rs.state_pool_compact.items():
            mask = rs.pos_states == state_key
            n_state = int(mask.sum())
            if n_state == 0:
                continue
            rand = rng.integers(0, len(pool), size=(n_state, num_neg))
            neg_compact[mask] = pool[rand]

        for k in range(num_neg):
            new_rest_idx[pos_slots + 1 + k] = neg_compact[:, k]

        return torch.from_numpy(new_rest_idx).long()

    def __getitem__(self, idx: int):
        ridx = self._rest_idx[idx]

        # Flat feature dict: scalars indexed by sample, compact arrays by indirection
        feats = {}
        for k, v in self._tensors.items():
            # Scalars are per-sample, compact arrays are per-restaurant
            if v.shape[0] == self._n_samples:
                feats[k] = v[idx]
            else:
                feats[k] = v[ridx]

        if self._pref_compact is not None and self._user_idx is not None:
            feats["preference_vec"] = self._pref_compact[self._user_idx[idx]]
        if self._onboarding_compact is not None and self._user_idx is not None:
            feats["onboarding_vec"] = self._onboarding_compact[self._user_idx[idx]]

        return feats, self._labels[idx]