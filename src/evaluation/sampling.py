"""Geographic negative sampling and test case construction for evaluation.

Provides geographic negative sampling so evaluation candidate sets are
geographically plausible. A user in Pittsburgh only sees Pittsburgh-area
candidates, not restaurants from Phoenix — matching the real-world
scenario where users browse nearby options.

The geographic scope (city-level vs state-level) is determined by the
caller's choice of indexes passed to build_test_cases(). Both train.py
and evaluate.py should use the same indexes via pipeline.build_eval_test_cases()
to prevent train/eval distribution mismatch.

Also provides pre-built test case construction: build_test_cases() samples
candidates once and returns reusable TestCase objects. This avoids redundant
negative sampling across epochs during training — only model scores change.

Evaluation vs. training negative stability:
    Evaluation candidate sets (built here) are sampled ONCE and frozen.
    This ensures metric comparisons across epochs and across models use
    identical test conditions — the only variable is the model's scores.

    Training negatives (managed by TwoTowerDataset.resample_negatives()
    in dataset.py) are reshuffled EVERY epoch. This is intentional: fixed
    training negatives let the model memorize specific pairings instead of
    learning generalizable feature interactions. The two strategies serve
    different purposes — stable measurement vs. robust learning.

Used by both train.py (in-training validation) and evaluate.py (final
post-training evaluation) to ensure consistent metrics across the pipeline.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def build_state_index(restaurants: pd.DataFrame) -> dict[str, list[str]]:
    """Build a mapping from state -> list of business_ids in that state.

    Used for sampling geographically plausible negatives.

    Args:
        restaurants: Processed restaurants DataFrame with 'business_id' and 'state'.

    Returns:
        Dict mapping state code -> list of business_ids in that state.
    """
    state_index: dict[str, list[str]] = {}
    for state, group in restaurants.groupby("state"):
        state_index[str(state)] = group["business_id"].tolist()
    return state_index


def build_city_index(
    restaurants: pd.DataFrame, min_restaurants: int = 10
) -> dict[str, list[str]]:
    """Build a mapping from city key -> list of business_ids in that city.

    Uses (state, city) as the key to handle duplicate city names across
    states. Only includes cities with at least min_restaurants restaurants,
    ensuring enough diversity for negative sampling.

    Args:
        restaurants: Processed restaurants DataFrame with 'business_id', 'state', 'city'.
        min_restaurants: Minimum restaurants a city must have to be included.

    Returns:
        Dict mapping "state::city" -> list of business_ids in that city.
    """
    city_index: dict[str, list[str]] = {}
    for (state, city), group in restaurants.groupby(["state", "city"]):
        if len(group) >= min_restaurants:
            key = f"{state}::{city}"
            city_index[key] = group["business_id"].tolist()
    return city_index


def build_restaurant_city_map(
    restaurants: pd.DataFrame, city_index: dict[str, list[str]]
) -> dict[str, str]:
    """Build a mapping from business_id -> city key.

    Only maps restaurants that belong to cities in the city_index
    (i.e., cities with enough restaurants for negative sampling).

    Args:
        restaurants: Processed restaurants DataFrame.
        city_index: Output of build_city_index().

    Returns:
        Dict mapping business_id -> "state::city" key.
    """
    valid_bids: dict[str, str] = {}
    for key, bids in city_index.items():
        for bid in bids:
            valid_bids[bid] = key
    return valid_bids


def build_restaurant_state_map(restaurants: pd.DataFrame) -> dict[str, str]:
    """Build a mapping from business_id -> state.

    Args:
        restaurants: Processed restaurants DataFrame with 'business_id' and 'state'.

    Returns:
        Dict mapping business_id -> state code.
    """
    return dict(restaurants.set_index("business_id")["state"])


def sample_negatives(
    positive_id: str,
    user_positives: set[str],
    state_pool: list[str],
    n_negatives: int,
    rng: np.random.Generator,
) -> list[str]:
    """Sample n_negatives business IDs from the same state, excluding positives.

    Ensures the candidate set is geographically plausible: all negatives
    come from the same state as the positive restaurant.

    Args:
        positive_id: The ground-truth business_id (excluded from negatives).
        user_positives: All business_ids this user has interacted with (excluded).
        state_pool: All business_ids in the same state as the positive.
        n_negatives: Number of negatives to sample.
        rng: NumPy random generator.

    Returns:
        List of negative business_ids.
    """
    eligible = [bid for bid in state_pool if bid not in user_positives]

    if len(eligible) < n_negatives:
        # If not enough negatives in the state (rare), sample with replacement
        return list(rng.choice(eligible, size=n_negatives, replace=True))

    return list(rng.choice(eligible, size=n_negatives, replace=False))


# ── Pre-built test cases ─────────────────────────────────────────────────────


@dataclass
class TestCase:
    """One evaluation instance: a positive review + geographically-scoped negatives.

    Constructed once via build_test_cases(), reused across epochs.
    Only model scores change between epochs — candidate sets are fixed.
    """
    user_id: str
    positive_id: str
    candidate_ids: list[str]   # positive + negatives, pre-shuffled
    relevant_ids: set[str]     # just the positive, for metric computation
    ctx: dict                  # dow_sin, dow_cos, is_weekend from the review


def build_test_cases(
    test_reviews: pd.DataFrame,
    all_reviews: pd.DataFrame,
    geo_index: dict[str, list[str]],
    restaurant_geo_map: dict[str, str],
    n_negatives: int = 99,
    rating_threshold: float = 3.0,
    max_cases: int | None = None,
    seed: int = 42,
) -> list[TestCase]:
    """Build evaluation test cases with geographic negative sampling.

    Each test case is one positive interaction + N geographically-scoped
    negatives. Candidate lists are pre-shuffled so position doesn't leak
    the answer.

    The geographic scope is determined entirely by the caller's choice of
    indexes: city-level (city_index + restaurant_city_map) for tighter
    candidate pools, or state-level (state_index + restaurant_state_map)
    for broader pools. This function is agnostic to which level is used.

    Designed to be called once and reused: during training, call this before
    the training loop, then re-score each epoch with score_test_cases().

    Vectorized: groups positives by geographic key and draws all negatives
    per group in one bulk rng.integers() call. Collisions with user
    positives are repaired in-place — with ~20 user positives in a pool
    of thousands, collision rate is <1% so repair is nearly free.

    Args:
        test_reviews: The test/val split DataFrame.
        all_reviews: ALL reviews for computing user positive sets to exclude
            from negative sampling.
        geo_index: Geographic key -> list of business_ids in that region.
            Typically city_index (from build_city_index) or state_index
            (from build_state_index).
        restaurant_geo_map: business_id -> geographic key. Must be
            consistent with geo_index (e.g. restaurant_city_map for
            city_index, restaurant_state_map for state_index).
        n_negatives: Number of negative candidates per test case.
        rating_threshold: Minimum star rating to count as a positive.
        max_cases: Cap on number of test cases (None = no cap). When set,
            positive reviews are sampled randomly up to this limit.
        seed: Random seed for reproducibility.

    Returns:
        List of TestCase instances, ready for repeated scoring.
    """
    rng = np.random.default_rng(seed)

    # Filter to positive interactions only
    positives = test_reviews[test_reviews["stars"] >= rating_threshold]
    if len(positives) == 0:
        return []

    # Optionally cap the number of test cases
    if max_cases is not None and len(positives) > max_cases:
        positives = positives.sample(n=max_cases, random_state=seed)

    # Pre-compute user positive sets (all interactions, not just test)
    user_positive_sets = (
        all_reviews.groupby("user_id")["business_id"]
        .apply(set)
        .to_dict()
    )

    # ── Extract arrays for vectorized access (avoid iterrows) ────────────
    pos_uids = positives["user_id"].values
    pos_bids = positives["business_id"].values
    pos_dow_sin = positives["dow_sin"].values.astype(np.float64)
    pos_dow_cos = positives["dow_cos"].values.astype(np.float64)
    pos_is_wknd = positives["is_weekend"].values.astype(np.float64)

    # Checkin-matched time-of-visit (zeros where unmatched)
    pos_checkin = {}
    for col in ("checkin_hour_sin", "checkin_hour_cos",
                "checkin_dow_sin", "checkin_dow_cos"):
        if col in positives.columns:
            pos_checkin[col] = positives[col].values.astype(np.float64)

    n_pos = len(positives)

    # Map each positive to its geographic key
    pos_geo_keys = np.array(
        [restaurant_geo_map.get(bid, "") for bid in pos_bids]
    )
    valid_mask = pos_geo_keys != ""

    # ── Vectorized per-region bulk sampling + collision repair ────────────
    # Instead of rebuilding an eligible list per case (O(pool_size) each),
    # draw all negatives for each region in one rng.integers() call, then
    # repair the ~<1% of samples that collide with user positives.

    neg_bids = np.empty((n_pos, n_negatives), dtype=object)

    # Pre-convert pools to numpy arrays for fancy indexing
    pool_arrays: dict[str, np.ndarray] = {
        key: np.array(pool) for key, pool in geo_index.items()
    }

    for geo_key, pool_arr in pool_arrays.items():
        geo_mask = pos_geo_keys == geo_key
        n_geo = int(geo_mask.sum())
        if n_geo == 0:
            continue

        pool_size = len(pool_arr)

        # One bulk draw for all cases in this region
        rand_idx = rng.integers(0, pool_size, size=(n_geo, n_negatives))
        sampled = pool_arr[rand_idx]

        # Collision repair: only touches samples that hit user positives.
        # With ~20 positives in a pool of thousands, <1% of cells need repair.
        geo_indices = np.where(geo_mask)[0]
        for i, global_idx in enumerate(geo_indices):
            user_pos = user_positive_sets.get(str(pos_uids[global_idx]), set())
            if not user_pos:
                continue
            for j in range(n_negatives):
                # Resample until we get a non-positive (usually 0 iterations)
                attempts = 0
                while sampled[i, j] in user_pos and attempts < 50:
                    sampled[i, j] = pool_arr[rng.integers(0, pool_size)]
                    attempts += 1

        neg_bids[geo_mask] = sampled

    # ── Build TestCase objects ───────────────────────────────────────────
    test_cases = []
    skipped = 0

    for i in range(n_pos):
        if not valid_mask[i]:
            skipped += 1
            continue

        candidates = [pos_bids[i]] + list(neg_bids[i])
        rng.shuffle(candidates)

        test_cases.append(TestCase(
            user_id=str(pos_uids[i]),
            positive_id=pos_bids[i],
            candidate_ids=list(candidates),
            relevant_ids={pos_bids[i]},
            ctx={
                "dow_sin": float(pos_dow_sin[i]),
                "dow_cos": float(pos_dow_cos[i]),
                "is_weekend": float(pos_is_wknd[i]),
                **{col: float(arr[i]) for col, arr in pos_checkin.items()},
            },
        ))

    if skipped > 0:
        print(f"  Skipped {skipped} test cases (restaurant not in geo index)")

    return test_cases