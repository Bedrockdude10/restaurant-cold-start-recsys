"""Shared data loading and feature preparation for train and evaluate.

Single source of truth for the setup logic that both scripts need:
    - Loading checkin profiles (conditional on temporal group)
    - Building biz_features, category vocab, state/city indexes
    - Computing user centroids and preferences (on city-filtered training data)
    - Building evaluation test cases (city-level negative sampling)
    - Augmenting cold-start user features for evaluation splits
    - Leave-one-out onboarding for cold-start user evaluation
    - Constructing baselines from training data only (no leakage)

Cold-start user onboarding — leave-one-out protocol:
    Cold-start users have no training reviews, so their onboarding vectors
    must be derived from test reviews. Naively computing onboarding from
    ALL test reviews leaks information: the onboarding signal is partially
    derived from the ground-truth positive being evaluated, artificially
    inflating cold-user metrics.

    build_loo_onboarding() fixes this by computing a per-test-case
    onboarding vector from all the user's test reviews EXCEPT the one
    containing the positive restaurant. The approach is vectorized:
    raw category counts are precomputed per cold user via a single
    merge + pivot, per-business category contributions are cached as
    index lists, and the per-case subtraction is a handful of integer
    decrements. Cuisine and food-type groups are renormalized separately
    (matching compute_user_preferences), then all cold-start cases are
    batch-sampled through the Gumbel-max onboarding sampler in one call.

    Cold-start users whose positive restaurant is their only source of
    preference signal receive an honest zero vector — the model must
    rely on other features (distance, day-of-week, restaurant tower
    quality) for those cases.

    This issue does NOT affect cold-start restaurants. The restaurant
    tower's features (categories, price, attributes, temporal profile)
    come from business metadata and checkin data, not from the reviews
    being evaluated. No review-derived signal leaks into the restaurant
    tower during cold-restaurant evaluation.

Both train.py and evaluate.py call prepare_features() instead of
duplicating ~30 lines of setup code with subtle divergence risk.
Both call build_eval_test_cases() for evaluation test case construction
to guarantee identical geographic scope for negative sampling.

Usage:
    feats = prepare_features(data_dir, train_reviews, restaurants, restaurant_group_names)
    test_cases = build_eval_test_cases(val_reviews, all_reviews, feats, seed=42)
    augmented = augment_cold_start_users(val_reviews, restaurants, feats, seed=42)
    loo = build_loo_onboarding(test_cases, val_reviews, restaurants, feats, seed=42)
    baselines = build_baselines(train_reviews, seed=42)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features import (
    PREFERENCE_FEATURES,
    build_category_vocab,
    compute_user_centroids,
    compute_user_preferences,
    sample_onboarding_selections,
)
from src.data.dataset import prepare_biz_features
from src.evaluation.baselines import RandomBaseline, PopularityBaseline, BaseRecommender
from src.evaluation.sampling import (
    build_state_index,
    build_restaurant_state_map,
    build_city_index,
    build_restaurant_city_map,
    build_test_cases,
    TestCase,
)


@dataclass
class PreparedFeatures:
    """Everything both train.py and evaluate.py need after data loading.

    All user-level features (centroids, preferences) are computed on
    city-filtered training reviews so train and eval see identical subsets.
    """
    # Restaurant features
    biz_features: dict[str, dict]
    category_vocab: dict[str, int]
    checkin_profiles: pd.DataFrame | None

    # User features (computed on city-filtered training data)
    user_centroids: dict[str, tuple[float, float]]
    user_preferences: pd.DataFrame
    user_onboarding: pd.DataFrame  # binary multi-hot onboarding selections (initial sample)

    # Geographic indexes
    state_index: dict[str, list[str]]
    restaurant_state_map: dict[str, str]
    city_index: dict[str, list[str]]
    restaurant_city_map: dict[str, str]

    # City-filtered training reviews (for dataset construction)
    train_reviews_filtered: pd.DataFrame


def prepare_features(
    processed_dir: Path,
    train_reviews: pd.DataFrame,
    restaurants: pd.DataFrame,
    restaurant_group_names: set[str],
    min_city_restaurants: int = 10,
    min_category_df: int = 10,
) -> PreparedFeatures:
    """Build all features needed for training and evaluation.

    This is the single entry point that replaces the duplicated setup
    blocks in train.py and evaluate.py. The city-filtering step is
    applied here so user centroids and preferences are always computed
    on the same subset.

    Args:
        processed_dir: Path to processed data directory (contains
            checkin_profiles.parquet when temporal features are enabled).
        train_reviews: Training split DataFrame.
        restaurants: Processed restaurants DataFrame.
        restaurant_group_names: Enabled restaurant tower groups (determines
            whether checkin profiles are loaded for temporal features).
        min_city_restaurants: Minimum restaurants per city for inclusion.
        min_category_df: Minimum document frequency for category vocab.

    Returns:
        PreparedFeatures with all indexes, feature dicts, and the
        city-filtered training reviews.
    """

    # ── Checkin profiles (conditional on temporal group) ──────────────
    checkin_profiles = None
    if "temporal" in restaurant_group_names:
        checkin_path = processed_dir / "checkin_profiles.parquet"
        if checkin_path.exists():
            checkin_profiles = pd.read_parquet(checkin_path)
            print(f"  Checkin profiles: {len(checkin_profiles):,}")
        else:
            print("  WARNING: temporal group enabled but checkin_profiles.parquet not found")
            print("  Temporal features will be zeros for all restaurants")

    # ── Restaurant features ──────────────────────────────────────────
    train_biz_ids = set(train_reviews["business_id"])
    category_vocab = build_category_vocab(restaurants, train_biz_ids, min_df=min_category_df)
    biz_features = prepare_biz_features(restaurants, checkin_profiles)

    # ── Geographic indexes ───────────────────────────────────────────
    state_index = build_state_index(restaurants)
    restaurant_state_map = build_restaurant_state_map(restaurants)
    city_index = build_city_index(restaurants, min_restaurants=min_city_restaurants)
    restaurant_city_map = build_restaurant_city_map(restaurants, city_index)

    # ── City-filtered training reviews ───────────────────────────────
    # Both train.py and evaluate.py need user features computed on the
    # same city-filtered subset. Centralizing this prevents divergence.
    valid_city_bids = set(restaurant_city_map.keys())
    train_filtered = train_reviews[
        train_reviews["business_id"].isin(valid_city_bids)
    ].copy()

    print(f"  Train after city filter: {len(train_filtered):,} / {len(train_reviews):,} "
          f"({len(train_filtered) / len(train_reviews) * 100:.1f}%)")

    # ── User features (on filtered training data) ────────────────────
    user_centroids = compute_user_centroids(train_filtered, restaurants)
    user_preferences = compute_user_preferences(train_filtered, restaurants)
    user_onboarding = sample_onboarding_selections(
        user_preferences, max_k=5, rng=np.random.default_rng(42),
    )

    print(f"  Category vocab: {len(category_vocab)} entries")
    print(f"  User centroids: {len(user_centroids):,}")
    print(f"  States: {len(state_index)} ({', '.join(sorted(state_index.keys()))})")
    print(f"  City pools: {len(city_index)} cities with {min_city_restaurants}+ restaurants")

    return PreparedFeatures(
        biz_features=biz_features,
        category_vocab=category_vocab,
        checkin_profiles=checkin_profiles,
        user_centroids=user_centroids,
        user_preferences=user_preferences,
        user_onboarding=user_onboarding,
        state_index=state_index,
        restaurant_state_map=restaurant_state_map,
        city_index=city_index,
        restaurant_city_map=restaurant_city_map,
        train_reviews_filtered=train_filtered,
    )


def build_eval_test_cases(
    test_reviews: pd.DataFrame,
    all_reviews: pd.DataFrame,
    feats: PreparedFeatures,
    n_negatives: int = 99,
    rating_threshold: float = 3.0,
    max_cases: int | None = None,
    seed: int = 42,
) -> list[TestCase]:
    """Build evaluation test cases using city-level negative sampling.

    Single entry point for both train.py (validation) and evaluate.py
    (final test). Eliminates the risk of train and eval using different
    geographic scopes for negative sampling.

    Args:
        test_reviews: The test/val split DataFrame.
        all_reviews: ALL reviews for computing user positive sets to
            exclude from negative sampling.
        feats: PreparedFeatures from prepare_features().
        n_negatives: Number of negative candidates per test case.
        rating_threshold: Minimum star rating to count as a positive.
        max_cases: Cap on number of test cases (None = no cap).
        seed: Random seed for reproducibility.

    Returns:
        List of TestCase instances, ready for repeated scoring.
    """
    return build_test_cases(
        test_reviews=test_reviews,
        all_reviews=all_reviews,
        geo_index=feats.city_index,
        restaurant_geo_map=feats.restaurant_city_map,
        n_negatives=n_negatives,
        rating_threshold=rating_threshold,
        max_cases=max_cases,
        seed=seed,
    )


@dataclass
class AugmentedUserFeatures:
    """User features augmented with cold-start user data for evaluation.

    When evaluating on splits that contain users not seen during training,
    user_centroids and user_onboarding need entries for those users.
    This dataclass wraps the augmented versions so callers don't mutate
    the original PreparedFeatures.
    """
    user_centroids: dict[str, tuple[float, float]]
    user_onboarding: pd.DataFrame


def augment_cold_start_users(
    test_reviews: pd.DataFrame,
    restaurants: pd.DataFrame,
    feats: PreparedFeatures,
    seed: int = 42,
) -> AugmentedUserFeatures:
    """Augment user features for cold-start users missing from training data.

    Cold-start users aren't in training data, so feats.user_onboarding and
    feats.user_centroids have no entries for them. Without augmentation they
    get zero onboarding vectors (defeating the feature's purpose) and (0,0)
    GPS coordinates (corrupting the distance feature).

    Onboarding vectors for cold-start users are set to zeros here. The
    actual leave-one-out onboarding is computed per-test-case by
    build_loo_onboarding() and injected into score_test_cases() via
    the case_onboarding_overrides array.

    Centroids are computed from held-out review locations — mirroring the
    real scenario where we know the user's location from their device.

    For warm splits where all users are in training, this is a no-op that
    returns the original features unchanged.

    Args:
        test_reviews: The test/val split DataFrame.
        restaurants: Processed restaurants DataFrame (for centroid computation).
        feats: PreparedFeatures from prepare_features().
        seed: Random seed (unused, kept for API compatibility).

    Returns:
        AugmentedUserFeatures with centroids and placeholder onboarding
        that cover all users in test_reviews.
    """
    eval_onboarding = feats.user_onboarding
    eval_centroids = feats.user_centroids

    test_user_ids = set(test_reviews["user_id"].astype(str))
    known_user_ids = set(feats.user_onboarding.index)
    cold_user_ids = test_user_ids - known_user_ids

    if cold_user_ids:
        cold_reviews = test_reviews[
            test_reviews["user_id"].astype(str).isin(cold_user_ids)
        ]
        print(f"  Augmenting features for {len(cold_user_ids)} cold-start users")

        # Onboarding: placeholder zeros — real values come from build_loo_onboarding()
        pref_dim = len(PREFERENCE_FEATURES)
        cold_onboarding = pd.DataFrame(
            np.zeros((len(cold_user_ids), pref_dim), dtype=np.float32),
            index=sorted(cold_user_ids),
            columns=list(PREFERENCE_FEATURES),
        )
        eval_onboarding = pd.concat([feats.user_onboarding, cold_onboarding])

        # Centroids: compute from held-out review locations
        cold_centroids = compute_user_centroids(cold_reviews, restaurants)
        eval_centroids = {**feats.user_centroids, **cold_centroids}

    return AugmentedUserFeatures(
        user_centroids=eval_centroids,
        user_onboarding=eval_onboarding,
    )


def build_loo_onboarding(
    test_cases: list[TestCase],
    test_reviews: pd.DataFrame,
    restaurants: pd.DataFrame,
    feats: PreparedFeatures,
    max_k: int = 5,
    seed: int = 42,
) -> np.ndarray | None:
    """Build leave-one-out onboarding vectors for cold-start test cases.

    For each test case belonging to a cold-start user, computes an onboarding
    vector from all that user's test reviews EXCEPT the one containing the
    positive being evaluated. This prevents information leakage where the
    onboarding signal is derived from the ground-truth answer.

    Warm users (present in training) are unaffected — their onboarding comes
    from training-only preferences via the global user_onboarding DataFrame.

    Vectorized approach:
        1. Precompute per-cold-user raw category counts from ALL their test
           reviews (one merge + pivot, not per-user loops).
        2. Precompute per-business category count vectors (which categories
           each restaurant contributes to a user's visit history).
        3. For each cold-start test case: subtract the positive restaurant's
           category contribution from the user's total counts, renormalize.
        4. Batch-sample onboarding across all cold-start cases at once via
           the existing Gumbel-max sampler.

    Args:
        test_cases: Pre-built TestCase list from build_eval_test_cases().
        test_reviews: The test/val split DataFrame.
        restaurants: Processed restaurants DataFrame.
        feats: PreparedFeatures (to identify cold-start users).
        max_k: Maximum cuisines per onboarding selection.
        seed: Random seed for reproducibility.

    Returns:
        Float32 array of shape (n_test_cases, PREFERENCE_DIM) with per-case
        onboarding vectors. Warm-user rows are all zeros (score_test_cases
        uses the global onboarding for those). Returns None if there are no
        cold-start users.
    """
    from src.data.features import (
        _CUISINES_LOWER,
        _FOOD_TYPES_LOWER,
        _sample_onboarding_array,
    )

    known_user_ids = set(feats.user_onboarding.index)
    test_user_ids = set(test_reviews["user_id"].astype(str))
    cold_user_ids = test_user_ids - known_user_ids

    if not cold_user_ids:
        return None

    pref_features = list(PREFERENCE_FEATURES)
    pref_dim = len(pref_features)
    feat_to_idx = {f: i for i, f in enumerate(pref_features)}
    all_pref_cats = _CUISINES_LOWER | _FOOD_TYPES_LOWER

    # ── Step 1: Build per-business category vectors ──────────────────────
    # For each restaurant, which PREFERENCE_FEATURES categories does it have?
    # This is the "contribution" that gets subtracted in leave-one-out.
    biz_cats = restaurants[["business_id", "categories"]].copy()
    biz_cats = biz_cats.explode("categories").dropna(subset=["categories"])
    biz_cats["categories"] = biz_cats["categories"].astype(str).str.strip().str.lower()
    biz_cats = biz_cats[biz_cats["categories"].isin(all_pref_cats)]

    # business_id -> list of feature indices
    biz_cat_indices: dict[str, list[int]] = {}
    for bid, group in biz_cats.groupby("business_id"):
        indices = [feat_to_idx[c] for c in group["categories"] if c in feat_to_idx]
        if indices:
            biz_cat_indices[str(bid)] = indices

    # ── Step 2: Build per-cold-user raw category counts ──────────────────
    # Raw visit counts (not normalized) so we can subtract analytically.
    cold_reviews_df = test_reviews[
        test_reviews["user_id"].astype(str).isin(cold_user_ids)
    ].copy()
    cold_reviews_df["user_id"] = cold_reviews_df["user_id"].astype(str)

    # Merge to get per-(user, visit) category rows
    cold_visits = cold_reviews_df[["user_id", "business_id"]].merge(
        biz_cats, on="business_id", how="left",
    ).dropna(subset=["categories"])

    # Pivot to raw counts: (user_id, category) -> count
    if len(cold_visits) > 0:
        user_cat_counts = cold_visits.pivot_table(
            index="user_id", columns="categories",
            aggfunc="size", fill_value=0,
        )
        user_cat_counts = user_cat_counts.reindex(
            columns=pref_features, fill_value=0,
        )
    else:
        user_cat_counts = pd.DataFrame(
            0, index=sorted(cold_user_ids), columns=pref_features,
        )

    # Convert to numpy for fast indexing (user_id -> row index)
    cold_uid_list = list(user_cat_counts.index)
    cold_uid_to_row = {uid: i for i, uid in enumerate(cold_uid_list)}
    counts_array = user_cat_counts.values.astype(np.float64)  # (n_cold_users, pref_dim)

    # ── Step 3: Build per-test-case LOO count vectors ────────────────────
    n_cases = len(test_cases)
    # Track which cases are cold-start (need LOO override)
    cold_case_indices: list[int] = []
    loo_counts_list: list[np.ndarray] = []

    for i, case in enumerate(test_cases):
        if case.user_id not in cold_user_ids:
            continue

        cold_case_indices.append(i)
        user_row = cold_uid_to_row.get(case.user_id)

        if user_row is None:
            # User had no matching preference categories at all
            loo_counts_list.append(np.zeros(pref_dim, dtype=np.float64))
            continue

        # Start with full counts, subtract positive restaurant's contribution
        loo = counts_array[user_row].copy()
        pos_indices = biz_cat_indices.get(case.positive_id, [])
        for idx in pos_indices:
            loo[idx] = max(0, loo[idx] - 1)

        loo_counts_list.append(loo)

    if not cold_case_indices:
        return None

    # ── Step 4: Normalize per cuisine/food-type group, then sample ───────
    # _build_preferences_table normalizes cuisine and food-type counts
    # SEPARATELY, so we must replicate that structure here.

    loo_matrix = np.array(loo_counts_list, dtype=np.float64)  # (n_cold_cases, pref_dim)

    # Identify which feature columns are cuisine vs food-type
    cuisine_mask = np.array([f in _CUISINES_LOWER for f in pref_features])
    food_mask = np.array([f in _FOOD_TYPES_LOWER for f in pref_features])

    # Normalize cuisine columns
    cuisine_sums = loo_matrix[:, cuisine_mask].sum(axis=1, keepdims=True)
    cuisine_sums = np.where(cuisine_sums > 0, cuisine_sums, 1.0)  # avoid div-by-zero
    loo_matrix[:, cuisine_mask] /= cuisine_sums

    # Normalize food-type columns
    food_sums = loo_matrix[:, food_mask].sum(axis=1, keepdims=True)
    food_sums = np.where(food_sums > 0, food_sums, 1.0)
    loo_matrix[:, food_mask] /= food_sums

    # Zero out rows where both groups had zero counts (truly no signal)
    no_signal = (loo_matrix.sum(axis=1) == 0)
    # (already zeros, but explicit for clarity)

    # Batch Gumbel-max sample across all cold-start cases
    rng = np.random.default_rng(seed)
    loo_onboarding = _sample_onboarding_array(loo_matrix, max_k=max_k, rng=rng)

    # ── Step 5: Build full output array (zeros for warm users) ───────────
    result = np.zeros((n_cases, pref_dim), dtype=np.float32)
    for j, case_idx in enumerate(cold_case_indices):
        result[case_idx] = loo_onboarding[j]

    n_zero = int(no_signal.sum())
    print(f"  LOO onboarding: {len(cold_case_indices)} cold-start cases "
          f"({n_zero} with no signal after LOO exclusion)")

    return result


def build_baselines(
    train_reviews: pd.DataFrame,
    seed: int = 42,
) -> dict[str, BaseRecommender]:
    """Construct baseline models from training data only.

    Uses training review counts for the popularity baseline, not the
    restaurant DataFrame's review_count column (which includes all
    reviews and would leak information for cold-start restaurants).

    Args:
        train_reviews: Training split DataFrame with business_id column.
        seed: Random seed for the random baseline.

    Returns:
        Dict mapping baseline name -> BaseRecommender instance.
    """
    train_counts = (
        train_reviews
        .groupby("business_id")
        .size()
        .reset_index(name="review_count")
    )
    return {
        "Random": RandomBaseline(seed=seed),
        "Popularity": PopularityBaseline(train_counts),
    }