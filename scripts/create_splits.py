"""Create train/val/test splits with cold start holdout sets.

Usage:
    python scripts/create_splits.py [--config configs/default.yaml]

Split design:
    2. Select cold start restaurants (stratified by review count) — ALL their
       reviews are removed from train/val so the model sees zero interactions.
    3. Select cold start users (stratified by review count) — ALL their
       reviews are removed from train/val.
    4. Temporal split on remaining reviews: train (oldest 80%), val (10%), test (10%).
    5. Filter warm test set to users/restaurants that appear in train,
       so "warm" evaluation is truly warm.
    6. Cold start test sets are drawn from the test time period only, so
       evaluation is temporally fair.

Outputs to data/splits/:
    - train.parquet                 Reviews for training (no cold start entities)
    - val.parquet                   Reviews for validation (no cold start entities)
    - test_warm.parquet             Test reviews where both user and restaurant
                                    have training history
    - test_cold_restaurant.parquet  Test reviews for held-out restaurants
                                    (reviewed by warm users only)
    - test_cold_user.parquet        Test reviews for held-out users
                                    (at warm restaurants only)
    - cold_start_restaurant_ids.csv Restaurant IDs held out for cold start eval
    - cold_start_user_ids.csv       User IDs held out for cold start eval
    - split_summary.txt             Human-readable summary of split statistics
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config


def stratified_sample(
    df: pd.DataFrame,
    id_col: str,
    count_col: str,
    n_sample: int,
    n_bins: int = 4,
    seed: int = 42,
) -> set:
    """Sample IDs stratified by a count column (e.g., review_count).

    Bins entities into quantile-based groups and samples proportionally
    from each bin, ensuring a mix of popular and niche entities.

    Args:
        df: DataFrame with entity info (one row per entity).
        id_col: Column name for entity ID.
        count_col: Column name for the count to stratify on.
        n_sample: Total number of entities to sample.
        n_bins: Number of quantile bins for stratification.
        seed: Random seed for reproducibility.

    Returns:
        Set of sampled entity IDs.
    """
    rng = np.random.RandomState(seed)

    # Bin by quantiles of the count column
    df = df.copy()
    df["_bin"] = pd.qcut(df[count_col], q=n_bins, labels=False, duplicates="drop")

    sampled_ids = []
    bin_counts = df["_bin"].value_counts().sort_index()

    for bin_label, bin_size in bin_counts.items():
        # Sample proportionally from each bin
        bin_n = max(1, int(n_sample * bin_size / len(df)))
        bin_df = df[df["_bin"] == bin_label]
        actual_n = min(bin_n, len(bin_df))
        sampled = bin_df.sample(n=actual_n, random_state=rng)
        sampled_ids.extend(sampled[id_col].tolist())

    # If rounding left us short, sample remaining from the full pool
    remaining = n_sample - len(sampled_ids)
    if remaining > 0:
        already_sampled = set(sampled_ids)
        pool = df[~df[id_col].isin(already_sampled)]
        if len(pool) > 0:
            extra = pool.sample(n=min(remaining, len(pool)), random_state=rng)
            sampled_ids.extend(extra[id_col].tolist())

    return set(sampled_ids[:n_sample])


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = Path(config["data"]["processed_dir"])
    splits_dir = Path(config["data"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    seed = config["training"]["seed"]
    n_cold_restaurants = config["evaluation"]["num_cold_start_businesses"]
    n_cold_users = config["evaluation"]["num_cold_start_users"]

    # ── Load processed data ──────────────────────────────────────────────
    print("Loading processed data...")
    reviews = pd.read_parquet(processed_dir / "reviews.parquet")
    restaurants = pd.read_parquet(processed_dir / "restaurants.parquet")
    users = pd.read_parquet(processed_dir / "users.parquet")

    reviews["date"] = pd.to_datetime(reviews["date"])
    reviews = reviews.sort_values("date").reset_index(drop=True)

    print(f"  Total reviews: {len(reviews):,}")
    print(f"  Date range: {reviews['date'].min()} → {reviews['date'].max()}")

    # ── Select cold start restaurants ────────────────────────────────────
    # Count reviews per restaurant in the filtered dataset
    biz_review_counts = (
        reviews.groupby("business_id").size().reset_index(name="n_reviews")
    )
    # Need enough reviews to have some in the test period
    eligible_restaurants = biz_review_counts[biz_review_counts["n_reviews"] >= 5]

    print(f"\nSelecting {n_cold_restaurants} cold start restaurants "
          f"(from {len(eligible_restaurants):,} eligible)...")
    cold_restaurant_ids = stratified_sample(
        eligible_restaurants, "business_id", "n_reviews",
        n_sample=n_cold_restaurants, seed=seed,
    )
    print(f"  Selected {len(cold_restaurant_ids)} restaurants")

    # ── Select cold start users ──────────────────────────────────────────
    user_review_counts = (
        reviews.groupby("user_id").size().reset_index(name="n_reviews")
    )
    eligible_users = user_review_counts[user_review_counts["n_reviews"] >= 5]

    print(f"\nSelecting {n_cold_users} cold start users "
          f"(from {len(eligible_users):,} eligible)...")
    cold_user_ids = stratified_sample(
        eligible_users, "user_id", "n_reviews",
        n_sample=n_cold_users, seed=seed,
    )
    print(f"  Selected {len(cold_user_ids)} users")

    # ── Separate cold start reviews from the pool ────────────────────────
    is_cold_restaurant = reviews["business_id"].isin(cold_restaurant_ids)
    is_cold_user = reviews["user_id"].isin(cold_user_ids)

    # Reviews involving ANY cold start entity are removed from train/val
    warm_reviews = reviews[~is_cold_restaurant & ~is_cold_user].copy()
    cold_restaurant_reviews = reviews[is_cold_restaurant & ~is_cold_user].copy()
    cold_user_reviews = reviews[is_cold_user & ~is_cold_restaurant].copy()
    # Reviews involving BOTH a cold start user and restaurant are dropped —
    # can't evaluate either side cleanly
    both_cold = reviews[is_cold_restaurant & is_cold_user]

    print(f"\n  Warm reviews:              {len(warm_reviews):,}")
    print(f"  Cold restaurant reviews:   {len(cold_restaurant_reviews):,}")
    print(f"  Cold user reviews:         {len(cold_user_reviews):,}")
    print(f"  Both cold (dropped):       {len(both_cold):,}")

    # ── Temporal split on warm reviews ───────────────────────────────────
    warm_reviews = warm_reviews.sort_values("date").reset_index(drop=True)
    n = len(warm_reviews)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    train = warm_reviews.iloc[:train_end]
    val = warm_reviews.iloc[train_end:val_end]
    test_warm_raw = warm_reviews.iloc[val_end:]

    # Determine the test period boundary for cold start sets
    test_start_date = test_warm_raw["date"].min()

    print(f"\n  Temporal split boundaries:")
    print(f"    Train:  {train['date'].min()} → {train['date'].max()}")
    print(f"    Val:    {val['date'].min()} → {val['date'].max()}")
    print(f"    Test:   {test_warm_raw['date'].min()} → {test_warm_raw['date'].max()}")

    # ── Filter warm test to truly warm entities ──────────────────────────
    # Only keep test reviews where both the user and restaurant appeared
    # in training, so "warm" evaluation is genuinely warm
    train_users = set(train["user_id"])
    train_restaurants = set(train["business_id"])

    test_warm = test_warm_raw[
        test_warm_raw["user_id"].isin(train_users)
        & test_warm_raw["business_id"].isin(train_restaurants)
    ].copy()

    n_filtered = len(test_warm_raw) - len(test_warm)
    print(f"\n  Warm test filtering:")
    print(f"    Before: {len(test_warm_raw):,} reviews")
    print(f"    After:  {len(test_warm):,} reviews")
    print(f"    Removed {n_filtered:,} reviews with unseen users/restaurants")

    # ── Cold start test sets (test period only) ──────────────────────────
    # Only evaluate cold start on reviews from the test time period
    # so evaluation is temporally fair
    test_cold_restaurant = cold_restaurant_reviews[
        cold_restaurant_reviews["date"] >= test_start_date
    ]
    test_cold_user = cold_user_reviews[
        cold_user_reviews["date"] >= test_start_date
    ]

    # Additionally filter cold start test sets to warm counterparts:
    # - Cold restaurant reviews should be from users in train
    # - Cold user reviews should be at restaurants in train
    test_cold_restaurant = test_cold_restaurant[
        test_cold_restaurant["user_id"].isin(train_users)
    ]
    test_cold_user = test_cold_user[
        test_cold_user["business_id"].isin(train_restaurants)
    ]

    # Verify cold start coverage
    cold_restaurants_with_test = test_cold_restaurant["business_id"].nunique()
    cold_users_with_test = test_cold_user["user_id"].nunique()

    print(f"\n  Cold start restaurants with test reviews: "
          f"{cold_restaurants_with_test} / {len(cold_restaurant_ids)}")
    print(f"  Cold start users with test reviews: "
          f"{cold_users_with_test} / {len(cold_user_ids)}")

    # ── Save splits ──────────────────────────────────────────────────────
    print(f"\nSaving splits to {splits_dir}/...")

    train.to_parquet(splits_dir / "train.parquet", index=False)
    val.to_parquet(splits_dir / "val.parquet", index=False)
    test_warm.to_parquet(splits_dir / "test_warm.parquet", index=False)
    test_cold_restaurant.to_parquet(splits_dir / "test_cold_restaurant.parquet", index=False)
    test_cold_user.to_parquet(splits_dir / "test_cold_user.parquet", index=False)

    # Save cold start entity IDs for downstream use
    pd.DataFrame({"business_id": list(cold_restaurant_ids)}).to_csv(
        splits_dir / "cold_start_restaurant_ids.csv", index=False
    )
    pd.DataFrame({"user_id": list(cold_user_ids)}).to_csv(
        splits_dir / "cold_start_user_ids.csv", index=False
    )

    # ── Summary ──────────────────────────────────────────────────────────
    train_sparsity = 1 - len(train) / (train["user_id"].nunique() * train["business_id"].nunique())

    summary_lines = [
        "Split Summary",
        "=" * 60,
        f"Seed:          {seed}",
        f"Train:                  {len(train):,} reviews, "
        f"{train['user_id'].nunique():,} users, "
        f"{train['business_id'].nunique():,} restaurants",
        f"  Date range:           {train['date'].min()} → {train['date'].max()}",
        f"",
        f"Validation:             {len(val):,} reviews, "
        f"{val['user_id'].nunique():,} users, "
        f"{val['business_id'].nunique():,} restaurants",
        f"  Date range:           {val['date'].min()} → {val['date'].max()}",
        f"",
        f"Test (warm):            {len(test_warm):,} reviews, "
        f"{test_warm['user_id'].nunique():,} users, "
        f"{test_warm['business_id'].nunique():,} restaurants",
        f"  Date range:           {test_warm['date'].min()} → {test_warm['date'].max()}",
        f"  (filtered to users & restaurants seen in train)",
        f"",
        f"Test (cold restaurant): {len(test_cold_restaurant):,} reviews, "
        f"{test_cold_restaurant['business_id'].nunique():,} restaurants",
        f"  Held out restaurants: {len(cold_restaurant_ids)}",
        f"  With test reviews:   {cold_restaurants_with_test}",
        f"",
        f"Test (cold user):      {len(test_cold_user):,} reviews, "
        f"{test_cold_user['user_id'].nunique():,} users",
        f"  Held out users:      {len(cold_user_ids)}",
        f"  With test reviews:   {cold_users_with_test}",
        f"",
        f"Dropped (both cold):   {len(both_cold):,}",
        f"",
        f"Sparsity (train):",
        f"  Users:        {train['user_id'].nunique():,}",
        f"  Restaurants:  {train['business_id'].nunique():,}",
        f"  Interactions: {len(train):,}",
        f"  Sparsity:     {train_sparsity:.6f}",
    ]
    summary = "\n".join(summary_lines)
    print(f"\n{summary}")

    with open(splits_dir / "split_summary.txt", "w") as f:
        f.write(summary)

    print(f"\nDone!")


if __name__ == "__main__":
    main()