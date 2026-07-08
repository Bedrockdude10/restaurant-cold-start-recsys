"""Preprocess extracted Yelp parquets into analysis-ready processed data.

Usage:
    python scripts/preprocess_yelp.py [--config configs/default.yaml]

Expected input in data/extracted/ (output of extract_yelp.py):
    - business.parquet
    - review.parquet
    - user.parquet
    - checkin.parquet
    - tip.parquet

Outputs to data/processed/:
    - restaurants.parquet       (filtered businesses with price_tier)
    - reviews.parquet           (filtered reviews with day-of-week features)
    - users.parquet             (filtered users)
    - checkin_profiles.parquet  (per-restaurant temporal profiles from checkins)
    - tips.parquet              (filtered tips)
"""

import argparse
from pathlib import Path

import pandas as pd

from src.data.preprocessing import (
    preprocess_businesses,
    preprocess_reviews,
    preprocess_users,
    preprocess_checkins,
    match_checkins_to_reviews,
    derive_checkin_time_features,
    filter_sparse_interactions,
    preprocess_tips,
    filter_sparse_states,
)
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Preprocess extracted Yelp data")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    extracted_dir = Path(config["data"]["extracted_dir"])
    out_dir = Path(config["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    min_reviews = config["data"].get("min_user_reviews", 5)
    date_cutoff = config["data"].get("date_cutoff", "2020-01-01")

    print(f"Date cutoff: {date_cutoff}")

    # ── Businesses ───────────────────────────────────────────────────────
    print("Loading businesses...")
    biz_df = pd.read_parquet(extracted_dir / "business.parquet")
    restaurants = preprocess_businesses(biz_df)
    restaurants = filter_sparse_states(restaurants, min_restaurants=100)
    valid_biz = set(restaurants["business_id"])
    print(f"  {len(restaurants):,} restaurants (from {len(biz_df):,} total businesses)")

    # ── Users ────────────────────────────────────────────────────────────
    print("Loading users...")
    user_df = pd.read_parquet(extracted_dir / "user.parquet")
    users = preprocess_users(user_df, min_reviews=min_reviews)
    valid_users = set(users["user_id"])
    print(f"  {len(users):,} users with {min_reviews}+ reviews (from {len(user_df):,} total)")

    # ── Reviews ──────────────────────────────────────────────────────────
    print("Loading reviews...")
    review_df = pd.read_parquet(extracted_dir / "review.parquet")
    reviews = preprocess_reviews(review_df, valid_biz, valid_users, date_cutoff=date_cutoff)
    print(f"  {len(reviews):,} reviews (from {len(review_df):,} total)")

    # ── Checkins ─────────────────────────────────────────────────────────
    checkin_path = extracted_dir / "checkin.parquet"
    checkin_df = None
    if checkin_path.exists():
        print("Loading checkins...")
        checkin_df = pd.read_parquet(checkin_path)
        checkin_profiles = preprocess_checkins(checkin_df, valid_biz, restaurants, date_cutoff=date_cutoff)
        n_with_checkins = len(checkin_profiles[checkin_profiles["business_id"].isin(valid_biz)])
        print(f"  {n_with_checkins:,} restaurants have checkin data "
              f"({n_with_checkins / len(restaurants) * 100:.1f}% coverage)")
    else:
        print("  SKIP checkins: checkin.parquet not found")
        checkin_profiles = None
        
    # ── Match checkins to reviews ────────────────────────────────────
    if checkin_df is not None:
        print("Matching checkins to reviews...")
        reviews = match_checkins_to_reviews(reviews, checkin_df, tolerance_days=1)
        match_rate = reviews["checkin_ts"].notna().mean()
        print(f"  {match_rate:.1%} of reviews matched to a checkin within 1 day")

        # Derive cyclical time-of-visit features from matched timestamps
        reviews = derive_checkin_time_features(reviews, restaurants)
    else:
        reviews["checkin_ts"] = pd.NaT
        reviews["checkin_hour_sin"] = 0.0
        reviews["checkin_hour_cos"] = 0.0
        reviews["checkin_dow_sin"] = 0.0
        reviews["checkin_dow_cos"] = 0.0

    # ── Tips ─────────────────────────────────────────────────────────────
    tip_path = extracted_dir / "tip.parquet"
    if tip_path.exists():
        print("Loading tips...")
        tip_df = pd.read_parquet(tip_path)
        tips = preprocess_tips(tip_df, valid_biz, restaurants, date_cutoff=date_cutoff)
        print(f"  {len(tips):,} tips (from {len(tip_df):,} total)")
    else:
        print("  SKIP tips: tip.parquet not found")
        tips = None

    # ── Interaction density filtering ────────────────────────────────
    print("Filtering sparse interactions...")
    pre_filter = len(reviews)
    reviews = filter_sparse_interactions(
        reviews,
        min_user_reviews=config["data"].get("min_user_interactions", 10),
        min_business_reviews=config["data"].get("min_business_interactions", 5),
    )
    print(f"  {len(reviews):,} reviews after density filter (from {pre_filter:,})")

    # ── Align all datasets to surviving interactions ─────────────────
    surviving_users = set(reviews["user_id"])
    surviving_biz = set(reviews["business_id"])
    users = users[users["user_id"].isin(surviving_users)].copy()
    restaurants = restaurants[restaurants["business_id"].isin(surviving_biz)].copy()
    if checkin_profiles is not None:
        checkin_profiles = checkin_profiles[checkin_profiles["business_id"].isin(surviving_biz)].copy()
    if tips is not None:
        tips = tips[tips["business_id"].isin(surviving_biz)].copy()
    print(f"  {len(users):,} users, {len(restaurants):,} restaurants after alignment")

    # ── Save ─────────────────────────────────────────────────────────────
    print("\nSaving to parquet...")

    restaurants.to_parquet(out_dir / "restaurants.parquet", index=False)
    print(f"  {out_dir / 'restaurants.parquet'}")

    users.to_parquet(out_dir / "users.parquet", index=False)
    print(f"  {out_dir / 'users.parquet'}")

    reviews.to_parquet(out_dir / "reviews.parquet", index=False)
    print(f"  {out_dir / 'reviews.parquet'}")

    if checkin_profiles is not None:
        checkin_profiles.to_parquet(out_dir / "checkin_profiles.parquet", index=False)
        print(f"  {out_dir / 'checkin_profiles.parquet'}")

    if tips is not None:
        tips.to_parquet(out_dir / "tips.parquet", index=False)
        print(f"  {out_dir / 'tips.parquet'}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nDone! Processed data saved to {out_dir}/")
    print(f"  Restaurants:      {len(restaurants):,}")
    print(f"  Users:            {len(users):,}")
    print(f"  Reviews:          {len(reviews):,}")
    if checkin_profiles is not None:
        print(f"  Checkin profiles: {len(checkin_profiles):,}")
    if tips is not None:
        print(f"  Tips:             {len(tips):,}")


if __name__ == "__main__":
    main()