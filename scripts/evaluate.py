"""Evaluate baselines and models on all three evaluation splits.

Usage:
    python scripts/evaluate.py --config configs/default.yaml [--checkpoint path/to/model.pt]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.baselines import BaseRecommender
from src.evaluation.metrics import score_test_cases, evaluate_recommendations
from src.evaluation.sampling import TestCase
from src.data.dataset import FEATURE_DTYPES
from src.data.pipeline import (
    prepare_features,
    build_baselines,
    build_eval_test_cases,
    augment_cold_start_users,
    build_loo_onboarding,
)
from src.models.two_tower import (
    TwoTowerModel,
    build_model,
)
from src.utils.config import load_config
from src.utils.device import get_device


# ── Two-Tower model loading ──────────────────────────────────────────────────


def load_two_tower(
    checkpoint_path: str | Path,
    restaurants: pd.DataFrame,
    train_reviews: pd.DataFrame,
) -> tuple[TwoTowerModel, dict]:
    """Load a trained TwoTower model from a checkpoint.

    Reconstructs the four-tower architecture from saved group names
    using build_model(), then loads the trained weights.
    """
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    category_vocab = ckpt["category_vocab"]
    restaurant_group_names = set(ckpt["restaurant_groups"])
    user_group_names = set(ckpt["user_groups"])
    saved_args = ckpt.get("args", {})

    model = build_model(
        restaurant_group_names=restaurant_group_names,
        user_group_names=user_group_names,
        num_categories=len(category_vocab),
        output_dim=saved_args.get("output_dim", 64),
        similarity=saved_args.get("similarity", "dot_product"),
        fusion=saved_args.get("fusion", "mlp"),
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, ckpt


# ── Evaluation runner ────────────────────────────────────────────────────────


def evaluate_model(
    model: BaseRecommender,
    test_cases: list[TestCase],
    hit_k: int = 5,
    ndcg_k: int = 10,
) -> dict[str, float | int]:
    all_ranked = []
    all_relevant = []
    for case in test_cases:
        user_context = {"user_id": case.user_id}
        scores = model.score_candidates(user_context, case.candidate_ids)
        ranked = sorted(case.candidate_ids, key=lambda bid: scores.get(bid, 0.0), reverse=True)
        all_ranked.append(ranked)
        all_relevant.append(case.relevant_ids)
    return evaluate_recommendations(all_ranked, all_relevant, hit_k, ndcg_k)


# ── Main ─────────────────────────────────────────────────────────────────────


SPLIT_NAMES = {
    "warm": "test_warm.parquet",
    "cold_restaurant": "test_cold_restaurant.parquet",
    "cold_user": "test_cold_user.parquet",
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate cold start recommendations")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to two-tower model checkpoint")
    parser.add_argument("--output", default="results/eval_results.json", help="Output JSON path")
    parser.add_argument("--n-negatives", type=int, default=99)
    parser.add_argument("--rating-threshold", type=float, default=3.0)
    parser.add_argument("--hit-k", type=int, default=5)
    parser.add_argument("--ndcg-k", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    splits_dir = Path(config["data"]["splits_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    seed = config["training"]["seed"]

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading data...")
    restaurants = pd.read_parquet(processed_dir / "restaurants.parquet")
    train = pd.read_parquet(splits_dir / "train.parquet")
    val = pd.read_parquet(splits_dir / "val.parquet")
    all_known_reviews = pd.concat([train, val], ignore_index=True)

    # Peek at checkpoint to determine which features to load
    restaurant_group_names: set[str] = set()
    user_group_names: set[str] = set()
    if args.checkpoint:
        ckpt_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        restaurant_group_names = set(ckpt_meta["restaurant_groups"])
        user_group_names = set(ckpt_meta["user_groups"])

    # ── Build features ───────────────────────────────────────────────────
    print("Building features...")
    feature_group_names = restaurant_group_names | user_group_names
    feats = prepare_features(processed_dir, train, restaurants, feature_group_names)

    print(f"  {len(restaurants):,} restaurants across {len(feats.state_index)} states")
    for state, bids in sorted(feats.state_index.items(), key=lambda x: -len(x[1])):
        print(f"    {state}: {len(bids):,} restaurants")

    # ── Initialize baselines ─────────────────────────────────────────────
    models: dict[str, BaseRecommender] = build_baselines(train, seed=seed)

    # ── Load two-tower model ─────────────────────────────────────────────
    two_tower_model = None
    two_tower_ckpt = None
    category_vocab = None

    if args.checkpoint:
        print(f"\nLoading TwoTower model from {args.checkpoint}...")
        two_tower_model, two_tower_ckpt = load_two_tower(
            args.checkpoint, restaurants, train,
        )
        category_vocab = two_tower_ckpt["category_vocab"]
        print(f"  Loaded model with groups: "
              f"restaurant={sorted(restaurant_group_names)}, "
              f"user={sorted(user_group_names)}")

    # ── Load cold start entity IDs for honest evaluation ────────────────
    cold_restaurant_ids_path = splits_dir / "cold_start_restaurant_ids.csv"
    cold_restaurant_ids: set[str] = set()
    if cold_restaurant_ids_path.exists():
        cold_restaurant_ids = set(
            pd.read_csv(cold_restaurant_ids_path)["business_id"].astype(str)
        )
        print(f"  Loaded {len(cold_restaurant_ids)} cold-start restaurant IDs")

    # ── Run evaluation across splits ─────────────────────────────────────
    results = {}

    for split_name, split_file in SPLIT_NAMES.items():
        split_path = splits_dir / split_file
        if not split_path.exists():
            print(f"\n  SKIP {split_name}: {split_path} not found")
            continue

        test_reviews = pd.read_parquet(split_path)
        print(f"\n{'='*60}")
        print(f"Split: {split_name}")
        print(f"  {len(test_reviews):,} test reviews")
        print(f"  {test_reviews['user_id'].nunique():,} unique users")
        print(f"  {test_reviews['business_id'].nunique():,} unique restaurants")

        all_reviews_for_exclusion = pd.concat(
            [all_known_reviews, test_reviews], ignore_index=True
        )

        print("  Building test cases...")
        t0 = time.time()
        test_cases = build_eval_test_cases(
            test_reviews=test_reviews,
            all_reviews=all_reviews_for_exclusion,
            feats=feats,
            n_negatives=args.n_negatives,
            rating_threshold=args.rating_threshold,
            seed=seed,
        )
        print(f"  Built {len(test_cases):,} test cases in {time.time() - t0:.1f}s")

        if not test_cases:
            print("  No valid test cases — skipping split")
            continue

        # Evaluate baselines
        results[split_name] = {}
        for model_name, model in models.items():
            t0 = time.time()
            metrics = evaluate_model(model, test_cases, hit_k=args.hit_k, ndcg_k=args.ndcg_k)
            elapsed = time.time() - t0
            results[split_name][model_name] = metrics
            print(f"  {model_name:20s}  "
                  f"Hit@{args.hit_k}: {metrics[f'Hit@{args.hit_k}']:.4f}  "
                  f"NDCG@{args.ndcg_k}: {metrics[f'NDCG@{args.ndcg_k}']:.4f}  "
                  f"({metrics['n_cases']} cases, {elapsed:.1f}s)")

        # Evaluate TwoTower model
        if two_tower_model is not None:
            assert category_vocab is not None
            enabled_keys = two_tower_model.expected_feature_keys()

            augmented = augment_cold_start_users(
                test_reviews, restaurants, feats, seed=seed,
            )
            loo_overrides = build_loo_onboarding(
                test_cases, test_reviews, restaurants, feats,
                max_k=5, seed=seed,
            )

            t0 = time.time()
            # For cold restaurant split, zero out temporal features to
            # simulate a new restaurant with no checkin history
            split_cold_ids = cold_restaurant_ids if split_name == "cold_restaurant" else None
            metrics = score_test_cases(
                model=two_tower_model,
                test_cases=test_cases,
                biz_features=feats.biz_features,
                user_centroids=augmented.user_centroids,
                category_vocab=category_vocab,
                enabled_keys=enabled_keys,
                feature_dtypes=FEATURE_DTYPES,
                user_preferences=feats.user_preferences,
                user_onboarding=augmented.user_onboarding,
                case_onboarding_overrides=loo_overrides,
                cold_restaurant_ids=split_cold_ids,
                hit_k=args.hit_k,
                ndcg_k=args.ndcg_k,
            )
            elapsed = time.time() - t0

            results[split_name]["TwoTower"] = metrics
            print(f"  {'TwoTower':20s}  "
                  f"Hit@{args.hit_k}: {metrics[f'Hit@{args.hit_k}']:.4f}  "
                  f"NDCG@{args.ndcg_k}: {metrics[f'NDCG@{args.ndcg_k}']:.4f}  "
                  f"({metrics['n_cases']} cases, {elapsed:.1f}s)")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    all_model_names = list(models.keys())
    if two_tower_model is not None:
        all_model_names.append("TwoTower")
    header = f"{'Split':<20s}"
    for name in all_model_names:
        header += f"  {name + ' H@' + str(args.hit_k):>14s}"
        header += f"  {name + ' N@' + str(args.ndcg_k):>14s}"
    print(header)
    print("-" * len(header))

    for split_name in SPLIT_NAMES:
        if split_name not in results:
            continue
        row = f"{split_name:<20s}"
        for name in all_model_names:
            m = results[split_name].get(name, {})
            h = m.get(f"Hit@{args.hit_k}", 0.0)
            n = m.get(f"NDCG@{args.ndcg_k}", 0.0)
            row += f"  {h:>14.4f}"
            row += f"  {n:>14.4f}"
        print(row)

    # ── Save results ─────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()