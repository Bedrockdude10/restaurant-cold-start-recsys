"""Train the two-tower cold start recommendation model.

Usage:
    python scripts/train.py --data-dir data --epochs 20 --batch-size 4096

    # Ablation: restaurant tower without temporal context
    python scripts/train.py --restaurant-groups categories price attributes

    # Ablation: user tower without preferences
    python scripts/train.py --user-groups day_of_week distance

Loads pre-built splits from data/splits/ and restaurant metadata from
data/processed/restaurants.parquet. Builds category vocab, extracts
attribute features, computes user centroids, and trains end-to-end.

Training loop uses on-device indexing: all pre-tensorized features are
moved to MPS/CUDA once, and batches are formed by indexing on-device.
No DataLoader, no CPU workers, no IPC, no per-batch CPU→GPU transfer.
Group-level shuffling preserves BPR (pos, neg_1, ..., neg_K) structure.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.features import ATTR_DIM
from src.data.dataset import (
    TwoTowerDataset,
    build_user_features,
    build_restaurant_features,
    FEATURE_DTYPES,
    USER_FEATURE_DTYPES,
    RESTAURANT_FEATURE_DTYPES,
)
from src.data.pipeline import prepare_features, build_eval_test_cases, augment_cold_start_users
from src.evaluation.metrics import score_test_cases
from src.evaluation.sampling import build_test_cases
from src.utils.geo import haversine_torch
from src.models.two_tower import (
    TEMPORAL_DIM,
    PREFERENCE_DIM,
    TwoTowerModel,
    build_model,
    RESTAURANT_ALL_GROUPS,
    USER_ALL_GROUPS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-negatives", type=int, default=20)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--similarity", type=str, default="dot_product",
                        choices=["dot_product", "cosine"])
    parser.add_argument("--fusion", type=str, default="mlp",
                        choices=["mlp", "gated"],
                        help="Fusion type: 'mlp' for standard, 'gated' for content-gated (EmerG-inspired)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--max-eval-cases", type=int, default=2000)
    parser.add_argument("--n-eval-negatives", type=int, default=99)
    parser.add_argument("--rating-threshold", type=float, default=3.0)
    parser.add_argument("--hit-k", type=int, default=5)
    parser.add_argument("--ndcg-k", type=int, default=10)

    parser.add_argument(
        "--restaurant-groups", nargs="+",
        default=sorted(RESTAURANT_ALL_GROUPS),
        choices=sorted(RESTAURANT_ALL_GROUPS),
        help="Restaurant content tower feature groups (default: all)",
    )
    parser.add_argument(
        "--user-groups", nargs="+",
        default=sorted(USER_ALL_GROUPS),
        choices=sorted(USER_ALL_GROUPS),
        help="User tower feature groups — split into context/content sub-towers (default: all)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    restaurant_group_names = set(args.restaurant_groups)
    user_group_names = set(args.user_groups)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading data...")
    train_reviews = pd.read_parquet(data_dir / "splits" / "train.parquet")
    val_reviews = pd.read_parquet(data_dir / "splits" / "val.parquet")
    restaurants = pd.read_parquet(data_dir / "processed" / "restaurants.parquet")

    print(f"  Train: {len(train_reviews):,} reviews")
    print(f"  Val:   {len(val_reviews):,} reviews")
    print(f"  Restaurants: {len(restaurants):,}")

    # ── Build features ───────────────────────────────────────────────────
    print("Building features...")
    feature_group_names = restaurant_group_names | user_group_names
    feats = prepare_features(data_dir / "processed", train_reviews, restaurants, feature_group_names)

    category_vocab = feats.category_vocab
    biz_features = feats.biz_features
    user_centroids = feats.user_centroids
    user_preferences = feats.user_preferences
    city_index = feats.city_index
    restaurant_city_map = feats.restaurant_city_map
    train_reviews = feats.train_reviews_filtered

    print(f"  Attribute dim: {ATTR_DIM}")
    print(f"  Temporal dim: {TEMPORAL_DIM}")
    print(f"  Preference dim: {PREFERENCE_DIM}")

    # ── Build model ──────────────────────────────────────────────────────
    print(f"\nRestaurant groups: {sorted(restaurant_group_names)}")
    print(f"User groups:       {sorted(user_group_names)}")

    model = build_model(
        restaurant_group_names=restaurant_group_names,
        user_group_names=user_group_names,
        num_categories=len(category_vocab),
        output_dim=args.output_dim,
        similarity=args.similarity,
        fusion=args.fusion,
    )
    model = model.to(device)

    enabled_keys = model.expected_feature_keys()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")

    # ── Pre-build validation test cases ──────────────────────────────────
    print("\nPre-building validation test cases...")
    all_known_reviews = pd.concat([train_reviews, val_reviews], ignore_index=True)
    t0 = time.time()
    val_test_cases = build_eval_test_cases(
        test_reviews=val_reviews,
        all_reviews=all_known_reviews,
        feats=feats,
        n_negatives=args.n_eval_negatives,
        rating_threshold=args.rating_threshold,
        max_cases=args.max_eval_cases,
        seed=args.seed,
    )
    print(f"  {len(val_test_cases):,} test cases built in {time.time() - t0:.1f}s")

    val_augmented = augment_cold_start_users(
        val_reviews, restaurants, feats, seed=args.seed,
    )

    # ── Build dataset ────────────────────────────────────────────────────
    print("\nBuilding training dataset...")
    city_pools = {city: np.array(bids) for city, bids in city_index.items()}

    t0 = time.time()
    train_dataset = TwoTowerDataset(
        interactions=train_reviews,
        biz_features=biz_features,
        user_centroids=user_centroids,
        category_vocab=category_vocab,
        user_preferences=user_preferences,
        enabled_keys=enabled_keys,
        restaurant_state_map=restaurant_city_map,
        state_pools=city_pools,
        num_negatives=args.num_negatives,
        seed=args.seed,
    )
    print(f"  {len(train_dataset):,} samples ({time.time() - t0:.1f}s)")

    # ── Move all training tensors to device (once) ───────────────────────
    print("\nMoving training tensors to device...")
    t0 = time.time()

    samples_per_pos = 1 + args.num_negatives
    n_samples = len(train_dataset)

    # Single flat dict of tensors on GPU. Per-sample scalars are indexed
    # directly; compact restaurant arrays are indexed via rest_idx.
    tensors_gpu = {k: v.to(device) for k, v in train_dataset._tensors.items()}

    rest_idx_gpu = train_dataset._rest_idx.to(device)
    labels_gpu = train_dataset._labels.to(device)

    user_idx_gpu = None
    if train_dataset._user_idx is not None:
        user_idx_gpu = train_dataset._user_idx.to(device)

    pref_gpu = None
    if train_dataset._pref_compact is not None:
        pref_gpu = train_dataset._pref_compact.to(device)

    onboarding_gpu = None
    if train_dataset._onboarding_compact is not None:
        onboarding_gpu = train_dataset._onboarding_compact.to(device)

    distance_on_gpu = "distance" in enabled_keys
    user_lat_gpu = user_lon_gpu = rest_lat_gpu = rest_lon_gpu = None
    if distance_on_gpu:
        assert isinstance(train_dataset._user_lat_compact, torch.Tensor)
        assert isinstance(train_dataset._user_lon_compact, torch.Tensor)
        assert isinstance(train_dataset._rest_lat_compact, torch.Tensor)
        assert isinstance(train_dataset._rest_lon_compact, torch.Tensor)
        user_lat_gpu = train_dataset._user_lat_compact.to(device)
        user_lon_gpu = train_dataset._user_lon_compact.to(device)
        rest_lat_gpu = train_dataset._rest_lat_compact.to(device)
        rest_lon_gpu = train_dataset._rest_lon_compact.to(device)

    n_groups = n_samples // samples_per_pos
    batch_size = args.batch_size - (args.batch_size % samples_per_pos)
    groups_per_batch = batch_size // samples_per_pos
    offsets = torch.arange(samples_per_pos, device=device)

    print(f"  Moved to {device} in {time.time() - t0:.1f}s")
    print(f"  {n_groups:,} groups × {samples_per_pos} samples/group = {n_samples:,} samples")
    print(f"  Batch size: {batch_size} ({groups_per_batch} groups/batch)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)

    # ── Training loop ────────────────────────────────────────────────────
    best_ndcg = 0.0
    patience = 15
    patience_counter = 0
    best_model_state = None
    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_rng = np.random.default_rng(args.seed + epoch)

        rest_idx_gpu = train_dataset.resample_negatives(epoch_rng).to(device)

        if onboarding_gpu is not None:
            resampled = train_dataset.resample_onboarding(epoch_rng)
            assert resampled is not None
            onboarding_gpu = resampled.to(device)

        group_perm = torch.randperm(n_groups, device=device)

        t0 = time.time()
        for batch_start in range(0, n_groups, groups_per_batch):
            batch_groups = group_perm[batch_start:batch_start + groups_per_batch]
            idx = (batch_groups * samples_per_pos).unsqueeze(1) + offsets
            idx = idx.reshape(-1)
            ridx = rest_idx_gpu[idx]

            # Build ONE flat batch dict — model routes keys internally
            batch = {}
            for k, v in tensors_gpu.items():
                if v.shape[0] == n_samples:
                    batch[k] = v[idx]       # per-sample scalars
                else:
                    batch[k] = v[ridx]      # compact restaurant arrays

            # Indexed user features
            if pref_gpu is not None and user_idx_gpu is not None:
                batch["preference_vec"] = pref_gpu[user_idx_gpu[idx]]
            if onboarding_gpu is not None and user_idx_gpu is not None:
                batch["onboarding_vec"] = onboarding_gpu[user_idx_gpu[idx]]

            # Distance: computed on GPU per-batch
            if distance_on_gpu:
                assert user_lat_gpu is not None and user_lon_gpu is not None
                assert rest_lat_gpu is not None and rest_lon_gpu is not None
                assert user_idx_gpu is not None
                u_lat = user_lat_gpu[user_idx_gpu[idx]]
                u_lon = user_lon_gpu[user_idx_gpu[idx]]
                r_lat = rest_lat_gpu[ridx]
                r_lon = rest_lon_gpu[ridx]
                batch["distance"] = haversine_torch(u_lat, u_lon, r_lat, r_lon).clamp(0, 200)

            scores = model(batch)
            scores = scores.view(-1, samples_per_pos)
            pos_scores = scores[:, 0:1]
            neg_scores = scores[:, 1:]
            loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        train_time = time.time() - t0

        # Evaluate
        t0 = time.time()
        val_metrics = score_test_cases(
            model=model,
            test_cases=val_test_cases,
            biz_features=biz_features,
            user_centroids=val_augmented.user_centroids,
            category_vocab=category_vocab,
            enabled_keys=enabled_keys,
            feature_dtypes=FEATURE_DTYPES,
            user_preferences=user_preferences,
            user_onboarding=val_augmented.user_onboarding,
            hit_k=args.hit_k,
            ndcg_k=args.ndcg_k,
        )
        eval_time = time.time() - t0
        scheduler.step(val_metrics[f"NDCG@{args.ndcg_k}"])
        hit5 = val_metrics[f"Hit@{args.hit_k}"]
        ndcg10 = val_metrics[f"NDCG@{args.ndcg_k}"]

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Hit@{args.hit_k}: {hit5:.4f} | "
            f"NDCG@{args.ndcg_k}: {ndcg10:.4f} | "
            f"Train: {train_time:.1f}s | "
            f"Eval: {eval_time:.1f}s"
        )

        if ndcg10 > best_ndcg:
            best_ndcg = ndcg10
            patience_counter = 0
            best_model_state = model.state_dict()
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "category_vocab": category_vocab,
                "restaurant_groups": sorted(restaurant_group_names),
                "user_groups": sorted(user_group_names),
                "args": vars(args),
            }
            torch.save(checkpoint, save_dir / "best_model.pt")
            print(f"Saved new best model with (NDCG@{args.ndcg_k}: {ndcg10:.4f})")
        else:
            patience_counter += 1
            if patience_counter == patience:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                print(f"Converged at Epoch {epoch} with NDCG@{args.ndcg_k}: {best_ndcg:.4f}")
                break

    final_checkpoint = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "category_vocab": category_vocab,
        "restaurant_groups": sorted(restaurant_group_names),
        "user_groups": sorted(user_group_names),
        "args": vars(args),
    }
    torch.save(final_checkpoint, save_dir / "final_model.pt")

    print(f"\nDone. Best NDCG@{args.ndcg_k}: {best_ndcg:.4f}")
    print(f"Checkpoints saved to {save_dir}/")


if __name__ == "__main__":
    main()