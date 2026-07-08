"""Full-corpus (unsampled) evaluation.

The default protocol (scripts/evaluate.py) ranks each positive against ~100
same-city *sampled* negatives. Sampled metrics are cheap and encode geographic
realism, but they can be inconsistent with full ranking (Krichene & Rendle,
"On Sampled Metrics for Item Recommendation", KDD 2020). This script removes the
sampling: each positive is ranked against EVERY restaurant in its candidate pool,
so the numbers are directly comparable to the sampled table and reveal whether the
sampling changed any conclusion.

Candidate pool:
    --corpus city   (default) all restaurants in the positive's city  (the honest
                    full candidate set for a geographically-constrained domain).
    --corpus global all restaurants in the dataset.
In both cases the user's other known positives are removed from the pool.

Methods: Random, Popularity, and the Two-Tower (with --checkpoint). Metrics match
the paper (Hit@5, NDCG@10) so sampled vs full is a like-for-like comparison. The
two-tower is scored with the SAME code path as scripts/evaluate.py
(score_test_cases), including cold-restaurant temporal zeroing and leave-one-out
onboarding for cold users. Processing is chunked over test cases to bound memory,
since the flattened (case, candidate) count is large under full ranking.

DropoutNet full-corpus is not included here: scripts/dropoutnet_baseline.py uses a
fixed-width candidate scorer and would need per-city grouping to rank full pools.

Usage:
    python scripts/evaluate_full_corpus.py --checkpoint results/ablation/full_model/checkpoints/best_model.pt
    python scripts/evaluate_full_corpus.py --smoke            # baselines only, few cases
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import FEATURE_DTYPES  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    hit_at_k, ndcg_at_k, score_test_cases,
)
from src.evaluation.sampling import TestCase  # noqa: E402
from src.data.pipeline import (  # noqa: E402
    prepare_features, augment_cold_start_users, build_loo_onboarding,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.device import get_device  # noqa: E402

SPLIT_FILES = {
    "warm": "test_warm.parquet",
    "cold_restaurant": "test_cold_restaurant.parquet",
    "cold_user": "test_cold_user.parquet",
}
CTX_COLS = ("dow_sin", "dow_cos", "is_weekend",
            "checkin_hour_sin", "checkin_hour_cos",
            "checkin_dow_sin", "checkin_dow_cos")


def load_two_tower(checkpoint_path):
    from src.models.two_tower import build_model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    model = build_model(
        restaurant_group_names=set(ckpt["restaurant_groups"]),
        user_group_names=set(ckpt["user_groups"]),
        num_categories=len(ckpt["category_vocab"]),
        output_dim=a.get("output_dim", 64),
        similarity=a.get("similarity", "dot_product"),
        fusion=a.get("fusion", "mlp"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(get_device())
    model.eval()
    return model, ckpt


def build_full_corpus_cases(chunk_df, pool_ids, restaurant_geo_map,
                            user_positive_sets, corpus):
    """One TestCase per positive; candidates = full pool minus the user's other positives."""
    cases = []
    global_pool = pool_ids if corpus == "global" else None
    for row in chunk_df.itertuples(index=False):
        bid = row.business_id
        if corpus == "city":
            key = restaurant_geo_map.get(bid)
            if key is None:
                continue
            pool = pool_ids[key]
        else:
            pool = global_pool
        upos = user_positive_sets.get(row.user_id, frozenset())
        candidates = [b for b in pool if b == bid or b not in upos]
        if bid not in candidates:      # safety (bid should be in its own pool)
            candidates.append(bid)
        ctx = {c: float(getattr(row, c)) for c in CTX_COLS if hasattr(row, c)}
        cases.append(TestCase(
            user_id=str(row.user_id), positive_id=bid,
            candidate_ids=candidates, relevant_ids={bid}, ctx=ctx,
        ))
    return cases


def baseline_metrics(cases, train_counts, seed, hit_k, ndcg_k):
    """Per-chunk Random + Popularity (returns sum-of-metric, n) for weighted merge."""
    rng = np.random.default_rng(seed)
    out = {"Random": [0.0, 0.0, 0], "Popularity": [0.0, 0.0, 0]}
    for case in cases:
        cand = case.candidate_ids
        # Random
        order = rng.permutation(len(cand))
        ranked = [cand[i] for i in order]
        out["Random"][0] += hit_at_k(ranked, case.relevant_ids, hit_k)
        out["Random"][1] += ndcg_at_k(ranked, case.relevant_ids, ndcg_k)
        out["Random"][2] += 1
        # Popularity (train review count; stable so ties keep pool order)
        scores = np.array([train_counts.get(b, 0) for b in cand], dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        ranked = [cand[i] for i in order]
        out["Popularity"][0] += hit_at_k(ranked, case.relevant_ids, hit_k)
        out["Popularity"][1] += ndcg_at_k(ranked, case.relevant_ids, ndcg_k)
        out["Popularity"][2] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--checkpoint", default=None, help="two-tower checkpoint (.pt)")
    ap.add_argument("--corpus", choices=["city", "global"], default="city")
    ap.add_argument("--chunk", type=int, default=256, help="test cases scored per batch")
    ap.add_argument("--rating-threshold", type=float, default=3.0)
    ap.add_argument("--hit-k", type=int, default=5)
    ap.add_argument("--ndcg-k", type=int, default=10)
    ap.add_argument("--max-cases", type=int, default=None, help="cap cases per split")
    ap.add_argument("--smoke", action="store_true", help="few cases, baselines only unless --checkpoint")
    ap.add_argument("--output", default=str(ROOT / "results/full_corpus_results.json"))
    args = ap.parse_args()
    if args.smoke and args.max_cases is None:
        args.max_cases = 100

    cfg = load_config(args.config)
    seed = cfg["training"]["seed"]
    processed = ROOT / cfg["data"]["processed_dir"]
    splits = ROOT / cfg["data"]["splits_dir"]

    restaurants = pd.read_parquet(processed / "restaurants.parquet")
    restaurants["business_id"] = restaurants["business_id"].astype(str)
    train = pd.read_parquet(splits / "train.parquet", columns=["user_id", "business_id"])
    val = pd.read_parquet(splits / "val.parquet", columns=["user_id", "business_id"])
    for df in (train, val):
        df["user_id"] = df["user_id"].astype(str)
        df["business_id"] = df["business_id"].astype(str)
    all_known = pd.concat([train, val], ignore_index=True)
    train_counts = train.groupby("business_id").size().to_dict()

    # Two-tower feature groups drive checkin loading; empty set is fine for baselines.
    group_names = set()
    model = ckpt = category_vocab = None
    if args.checkpoint:
        model, ckpt = load_two_tower(args.checkpoint)
        group_names = set(ckpt["restaurant_groups"]) | set(ckpt["user_groups"])
        category_vocab = ckpt["category_vocab"]

    feats = prepare_features(processed, train, restaurants, group_names)

    if args.corpus == "city":
        pool_ids = feats.city_index
        geo_map = feats.restaurant_city_map
    else:
        pool_ids = restaurants["business_id"].tolist()
        geo_map = None
    avg_pool = (np.mean([len(v) for v in feats.city_index.values()])
                if args.corpus == "city" else len(pool_ids))
    print(f"corpus={args.corpus}  avg pool size ~{avg_pool:.0f}  "
          f"checkpoint={'yes' if model else 'no (baselines only)'}")

    cold_ids = set()
    p = splits / "cold_start_restaurant_ids.csv"
    if p.exists():
        cold_ids = set(pd.read_csv(p)["business_id"].astype(str))

    enabled_keys = model.expected_feature_keys() if model else set()
    results = {}

    for split, fname in SPLIT_FILES.items():
        path = splits / fname
        if not path.exists():
            continue
        tr = pd.read_parquet(path)
        tr["user_id"] = tr["user_id"].astype(str)
        tr["business_id"] = tr["business_id"].astype(str)
        pos = tr[tr["stars"] >= args.rating_threshold]
        if args.max_cases and len(pos) > args.max_cases:
            pos = pos.sample(n=args.max_cases, random_state=seed)

        # user positive sets to exclude from the pool (all known + this split)
        allr = pd.concat([all_known, tr[["user_id", "business_id"]]], ignore_index=True)
        user_pos = allr.groupby("user_id")["business_id"].apply(frozenset).to_dict()

        aug = augment_cold_start_users(tr, restaurants, feats, seed=seed) if model else None

        acc = {}  # method -> [sum_hit, sum_ndcg, n]
        def add(name, s_hit, s_ndcg, n):
            a = acc.setdefault(name, [0.0, 0.0, 0])
            a[0] += s_hit; a[1] += s_ndcg; a[2] += n

        # cold_user: build all cases + LOO onboarding ONCE (rebuilding LOO per
        # chunk repeats a full merge/pivot over all cold reviews). Other splits
        # build cases lazily per chunk to bound memory on large warm pools.
        prebuilt = loo_all = None
        if split == "cold_user" and model is not None:
            prebuilt = build_full_corpus_cases(pos, pool_ids, geo_map, user_pos, args.corpus)
            loo_all = build_loo_onboarding(prebuilt, tr, restaurants, feats, max_k=5, seed=seed)

        def chunks():
            if prebuilt is not None:
                for i in range(0, len(prebuilt), args.chunk):
                    lo = loo_all[i:i + args.chunk] if loo_all is not None else None
                    yield prebuilt[i:i + args.chunk], lo
            else:
                for start in range(0, len(pos), args.chunk):
                    yield build_full_corpus_cases(
                        pos.iloc[start:start + args.chunk], pool_ids, geo_map,
                        user_pos, args.corpus), None

        t0 = time.time()
        for cases, loo in chunks():
            if not cases:
                continue
            bm = baseline_metrics(cases, train_counts, seed, args.hit_k, args.ndcg_k)
            for name, (sh, sn, n) in bm.items():
                add(name, sh, sn, n)
            if model:
                m = score_test_cases(
                    model=model, test_cases=cases,
                    biz_features=feats.biz_features,
                    user_centroids=aug.user_centroids,
                    category_vocab=category_vocab, enabled_keys=enabled_keys,
                    feature_dtypes=FEATURE_DTYPES,
                    user_preferences=feats.user_preferences,
                    user_onboarding=aug.user_onboarding,
                    case_onboarding_overrides=loo,
                    cold_restaurant_ids=cold_ids if split == "cold_restaurant" else None,
                    hit_k=args.hit_k, ndcg_k=args.ndcg_k,
                )
                n = m["n_cases"]
                add("TwoTower", m[f"Hit@{args.hit_k}"] * n, m[f"NDCG@{args.ndcg_k}"] * n, n)

        res = {}
        for name, (sh, sn, n) in acc.items():
            res[name] = {f"Hit@{args.hit_k}": sh / n, f"NDCG@{args.ndcg_k}": sn / n,
                         "n_cases": n}
        results[split] = res
        print(f"\n[{split}]  ({time.time()-t0:.0f}s)")
        for name, m in res.items():
            print(f"    {name:12s}  Hit@{args.hit_k}={m[f'Hit@{args.hit_k}']:.4f}  "
                  f"NDCG@{args.ndcg_k}={m[f'NDCG@{args.ndcg_k}']:.4f}  (n={m['n_cases']})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"corpus": args.corpus, "results": results}, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
