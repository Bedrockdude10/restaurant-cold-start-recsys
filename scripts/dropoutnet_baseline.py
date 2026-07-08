"""DropoutNet cold-start baseline (Volkovs et al., NeurIPS 2017).

A learned cold-start baseline for direct comparison with the two-tower model,
evaluated on the SAME frozen test cases and metrics as scripts/evaluate.py.

Recipe (lightweight, dependency-free):
    1. Collaborative latent factors U (users) and V (items) from a truncated
       SVD of the binary train interaction matrix  (WMF substitute).
    2. Two MLP "transforms": phi_U([U_u ; content_u]),  phi_I([V_i ; content_i]).
       Content = the SAME side information the two-tower uses:
         user  -> 47-d cuisine/food-type preference vector (LOO onboarding for cold users)
         item  -> category multi-hot (train vocab) + price tier.
    3. Trained to reconstruct the CF affinity U_u . V_i (distillation) while
       randomly zeroing U_u or V_i on each sample (input dropout). This forces
       the content pathway to carry the signal when a latent is missing — i.e.
       exactly the cold-start case.
    4. Inference:
         warm            -> both latents present.
         cold user       -> zero the user latent  (rely on onboarding content).
         cold restaurant -> zero the item latent for ALL candidates
                            (content-only ranking, mirroring the two-tower's
                            temporal-zeroing protocol for cold-restaurant eval).

Validation: recomputes Random + Popularity on the regenerated test cases; the
deterministic Popularity numbers must match the committed eval_results.json,
proving the test cases are identical to the two-tower's.

Usage:
    python scripts/dropoutnet_baseline.py                # full run
    python scripts/dropoutnet_baseline.py --smoke        # fast smoke test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.features import PREFERENCE_FEATURES, encode_categories  # noqa: E402
from src.data.pipeline import (  # noqa: E402
    prepare_features,
    build_eval_test_cases,
    augment_cold_start_users,
    build_loo_onboarding,
)
from src.utils.config import load_config  # noqa: E402

SPLIT_FILES = {
    "warm": "test_warm.parquet",
    "cold_restaurant": "test_cold_restaurant.parquet",
    "cold_user": "test_cold_user.parquet",
}


# ── Metrics (single relevant item per case; identical to metrics.py) ──────────

def metrics_from_scores(scores: np.ndarray, pos_col: np.ndarray) -> dict:
    """scores: (n_cases, n_cand); pos_col: index of the positive per case.

    Rank of the positive = #candidates strictly better-scored. With one
    relevant item this reproduces Hit@5 and NDCG@10 from metrics.py exactly.
    """
    n, ncand = scores.shape
    pos_scores = scores[np.arange(n), pos_col]
    # Rank with the SAME tie-breaking as the repo's stable descending sort
    # (metrics.py): candidates strictly above the positive rank higher, and
    # tied candidates rank higher iff they precede the positive in the
    # (pre-shuffled) candidate order.
    greater = (scores > pos_scores[:, None]).sum(1)
    tied_before = ((scores == pos_scores[:, None])
                   & (np.arange(ncand)[None, :] < pos_col[:, None])).sum(1)
    rank = greater + tied_before  # 0 = top
    hit5 = (rank < 5).mean()
    ndcg = np.where(rank < 10, 1.0 / np.log2(rank + 2), 0.0).mean()
    return {"Hit@5": float(hit5), "NDCG@10": float(ndcg), "n_cases": int(n)}


# ── Candidate/positive tensors from TestCase list ─────────────────────────────

def cases_to_arrays(test_cases):
    n_cand = len(test_cases[0].candidate_ids)
    assert all(len(c.candidate_ids) == n_cand for c in test_cases), "ragged candidates"
    cand = np.array([c.candidate_ids for c in test_cases], dtype=object)
    user_ids = [c.user_id for c in test_cases]
    pos_col = np.array(
        [c.candidate_ids.index(c.positive_id) for c in test_cases], dtype=np.int64
    )
    return cand, user_ids, pos_col, n_cand


# ── DropoutNet ────────────────────────────────────────────────────────────────

class Transform(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), out_dim=64, dropout=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_item_content(restaurants, category_vocab):
    """bid -> float32 vector: category multi-hot (vocab) ++ price_tier/4."""
    V = len(category_vocab)
    dim = V + 1
    content = {}
    price = restaurants.set_index("business_id")["price_tier"].to_dict()
    cats = restaurants.set_index("business_id")["categories"].to_dict()
    for bid in restaurants["business_id"].values:
        vec = np.zeros(dim, dtype=np.float32)
        for idx in encode_categories(cats.get(bid), category_vocab):
            vec[idx] = 1.0
        pt = price.get(bid, 2)
        vec[V] = (float(pt) if pt is not None and not pd.isna(pt) else 2.0) / 4.0
        content[str(bid)] = vec
    return content, dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--k", type=int, default=64, help="CF latent dim")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-negatives", type=int, default=99)
    ap.add_argument("--rating-threshold", type=float, default=3.0)
    ap.add_argument("--output", default=str(ROOT / "results/dropoutnet_results.json"))
    ap.add_argument("--smoke", action="store_true", help="tiny/fast validation run")
    args = ap.parse_args()

    if args.smoke:
        args.k, args.epochs = 32, 2

    cfg = load_config(args.config)
    seed = cfg["training"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    processed = ROOT / cfg["data"]["processed_dir"]
    splits = ROOT / cfg["data"]["splits_dir"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}  k={args.k}  epochs={args.epochs}  smoke={args.smoke}")

    # ── Load data (only the columns we need) ─────────────────────────────────
    t0 = time.time()
    restaurants = pd.read_parquet(processed / "restaurants.parquet")
    restaurants["business_id"] = restaurants["business_id"].astype(str)
    train = pd.read_parquet(splits / "train.parquet", columns=["user_id", "business_id"])
    val = pd.read_parquet(splits / "val.parquet", columns=["user_id", "business_id"])
    for df in (train, val):
        df["user_id"] = df["user_id"].astype(str)
        df["business_id"] = df["business_id"].astype(str)
    all_known = pd.concat([train, val], ignore_index=True)
    print(f"  loaded data in {time.time()-t0:.1f}s  "
          f"(train={len(train):,}, restaurants={len(restaurants):,})")

    # ── Shared feature prep (city index, prefs, centroids, vocab) ────────────
    feats = prepare_features(processed, train, restaurants, set())
    category_vocab = feats.category_vocab
    item_content, item_dim = build_item_content(restaurants, category_vocab)
    pref_lookup = {str(u): row.astype(np.float32)
                   for u, row in zip(feats.user_preferences.index,
                                     feats.user_preferences.values)}
    train_counts = train.groupby("business_id").size().to_dict()

    # ── Collaborative latent factors via truncated SVD ───────────────────────
    t0 = time.time()
    u_ids = train["user_id"].values
    i_ids = train["business_id"].values
    uu = {u: idx for idx, u in enumerate(pd.unique(u_ids))}
    ii = {b: idx for idx, b in enumerate(pd.unique(i_ids))}
    rows = np.fromiter((uu[u] for u in u_ids), dtype=np.int32, count=len(u_ids))
    cols = np.fromiter((ii[b] for b in i_ids), dtype=np.int32, count=len(i_ids))
    R = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                   shape=(len(uu), len(ii)))
    svd = TruncatedSVD(n_components=args.k, random_state=seed)
    U = svd.fit_transform(R).astype(np.float32)          # (n_users, k)
    Vf = svd.components_.T.astype(np.float32)            # (n_items, k)
    print(f"  SVD({args.k}) on {R.shape} in {time.time()-t0:.1f}s  "
          f"(explained var={svd.explained_variance_ratio_.sum():.3f})")

    # ── Aligned training tensors ─────────────────────────────────────────────
    n_u, n_i = len(uu), len(ii)
    user_content_tr = np.zeros((n_u, len(PREFERENCE_FEATURES)), np.float32)
    for u, idx in uu.items():
        if u in pref_lookup:
            user_content_tr[idx] = pref_lookup[u]
    item_content_tr = np.zeros((n_i, item_dim), np.float32)
    for b, idx in ii.items():
        item_content_tr[idx] = item_content[b]

    dev = device
    U_t = torch.from_numpy(U).to(dev)
    V_t = torch.from_numpy(Vf).to(dev)
    UC_t = torch.from_numpy(user_content_tr).to(dev)
    IC_t = torch.from_numpy(item_content_tr).to(dev)

    user_net = Transform(args.k + len(PREFERENCE_FEATURES)).to(dev)
    item_net = Transform(args.k + item_dim).to(dev)
    opt = torch.optim.Adam(
        list(user_net.parameters()) + list(item_net.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )
    mse = nn.MSELoss()

    # Observed positive pairs (u_row, i_row)
    obs_u = torch.from_numpy(rows.astype(np.int64))
    obs_i = torch.from_numpy(cols.astype(np.int64))
    n_obs = len(obs_u)
    rng = np.random.default_rng(seed)

    # ── Train (distill CF affinity with latent input-dropout) ────────────────
    t0 = time.time()
    for epoch in range(args.epochs):
        # each epoch: all observed positives + equal # random pairs
        ru = torch.from_numpy(rng.integers(0, n_u, n_obs))
        ri = torch.from_numpy(rng.integers(0, n_i, n_obs))
        bu = torch.cat([obs_u, ru]); bi = torch.cat([obs_i, ri])
        perm = torch.randperm(len(bu))
        bu, bi = bu[perm], bi[perm]
        user_net.train(); item_net.train()
        total, nb = 0.0, 0
        for s in range(0, len(bu), args.batch_size):
            u_idx = bu[s:s + args.batch_size].to(dev)
            i_idx = bi[s:s + args.batch_size].to(dev)
            Ul, Vl = U_t[u_idx], V_t[i_idx]
            target = (Ul * Vl).sum(1)                       # true CF affinity
            # input dropout: per-sample mode 0=none,1=drop user,2=drop item
            mode = torch.randint(0, 3, (len(u_idx),), device=dev)
            Ul_in = Ul * (mode != 1).float().unsqueeze(1)
            Vl_in = Vl * (mode != 2).float().unsqueeze(1)
            phiU = user_net(torch.cat([Ul_in, UC_t[u_idx]], 1))
            phiI = item_net(torch.cat([Vl_in, IC_t[i_idx]], 1))
            pred = (phiU * phiI).sum(1)
            loss = mse(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        print(f"  epoch {epoch+1}/{args.epochs}  mse={total/nb:.4f}  "
              f"({time.time()-t0:.0f}s)")

    user_net.eval(); item_net.eval()

    # ── Embedding helpers ────────────────────────────────────────────────────
    @torch.no_grad()
    def embed_items(bids, zero_latent):
        lat = np.zeros((len(bids), args.k), np.float32)
        con = np.zeros((len(bids), item_dim), np.float32)
        for j, b in enumerate(bids):
            if not zero_latent and b in ii:
                lat[j] = Vf[ii[b]]
            con[j] = item_content.get(b, np.zeros(item_dim, np.float32))
        x = torch.cat([torch.from_numpy(lat), torch.from_numpy(con)], 1).to(dev)
        return item_net(x).cpu().numpy()

    @torch.no_grad()
    def embed_users_warm(user_ids):
        lat = np.zeros((len(user_ids), args.k), np.float32)
        con = np.zeros((len(user_ids), len(PREFERENCE_FEATURES)), np.float32)
        for j, u in enumerate(user_ids):
            if u in uu:
                lat[j] = U[uu[u]]
            if u in pref_lookup:
                con[j] = pref_lookup[u]
        x = torch.cat([torch.from_numpy(lat), torch.from_numpy(con)], 1).to(dev)
        return user_net(x).cpu().numpy()

    @torch.no_grad()
    def embed_users_cold(loo_content):  # latent zeroed
        lat = np.zeros((len(loo_content), args.k), np.float32)
        x = torch.cat([torch.from_numpy(lat), torch.from_numpy(loo_content)], 1).to(dev)
        return user_net(x).cpu().numpy()

    def score_matrix(user_emb_per_case, item_emb, bid_to_row, cand):
        """Return (n_cases, n_cand) DropoutNet scores, chunked to bound memory."""
        n = len(cand)
        n_cand = len(cand[0])
        cand_rows = np.array(
            [[bid_to_row[b] for b in row] for row in cand], dtype=np.int64
        )
        out = np.empty((n, n_cand), np.float32)
        step = 4000
        for s in range(0, n, step):
            e = min(s + step, n)
            ie = item_emb[cand_rows[s:e]]                 # (chunk, n_cand, d)
            ue = user_emb_per_case[s:e][:, None, :]       # (chunk, 1, d)
            out[s:e] = (ie * ue).sum(-1)
        return out

    # ── Evaluate every split ─────────────────────────────────────────────────
    results = {}
    for split, fname in SPLIT_FILES.items():
        path = splits / fname
        if not path.exists():
            continue
        test_reviews = pd.read_parquet(path)
        test_reviews["user_id"] = test_reviews["user_id"].astype(str)
        test_reviews["business_id"] = test_reviews["business_id"].astype(str)
        all_excl = pd.concat([all_known, test_reviews[["user_id", "business_id"]]],
                             ignore_index=True)
        max_cases = 2000 if args.smoke else None
        test_cases = build_eval_test_cases(
            test_reviews, all_excl, feats, n_negatives=args.n_negatives,
            rating_threshold=args.rating_threshold, max_cases=max_cases, seed=seed,
        )
        if not test_cases:
            continue
        cand, user_ids, pos_col, n_cand = cases_to_arrays(test_cases)
        uniq_bids = sorted({b for row in cand for b in row})
        bid_row = {b: r for r, b in enumerate(uniq_bids)}

        # baselines (validation) — same cases
        rand_scores = np.random.default_rng(seed).random((len(cand), n_cand))
        pop_scores = np.array(
            [[float(train_counts.get(b, 0)) for b in row] for row in cand],
            dtype=np.float64,
        )
        res = {
            "Random": metrics_from_scores(rand_scores, pos_col),
            "Popularity": metrics_from_scores(pop_scores, pos_col),
        }

        # DropoutNet
        zero_item = (split == "cold_restaurant")
        item_emb = embed_items(uniq_bids, zero_latent=zero_item)
        if split == "cold_user":
            loo = build_loo_onboarding(test_cases, test_reviews, restaurants, feats,
                                       max_k=5, seed=seed)
            if loo is None:
                loo = np.zeros((len(test_cases), len(PREFERENCE_FEATURES)), np.float32)
            user_emb = embed_users_cold(loo.astype(np.float32))
        else:
            uniq_users = list(dict.fromkeys(user_ids))
            emb = embed_users_warm(uniq_users)
            umap = {u: k for k, u in enumerate(uniq_users)}
            user_emb = emb[np.array([umap[u] for u in user_ids])]
        dn_scores = score_matrix(user_emb, item_emb, bid_row, cand)
        res["DropoutNet"] = metrics_from_scores(dn_scores, pos_col)

        results[split] = res
        print(f"\n[{split}]  ({len(test_cases):,} cases)")
        for name, m in res.items():
            print(f"    {name:12s}  Hit@5={m['Hit@5']:.4f}  NDCG@10={m['NDCG@10']:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
