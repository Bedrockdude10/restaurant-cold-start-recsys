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

Multi-seed: the model (SVD + input-dropout net) is stochastic, and DropoutNet is
a *learned* baseline that competes with the two-tower, so it gets the same
variance treatment. It is trained once per seed (default: the canonical SEEDS
from src.experiments), writing results/dropoutnet/seed_<N>/results.json. The
evaluation seed — which fixes the test cases, the Random baseline, and cold-user
leave-one-out onboarding — is held at the config seed for ALL model seeds, so the
test set stays identical to the two-tower's and across DropoutNet seeds; only the
trained model varies. scripts/summarize_ablation.py aggregates (mean ± std).

Usage:
    python scripts/dropoutnet_baseline.py                # sweep the canonical SEEDS
    python scripts/dropoutnet_baseline.py --seeds 42     # single seed
    python scripts/dropoutnet_baseline.py --smoke        # fast smoke test
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
    build_loo_onboarding,
)
from src.experiments import SEEDS  # noqa: E402  (single source of truth for the seed sweep)
from src.utils.config import load_config  # noqa: E402
from src.evaluation.significance import (  # noqa: E402
    wilson_ci, bootstrap_ci, paired_bootstrap_diff, mcnemar,
)
from src.utils.device import get_device  # noqa: E402

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


def percase_metrics(scores: np.ndarray, pos_col: np.ndarray):
    """Per-case (hit5, ndcg) arrays using the same ranking as metrics_from_scores."""
    n, ncand = scores.shape
    pos_scores = scores[np.arange(n), pos_col]
    greater = (scores > pos_scores[:, None]).sum(1)
    tied_before = ((scores == pos_scores[:, None])
                   & (np.arange(ncand)[None, :] < pos_col[:, None])).sum(1)
    rank = greater + tied_before
    hit5 = (rank < 5).astype(np.float64)
    ndcg = np.where(rank < 10, 1.0 / np.log2(rank + 2), 0.0)
    return hit5, ndcg


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


def score_matrix(user_emb_per_case, item_emb, bid_to_row, cand):
    """Return (n_cases, n_cand) DropoutNet scores, chunked to bound memory."""
    n = len(cand)
    n_cand = len(cand[0])
    cand_rows = np.array([[bid_to_row[b] for b in row] for row in cand], dtype=np.int64)
    out = np.empty((n, n_cand), np.float32)
    step = 4000
    for s in range(0, n, step):
        e = min(s + step, n)
        ie = item_emb[cand_rows[s:e]]                 # (chunk, n_cand, d)
        ue = user_emb_per_case[s:e][:, None, :]       # (chunk, 1, d)
        out[s:e] = (ie * ue).sum(-1)
    return out


@dataclass
class Shared:
    """Seed-independent inputs + frozen per-split eval artifacts.

    Built once and reused for every model seed. Everything here is a function of
    the data and the *evaluation* seed only, so it is identical across model
    seeds (and matches the two-tower's frozen test cases).
    """
    restaurants: pd.DataFrame
    item_content: dict
    item_dim: int
    pref_lookup: dict
    uu: dict
    ii: dict
    rows: np.ndarray
    cols: np.ndarray
    R: csr_matrix
    user_content_tr: np.ndarray
    item_content_tr: np.ndarray
    splits_data: dict  # split -> frozen cases/baselines/loo


def load_shared(cfg, args, eval_seed: int) -> Shared:
    """Load data and build all seed-invariant artifacts, including frozen test cases."""
    processed = ROOT / cfg["data"]["processed_dir"]
    splits = ROOT / cfg["data"]["splits_dir"]

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

    feats = prepare_features(processed, train, restaurants, set())
    category_vocab = feats.category_vocab
    item_content, item_dim = build_item_content(restaurants, category_vocab)
    pref_lookup = {str(u): row.astype(np.float32)
                   for u, row in zip(feats.user_preferences.index,
                                     feats.user_preferences.values)}
    train_counts = train.groupby("business_id").size().to_dict()

    # Index maps + sparse interaction matrix (seed-independent).
    u_ids = train["user_id"].values
    i_ids = train["business_id"].values
    uu = {u: idx for idx, u in enumerate(pd.unique(u_ids))}
    ii = {b: idx for idx, b in enumerate(pd.unique(i_ids))}
    rows = np.fromiter((uu[u] for u in u_ids), dtype=np.int32, count=len(u_ids))
    cols = np.fromiter((ii[b] for b in i_ids), dtype=np.int32, count=len(i_ids))
    R = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                   shape=(len(uu), len(ii)))

    n_u, n_i = len(uu), len(ii)
    user_content_tr = np.zeros((n_u, len(PREFERENCE_FEATURES)), np.float32)
    for u, idx in uu.items():
        if u in pref_lookup:
            user_content_tr[idx] = pref_lookup[u]
    item_content_tr = np.zeros((n_i, item_dim), np.float32)
    for b, idx in ii.items():
        item_content_tr[idx] = item_content[b]

    # Frozen per-split eval artifacts — built once with the EVAL seed so the test
    # cases, Random baseline, and cold-user LOO are identical across model seeds.
    max_cases = 2000 if args.smoke else None
    splits_data = {}
    for split, fname in SPLIT_FILES.items():
        path = splits / fname
        if not path.exists():
            continue
        tr = pd.read_parquet(path)
        tr["user_id"] = tr["user_id"].astype(str)
        tr["business_id"] = tr["business_id"].astype(str)
        all_excl = pd.concat([all_known, tr[["user_id", "business_id"]]], ignore_index=True)
        test_cases = build_eval_test_cases(
            tr, all_excl, feats, n_negatives=args.n_negatives,
            rating_threshold=args.rating_threshold, max_cases=max_cases, seed=eval_seed,
        )
        if not test_cases:
            continue
        cand, user_ids, pos_col, n_cand = cases_to_arrays(test_cases)
        uniq_bids = sorted({b for row in cand for b in row})
        bid_row = {b: r for r, b in enumerate(uniq_bids)}
        rand_scores = np.random.default_rng(eval_seed).random((len(cand), n_cand))
        pop_scores = np.array(
            [[float(train_counts.get(b, 0)) for b in row] for row in cand], dtype=np.float64
        )
        loo = None
        if split == "cold_user":
            loo = build_loo_onboarding(test_cases, tr, restaurants, feats, max_k=5, seed=eval_seed)
            if loo is None:
                loo = np.zeros((len(test_cases), len(PREFERENCE_FEATURES)), np.float32)
            loo = loo.astype(np.float32)
        splits_data[split] = dict(
            n_cases=len(test_cases), cand=cand, user_ids=user_ids, pos_col=pos_col,
            uniq_bids=uniq_bids, bid_row=bid_row, rand_scores=rand_scores,
            pop_scores=pop_scores, loo=loo,
        )
        print(f"  [{split}] {len(test_cases):,} frozen test cases (eval seed {eval_seed})")

    return Shared(restaurants, item_content, item_dim, pref_lookup, uu, ii,
                  rows, cols, R, user_content_tr, item_content_tr, splits_data)


def run_seed(shared: Shared, model_seed: int, args, device, seed_dir: Path) -> dict:
    """Fit SVD + input-dropout nets at one model seed and evaluate every split."""
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    k, dev = args.k, device

    # ── Collaborative latent factors via truncated SVD ───────────────────────
    t0 = time.time()
    svd = TruncatedSVD(n_components=k, random_state=model_seed)
    U = svd.fit_transform(shared.R).astype(np.float32)   # (n_users, k)
    Vf = svd.components_.T.astype(np.float32)            # (n_items, k)
    print(f"    SVD({k}) on {shared.R.shape} in {time.time()-t0:.1f}s "
          f"(explained var={svd.explained_variance_ratio_.sum():.3f})")

    U_t = torch.from_numpy(U).to(dev)
    V_t = torch.from_numpy(Vf).to(dev)
    UC_t = torch.from_numpy(shared.user_content_tr).to(dev)
    IC_t = torch.from_numpy(shared.item_content_tr).to(dev)

    user_net = Transform(k + len(PREFERENCE_FEATURES)).to(dev)
    item_net = Transform(k + shared.item_dim).to(dev)
    opt = torch.optim.Adam(
        list(user_net.parameters()) + list(item_net.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )
    mse = nn.MSELoss()

    obs_u = torch.from_numpy(shared.rows.astype(np.int64))
    obs_i = torch.from_numpy(shared.cols.astype(np.int64))
    n_obs = len(obs_u)
    n_u, n_i = len(shared.uu), len(shared.ii)
    rng = np.random.default_rng(model_seed)

    # ── Train (distill CF affinity with latent input-dropout) ────────────────
    t0 = time.time()
    for epoch in range(args.epochs):
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
            mode = torch.randint(0, 3, (len(u_idx),), device=dev)  # 0=none,1=drop U,2=drop V
            Ul_in = Ul * (mode != 1).float().unsqueeze(1)
            Vl_in = Vl * (mode != 2).float().unsqueeze(1)
            phiU = user_net(torch.cat([Ul_in, UC_t[u_idx]], 1))
            phiI = item_net(torch.cat([Vl_in, IC_t[i_idx]], 1))
            loss = mse((phiU * phiI).sum(1), target)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        print(f"    epoch {epoch+1}/{args.epochs}  mse={total/nb:.4f}  ({time.time()-t0:.0f}s)")

    user_net.eval(); item_net.eval()

    # ── Embedding helpers (bound to this seed's factors + nets) ──────────────
    @torch.no_grad()
    def embed_items(bids, zero_latent):
        lat = np.zeros((len(bids), k), np.float32)
        con = np.zeros((len(bids), shared.item_dim), np.float32)
        for j, b in enumerate(bids):
            if not zero_latent and b in shared.ii:
                lat[j] = Vf[shared.ii[b]]
            con[j] = shared.item_content.get(b, np.zeros(shared.item_dim, np.float32))
        x = torch.cat([torch.from_numpy(lat), torch.from_numpy(con)], 1).to(dev)
        return item_net(x).cpu().numpy()

    @torch.no_grad()
    def embed_users_warm(user_ids):
        lat = np.zeros((len(user_ids), k), np.float32)
        con = np.zeros((len(user_ids), len(PREFERENCE_FEATURES)), np.float32)
        for j, u in enumerate(user_ids):
            if u in shared.uu:
                lat[j] = U[shared.uu[u]]
            if u in shared.pref_lookup:
                con[j] = shared.pref_lookup[u]
        x = torch.cat([torch.from_numpy(lat), torch.from_numpy(con)], 1).to(dev)
        return user_net(x).cpu().numpy()

    @torch.no_grad()
    def embed_users_cold(loo_content):  # latent zeroed
        lat = np.zeros((len(loo_content), k), np.float32)
        x = torch.cat([torch.from_numpy(lat), torch.from_numpy(loo_content)], 1).to(dev)
        return user_net(x).cpu().numpy()

    # ── Evaluate every split on the frozen cases ─────────────────────────────
    results = {}
    for split, sd in shared.splits_data.items():
        cand, pos_col = sd["cand"], sd["pos_col"]
        res = {
            "Random": metrics_from_scores(sd["rand_scores"], pos_col),
            "Popularity": metrics_from_scores(sd["pop_scores"], pos_col),
        }
        item_emb = embed_items(sd["uniq_bids"], zero_latent=(split == "cold_restaurant"))
        if split == "cold_user":
            user_emb = embed_users_cold(sd["loo"])
        else:
            uniq_users = list(dict.fromkeys(sd["user_ids"]))
            emb = embed_users_warm(uniq_users)
            umap = {u: idx for idx, u in enumerate(uniq_users)}
            user_emb = emb[np.array([umap[u] for u in sd["user_ids"]])]
        dn_scores = score_matrix(user_emb, item_emb, sd["bid_row"], cand)
        res["DropoutNet"] = metrics_from_scores(dn_scores, pos_col)

        m = res["DropoutNet"]
        print(f"    [{split}] DropoutNet Hit@5={m['Hit@5']:.4f}  NDCG@10={m['NDCG@10']:.4f}")

        if args.significance or args.dump_percase:
            pc = {
                "Random": percase_metrics(sd["rand_scores"], pos_col),
                "Popularity": percase_metrics(sd["pop_scores"], pos_col),
                "DropoutNet": percase_metrics(dn_scores, pos_col),
            }
            if args.dump_percase:
                np.savez(
                    seed_dir / f"percase_{split}.npz",
                    **{f"{name}_hit5": v[0] for name, v in pc.items()},
                    **{f"{name}_ndcg": v[1] for name, v in pc.items()},
                )
            if args.significance:
                sig = {}
                for name, (h, nd) in pc.items():
                    _, wlo, whi = wilson_ci(int(h.sum()), len(h))
                    _, blo, bhi = bootstrap_ci(nd, seed=model_seed)
                    sig[name] = {
                        "hit5_wilson95": [round(wlo, 4), round(whi, 4)],
                        "ndcg_bootstrap95": [round(blo, 4), round(bhi, 4)],
                    }
                for ref in ("Popularity", "Random"):
                    d, lo, hi, p = paired_bootstrap_diff(
                        pc["DropoutNet"][1], pc[ref][1], seed=model_seed)
                    _, _, pm = mcnemar(
                        pc["DropoutNet"][0].astype(bool), pc[ref][0].astype(bool))
                    sig[f"DropoutNet_vs_{ref}"] = {
                        "ndcg_delta": round(d, 4),
                        "ndcg_delta_ci95": [round(lo, 4), round(hi, 4)],
                        "ndcg_p": round(p, 4),
                        "hit5_mcnemar_p": round(pm, 4),
                    }
                res["significance"] = sig

        results[split] = res
    return results


def main():
    # Output uses a non-ASCII glyph (→); make stdout robust on Windows cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--k", type=int, default=64, help="CF latent dim")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-negatives", type=int, default=99)
    ap.add_argument("--rating-threshold", type=float, default=3.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                    help=f"Model seeds to train (default: {SEEDS})")
    ap.add_argument("--out-dir", default=str(ROOT / "results/dropoutnet"),
                    help="Output dir; each seed writes <out-dir>/seed_<N>/results.json")
    ap.add_argument("--smoke", action="store_true", help="tiny/fast validation run")
    ap.add_argument("--significance", action="store_true",
                    help="report Wilson CIs (Hit@5), bootstrap CIs (NDCG@10), and paired tests")
    ap.add_argument("--dump-percase", action="store_true",
                    help="save per-case hit5/ndcg arrays (.npz) into each seed dir")
    args = ap.parse_args()

    if args.smoke:
        args.k, args.epochs = 32, 2

    cfg = load_config(args.config)
    eval_seed = cfg["training"]["seed"]  # fixed across model seeds → frozen test cases
    device = get_device()
    print(f"device={device}  k={args.k}  epochs={args.epochs}  "
          f"seeds={args.seeds}  eval_seed={eval_seed}  smoke={args.smoke}")

    shared = load_shared(cfg, args, eval_seed)

    out_dir = Path(args.out_dir)
    for model_seed in args.seeds:
        print(f"\n=== model seed {model_seed} ===")
        seed_dir = out_dir / f"seed_{model_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        results = run_seed(shared, model_seed, args, device, seed_dir)
        out_path = seed_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"    saved → {out_path}")

    print(f"\nDone. {len(args.seeds)} seed(s) → {out_dir}")


if __name__ == "__main__":
    main()
