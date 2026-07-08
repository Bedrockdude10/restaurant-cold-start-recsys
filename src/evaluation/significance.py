"""Confidence intervals and paired significance tests for ranking metrics.

No SciPy dependency (chi-square survival is computed via erfc).

Which tool for which metric:
  - Hit@K is a per-case Bernoulli (the positive is in the top-K, or not), so a
    proportion CI is exact-in-form: use `wilson_ci`. To compare two methods'
    Hit@K on the *same* frozen cases, use `mcnemar` (paired binary).
  - NDCG@K is a per-case real value in [0, 1]; use `bootstrap_ci` for a single
    method and `paired_bootstrap_diff` to compare two methods on the same cases.

`wilson_ci` needs only the aggregate rate and n, so it can be computed from
already-reported numbers. The bootstrap/McNemar helpers need per-case arrays,
which the eval scripts can dump (see scripts/dropoutnet_baseline.py --significance
and DROPOUTNET_RUNBOOK.md).
"""

from __future__ import annotations

import math

import numpy as np

Z95 = 1.959963984540054  # standard normal 97.5th percentile


# ── Proportion CI (Hit@K) ────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """Wilson score CI for a binomial proportion. Returns (rate, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def wilson_ci_from_rate(rate: float, n: int, z: float = Z95) -> tuple[float, float, float]:
    """Wilson CI when you only have the reported rate (in [0,1]) and n."""
    return wilson_ci(int(round(rate * n)), n, z)


# ── Mean CI (NDCG@K) ─────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot: int = 10000, seed: int = 0,
                 ci: float = 0.95) -> tuple[float, float, float]:
    """Percentile-bootstrap CI for the mean of per-case values. (mean, lo, hi)."""
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return (float(v.mean()), float(lo), float(hi))


def paired_bootstrap_diff(a, b, n_boot: int = 10000, seed: int = 0,
                          ci: float = 0.95) -> tuple[float, float, float, float]:
    """Paired percentile-bootstrap for mean(a) - mean(b) over the same cases.

    Returns (delta, lo, hi, two_sided_p). `a` and `b` must be aligned per-case.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must be the same length (paired per-case)")
    n = a.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return (float(a.mean() - b.mean()), float(lo), float(hi), min(1.0, float(p)))


def _chi2_sf_1df(x: float) -> float:
    """Survival function of chi-square with 1 dof: P(X > x) = erfc(sqrt(x/2))."""
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar(a_hits, b_hits, correction: bool = True) -> tuple[int, int, float]:
    """McNemar's test for paired binary outcomes (e.g. Hit@K of two methods).

    Returns (n_a_only, n_b_only, p_value). Uses continuity correction by default.
    """
    a = np.asarray(a_hits, dtype=bool)
    b = np.asarray(b_hits, dtype=bool)
    n_a_only = int(np.sum(a & ~b))   # a hit, b missed
    n_b_only = int(np.sum(~a & b))   # b hit, a missed
    disc = n_a_only + n_b_only
    if disc == 0:
        return (n_a_only, n_b_only, 1.0)
    c = 1.0 if correction else 0.0
    stat = (abs(n_a_only - n_b_only) - c) ** 2 / disc
    return (n_a_only, n_b_only, min(1.0, _chi2_sf_1df(stat)))


# ── Per-case metric helpers (match src/evaluation/metrics.py conventions) ─────

def hit_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    return 1.0 if any(r in relevant_ids for r in ranked_ids[:k]) else 0.0


def ndcg_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    """NDCG@k with a single relevant item (IDCG = 1)."""
    for i, rid in enumerate(ranked_ids[:k]):
        if rid in relevant_ids:
            return 1.0 / math.log2(i + 2)
    return 0.0


# ── Self-test + report reported Hit@5 CIs ────────────────────────────────────

def _selftest() -> None:
    # constant array → CI collapses to the mean
    m, lo, hi = bootstrap_ci([0.3] * 200, n_boot=2000, seed=1)
    assert abs(m - 0.3) < 1e-9 and abs(lo - 0.3) < 1e-9 and abs(hi - 0.3) < 1e-9
    # wilson sanity: 50/100 ~ [0.404, 0.596]
    _, wlo, whi = wilson_ci(50, 100)
    assert abs(wlo - 0.404) < 0.005 and abs(whi - 0.596) < 0.005
    # paired bootstrap: a strictly > b → tiny p, positive delta CI
    rng = np.random.default_rng(0)
    a = rng.random(2000); b = a - 0.1
    d, lo, hi, p = paired_bootstrap_diff(a, b, n_boot=2000, seed=2)
    assert d > 0 and lo > 0 and p < 0.01
    # mcnemar: symmetric discordance → p ~ 1
    hits_a = np.array([1, 0] * 100, dtype=bool)
    hits_b = np.array([0, 1] * 100, dtype=bool)
    _, _, pm = mcnemar(hits_a, hits_b)
    assert pm > 0.5
    # ndcg: rank 1 → 1.0, rank 2 → 1/log2(3)
    assert ndcg_at_k(["x"], {"x"}, 10) == 1.0
    assert abs(ndcg_at_k(["a", "x"], {"x"}, 10) - 1 / math.log2(3)) < 1e-12
    print("  self-test: PASS")


# Reported Hit@5 rates (%) and case counts (Table 1) — for Wilson CIs from aggregates.
_HIT5 = {
    "Warm (n=80057)": (80057, {"Random": 5.4, "Popularity": 27.2, "DropoutNet": 18.0, "Two-Tower": 28.7}),
    "Cold-Restaurant (n=1369)": (1369, {"Random": 4.1, "Popularity": 0.0, "DropoutNet": 12.5, "Two-Tower": 10.0}),
    "Cold-User (n=1732)": (1732, {"Random": 6.2, "Popularity": 27.1, "DropoutNet": 11.8, "Two-Tower": 31.8}),
}


def _report_hit5_cis() -> None:
    print("\nWilson 95% CIs for Hit@5 (from reported rate + n):")
    for split, (n, methods) in _HIT5.items():
        print(f"\n{split}")
        for name, rate in methods.items():
            _, lo, hi = wilson_ci_from_rate(rate / 100.0, n)
            print(f"    {name:12s} {rate:5.1f}%  95% CI [{lo*100:5.1f}, {hi*100:5.1f}]")


if __name__ == "__main__":
    _selftest()
    _report_hit5_cis()
