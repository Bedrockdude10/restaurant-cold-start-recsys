"""Seed-level significance tests for the ablation study.

For each headline comparison (``full_model`` vs. an ablation) this tests whether
the Hit@5 / NDCG@10 difference is larger than training-seed noise, using the
per-seed runs of the multi-seed sweep. Where per-seed values are available for
BOTH sides over the same seed set it runs a **paired t-test** (tighter, seeds are
a shared blocking factor); otherwise it falls back to **Welch's two-sample
t-test** from the aggregated mean/std/n in ``summary.csv``.

    NB: ``full_model``'s per-seed ``eval_results.json`` are not currently committed
    (only its aggregate row in ``summary.csv``), so its comparisons run in Welch
    mode. Drop the per-seed files in and this script upgrades those comparisons to
    the paired test automatically — no code change needed.

This is *seed-level* significance: robustness of an effect to random
initialisation / training noise. It is complementary to the *per-case*
significance (McNemar / paired bootstrap over individual test cases) that
``scripts/dropoutnet_baseline.py --significance`` computes. Reproducing that for
the two-tower ablations needs per-case outcome arrays, which requires re-running
eval with ``--dump-percase`` (see ``scripts/evaluate.py``).

p-values are Holm–Bonferroni corrected across the family of reported tests.

Usage:
    python scripts/significance.py --results-dir results/ablation
    python scripts/significance.py --results-dir results/ablation --csv sig.csv --md sig.md
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

# Ensure the project root is importable so `src` resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments import (  # noqa: E402
    EXPERIMENT_LABELS,
    KEY_COMPARISONS,
    SPLITS,
)

# Metrics tested (must match scripts/evaluate.py defaults + summary.csv columns).
METRICS = [("Hit@5", "hit5"), ("NDCG@10", "ndcg10")]
MODEL_KEY = "TwoTower"
ALPHA = 0.05


# ── Numerics: regularized incomplete beta → Student-t p-values ────────────────
# Pure-stdlib so this runs without numpy/scipy. betacf/betainc follow the
# standard continued-fraction evaluation (Numerical Recipes, §6.4).


def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value P(|T| >= |t|) for a Student-t with df degrees of freedom."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    x = df / (df + t * t)
    return betainc(df / 2.0, 0.5, x)


def t_crit(df: float, conf: float = 0.95) -> float:
    """Two-sided critical t value at the given confidence (e.g. 2.776 for df=4)."""
    target = 1.0 - conf
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── Loading ───────────────────────────────────────────────────────────────────


def load_per_seed(results_dir: Path, exp: str, split: str, metric: str) -> dict[int, float]:
    """{seed: metric value} from <exp>/seed_<N>/eval_results.json, if present."""
    out: dict[int, float] = {}
    for sd in sorted((results_dir / exp).glob("seed_*")):
        f = sd / "eval_results.json"
        if not f.exists():
            continue
        try:
            seed = int(sd.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        data = json.load(open(f))
        data = data.get("metrics", data)
        val = data.get(split, {}).get(MODEL_KEY, {}).get(metric)
        if val is not None:
            out[seed] = float(val)
    return out


def load_summary(results_dir: Path) -> dict[str, dict]:
    """{experiment: row dict} from summary.csv (aggregate mean/std/n_seeds)."""
    path = results_dir / "summary.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["experiment"]: r for r in csv.DictReader(f)}


def _agg_from_summary(row: dict, split: str, col: str) -> tuple[float, float, int] | None:
    """(mean, sample_std, n_seeds) for one split/metric from a summary.csv row."""
    if not row:
        return None
    try:
        mean = float(row[f"{split}_{col}"])
        std = float(row[f"{split}_{col}_std"])
        n = int(row["n_seeds"])
    except (KeyError, ValueError, TypeError):
        return None
    return mean, std, n


# ── Tests ─────────────────────────────────────────────────────────────────────


def paired_t(diffs: list[float]) -> dict:
    """Paired t-test on per-seed differences (full - variant)."""
    n = len(diffs)
    mean_d = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else float("inf")
    df = n - 1
    t = mean_d / se if se > 0 else (0.0 if mean_d == 0 else math.inf)
    p = t_two_sided_p(t, df) if se > 0 else (1.0 if mean_d == 0 else 0.0)
    tc = t_crit(df) if df > 0 else float("nan")
    return {
        "method": "paired-t", "n": n, "delta": mean_d,
        "ci_lo": mean_d - tc * se, "ci_hi": mean_d + tc * se, "p": p,
    }


def welch_t(a: tuple[float, float, int], b: tuple[float, float, int]) -> dict:
    """Welch's two-sample t-test on (mean, std, n) summaries. delta = a - b."""
    (ma, sa, na), (mb, sb, nb) = a, b
    va, vb = sa * sa, sb * sb
    se = math.sqrt(va / na + vb / nb)
    delta = ma - mb
    if se == 0:
        return {"method": "welch-t", "n": min(na, nb), "delta": delta,
                "ci_lo": delta, "ci_hi": delta, "p": 1.0 if delta == 0 else 0.0}
    df_num = (va / na + vb / nb) ** 2
    df_den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = df_num / df_den if df_den > 0 else min(na, nb) - 1
    t = delta / se
    tc = t_crit(df)
    return {"method": "welch-t", "n": min(na, nb), "delta": delta,
            "ci_lo": delta - tc * se, "ci_hi": delta + tc * se, "p": t_two_sided_p(t, df)}


def holm(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values (order preserved)."""
    idx = [i for i, p in enumerate(pvals) if p == p]  # drop NaNs from correction
    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    adj = [float("nan")] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def run(results_dir: Path) -> list[dict]:
    summary = load_summary(results_dir)
    rows: list[dict] = []

    for cmp in KEY_COMPARISONS:
        base, var = cmp.baseline, cmp.variant  # base=full_model, var=ablation
        for metric, col in METRICS:
            for split in SPLITS:
                base_seeds = load_per_seed(results_dir, base, split, metric)
                var_seeds = load_per_seed(results_dir, var, split, metric)
                shared = sorted(set(base_seeds) & set(var_seeds))

                if len(shared) >= 2:
                    diffs = [base_seeds[s] - var_seeds[s] for s in shared]
                    res = paired_t(diffs)
                    mean_base = statistics.mean(base_seeds[s] for s in shared)
                    mean_var = statistics.mean(var_seeds[s] for s in shared)
                else:
                    a = _agg_from_summary(summary.get(base, {}), split, col)
                    b = _agg_from_summary(summary.get(var, {}), split, col)
                    if a is None or b is None:
                        continue
                    res = welch_t(a, b)
                    mean_base, mean_var = a[0], b[0]

                rows.append({
                    "comparison": cmp.title, "baseline": base, "variant": var,
                    "split": split, "metric": metric,
                    "full": mean_base, "ablation": mean_var, **res,
                })

    for r, padj in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = padj
        r["sig"] = (padj == padj) and padj < ALPHA
    return rows


# ── Reporting ───────────────────────────────────────────────────────────────


def _fmt_p(p: float) -> str:
    if p != p:
        return "  n/a "
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("  ABLATION SIGNIFICANCE — full_model vs. ablation (Δ = full − ablation)")
    print("  Paired t-test where both sides have per-seed runs, else Welch two-sample.")
    print("  p adjusted Holm–Bonferroni across the family; ★ = significant at α=0.05.")
    print("=" * 100)
    cur = None
    for r in rows:
        if r["comparison"] != cur:
            cur = r["comparison"]
            print(f"\n  {cur}   [{r['method']}, n={r['n']}]")
            print(f"    {'metric/split':<22s}{'full':>8s}{'abl.':>8s}{'Δ':>9s}"
                  f"{'95% CI':>18s}{'p':>8s}{'p_holm':>9s}  sig")
        ci = f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
        star = " ★" if r["sig"] else ""
        print(f"    {r['metric'] + ' ' + r['split']:<22s}{r['full']:>8.4f}{r['ablation']:>8.4f}"
              f"{r['delta']:>+9.4f}{ci:>18s}{_fmt_p(r['p']):>8s}{_fmt_p(r['p_holm']):>9s}{star}")
    n_sig = sum(1 for r in rows if r["sig"])
    print(f"\n  {n_sig}/{len(rows)} tests significant after Holm correction.")
    if any(r["method"] == "welch-t" for r in rows):
        print("  Note: Welch (unpaired) rows use full_model's aggregate mean/std from "
              "summary.csv;\n        commit its per-seed eval_results.json to upgrade "
              "them to the paired test.")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["comparison", "baseline", "variant", "split", "metric", "method", "n",
            "full", "ablation", "delta", "ci_lo", "ci_hi", "p", "p_holm", "sig"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    print(f"\nSignificance table written to {path}")


def write_md(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Ablation significance (full_model vs. ablation)", "",
             "Δ = full − ablation. Paired t-test where per-seed runs exist for both "
             "sides, else Welch two-sample from `summary.csv`. p-values Holm–Bonferroni "
             "corrected; **bold** = significant at α=0.05.", "",
             "| Comparison | Split | Metric | full | abl. | Δ | 95% CI | p | p (Holm) | method |",
             "|---|---|---|---:|---:|---:|:---:|---:|---:|---|"]
    for r in rows:
        d = f"**{r['delta']:+.4f}**" if r["sig"] else f"{r['delta']:+.4f}"
        ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
        lines.append(
            f"| {r['comparison']} | {r['split']} | {r['metric']} | {r['full']:.4f} | "
            f"{r['ablation']:.4f} | {d} | {ci} | {_fmt_p(r['p'])} | {_fmt_p(r['p_holm'])} | "
            f"{r['method']} |")
    path.write_text("\n".join(lines) + "\n")
    print(f"Markdown table written to {path}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Seed-level significance for the ablation study")
    ap.add_argument("--results-dir", default="results/ablation")
    ap.add_argument("--csv", default=None, help="Optional: write the table to CSV")
    ap.add_argument("--md", default=None, help="Optional: write a Markdown table (for the paper)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    _ = EXPERIMENT_LABELS  # imported for parity with the summarizer's registry use

    rows = run(results_dir)
    if not rows:
        print("No comparable results found (need summary.csv or per-seed eval_results.json).")
        return
    print_table(rows)
    if args.csv:
        write_csv(rows, Path(args.csv))
    if args.md:
        write_md(rows, Path(args.md))


if __name__ == "__main__":
    main()
