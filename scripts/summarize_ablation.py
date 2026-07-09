"""Summarize ablation study results into comparison tables.

Reads per-seed eval_results.json from each experiment directory and produces:
    1. Console table with all experiments × splits × metrics (mean ± std over seeds)
    2. CSV for further analysis / paper figures
    3. Key comparisons highlighted (content vs content+context)
    4. Sampled-vs-full-corpus comparison (when full-corpus results are present)

Expected layout (written by scripts/run_ablation.py):
    results/ablation/{experiment}/seed_{N}/eval_results.json
    results/ablation/{experiment}/seed_{N}/full_corpus_results.json   (optional)
A legacy flat layout (results/ablation/{experiment}/eval_results.json) is still
read, treated as a single primary-seed run.

Usage:
    python scripts/summarize_ablation.py --results-dir results/ablation
    python scripts/summarize_ablation.py --results-dir results/ablation --csv out.csv
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is importable so `src` resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments import (  # noqa: E402
    DISPLAY_ORDER,
    EXPERIMENT_LABELS,
    KEY_COMPARISONS,
    PRIMARY_SEED,
    SHORT_LABELS,
    SPLITS,
)

SPLIT_LABELS = {
    "warm": "Warm",
    "cold_restaurant": "Cold Rest.",
    "cold_user": "Cold User",
}

# Metrics reported in the tables (must match scripts/evaluate.py defaults).
HIT = "Hit@5"
NDCG = "NDCG@10"


# ── Loading ────────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    # Tolerate an outer {metrics: ...} wrapper.
    return data.get("metrics", data)


def _load_full_corpus(path: Path) -> dict | None:
    """Load full_corpus_results.json, returning its per-split results dict."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("results", data)


def load_experiment_results(results_dir: Path) -> dict[str, dict]:
    """Load every experiment's per-seed results from ``results_dir``.

    Returns a dict mapping experiment_name -> {
        "seeds": {seed_int: <per-split metrics>},
        "full_corpus": <per-split full-corpus results> | None,
    }. Prefers the per-seed layout (``<exp>/seed_<N>/``); falls back to the
    legacy flat layout (``<exp>/eval_results.json``) as a single primary seed.
    """
    experiments: dict[str, dict] = {}
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        seeds: dict[int, dict] = {}
        full_corpus = None

        seed_dirs = sorted(exp_dir.glob("seed_*"))
        if seed_dirs:
            for sd in seed_dirs:
                try:
                    seed = int(sd.name.split("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                eval_file = sd / "eval_results.json"
                if eval_file.exists():
                    seeds[seed] = _load_json(eval_file)
            # Full-corpus lives with the primary seed (fall back to any seed dir).
            for sd in [exp_dir / f"seed_{PRIMARY_SEED}", *seed_dirs]:
                fc = _load_full_corpus(sd / "full_corpus_results.json")
                if fc:
                    full_corpus = fc
                    break
        else:
            # Legacy flat layout: one run, treated as the primary seed.
            eval_file = exp_dir / "eval_results.json"
            if eval_file.exists():
                seeds[PRIMARY_SEED] = _load_json(eval_file)
            full_corpus = _load_full_corpus(exp_dir / "full_corpus_results.json")

        if seeds:
            experiments[exp_dir.name] = {"seeds": seeds, "full_corpus": full_corpus}
    return experiments


# ── Table construction ───────────────────────────────────────────────────────


def _agg(values: list) -> tuple[float | None, float | None]:
    """Mean and sample std of the non-null values. std is None for n < 2."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else None
    return mean, std


def _aggregate_row(name: str, label: str, short: str, seed_metrics: dict[int, dict],
                   model_key: str = "TwoTower", full_corpus: dict | None = None,
                   include_baselines: bool = True) -> dict:
    """Aggregate one model's metrics across seeds into a flat table row.

    The model's Hit@5/NDCG@10 are averaged over seeds (with sample std). Baselines,
    case counts, and full-corpus numbers are seed-invariant, so they're taken from
    the primary seed (or the lowest available seed).
    """
    seeds = sorted(seed_metrics)
    primary = seed_metrics.get(PRIMARY_SEED, seed_metrics[seeds[0]])

    row = {"experiment": name, "label": label, "short_label": short, "n_seeds": len(seeds)}
    for split in SPLITS:
        hits = [seed_metrics[s].get(split, {}).get(model_key, {}).get(HIT) for s in seeds]
        ndcgs = [seed_metrics[s].get(split, {}).get(model_key, {}).get(NDCG) for s in seeds]
        mean_h, std_h = _agg(hits)
        mean_n, std_n = _agg(ndcgs)
        row[f"{split}_hit5"] = mean_h          # mean (kept for dashboard compatibility)
        row[f"{split}_hit5_std"] = std_h
        row[f"{split}_ndcg10"] = mean_n
        row[f"{split}_ndcg10_std"] = std_n
        row[f"{split}_n_cases"] = primary.get(split, {}).get(model_key, {}).get("n_cases")

        if include_baselines:
            for baseline in ("Random", "Popularity"):
                bl = primary.get(split, {}).get(baseline, {})
                if bl:
                    row[f"{split}_{baseline.lower()}_hit5"] = bl.get(HIT)
                    row[f"{split}_{baseline.lower()}_ndcg10"] = bl.get(NDCG)

        if full_corpus:
            fc = full_corpus.get(split, {}).get(model_key, {})
            if fc:
                row[f"{split}_hit5_full"] = fc.get(HIT)
                row[f"{split}_ndcg10_full"] = fc.get(NDCG)

    return row


def build_results_table(experiments: dict[str, dict],
                        dropoutnet_seeds: dict[int, dict] | None = None) -> pd.DataFrame:
    """Build a flat DataFrame with one row per experiment (+ DropoutNet)."""
    def row_for(name: str, exp: dict) -> dict:
        return _aggregate_row(
            name, EXPERIMENT_LABELS.get(name, name), SHORT_LABELS.get(name, name),
            exp["seeds"], full_corpus=exp.get("full_corpus"),
        )

    rows = [row_for(name, experiments[name]) for name in DISPLAY_ORDER if name in experiments]
    rows += [row_for(name, exp) for name, exp in experiments.items() if name not in DISPLAY_ORDER]

    if dropoutnet_seeds:
        # DropoutNet is a learned baseline with its own seed sweep and model key,
        # aggregated the same way (mean ± std) as the two-tower experiments.
        rows.append(_aggregate_row(
            "dropoutnet", "DropoutNet (cold-start baseline)", "DropoutNet",
            dropoutnet_seeds, model_key="DropoutNet", include_baselines=False,
        ))

    return pd.DataFrame(rows)


def load_dropoutnet(results_dir: Path, explicit: Path | None = None) -> dict[int, dict] | None:
    """Load the DropoutNet baseline's per-seed results as {seed: metrics}.

    DropoutNet is evaluated by scripts/dropoutnet_baseline.py on the SAME frozen
    test cases as the ablation eval (verified by matching Popularity). Prefers the
    multi-seed layout ``<results_dir>/../dropoutnet/seed_<N>/results.json``; falls
    back to the legacy single file ``<results_dir>/../dropoutnet_results.json``.
    An explicit path (file or directory) overrides the search.
    """
    def seeds_from_dir(d: Path) -> dict[int, dict]:
        out = {}
        for sd in sorted(d.glob("seed_*")):
            f = sd / "results.json"
            if f.exists():
                try:
                    out[int(sd.name.split("_", 1)[1])] = _load_json(f)
                except (IndexError, ValueError):
                    continue
        return out

    if explicit is not None:
        if explicit.is_dir():
            return seeds_from_dir(explicit) or None
        return {PRIMARY_SEED: _load_json(explicit)} if explicit.exists() else None

    dn_dir = results_dir.parent / "dropoutnet"
    if dn_dir.is_dir():
        seeds = seeds_from_dir(dn_dir)
        if seeds:
            return seeds
    legacy = results_dir.parent / "dropoutnet_results.json"
    return {PRIMARY_SEED: _load_json(legacy)} if legacy.exists() else None


# ── Printing ───────────────────────────────────────────────────────────────


def _fmt(mean: float | None, std: float | None = None) -> str:
    """Format a metric cell as ``mean`` or ``mean±std``."""
    if mean is None:
        return "—"
    return f"{mean:.4f}±{std:.3f}" if std is not None else f"{mean:.4f}"


def print_main_table(df: pd.DataFrame) -> None:
    """Print the primary results table to console (mean ± std over seeds)."""
    print("\n" + "=" * 112)
    print("  ABLATION STUDY RESULTS — Two-Tower Cold Start Recommendation Model")
    print("  Cells are mean ± sample std over seeds (std omitted for single-seed rows).")
    print("=" * 112)

    print(f"\n{'Experiment':<40s}{'seeds':>6s}", end="")
    for split in SPLITS:
        label = SPLIT_LABELS[split]
        print(f"  {label + ' H@5':>14s}  {label + ' N@10':>14s}", end="")
    print()
    print("-" * 130)

    for _, row in df.iterrows():
        label = row["label"]
        if len(label) > 39:
            label = label[:36] + "..."
        print(f"{label:<40s}{int(row.get('n_seeds', 1)):>6d}", end="")
        for split in SPLITS:
            h = _fmt(row.get(f"{split}_hit5"), row.get(f"{split}_hit5_std"))
            n = _fmt(row.get(f"{split}_ndcg10"), row.get(f"{split}_ndcg10_std"))
            print(f"  {h:>14s}  {n:>14s}", end="")
        print()
    print("-" * 130)

    # Baselines (seed-invariant) from the first row that carries them.
    for _, first in df.iterrows():
        if first.get("warm_random_hit5") is not None:
            print(f"\n{'Baselines:':<46s}")
            for baseline in ("random", "popularity"):
                print(f"  {baseline.capitalize():<44s}", end="")
                for split in SPLITS:
                    h = _fmt(first.get(f"{split}_{baseline}_hit5"))
                    n = _fmt(first.get(f"{split}_{baseline}_ndcg10"))
                    print(f"  {h:>14s}  {n:>14s}", end="")
                print()
            break


def _print_comparison(df: pd.DataFrame, exp_lookup: dict,
                      title: str, exp_a: str, exp_b: str, question: str) -> None:
    """Print a single A-vs-B Hit@5 comparison block, if both rows exist."""
    if exp_a not in exp_lookup or exp_b not in exp_lookup:
        return
    row_a = df.iloc[exp_lookup[exp_a]]
    row_b = df.iloc[exp_lookup[exp_b]]

    print(f"\n  {title}")
    print(f"  Question: {question}")
    print(f"  {'':20s}  {'Warm H@5':>10s}  {'Cold-R H@5':>10s}  {'Cold-U H@5':>10s}")

    for row in (row_a, row_b):
        short_label = row["label"][:20]
        vals = [f"{row.get(f'{s}_hit5'):.4f}" if row.get(f"{s}_hit5") is not None
                else "  —   " for s in SPLITS]
        print(f"  {short_label:<20s}  {vals[0]:>10s}  {vals[1]:>10s}  {vals[2]:>10s}")

    deltas = []
    for split in SPLITS:
        ha, hb = row_a.get(f"{split}_hit5"), row_b.get(f"{split}_hit5")
        if ha is not None and hb is not None:
            deltas.append(f"{'+' if ha - hb >= 0 else ''}{ha - hb:.4f}")
        else:
            deltas.append("  —   ")
    print(f"  {'Δ (A - B)':<20s}  {deltas[0]:>10s}  {deltas[1]:>10s}  {deltas[2]:>10s}")


def print_key_comparisons(df: pd.DataFrame) -> None:
    """Print the key ablation comparisons that tell the paper's story."""
    print("\n" + "=" * 80)
    print("  KEY COMPARISONS (Hit@5, mean over seeds)")
    print("=" * 80)

    exp_lookup = dict(zip(df["experiment"], range(len(df))))
    for cmp in KEY_COMPARISONS:
        _print_comparison(df, exp_lookup, cmp.title, cmp.baseline, cmp.variant, cmp.question)

    # DropoutNet is a separate model, not in the registry — compare when present.
    _print_comparison(
        df, exp_lookup, "Full model vs DropoutNet", "full_model", "dropoutnet",
        "How does the two-tower compare to the DropoutNet cold-start baseline?",
    )


def print_full_corpus_comparison(df: pd.DataFrame) -> None:
    """Print sampled-vs-full-corpus Hit@5 for experiments that have both."""
    have_full = df[df["warm_hit5_full"].notna()] if "warm_hit5_full" in df else df.iloc[0:0]
    if have_full.empty:
        return

    print("\n" + "=" * 96)
    print("  SAMPLED vs FULL-CORPUS (Hit@5) — does the eval protocol change conclusions?")
    print("=" * 96)
    print(f"\n{'Experiment':<28s}", end="")
    for split in SPLITS:
        print(f"  {SPLIT_LABELS[split] + ' smp':>12s}  {SPLIT_LABELS[split] + ' full':>12s}", end="")
    print()
    print("-" * 108)
    for _, row in have_full.iterrows():
        print(f"{row['label'][:27]:<28s}", end="")
        for split in SPLITS:
            smp = _fmt(row.get(f"{split}_hit5"))
            full = _fmt(row.get(f"{split}_hit5_full"))
            print(f"  {smp:>12s}  {full:>12s}", end="")
        print()


# ── Entry point ────────────────────────────────────────────────────────────


def summarize(results_dir: Path, csv_path: Path | None = None,
              dropoutnet_path: Path | None = None) -> None:
    """Load results and print summary tables. Callable from run_ablation or CLI."""
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    experiments = load_experiment_results(results_dir)
    if not experiments:
        print(f"No eval_results.json found under {results_dir}/")
        return

    seed_counts = {name: len(e["seeds"]) for name, e in experiments.items()}
    print(f"Found {len(experiments)} experiments "
          f"({min(seed_counts.values())}–{max(seed_counts.values())} seeds each)")

    dropoutnet_seeds = load_dropoutnet(results_dir, explicit=dropoutnet_path)
    if dropoutnet_seeds:
        print(f"Including DropoutNet baseline ({len(dropoutnet_seeds)} seed(s))")

    df = build_results_table(experiments, dropoutnet_seeds=dropoutnet_seeds)
    print_main_table(df)
    print_key_comparisons(df)
    print_full_corpus_comparison(df)

    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

    missing = [name for name in DISPLAY_ORDER if name not in experiments]
    if missing:
        print(f"\nMissing experiments ({len(missing)}):")
        for name in missing:
            print(f"  - {name}: {EXPERIMENT_LABELS.get(name, name)}")


def main():
    # Console tables use non-ASCII glyphs (×, Δ, ±); make stdout robust on
    # Windows cp1252 terminals so a print can never abort the CSV write.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Summarize ablation study results")
    parser.add_argument("--results-dir", default="results/ablation",
                        help="Directory containing experiment subdirectories")
    parser.add_argument("--csv", default=None,
                        help="Optional: save results table to CSV")
    parser.add_argument("--dropoutnet", default=None,
                        help="Optional: path to dropoutnet_results.json "
                             "(default: <results-dir>/../dropoutnet_results.json)")
    args = parser.parse_args()

    summarize(
        results_dir=Path(args.results_dir),
        csv_path=Path(args.csv) if args.csv else None,
        dropoutnet_path=Path(args.dropoutnet) if args.dropoutnet else None,
    )


if __name__ == "__main__":
    main()
