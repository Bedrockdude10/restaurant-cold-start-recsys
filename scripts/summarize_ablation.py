"""Summarize ablation study results into comparison tables.

Reads eval_results.json from each experiment directory and produces:
    1. Console table with all experiments × splits × metrics
    2. CSV for further analysis / paper figures
    3. Key comparisons highlighted (content vs content+context)

Usage:
    python scripts/summarize_ablation.py --results-dir results/ablation
    python scripts/summarize_ablation.py --results-dir results/ablation --csv results/ablation_summary.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is importable so `src` resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments import (  # noqa: E402
    DISPLAY_ORDER,
    EXPERIMENT_LABELS,
    KEY_COMPARISONS,
    SHORT_LABELS,
    SPLITS,
)

SPLIT_LABELS = {
    "warm": "Warm",
    "cold_restaurant": "Cold Rest.",
    "cold_user": "Cold User",
}


def load_experiment_results(results_dir: Path) -> dict[str, dict]:
    """Load all eval_results.json files from the ablation directory.

    Returns:
        Dict mapping experiment_name -> {split_name -> {metric -> value}}.
    """
    experiments = {}
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        eval_file = exp_dir / "eval_results.json"
        if not eval_file.exists():
            continue
        with open(eval_file) as f:
            data = json.load(f)
        # Handle both flat format and nested {metrics: ..., config: ...}
        metrics = data.get("metrics", data)
        experiments[exp_dir.name] = metrics
    return experiments


def load_dropoutnet(path: Path | None) -> dict | None:
    """Load the DropoutNet baseline results, if the file exists.

    DropoutNet is evaluated by scripts/dropoutnet_baseline.py on the SAME
    frozen test cases as the ablation eval (verified by matching Popularity),
    but stored separately in results/dropoutnet_results.json.
    """
    if not path or not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("metrics", data)


def _result_row(name: str, label: str, short: str, data: dict,
                model_key: str = "TwoTower", include_baselines: bool = False) -> dict:
    """Build one flat table row from a per-split results dict.

    Args:
        model_key: which model's scores to read per split (e.g. "TwoTower"
            for ablations, "DropoutNet" for the DropoutNet baseline).
        include_baselines: also emit Random/Popularity columns when present.
    """
    row = {"experiment": name, "label": label, "short_label": short}
    for split in SPLITS:
        split_data = data.get(split, {})
        m = split_data.get(model_key, {})
        row[f"{split}_hit5"] = m.get("Hit@5", None)
        row[f"{split}_ndcg10"] = m.get("NDCG@10", None)
        row[f"{split}_n_cases"] = m.get("n_cases", None)
        if include_baselines:
            for baseline in ["Random", "Popularity"]:
                bl = split_data.get(baseline, {})
                if bl:
                    row[f"{split}_{baseline.lower()}_hit5"] = bl.get("Hit@5", None)
                    row[f"{split}_{baseline.lower()}_ndcg10"] = bl.get("NDCG@10", None)
    return row


def build_results_table(experiments: dict[str, dict],
                        dropoutnet: dict | None = None) -> pd.DataFrame:
    """Build a flat DataFrame with one row per experiment.

    Columns: experiment, label, {split}_{metric} for each split × metric.
    The DropoutNet baseline (if provided) is appended as a final row.
    """
    rows = []
    for exp_name in DISPLAY_ORDER:
        if exp_name not in experiments:
            continue
        rows.append(_result_row(
            exp_name, EXPERIMENT_LABELS.get(exp_name, exp_name),
            SHORT_LABELS.get(exp_name, exp_name), experiments[exp_name],
            include_baselines=True,
        ))

    # Add any experiments not in DISPLAY_ORDER
    for exp_name, data in experiments.items():
        if exp_name not in DISPLAY_ORDER:
            rows.append(_result_row(
                exp_name, EXPERIMENT_LABELS.get(exp_name, exp_name),
                SHORT_LABELS.get(exp_name, exp_name), data,
                include_baselines=True,
            ))

    # Append the DropoutNet baseline as its own row (different model key).
    if dropoutnet:
        rows.append(_result_row(
            "dropoutnet", "DropoutNet (cold-start baseline)", "DropoutNet",
            dropoutnet, model_key="DropoutNet",
        ))

    return pd.DataFrame(rows)


def print_main_table(df: pd.DataFrame) -> None:
    """Print the primary results table to console."""
    print("\n" + "=" * 100)
    print("  ABLATION STUDY RESULTS — Two-Tower Cold Start Recommendation Model")
    print("=" * 100)

    # Header
    print(f"\n{'Experiment':<45s}", end="")
    for split in SPLITS:
        label = SPLIT_LABELS[split]
        print(f"  {label + ' H@5':>10s}  {label + ' N@10':>10s}", end="")
    print()
    print("-" * 105)

    for _, row in df.iterrows():
        label = row["label"]
        if len(label) > 44:
            label = label[:41] + "..."
        print(f"{label:<45s}", end="")
        for split in SPLITS:
            h = row.get(f"{split}_hit5")
            n = row.get(f"{split}_ndcg10")
            h_str = f"{h:.4f}" if h is not None else "  —   "
            n_str = f"{n:.4f}" if n is not None else "  —   "
            print(f"  {h_str:>10s}  {n_str:>10s}", end="")
        print()

    print("-" * 105)

    # Print baselines if available
    first_row = df.iloc[0] if len(df) > 0 else None
    if first_row is not None and first_row.get("warm_random_hit5") is not None:
        print(f"\n{'Baselines:':<45s}")
        for baseline in ["random", "popularity"]:
            name = baseline.capitalize()
            print(f"  {name:<43s}", end="")
            for split in SPLITS:
                h = first_row.get(f"{split}_{baseline}_hit5")
                n = first_row.get(f"{split}_{baseline}_ndcg10")
                h_str = f"{h:.4f}" if h is not None else "  —   "
                n_str = f"{n:.4f}" if n is not None else "  —   "
                print(f"  {h_str:>10s}  {n_str:>10s}", end="")
            print()
        print()


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

    for row, label in [(row_a, row_a["label"]), (row_b, row_b["label"])]:
        short_label = label[:20] if len(label) > 20 else label
        vals = []
        for split in SPLITS:
            h = row.get(f"{split}_hit5")
            vals.append(f"{h:.4f}" if h is not None else "  —   ")
        print(f"  {short_label:<20s}  {vals[0]:>10s}  {vals[1]:>10s}  {vals[2]:>10s}")

    # Delta
    deltas = []
    for split in SPLITS:
        ha = row_a.get(f"{split}_hit5")
        hb = row_b.get(f"{split}_hit5")
        if ha is not None and hb is not None:
            delta = ha - hb
            sign = "+" if delta >= 0 else ""
            deltas.append(f"{sign}{delta:.4f}")
        else:
            deltas.append("  —   ")
    print(f"  {'Δ (A - B)':<20s}  {deltas[0]:>10s}  {deltas[1]:>10s}  {deltas[2]:>10s}")


def print_key_comparisons(df: pd.DataFrame) -> None:
    """Print the key ablation comparisons that tell the paper's story."""
    print("\n" + "=" * 80)
    print("  KEY COMPARISONS")
    print("=" * 80)

    exp_lookup = dict(zip(df["experiment"], range(len(df))))

    for cmp in KEY_COMPARISONS:
        _print_comparison(df, exp_lookup, cmp.title, cmp.baseline, cmp.variant, cmp.question)

    # DropoutNet is a separate model, not in the registry — compare it against
    # the full two-tower when its results are present.
    _print_comparison(
        df, exp_lookup, "Full model vs DropoutNet", "full_model", "dropoutnet",
        "How does the two-tower compare to the DropoutNet cold-start baseline?",
    )


def summarize(results_dir: Path, csv_path: Path | None = None,
              dropoutnet_path: Path | None = None) -> None:
    """Load results and print summary tables.

    Can be called from run_ablation.py or standalone via CLI.
    """
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    experiments = load_experiment_results(results_dir)
    if not experiments:
        print(f"No eval_results.json found in {results_dir}/*/")
        return

    print(f"Found {len(experiments)} completed experiments")

    # DropoutNet baseline lives alongside the ablation dir (results/) by default.
    if dropoutnet_path is None:
        dropoutnet_path = results_dir.parent / "dropoutnet_results.json"
    dropoutnet = load_dropoutnet(dropoutnet_path)
    if dropoutnet:
        print(f"Including DropoutNet baseline from {dropoutnet_path}")

    df = build_results_table(experiments, dropoutnet=dropoutnet)
    print_main_table(df)
    print_key_comparisons(df)

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
    # Console tables use non-ASCII glyphs (×, Δ); make stdout robust on
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