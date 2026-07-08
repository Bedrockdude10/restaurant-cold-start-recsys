"""Run all ablation experiments for the two-tower cold start model.

Trains each configuration to convergence (early stopping), then evaluates
on all three test splits. Skips already-completed experiments so it's safe
to restart after interruption.

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --data-dir data --epochs 50 --results-dir results/ablation
    python scripts/run_ablation.py --only full_model rest_content_only content_x_content
    python scripts/run_ablation.py --force              # re-run all, overwriting existing results
    python scripts/run_ablation.py --force --only full_model  # re-run just one

After completion:
    python scripts/summarize_ablation.py --results-dir results/ablation

Output structure:
    results/ablation/
      {experiment_name}/
        checkpoints/best_model.pt
        checkpoints/final_model.pt
        eval_results.json
        train.log
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


# ── Experiment definitions ───────────────────────────────────────────────────


@dataclass
class Experiment:
    """One ablation configuration."""
    name: str
    label: str
    restaurant_groups: list[str]
    user_groups: list[str]
    extra_args: list[str] = field(default_factory=list)


EXPERIMENTS: list[Experiment] = [
    # ── v4 ablation matrix ───────────────────────────────────────────────
    #
    # Restaurant content tower: categories, price, attributes
    # Restaurant context tower: temporal (24 hour_dist + 7 dow_dist)
    # User context tower:       day_of_week, distance
    # User content tower:       preferences, onboarding
    #
    # Fusion MLPs combine content + context on each side.
    #
    # Ablation axes:
    #   1. Restaurant context: temporal vs none
    #   2. User content: all vs preferences-only vs onboarding-only vs none
    #   3. User context: all vs none
    #   4. Cross-concern: content×content, context-heavy, etc.

    # ── Full model ───────────────────────────────────────────────────────
    Experiment(
        name="full_model",
        label="Full model (all features)",
        restaurant_groups=["attributes", "categories", "price", "temporal"],
        user_groups=["day_of_week", "distance", "onboarding", "preferences"],
    ),

    # ── Restaurant context ablations ─────────────────────────────────────
    Experiment(
        name="no_temporal",
        label="No temporal (drop restaurant checkin profile)",
        restaurant_groups=["attributes", "categories", "price"],
        user_groups=["day_of_week", "distance", "onboarding", "preferences"],
    ),

    # ── User context ablations ───────────────────────────────────────────
    Experiment(
        name="no_user_context",
        label="No user context (drop dow/distance)",
        restaurant_groups=["attributes", "categories", "price", "temporal"],
        user_groups=["onboarding", "preferences"],
    ),

    # ── User content ablations ───────────────────────────────────────────
    Experiment(
        name="no_user_content",
        label="No user content (drop preferences + onboarding)",
        restaurant_groups=["attributes", "categories", "price", "temporal"],
        user_groups=["day_of_week", "distance"],
    ),
    Experiment(
        name="no_onboarding",
        label="No onboarding (preferences only)",
        restaurant_groups=["attributes", "categories", "price", "temporal"],
        user_groups=["day_of_week", "distance", "preferences"],
    ),
    Experiment(
        name="no_preferences",
        label="No preferences (onboarding only)",
        restaurant_groups=["attributes", "categories", "price", "temporal"],
        user_groups=["day_of_week", "distance", "onboarding"],
    ),

    # ── Cross-concern ablations ──────────────────────────────────────────
    Experiment(
        name="content_x_content",
        label="Content × Content (no context either side)",
        restaurant_groups=["attributes", "categories", "price"],
        user_groups=["onboarding", "preferences"],
    ),
    Experiment(
        name="context_x_context",
        label="Context × Context (no content either side — restaurant content still required)",
        restaurant_groups=["attributes", "categories", "price", "temporal"],
        user_groups=["day_of_week", "distance"],
    ),

    # ── User checkin ablations ───────────────────────────────────────────
    Experiment(
        name="with_user_checkin",
        label="Full + user checkin time-of-visit",
        restaurant_groups=["attributes", "categories", "price", "temporal"],
        user_groups=["day_of_week", "distance", "onboarding", "preferences", "user_checkin"],
    ),
    Experiment(
        name="checkin_replaces_dow",
        label="User checkin replaces day-of-week",
        restaurant_groups=["attributes", "categories", "price", "temporal"],
        user_groups=["distance", "onboarding", "preferences", "user_checkin"],
    ),

    # ── Fusion ablations ─────────────────────────────────────────────────
    Experiment(
        name="gated_fusion",
        label="Gated fusion (full model, content-gated context mixing)",
        restaurant_groups=["attributes", "categories", "price", "temporal"],
        user_groups=["day_of_week", "distance", "onboarding", "preferences"],
        extra_args=["--fusion", "gated"],
    ),
    Experiment(
        name="gated_fusion_with_checkin",
        label="Gated fusion + user checkin time-of-visit",
        restaurant_groups=["attributes", "categories", "price", "temporal"],
        user_groups=["day_of_week", "distance", "onboarding", "preferences", "user_checkin"],
        extra_args=["--fusion", "gated"],
    ),
]

EXPERIMENT_MAP = {exp.name: exp for exp in EXPERIMENTS}


# ── Helpers ──────────────────────────────────────────────────────────────────


def fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs}s"


def run_command(cmd: list[str], log_path: Path) -> tuple[bool, float]:
    """Run a command, logging all output to a file.

    Ensures PYTHONPATH includes the project root so subprocess can
    find the src package.

    Returns:
        (success, elapsed_seconds)
    """
    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parent.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = project_root if not existing else f"{project_root}:{existing}"

    t0 = time.time()
    with open(log_path, "a") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0
    return result.returncode == 0, elapsed


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Run ablation study experiments")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--results-dir", default="results/ablation")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-negatives", type=int, default=20)
    parser.add_argument("--n-eval-negatives", type=int, default=99)
    parser.add_argument("--max-eval-cases", type=int, default=2000)
    parser.add_argument(
        "--only", nargs="+", default=None,
        choices=[e.name for e in EXPERIMENTS],
        help="Run only these experiments (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run experiments even if results already exist",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = EXPERIMENTS
    if args.only:
        experiments = [EXPERIMENT_MAP[name] for name in args.only]

    total = len(experiments)
    passed = 0
    failed = 0
    skipped = 0
    timings: dict[str, float] = {}

    print("=" * 72)
    print(f"  Ablation Study: {total} experiments")
    print(f"  Data:    {args.data_dir}")
    print(f"  Config:  {args.config}")
    print(f"  Output:  {results_dir}")
    print(f"  Epochs:  {args.epochs} (with early stopping, patience=15)")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    for i, exp in enumerate(experiments):
        run_num = i + 1
        exp_dir = results_dir / exp.name
        ckpt_dir = exp_dir / "checkpoints"
        log_path = exp_dir / "train.log"
        eval_output = exp_dir / "eval_results.json"

        print(f"\n[{run_num}/{total}] {exp.name} — {exp.label}")
        print(f"  Restaurant: {exp.restaurant_groups}")
        print(f"  User:       {exp.user_groups}")

        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already completed (unless --force)
        if eval_output.exists() and not args.force:
            print(f"  SKIP (already completed, use --force to re-run)")
            skipped += 1
            continue

        # Clear previous results when re-running
        if eval_output.exists():
            eval_output.unlink()
        if log_path.exists():
            log_path.unlink()

        # ── Train ────────────────────────────────────────────────────────
        train_cmd = [
            sys.executable, "scripts/train.py",
            "--data-dir", args.data_dir,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--num-negatives", str(args.num_negatives),
            "--max-eval-cases", str(args.max_eval_cases),
            "--n-eval-negatives", str(args.n_eval_negatives),
            "--save-dir", str(ckpt_dir),
            "--restaurant-groups", *exp.restaurant_groups,
            "--user-groups", *exp.user_groups,
            *exp.extra_args,
        ]

        print(f"  Training...", end="", flush=True)
        success, elapsed = run_command(train_cmd, log_path)

        if not success:
            print(f" FAILED ({fmt_duration(elapsed)}). See {log_path}")
            failed += 1
            timings[exp.name] = elapsed
            continue

        print(f" done ({fmt_duration(elapsed)})")
        train_elapsed = elapsed

        # ── Evaluate ─────────────────────────────────────────────────────
        best_ckpt = ckpt_dir / "best_model.pt"
        if not best_ckpt.exists():
            best_ckpt = ckpt_dir / "final_model.pt"

        if not best_ckpt.exists():
            print(f"  FAILED: No checkpoint found")
            failed += 1
            timings[exp.name] = train_elapsed
            continue

        eval_cmd = [
            sys.executable, "scripts/evaluate.py",
            "--config", args.config,
            "--checkpoint", str(best_ckpt),
            "--output", str(eval_output),
            "--n-negatives", str(args.n_eval_negatives),
        ]

        print(f"  Evaluating...", end="", flush=True)
        success, elapsed = run_command(eval_cmd, log_path)

        total_elapsed = train_elapsed + elapsed
        timings[exp.name] = total_elapsed

        if success:
            print(f" done ({fmt_duration(elapsed)}). Total: {fmt_duration(total_elapsed)}")
            passed += 1
        else:
            print(f" FAILED ({fmt_duration(elapsed)}). See {log_path}")
            failed += 1

    # ── Final summary ────────────────────────────────────────────────────
    total_time = sum(timings.values())

    print(f"\n{'=' * 72}")
    print(f"  Ablation Study Complete")
    print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time: {fmt_duration(total_time)}")
    print(f"  Passed: {passed}  Failed: {failed}  Skipped: {skipped}")
    print(f"{'=' * 72}")

    if timings:
        print(f"\n  Timing:")
        for name, t in sorted(timings.items(), key=lambda x: -x[1]):
            print(f"    {name:<30s}  {fmt_duration(t)}")

    # ── Results table ────────────────────────────────────────────────────
    print(f"\nGenerating summary...")
    summary_cmd = [
        sys.executable, "scripts/summarize_ablation.py",
        "--results-dir", str(results_dir),
        "--csv", str(results_dir / "summary.csv"),
    ]
    result = subprocess.run(summary_cmd)
    if result.returncode != 0:
        print(f"  Summary failed — run manually:")
        print(f"    python scripts/summarize_ablation.py --results-dir {results_dir}")


if __name__ == "__main__":
    main()