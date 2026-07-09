"""Run all ablation experiments for the two-tower cold start model.

Trains each configuration to convergence (early stopping) once per seed, then
evaluates on all three test splits. Skips already-completed (experiment, seed)
pairs so it's safe to restart after interruption.

The set of experiments and seeds is defined once in ``src/experiments.py`` and
imported here, so the runner and the summarizer can never disagree on what the
study contains.

Usage:
    python scripts/run_ablation.py                       # all experiments, all SEEDS
    python scripts/run_ablation.py --seeds 42            # single-seed (quick)
    python scripts/run_ablation.py --only full_model no_temporal
    python scripts/run_ablation.py --force               # re-run, overwriting results
    python scripts/run_ablation.py --full-corpus         # + full-corpus eval (primary seed)

After completion:
    python scripts/summarize_ablation.py --results-dir results/ablation

Output structure:
    results/ablation/
      {experiment_name}/
        seed_{N}/
          checkpoints/best_model.pt
          checkpoints/final_model.pt
          eval_results.json           # sampled-negative metrics
          full_corpus_results.json    # full-corpus metrics (primary seed, --full-corpus)
          train.log
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure the project root is importable so `src` resolves when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiments import (  # noqa: E402
    EXPERIMENTS,
    EXPERIMENT_MAP,
    PRIMARY_SEED,
    SEEDS,
)


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


def child_env() -> dict:
    """Environment for subprocesses: project root on PYTHONPATH + UTF-8 I/O.

    PYTHONUTF8 forces UTF-8 stdout so non-ASCII glyphs in child output can't
    crash a run when logs are redirected on a Windows cp1252 console.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}{os.pathsep}{existing}"
    env["PYTHONUTF8"] = "1"
    return env


def run_command(cmd: list[str], log_path: Path) -> tuple[bool, float]:
    """Run a command, appending all output to a log file.

    Returns (success, elapsed_seconds).
    """
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=child_env())
    return result.returncode == 0, time.time() - t0


# ── Single (experiment, seed) run ──────────────────────────────────────────────


def run_experiment_seed(exp, seed: int, seed_dir: Path, args) -> tuple[str, float]:
    """Train + evaluate one experiment at one seed.

    Returns (status, elapsed) where status is "passed" | "failed" | "skipped".
    """
    ckpt_dir = seed_dir / "checkpoints"
    log_path = seed_dir / "train.log"
    eval_output = seed_dir / "eval_results.json"
    fc_output = seed_dir / "full_corpus_results.json"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Skip if this (experiment, seed) already completed.
    if eval_output.exists() and not args.force:
        print("    SKIP (already completed, use --force to re-run)")
        return "skipped", 0.0

    # Clear stale artifacts when re-running.
    for stale in (eval_output, fc_output, log_path):
        if stale.exists():
            stale.unlink()

    # ── Train ──────────────────────────────────────────────────────────────
    train_cmd = [
        sys.executable, "scripts/train.py",
        "--data-dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(seed),
        "--num-negatives", str(args.num_negatives),
        "--max-eval-cases", str(args.max_eval_cases),
        "--n-eval-negatives", str(args.n_eval_negatives),
        "--save-dir", str(ckpt_dir),
        "--restaurant-groups", *exp.restaurant_groups,
        "--user-groups", *exp.user_groups,
        *exp.extra_args,
    ]
    print("    Training...", end="", flush=True)
    success, elapsed = run_command(train_cmd, log_path)
    if not success:
        print(f" FAILED ({fmt_duration(elapsed)}). See {log_path}")
        return "failed", elapsed
    print(f" done ({fmt_duration(elapsed)})")
    total_elapsed = elapsed

    # ── Locate checkpoint ────────────────────────────────────────────────────
    best_ckpt = ckpt_dir / "best_model.pt"
    if not best_ckpt.exists():
        best_ckpt = ckpt_dir / "final_model.pt"
    if not best_ckpt.exists():
        print("    FAILED: no checkpoint produced")
        return "failed", total_elapsed

    # ── Evaluate (sampled negatives) ─────────────────────────────────────────
    eval_cmd = [
        sys.executable, "scripts/evaluate.py",
        "--config", args.config,
        "--checkpoint", str(best_ckpt),
        "--output", str(eval_output),
        "--n-negatives", str(args.n_eval_negatives),
    ]
    print("    Evaluating (sampled)...", end="", flush=True)
    success, elapsed = run_command(eval_cmd, log_path)
    total_elapsed += elapsed
    if not success:
        print(f" FAILED ({fmt_duration(elapsed)}). See {log_path}")
        return "failed", total_elapsed
    print(f" done ({fmt_duration(elapsed)})")

    # ── Evaluate (full corpus) — primary seed only, opt-in ───────────────────
    if args.full_corpus and seed == PRIMARY_SEED:
        fc_cmd = [
            sys.executable, "scripts/evaluate_full_corpus.py",
            "--config", args.config,
            "--checkpoint", str(best_ckpt),
            "--corpus", args.corpus,
            "--output", str(fc_output),
        ]
        print("    Evaluating (full corpus)...", end="", flush=True)
        success, elapsed = run_command(fc_cmd, log_path)
        total_elapsed += elapsed
        if success:
            print(f" done ({fmt_duration(elapsed)})")
        else:
            # Full-corpus is supplementary; a failure shouldn't fail the run.
            print(f" WARN failed ({fmt_duration(elapsed)}). See {log_path}")

    return "passed", total_elapsed


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Run ablation study experiments")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--results-dir", default="results/ablation")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS,
        help=f"Seeds to train each experiment with (default: {SEEDS})",
    )
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
        help="Re-run (experiment, seed) pairs even if results already exist",
    )
    parser.add_argument(
        "--full-corpus", action="store_true",
        help="Also run full-corpus (unsampled) eval for the primary seed",
    )
    parser.add_argument(
        "--corpus", choices=["city", "global"], default="city",
        help="Candidate pool for full-corpus eval (default: city)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = EXPERIMENTS
    if args.only:
        experiments = [EXPERIMENT_MAP[name] for name in args.only]

    seeds = args.seeds
    total = len(experiments) * len(seeds)
    passed = failed = skipped = 0
    timings: dict[str, float] = {}

    print("=" * 72)
    print(f"  Ablation Study: {len(experiments)} experiments × {len(seeds)} seeds "
          f"= {total} runs")
    print(f"  Seeds:   {seeds}")
    print(f"  Data:    {args.data_dir}")
    print(f"  Config:  {args.config}")
    print(f"  Output:  {results_dir}")
    print(f"  Epochs:  {args.epochs} (with early stopping, patience=15)")
    if args.full_corpus:
        print(f"  Full-corpus eval: on (primary seed {PRIMARY_SEED}, corpus={args.corpus})")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    run_num = 0
    for exp in experiments:
        print(f"\n{exp.name} — {exp.label}")
        print(f"  Restaurant: {exp.restaurant_groups}")
        print(f"  User:       {exp.user_groups}")
        for seed in seeds:
            run_num += 1
            seed_dir = results_dir / exp.name / f"seed_{seed}"
            print(f"  [{run_num}/{total}] seed {seed}")
            status, elapsed = run_experiment_seed(exp, seed, seed_dir, args)
            timings[f"{exp.name}/seed_{seed}"] = elapsed
            if status == "passed":
                passed += 1
            elif status == "failed":
                failed += 1
            else:
                skipped += 1

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  Ablation Study Complete")
    print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time: {fmt_duration(sum(timings.values()))}")
    print(f"  Passed: {passed}  Failed: {failed}  Skipped: {skipped}")
    print(f"{'=' * 72}")

    slowest = sorted(((t, k) for k, t in timings.items() if t > 0), reverse=True)[:10]
    if slowest:
        print("\n  Slowest runs:")
        for t, k in slowest:
            print(f"    {k:<40s}  {fmt_duration(t)}")

    # ── Results table ────────────────────────────────────────────────────────
    print("\nGenerating summary...")
    summary_cmd = [
        sys.executable, "scripts/summarize_ablation.py",
        "--results-dir", str(results_dir),
        "--csv", str(results_dir / "summary.csv"),
    ]
    if subprocess.run(summary_cmd, env=child_env()).returncode != 0:
        print("  Summary failed — run manually:")
        print(f"    python scripts/summarize_ablation.py --results-dir {results_dir}")


if __name__ == "__main__":
    main()
