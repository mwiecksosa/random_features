"""
Run all experiments multiple times.

Outputs are saved in outputs/<timestamp>_<n>runs/run_000/, run_001/, etc.

Usage:
    # Default: 100 runs with base_seed=42
    python run_repeated_experiments.py

    # If you need to split up the computations, use different base_seed on each batch of runs
    # Example: split 1000 runs into 10 batches of 100 runs, for MYSEEDNUMBER = 2026042, 2026142, ..., 2026942, do:  
    python run_repeated_experiments.py --base-seed MYSEEDNUMBER --n-runs 100  

Use combine_repeated_experiments.py to merge results from multiple batches of runs.
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from experiments import main, OUTPUT_BASE
from experiment_configs import RUN_EXPERIMENTS

EXPERIMENT_NAMES = RUN_EXPERIMENTS


def load_estimates_from_run_dirs(parent_dir: Path) -> dict:
    """
    Load theta_hat and true_theta from experiment_results.json in each run subdir.

    Returns:
        Dict: {experiment_name: {n_key: {"theta_hats": array, "true_theta": [...], "param_names": [...]}}}
        where n_key is "100", "1000", "10000" when by_n is present, or "default" for legacy.
    """
    # estimates[name][n_key] = { theta_hats: list, true_theta, param_names }
    estimates = {name: {} for name in EXPERIMENT_NAMES}

    run_dirs = sorted(parent_dir.glob("run_*"))
    for run_dir in run_dirs:
        results_path = run_dir / "experiment_results.json"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            data = json.load(f)

        for name in EXPERIMENT_NAMES:
            res = data.get(name)
            if res is None or "error" in res:
                continue
            by_n = res.get("by_n")
            if by_n:
                for n_str, sub in by_n.items():
                    theta_hat = sub.get("theta_hat")
                    if theta_hat is not None:
                        if n_str not in estimates[name]:
                            estimates[name][n_str] = {"theta_hats": [], "true_theta": res.get("true_theta"), "param_names": res.get("param_names")}
                        estimates[name][n_str]["theta_hats"].append(theta_hat)
                        if estimates[name][n_str]["param_names"] is None:
                            estimates[name][n_str]["param_names"] = [f"θ{i}" for i in range(len(theta_hat))]
            else:
                theta_hat = res.get("theta_hat")
                if theta_hat is not None:
                    n_key = "default"
                    if n_key not in estimates[name]:
                        estimates[name][n_key] = {"theta_hats": [], "true_theta": res.get("true_theta"), "param_names": res.get("param_names")}
                    estimates[name][n_key]["theta_hats"].append(theta_hat)
                    if estimates[name][n_key]["param_names"] is None:
                        estimates[name][n_key]["param_names"] = [f"θ{i}" for i in range(len(theta_hat))]

    for name in EXPERIMENT_NAMES:
        for n_key in list(estimates[name].keys()):
            th = estimates[name][n_key]["theta_hats"]
            if th:
                estimates[name][n_key]["theta_hats"] = np.array(th)
                if estimates[name][n_key]["param_names"] is None:
                    estimates[name][n_key]["param_names"] = [f"θ{i}" for i in range(len(th[0]))]
            else:
                estimates[name][n_key]["theta_hats"] = np.array([])

    return estimates


def run_repeated_experiments(
    n_runs: int = 100,
    base_seed: int = 42,
) -> dict:
    """
    Run all experiments n_runs times with different seeds.
    
    Args:
        n_runs: Number of repeated runs.
        base_seed: Base seed for run 0. Run i uses seed = base_seed + i.
        
    Returns:
        Dictionary with aggregated results across all runs.
    """
    parent_dir = OUTPUT_BASE / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{n_runs}runs"
    parent_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running {n_runs} repeated experiment runs")
    print(f"Output directory: {parent_dir}")
    print(f"Base seed: {base_seed} (run i uses seed {base_seed}+i)")
    print("=" * 70)

    t_start = time.perf_counter()
    all_results = []

    for i in range(n_runs):
        run_seed = base_seed + i
        run_dir = parent_dir / f"run_{i:03d}"
        
        print(f"\n{'#'*70}")
        print(f"RUN {i+1}/{n_runs} (seed={run_seed})")
        print(f"{'#'*70}")
        
        try:
            results = main(seed=run_seed, output_dir=run_dir)
            all_results.append({
                "run": i,
                "seed": run_seed,
                "results": results,
                "error": None,
            })
        except Exception as e:
            print(f"Run {i} failed: {e}")
            all_results.append({
                "run": i,
                "seed": run_seed,
                "results": None,
                "error": str(e),
            })

    elapsed_seconds = time.perf_counter() - t_start
    experiment_names = EXPERIMENT_NAMES
    run_time_seconds_per_run = []
    for r in all_results:
        if r["error"] is not None or not r["results"]:
            continue
        meta = r["results"].get("_run_metadata") or {}
        t = meta.get("run_time_seconds")
        if t is not None:
            run_time_seconds_per_run.append(t)

    aggregated = {}
    for name in experiment_names:
        rmses = []
        successes = []
        times = []
        for r in all_results:
            if r["error"] is not None:
                continue
            res = r["results"].get(name)
            if res is not None and "error" not in res:
                rmses.append(res["rmse"])
                successes.append(res["success"])
                if "time_seconds" in res:
                    times.append(res["time_seconds"])
        
        agg_entry = {
            "n_successful": sum(successes),
            "n_runs": len(rmses),
            "rmse_mean": float(np.mean(rmses)) if rmses else None,
            "rmse_std": float(np.std(rmses)) if rmses else None,
            "rmse_median": float(np.median(rmses)) if rmses else None,
        }
        if times:
            agg_entry["time_seconds_mean"] = round(float(np.mean(times)), 2)
            agg_entry["time_seconds_std"] = round(float(np.std(times)), 2)
        aggregated[name] = agg_entry

    # Save aggregated summary
    summary = {
        "n_runs": n_runs,
        "base_seed": base_seed,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "elapsed_human": _format_elapsed(elapsed_seconds),
        "run_time_seconds_per_run": run_time_seconds_per_run,
        "mean_run_time_seconds": round(float(np.mean(run_time_seconds_per_run)), 2) if run_time_seconds_per_run else None,
        "aggregated": aggregated,
        "run_summary": [
            {
                "run": r["run"],
                "seed": r["seed"],
                "error": r["error"],
                "run_time_seconds": (r["results"] or {}).get("_run_metadata", {}).get("run_time_seconds"),
                "rmses": {
                    k: v["rmse"] for k, v in (r["results"] or {}).items()
                    if k != "_run_metadata" and "error" not in (v or {}) and "rmse" in (v or {})
                } if r["results"] else None,
                "time_seconds": {
                    k: v["time_seconds"] for k, v in (r["results"] or {}).items()
                    if k != "_run_metadata" and (v or {}).get("time_seconds") is not None
                } if r["results"] else None,
            }
            for r in all_results
        ],
    }
    
    summary_path = parent_dir / "aggregated_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS (all runs)")
    print(f"{'='*70}")
    for name, agg in aggregated.items():
        print(f"\n{name}:")
        print(f"  Successful runs: {agg['n_successful']}/{agg['n_runs']}")
        if agg["n_runs"] > 0:
            print(f"  RMSE mean ± std: {agg['rmse_mean']:.6f} ± {agg['rmse_std']:.6f}")
            print(f"  RMSE median: {agg['rmse_median']:.6f}")
            if "time_seconds_mean" in agg:
                print(f"  Mean time per run: {agg['time_seconds_mean']} s (± {agg['time_seconds_std']} s)")
    
    print(f"\nTotal wall time: {summary['elapsed_human']} ({summary['elapsed_seconds']:.1f} s)")
    if summary.get("mean_run_time_seconds") is not None:
        print(f"Mean time per single run: {summary['mean_run_time_seconds']} s")
    print(f"Aggregated results saved to: {summary_path}")
    return summary


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as human-readable string (e.g. '2h 15m 3s')."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full experiment suite multiple times with different seeds."
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed for run 0. Run i uses seed = base_seed + i (default: 42)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=100,
        help="Number of repeated runs (default: 100)",
    )
    args = parser.parse_args()
    run_repeated_experiments(n_runs=args.n_runs, base_seed=args.base_seed)
