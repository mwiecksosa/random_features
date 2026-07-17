"""
Combine results from multiple batches of experiments.

Pass the parent directory. The script finds all subdirs that contain run_000/,
run_001/, ... and combines them.

Usage:
    python combine_repeated_experiments.py <parent_dir>

For example:
    python combine_repeated_experiments.py outputs

Combined density plots and summary are saved to <parent_dir>/combined/
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from run_repeated_experiments import (
    EXPERIMENT_NAMES,
    load_estimates_from_run_dirs,
)


def discover_repeated_experiment_dirs(parent_dir: Path) -> list[Path]:
    """
    Find all subdirs of parent_dir that look like repeated experiments
    (contain run_*/ with experiment_results.json).
    """
    parent_dir = Path(parent_dir)
    if not parent_dir.exists() or not parent_dir.is_dir():
        return []
    found = []
    for subdir in sorted(parent_dir.iterdir()):
        if not subdir.is_dir():
            continue
        run_dirs = list(subdir.glob("run_*"))
        if not run_dirs:
            continue
        if any((rd / "experiment_results.json").exists() for rd in run_dirs):
            found.append(subdir)
    return found


def load_estimates_from_multiple_parents(parent_dirs: list[Path]) -> dict:
    """
    Load and combine theta_hat from run_* subdirs across multiple parent directories.
    Supports by_n structure: combined[name][n_key] = { theta_hats, true_theta, param_names }.

    Returns:
        Dict: {experiment_name: {n_key: {"theta_hats": array, "true_theta": [...], "param_names": [...]}}}
    """
    combined = {name: {} for name in EXPERIMENT_NAMES}

    for parent_dir in parent_dirs:
        parent_dir = Path(parent_dir)
        if not parent_dir.exists():
            print(f"Warning: skipping missing directory {parent_dir}")
            continue
        estimates = load_estimates_from_run_dirs(parent_dir)
        n_runs = 0
        for name in EXPERIMENT_NAMES:
            for n_key, est in estimates.get(name, {}).items():
                th = est["theta_hats"]
                if len(th) > 0:
                    if n_key not in combined[name]:
                        combined[name][n_key] = {"theta_hats": [], "true_theta": est.get("true_theta"), "param_names": est.get("param_names")}
                    combined[name][n_key]["theta_hats"].extend(th.tolist())
                    n_runs = max(n_runs, len(th))
                    if combined[name][n_key]["true_theta"] is None:
                        combined[name][n_key]["true_theta"] = est.get("true_theta")
                        combined[name][n_key]["param_names"] = est.get("param_names")
        if n_runs > 0:
            print(f"  Loaded runs from {parent_dir.name}")

    for name in EXPERIMENT_NAMES:
        for n_key in list(combined[name].keys()):
            th = combined[name][n_key]["theta_hats"]
            if th:
                combined[name][n_key]["theta_hats"] = np.array(th)
                if combined[name][n_key]["param_names"] is None:
                    combined[name][n_key]["param_names"] = [f"θ{i}" for i in range(len(th[0]))]
            else:
                combined[name][n_key]["theta_hats"] = np.array([])

    return combined


def _sorted_n_keys(by_n: dict) -> list:
    """Sort n_key by numeric value when possible (e.g., '10', '25', '1000')."""
    def key(k):
        try:
            return int(k)
        except ValueError:
            return -1
    return sorted(by_n.keys(), key=key)


# Display names for plot titles: "Estimates of <param> from <model name>"
EXPERIMENT_DISPLAY_NAMES = {
    "iid_gaussian": "IID Gaussian",
    "iid_gaussian_3dim_powers": "IID Gaussian 3-dim powers",
    "iid_gaussian_2dim": "IID Gaussian 2-dim",
    "ma1_gaussian": "MA(1) Gaussian",
    "ar1_gaussian": "AR(1) Gaussian",
    "moving_average": "MA(1) g-and-k",
    "autoregressive": "AR(1) g-and-k",
    "logistic_map": "Logistic Map",
    "state_space": "State-Space Model",
    "lorenz63": "Lorenz-63",
    "lorenz63_fixed_param_vary_rho": "Lorenz-63 (fixed σ=10, β=8/3, λ₁=λ₂=λ₃=1)",
    "henon_map": "Henon Map",
    "henon_map_fixed_b_sigma": "Henon Map (fixed b=0.3, σ=0.1)",
    "sir": "SIR",
    "sir_binomial_testing": "SIR Binomial Testing",
    "lotka_volterra": "Lotka-Volterra",
    "structural_timeseries": "Structural Time Series",
}


def plot_estimate_densities(output_dir: Path, estimates: dict) -> None:
    """
    For each model, one plot with up to 4 subplots per row, 2 rows max.
    "Density" label once on the left, legend (sample sizes) once on the right.
    Each subplot has different x labels and different x/y ticks.
    Overlay KDEs of parameter estimates for each sample size (different colors).
    Saves PDFs with larger fonts and thicker lines.
    """
    colors = plt.cm.tab10.colors
    inch_per_subplot = 6.5
    n_cols = 4
    n_rows_per_fig = 2
    dpi = 300
    ext = "pdf"
    lw_curve = 1.8
    lw_true = 1.5
    fontsize_label = 20
    fontsize_tick = 16
    fontsize_legend = 16
    _sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    def _fallback_param(j: int) -> str:
        return "θ" + str(j).translate(_sub)

    def _format_param(name: str) -> str:
        """Convert trailing digits to subscript (e.g. μ1 -> μ₁)."""
        if not name or len(name) < 2:
            return name
        idx = len(name)
        while idx > 0 and name[idx - 1].isdigit():
            idx -= 1
        if idx < len(name):
            return name[:idx] + name[idx:].translate(_sub)
        return name

    for exp_name in EXPERIMENT_NAMES:
        by_n = estimates.get(exp_name, {})
        if not by_n:
            continue
        n_keys_sorted = _sorted_n_keys(by_n)
        n_params = None
        param_names = None
        true_theta = None
        for n_key in n_keys_sorted:
            est = by_n[n_key]
            if len(est["theta_hats"]) > 0:
                n_params = est["theta_hats"].shape[1]
                param_names = est.get("param_names") or [_fallback_param(i) for i in range(n_params)]
                true_theta = est.get("true_theta")
                break
        if n_params is None:
            continue

        n_params_per_fig = n_cols * n_rows_per_fig
        n_figs = (n_params + n_params_per_fig - 1) // n_params_per_fig

        for fig_idx in range(n_figs):
            start_param = fig_idx * n_params_per_fig
            end_param = min(start_param + n_params_per_fig, n_params)
            n_this_fig = end_param - start_param

            n_rows = (n_this_fig + n_cols - 1) // n_cols
            n_cols_this = min(n_cols, n_this_fig)
            fig, axes = plt.subplots(n_rows, n_cols_this, figsize=(inch_per_subplot * n_cols_this, inch_per_subplot * n_rows), squeeze=False)
            axes = axes.flatten()
            for sub_idx, j in enumerate(range(start_param, end_param)):
                ax = axes[sub_idx]
                for idx, n_key in enumerate(n_keys_sorted):
                    est = by_n[n_key]
                    theta_hats = est["theta_hats"]
                    if len(theta_hats) == 0:
                        continue
                    x_j = theta_hats[:, j]
                    if len(np.unique(x_j)) < 2:
                        ax.axvline(x_j[0], color=colors[idx % len(colors)], linestyle="-", linewidth=lw_curve, label=f"n = {n_key}")
                        continue
                    kde = stats.gaussian_kde(x_j)
                    x_min, x_max = x_j.min(), x_j.max()
                    pad = 0.1 * (x_max - x_min) or 0.5
                    x_grid = np.linspace(x_min - pad, x_max + pad, 200)
                    ax.plot(x_grid, kde(x_grid), color=colors[idx % len(colors)], linewidth=lw_curve, label=f"n = {n_key}")
                cur_true_val = true_theta[j] if (true_theta is not None and j < len(true_theta)) else None
                if cur_true_val is not None:
                    ax.axvline(cur_true_val, color="black", linestyle=":", linewidth=lw_true, label="True value")
                ax.set_xlabel(_format_param(param_names[j] if param_names else _fallback_param(j)), fontsize=fontsize_label)
                ax.tick_params(axis="both", labelsize=fontsize_tick, length=5, width=1.5)
                if sub_idx % n_cols == 0:
                    ax.set_ylabel("Density", fontsize=fontsize_label)
                else:
                    ax.set_ylabel("")
                ax.grid(True, alpha=0.3)
            for idx in range(n_this_fig, len(axes)):
                axes[idx].set_visible(False)
            handles, labels = axes[0].get_legend_handles_labels()
            # 1 row: legend in top right of rightmost subplot; 2 rows: legend in top right of top-right subplot
            ax_legend = axes[n_this_fig - 1] if n_rows == 1 else axes[n_cols_this - 1]
            ax_legend.legend(handles, labels, loc="upper right", fontsize=fontsize_legend, frameon=True)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            suffix = f"params{start_param}-{end_param-1}" if n_params > 1 else "param0"
            save_path = output_dir / f"{exp_name}_estimate_densities_{suffix}.{ext}"
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {save_path}")



def aggregate_rmse_from_parents(parent_dirs: list[Path]) -> dict:
    """Load aggregated_results.json from each parent and combine RMSE stats."""
    all_rmses = {name: [] for name in EXPERIMENT_NAMES}

    for parent_dir in parent_dirs:
        parent_dir = Path(parent_dir)
        summary_path = parent_dir / "aggregated_results.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            data = json.load(f)
        for r in data.get("run_summary", []):
            rmses = r.get("rmses")
            if rmses and r.get("error") is None:
                for name in EXPERIMENT_NAMES:
                    if name in rmses:
                        all_rmses[name].append(rmses[name])

    aggregated = {}
    for name in EXPERIMENT_NAMES:
        rmses = all_rmses[name]
        if rmses:
            aggregated[name] = {
                "n_runs": len(rmses),
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "rmse_median": float(np.median(rmses)),
            }
        else:
            aggregated[name] = {"n_runs": 0}
    return aggregated


def combine_repeated_experiments(
    parent_dir: Path,
    output_dir: Path = None,
) -> dict:
    """
    Find all repeated-experiment subdirs under parent_dir, combine results, create density plots.

    Args:
        parent_dir: Top-level directory containing repeated-experiment subdirs (each with run_000/, run_001/, ...).
        output_dir: Where to save combined plots. Default: parent_dir/combined/

    Returns:
        Summary dict with combined estimates count and optional RMSE stats.
    """
    parent_dir = Path(parent_dir)
    parent_dirs = discover_repeated_experiment_dirs(parent_dir)
    if not parent_dirs:
        raise ValueError(
            f"No repeated-experiment directories found under {parent_dir}. "
            "Each subdir should contain run_000/, run_001/, ... with experiment_results.json."
        )

    if output_dir is None:
        output_dir = parent_dir / "combined"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(parent_dirs)} repeated-experiment directories under {parent_dir}")
    for d in parent_dirs:
        print(f"  - {d.name}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    estimates = load_estimates_from_multiple_parents(parent_dirs)

    n_total = {}
    for name in EXPERIMENT_NAMES:
        n_total[name] = {n_key: len(est["theta_hats"]) for n_key, est in estimates.get(name, {}).items() if len(est["theta_hats"]) > 0}
    print(f"\nTotal runs per experiment (by sample size n): {n_total}")

    plot_estimate_densities(output_dir, estimates)
    aggregated = aggregate_rmse_from_parents(parent_dirs)
    if any(agg.get("n_runs", 0) > 0 for agg in aggregated.values()):
        summary = {
            "parent_dirs": [str(p) for p in parent_dirs],
            "n_total_per_experiment": n_total,
            "aggregated_rmse": aggregated,
        }
        summary_path = output_dir / "combined_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved combined summary to {summary_path}")
        for name in EXPERIMENT_NAMES:
            agg = aggregated.get(name, {})
            if agg.get("n_runs", 0) > 0:
                print(f"  {name}: n={agg['n_runs']}, RMSE mean={agg['rmse_mean']:.6f} ± {agg['rmse_std']:.6f}")
    else:
        summary = {"parent_dirs": [str(p) for p in parent_dirs], "n_total_per_experiment": n_total}

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Combine repeated-experiment results from all subdirs of a parent directory."
    )
    parser.add_argument(
        "parent_dir",
        type=Path,
        help="Parent directory containing repeated-experiment subdirs (each with run_000/, run_001/, ...)",
    )
    args = parser.parse_args()
    combine_repeated_experiments(args.parent_dir)


if __name__ == "__main__":
    main()
