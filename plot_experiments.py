"""
Generate plots for experiments: timeseries, objective sweeps, 3D feature trajectories.

Usage:
    python plot_experiments.py
    python plot_experiments.py --seed 42
    python plot_experiments.py --seed 42 --output-dir path/to/output

Plots are written to the output directory (default: plots/ in package base directory).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path
from typing import Dict, Optional, Callable, List, Any

from random_feature_estimators import RandomFeatures, TimeAverageEstimator, RollingWindowEstimator
from experiment_configs import SAMPLE_SIZES, get_experiment_config


### Experiments for RFF trajectory plots for estimation paper
#PLOT_EXPERIMENTS: List[str] = [
#    "iid_gaussian",
#    "moving_average",
#    "autoregressive",
#    "logistic_map",
#    "state_space",
#    "sir_binomial_testing",
#    "lotka_volterra",
#    "structural_timeseries",
#]

### Experiments for RFF trajectory plots for identification paper
PLOT_EXPERIMENTS: List[str] = [
    "iid_gaussian_3dim_powers",
    "lorenz63_fixed_param_vary_rho",
]
# "henon_map_fixed_b_sigma"
# Sample size used for 3D random-feature trajectory plots.


def plot_timeseries(X: np.ndarray, title: str, save_path: str) -> None:
    """Plot all dimensions of time series data to PDF."""
    n_dims = X.shape[1]
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 2.5 * n_dims), squeeze=False)
    for i in range(n_dims):
        ax = axes[i, 0]
        ax.plot(X[:, i])
        ax.set_ylabel(f'Dimension {i+1}')
        ax.grid(True, alpha=0.3)
        if i == n_dims - 1:
            ax.set_xlabel('Time')
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _get_per_feature_squared_errors_ta(estimator, theta: np.ndarray, X_obs: np.ndarray) -> np.ndarray:
    """Per-feature squared errors (F_obs - F_sim)^2 for TimeAverageEstimator. Shape (k,)."""
    F_obs = estimator.compute_observed_average(X_obs)
    F_sim = estimator.compute_simulated_average(theta, len(X_obs))
    return (F_obs - F_sim) ** 2


def _get_per_feature_squared_errors_rw(estimator, theta: np.ndarray, X_obs: np.ndarray) -> np.ndarray:
    """Time-averaged per-feature squared errors from main term for RollingWindowEstimator. Shape (k,)."""
    F_obs = estimator.compute_observed_rolling_features(X_obs)
    F_sim, _ = estimator.compute_simulated_rolling_features(theta, len(X_obs))
    n, m, tau = len(X_obs), estimator.n_lags, estimator.initial_offset
    start, end = tau, n - m
    if end <= start:
        return np.zeros(estimator.random_features.k)
    diff_sq = (F_obs[start:end] - F_sim[start:end]) ** 2
    return np.mean(diff_sq, axis=0)


def plot_objective_sweeps(
    name: str,
    estimator,
    X_obs: np.ndarray,
    true_theta: np.ndarray,
    bounds: list,
    param_names: list,
    plot_prefix: str,
    output_dir: Path,
    theta_hat: Optional[np.ndarray] = None,
    theta_hat_by_n: Optional[Dict[int, np.ndarray]] = None,
    n_grid: int = 20,
) -> None:
    """
    For each parameter, sweep that parameter (others fixed at true_theta) and plot
    total objective and per-feature squared errors. If theta_hat_by_n is provided,
    plot vertical lines for each sample size n. Saves two PNGs to output_dir.
    """
    param_dim = len(true_theta)
    is_ta = hasattr(estimator, 'compute_simulated_average')
    n_estimates = theta_hat_by_n if theta_hat_by_n is not None else {}
    colors_n = {100: 'C0', 1000: 'C1', 10000: 'C2'}

    grids = []
    objectives_total = []
    objectives_per_feature = []

    for j in range(param_dim):
        grid = np.linspace(bounds[j][0], bounds[j][1], n_grid)
        grids.append(grid)
        obj_vals = []
        feat_vals = []
        for v in grid:
            theta = true_theta.copy()
            theta[j] = v
            obj_vals.append(estimator.objective(theta, X_obs))
            if is_ta:
                feat_vals.append(_get_per_feature_squared_errors_ta(estimator, theta, X_obs))
            else:
                feat_vals.append(_get_per_feature_squared_errors_rw(estimator, theta, X_obs))
        objectives_total.append(np.array(obj_vals))
        objectives_per_feature.append(np.array(feat_vals))

    k = objectives_per_feature[0].shape[1]
    n_cols = min(4, param_dim)
    n_rows = (param_dim + n_cols - 1) // n_cols

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if param_dim == 1:
        axes1 = np.array([axes1])
    axes1 = axes1.flatten()
    for j in range(param_dim):
        ax = axes1[j]
        ax.plot(grids[j], objectives_total[j], 'b-', linewidth=2, label='Total objective')
        ax.axvline(true_theta[j], color='black', linestyle=':', linewidth=1.5, label='True value')
        if theta_hat is not None and not n_estimates:
            ax.axvline(theta_hat[j], color='orange', linestyle=':', linewidth=1.5, label='Estimated')
        for n_val in sorted(n_estimates.keys()):
            th = n_estimates[n_val]
            if th is not None and j < len(th):
                ax.axvline(th[j], color=colors_n.get(n_val, 'gray'), linestyle=':', linewidth=1.2, label=f'n = {n_val}')
        ax.set_xlabel(param_names[j])
        ax.set_ylabel('Objective')
        ax.set_title(f'Sweep Over {param_names[j]}')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(param_dim, len(axes1)):
        axes1[j].set_visible(False)
    fig1.suptitle(f'{name}: Objective vs Each Parameter (Others at True Value)', fontsize=12)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    path1 = output_dir / f"{plot_prefix}_objective.pdf"
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved objective sweep plot to {path1}")

    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if param_dim == 1:
        axes2 = np.array([axes2])
    axes2 = axes2.flatten()
    for j in range(param_dim):
        ax = axes2[j]
        for i in range(k):
            ax.plot(grids[j], objectives_per_feature[j][:, i], alpha=0.6, linewidth=0.8, label=f'Feature {i}' if k <= 10 else None)
        ax.axvline(true_theta[j], color='black', linestyle=':', linewidth=1, label='True value' if k > 10 else None)
        if theta_hat is not None and not n_estimates:
            ax.axvline(theta_hat[j], color='orange', linestyle=':', linewidth=1, label='Estimated' if k > 10 else None)
        for n_val in sorted(n_estimates.keys()):
            th = n_estimates[n_val]
            if th is not None and j < len(th):
                ax.axvline(th[j], color=colors_n.get(n_val, 'gray'), linestyle=':', linewidth=1, label=f'n = {n_val}' if k > 10 else None)
        ax.set_xlabel(param_names[j])
        ax.set_ylabel('Squared Error')
        ax.set_title(f'Per-Feature (k = {k})')
        if k <= 10:
            ax.legend(loc='best', fontsize=6)
        ax.grid(True, alpha=0.3)
    for j in range(param_dim, len(axes2)):
        axes2[j].set_visible(False)
    fig2.suptitle(f'{name}: Per-Feature Objectives vs Each Parameter', fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    path2 = output_dir / f"{plot_prefix}_objective_components.pdf"
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved objective components plot to {path2}")


def plot_feature_trajectory_3d(
    name: str,
    simulator: Callable,
    param_bounds: tuple,
    param_name: str,
    plot_prefix: str,
    output_dir: Path,
    time: int,
    n_steps: Optional[int] = None,
    n_grid: int = 1000,
    n_lags: int = 1,
    obs_dim: int = 1,
    n_simulations: int = 10,
    seed: int = 42,
) -> None:
    """
    Plot 3D trajectory of 3 random Fourier features as the single parameter θ varies.

    At each θ, features are averaged over n_simulations at the given observation time
    (same random features for all simulations).
    """
    if n_steps is None:
        raise ValueError("n_steps is required")
 
    theta_grid = np.linspace(param_bounds[0], param_bounds[1], n_grid)
    fig = plt.figure(figsize=(16, 13))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.viridis

    F_trajectory_avg = []
    theta_used = []
    
    ### same random features for all trajectories
    feat_seed = 81
    rf = RandomFeatures.generate(k=3, m=n_lags, d=obs_dim, seed=feat_seed)
    for theta_val in theta_grid:
        theta = np.array([theta_val])
        try:
            X_batch = simulator(theta, n_steps, seed=seed, n_simulations=n_simulations)
            features_by_sim = []
            for X in X_batch:
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                features_by_sim.append(rf.compute_features_timeseries(X))
            # Average over simulations at the chosen observation time.
            features_by_time = np.mean(features_by_sim, axis=0)
            F_trajectory_avg.append(features_by_time[time]) # at a particular time t
            theta_used.append(theta_val)
        except ValueError:
            continue
    if len(F_trajectory_avg) < 2:
        raise ValueError(
            f"{name}: fewer than 2 stable parameter values on sweep over {param_name}"
        )
    F_trajectory_avg = np.array(F_trajectory_avg)
    theta_used = np.array(theta_used)
    norm = plt.Normalize(vmin=theta_used.min(), vmax=theta_used.max())

    points = F_trajectory_avg.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_colors = (theta_used[:-1] + theta_used[1:]) / 2
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidths=2, alpha=0.9)
    lc.set_array(segment_colors)
    ax.add_collection3d(lc)
    sc_pts = ax.scatter(
        F_trajectory_avg[:, 0], F_trajectory_avg[:, 1], F_trajectory_avg[:, 2],
        c=theta_used, cmap=cmap, norm=norm, s=24, alpha=0.95, edgecolors='none'
    )
    # Match density plot font sizes, se larger for 3D which often displays smaller
    fontsize_label = 28
    fontsize_tick = 22
    labelpad = 24
    ax.set_xlabel(r'$\varphi_1$', fontsize=fontsize_label, labelpad=labelpad)
    ax.set_ylabel(r'$\varphi_2$', fontsize=fontsize_label, labelpad=labelpad)
    ax.set_zlabel(r'$\varphi_3$', fontsize=fontsize_label, labelpad=labelpad)
    ax.tick_params(axis='both', labelsize=fontsize_tick, pad=8)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.set_xlim(F_trajectory_avg[:, 0].min(), F_trajectory_avg[:, 0].max())
    ax.set_ylim(F_trajectory_avg[:, 1].min(), F_trajectory_avg[:, 1].max())
    ax.set_zlim(F_trajectory_avg[:, 2].min(), F_trajectory_avg[:, 2].max())
    cbar = fig.colorbar(sc_pts, ax=ax, shrink=0.6)
    cbar.set_label(param_name, rotation=0, labelpad=12, fontsize=fontsize_label)
    cbar.ax.tick_params(labelsize=fontsize_tick)
    # ax.set_title(f'{name}: Three Random Features vs {param_name} (n = {n_steps}, Mean of {n_simulations} Trajectories)')
    fig.tight_layout()
    path = output_dir / f"{plot_prefix}_feature_trajectory_3d.pdf"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 3D feature trajectory to {path}")


def run_experiment_and_plot(
    name: str,
    simulator: Callable,
    true_theta: np.ndarray,
    theta_init: np.ndarray,
    bounds: list,
    param_names: list,
    estimator_type: str,
    plot_prefix: str,
    output_dir: Path,
    seed: int,
    n_steps_list: Optional[List[int]] = None,
    **estimator_kwargs
) -> None:
    """Run one experiment (simulate + fit for each n) and save timeseries, objective-sweep, and (if 1-param) 3D plots."""
    if n_steps_list is None:
        n_steps_list = SAMPLE_SIZES
    n_steps_list = sorted(n_steps_list)
    prefix = plot_prefix if plot_prefix else name.lower().replace(" ", "_")
    param_dim = len(true_theta)
    kwargs = dict(estimator_kwargs)
    n_features = kwargs.get("n_features")
    if n_features is None:
        n_features = 2 * param_dim + 1
    n_lags = kwargs.get("n_lags", 1)
    kwargs["seed"] = seed

    window_size_fixed = kwargs.pop("window_size", None) if estimator_type == "rolling_window" else None
    initial_offset_fixed = kwargs.pop("initial_offset", None) if estimator_type == "rolling_window" else None
    lag_L_fixed = kwargs.pop("lag_L", None) if estimator_type == "rolling_window" else None
    if estimator_type == "rolling_window" and window_size_fixed is not None:
        kwargs["window_size"] = window_size_fixed
        if initial_offset_fixed is not None:
            kwargs["initial_offset"] = initial_offset_fixed
        if lag_L_fixed is not None:
            kwargs["lag_L"] = lag_L_fixed

    fit_options = kwargs.pop("fit_options", None)
    if fit_options is None:
        fit_options = {"maxiter": 200, "maxfun": 8000, "n_starts": 10, "n_jobs": 10}
    fit_options = dict(fit_options)
    n_starts = fit_options.pop("n_starts", 10)
    n_jobs = fit_options.pop("n_jobs", 10)
    # Optimizer options: 'L-BFGS-B', 'Nelder-Mead', 'Powell', 'differential_evolution', 'dual_annealing', 'basinhopping'
    optimizer = fit_options.pop("optimizer", fit_options.pop("method", "differential_evolution"))

    n_max = max(n_steps_list)
    X_obs_max = simulator(true_theta, n_max, seed=seed)[0]
    obs_dim = X_obs_max.shape[-1]

    estimator = None
    by_n: Dict[int, Dict[str, Any]] = {}
    theta_hat_by_n: Dict[int, np.ndarray] = {}
    X_obs_last = None

    for n_val in n_steps_list:
        X_obs = simulator(true_theta, n_val, seed=seed)[0]
        X_obs_last = X_obs

        if estimator_type == "rolling_window" and window_size_fixed is None:
            estimator = RollingWindowEstimator(
                simulator=simulator,
                param_dim=param_dim,
                obs_dim=obs_dim,
                window_size=None,
                initial_offset=None,
                lag_L=None,
                **kwargs
            )
        elif estimator_type == "time_average":
            if estimator is None:
                estimator = TimeAverageEstimator(
                    simulator=simulator,
                    param_dim=param_dim,
                    obs_dim=obs_dim,
                    **kwargs
                )
        elif estimator_type == "rolling_window":
            estimator = RollingWindowEstimator(
                simulator=simulator,
                param_dim=param_dim,
                obs_dim=obs_dim,
                **kwargs
            )
        ### don't need to fit, not plotting objective sweeps
        #result = estimator.fit(
        #    X_obs=X_obs,
        #    theta_init=theta_init,
        #    bounds=bounds,
        #    optimizer=optimizer,
        #    options=fit_options,
        #    n_starts=n_starts,
        #    n_jobs=n_jobs,
        #)
        #theta_hat = np.array(result["theta_hat"])
        #theta_hat_by_n[n_val] = theta_hat
        #by_n[n_val] = result

    #result_max = by_n[n_max]
    #theta_hat = result_max["theta_hat"]
    #if not isinstance(theta_hat, np.ndarray):
    #    theta_hat = np.asarray(theta_hat)

    # Save plots
    plot_timeseries(
        X_obs_last,
        title=name,
        save_path=str(output_dir / f"{prefix}_timeseries.pdf"),
    )
    #plot_objective_sweeps(
    #    name=name,
    #    estimator=estimator,
    #    X_obs=X_obs_last,
    #    true_theta=true_theta,
    #    bounds=bounds,
    #    param_names=param_names,
    #    plot_prefix=prefix,
    #    output_dir=output_dir,
    #    theta_hat=theta_hat,
    #    theta_hat_by_n=theta_hat_by_n,
    #)
    if name == "IID Gaussian 3-dim powers (unknown μ, known σ)":
        plot_feature_trajectory_3d(
            name=name,
            simulator=simulator,
            param_bounds=bounds[0],
            param_name=param_names[0],
            plot_prefix=prefix,
            output_dir=output_dir,
            time=-1,
            n_lags=getattr(estimator, "n_lags", 1),
            obs_dim=obs_dim,
            n_simulations=10,
            n_steps=3,
            seed=seed,
        )
    if name == "Lorenz-63 System (fixed σ=10, β=8/3, λ₁=λ₂=λ₃=1)":
        plot_feature_trajectory_3d(
            name=name,
            simulator=simulator,
            param_bounds=bounds[0],
            param_name=param_names[0],
            plot_prefix=prefix,
            output_dir=output_dir,
            time=-1,
            n_lags=getattr(estimator, "n_lags", 1),
            obs_dim=obs_dim,
            n_simulations=10,
            n_steps=3,
            seed=seed,
        )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the experiment suite and save plots."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for this run (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for results and plots (default: plots/ in package base directory)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in PLOT_EXPERIMENTS:
        cfg = get_experiment_config(name)
        print(f"\n{'='*70}\nRunning and plotting: {cfg['name']}\n{'='*70}")
        try:
            run_experiment_and_plot(
                output_dir=output_dir,
                seed=args.seed,
                **cfg,
            )
        except Exception as e:
            print(f"  Error: {e}")
