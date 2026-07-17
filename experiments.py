"""
Experiments for time-average and rolling-window estimators.

For running the experiments, see run_repeated_experiments.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable

from random_feature_estimators import TimeAverageEstimator, RollingWindowEstimator
from experiment_configs import SAMPLE_SIZES, RUN_EXPERIMENTS, get_experiment_config

OUTPUT_BASE = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR = OUTPUT_BASE


def run_experiment(
    name: str,
    simulator: callable,
    true_theta: np.ndarray,
    theta_init: np.ndarray,
    bounds: list,
    n_steps_list: List[int],
    estimator_type: str,
    param_names: list,
    plot_prefix: str = None,
    seed: int = None,
    **estimator_kwargs
) -> Dict[str, Any]:
    """
    Run a single experiment for one or more sample sizes.

    Args:
        name: Experiment name
        simulator: Simulation function
        true_theta: True parameter values
        theta_init: Initial parameter guess
        bounds: Parameter bounds
        n_steps_list: List of sample sizes n to use from SAMPLE_SIZES.
        estimator_type: 'time_average' or 'rolling_window'
        param_names: Names of parameters
        plot_prefix: prefix for plot file name
        seed: Seed for this run.
        **estimator_kwargs: Additional arguments for estimator. If n_features not set, estimators use 2*param_dim+1.

    Returns:
        Dictionary with 'by_n' (results per n), 'true_theta', 'param_names',
        and top-level 'theta_hat'/'rmse' from largest n.

    Note: Simulated data inside the optimizer always uses fixed seeds so
    the same noise inputs are used for every theta. Only the parameters change the trajectory.
    """
    n_steps_list = sorted(n_steps_list)
    prefix = plot_prefix if plot_prefix else name.lower().replace(" ", "_")

    # Collect all log lines for this experiment while still printing to stdout
    log_lines: List[str] = []

    def log(msg: str):
        """Print and also store message for writing to text files/JSON."""
        print(msg)
        log_lines.append(msg)

    param_dim = len(true_theta)
    n_features = estimator_kwargs.get('n_features')
    if n_features is None:
        n_features = 2 * param_dim + 1
    n_simulations = estimator_kwargs.get('n_simulations', 10)
    n_lags = estimator_kwargs.get('n_lags', 1)
    log(f"\n{'='*70}")
    log(f"Running experiment: {name}")
    log(f"{'='*70}")
    log(f"Parameter dimension (param_dim): {param_dim}")
    log(f"Number of random features (n_features): {n_features}")
    log(f"Number of simulations (n_simulations): {n_simulations}")
    log(f"Number of lags in random features (n_lags): {n_lags}")
    log(f"True parameters: {dict(zip(param_names, true_theta))}")
    log(f"Sample sizes: {n_steps_list}")

    estimator_kwargs = {**estimator_kwargs, 'seed': seed}

    window_size_fixed = estimator_kwargs.pop('window_size', None) if estimator_type == 'rolling_window' else None
    initial_offset_fixed = estimator_kwargs.pop('initial_offset', None) if estimator_type == 'rolling_window' else None
    lag_L_fixed = estimator_kwargs.pop('lag_L', None) if estimator_type == 'rolling_window' else None
    if estimator_type == 'rolling_window' and window_size_fixed is not None:
        estimator_kwargs['window_size'] = window_size_fixed
        if initial_offset_fixed is not None:
            estimator_kwargs['initial_offset'] = initial_offset_fixed
        if lag_L_fixed is not None:
            estimator_kwargs['lag_L'] = lag_L_fixed

    fit_options = estimator_kwargs.pop('fit_options', None)
    if fit_options is None:
        fit_options = {'maxiter': 200, 'maxfun': 8000, 'n_starts': 10, 'n_jobs': 10}
    fit_options = dict(fit_options)
    n_starts = fit_options.pop('n_starts', 10)
    n_jobs = fit_options.pop('n_jobs', 10)
    # Optimizer options: 'L-BFGS-B', 'Nelder-Mead', 'Powell', 'differential_evolution', 'dual_annealing', 'basinhopping'
    optimizer = fit_options.pop('optimizer', fit_options.pop('method', 'differential_evolution'))

    # Get obs_dim from one simulation (largest n)
    n_max = max(n_steps_list)
    X_obs_max = simulator(true_theta, n_max, seed=seed)[0]
    obs_dim = X_obs_max.shape[-1]
    estimator = None
    if estimator_type == 'time_average':
        estimator = TimeAverageEstimator(
            simulator=simulator,
            param_dim=param_dim,
            obs_dim=obs_dim,
            **estimator_kwargs
        )
    elif window_size_fixed is not None:
        estimator = RollingWindowEstimator(
            simulator=simulator,
            param_dim=param_dim,
            obs_dim=obs_dim,
            **estimator_kwargs
        )

    by_n: Dict[int, Dict[str, Any]] = {}
    theta_hat_by_n: Dict[int, np.ndarray] = {}

    for n_val in n_steps_list:
        log(f"\n--- n = {n_val} ---")
        X_obs = simulator(true_theta, n_val, seed=seed)[0]

        if estimator_type == 'rolling_window' and window_size_fixed is None:
            estimator = RollingWindowEstimator(
                simulator=simulator,
                param_dim=param_dim,
                obs_dim=obs_dim,
                window_size=None,
                initial_offset=None,
                lag_L=None,
                **estimator_kwargs
            )

        log(f"Fitting (optimizer={optimizer}, n_starts={n_starts})...")
        result = estimator.fit(
            X_obs=X_obs,
            theta_init=theta_init,
            bounds=bounds,
            optimizer=optimizer,
            options=fit_options,
            n_starts=n_starts,
            n_jobs=n_jobs,
        )
        theta_hat = np.array(result['theta_hat'])
        errors = [theta_hat[i] - true_theta[i] for i in range(param_dim)]
        rmse = float(np.sqrt(np.mean(np.array(errors)**2)))
        theta_hat_by_n[n_val] = theta_hat
        by_n[n_val] = {
            'theta_hat': result['theta_hat'],
            'rmse': rmse,
            'success': result['success'],
            'n_iterations': result['n_iterations'],
            'objective_value': result['objective_value'],
            'X_obs': X_obs,
        }
        if estimator_type == 'rolling_window':
            # Save the selected window size used for this fit
            by_n[n_val]['window_size'] = int(getattr(estimator, 'window_size', 0) or 0)
        if result.get('n_starts', 1) > 1:
            by_n[n_val]['n_starts'] = result['n_starts']
            by_n[n_val]['total_iterations'] = result.get('total_iterations')
        log(f"  theta_hat = {theta_hat.tolist()}, RMSE = {rmse:.6f}")

    # Use largest n for top-level summary
    result_max = by_n[n_max]
    theta_hat = result_max['theta_hat']
    rmse = result_max['rmse']
    log(f"\n{'='*70}")
    log("RESULTS (summary)")
    log(f"{'='*70}")
    for n_val in n_steps_list:
        r = by_n[n_val]
        log(f"  n={n_val}: theta_hat={r['theta_hat']}, RMSE={r['rmse']:.6f}, success={r['success']}")

    def _to_list(x):
        """Convert array or iterable to list of Python floats for JSON."""
        return np.asarray(x).tolist()

    by_n_serializable = {}
    for n_val, r in by_n.items():
        by_n_serializable[str(n_val)] = {
            'theta_hat': _to_list(r['theta_hat']),
            'rmse': float(r['rmse']),
            'success': bool(r['success']),
            'n_iterations': int(r['n_iterations']),
            'objective_value': float(r['objective_value']),
            'X_obs': _to_list(r['X_obs']),
        }
        if estimator_type == 'rolling_window' and r.get('window_size') is not None:
            by_n_serializable[str(n_val)]['window_size'] = int(r['window_size'])
        if r.get('n_starts'):
            by_n_serializable[str(n_val)]['n_starts'] = int(r['n_starts'])
            ti = r.get('total_iterations')
            if ti is not None:
                by_n_serializable[str(n_val)]['total_iterations'] = int(ti)

    out = {
        'name': name,
        'param_names': param_names,
        'true_theta': true_theta.tolist(),
        'theta_hat': _to_list(theta_hat),
        'rmse': float(rmse),
        'success': bool(result_max['success']),
        'n_iterations': int(result_max['n_iterations']),
        'optimizer': optimizer,
        'by_n': by_n_serializable,
    }
    if result_max.get('n_starts', 1) > 1:
        out['n_starts'] = int(result_max.get('n_starts'))
        ti = result_max.get('total_iterations')
        if ti is not None:
            out['total_iterations'] = int(ti)
    # Full log of all printed output for this experiment
    out['log'] = "\n".join(log_lines)
    if estimator_type == 'rolling_window':
        out['window_size'] = int(getattr(estimator, 'window_size', 0) or 0)
    return out


def main(seed: int, output_dir: Path = None):
    """
    Run experiments. Estimates are saved to experiment_results.json.

    Args:
        seed: Seed for this run. Call from run_repeated_experiments.py with different seeds per run.
        output_dir: If provided, use this directory instead of creating a timestamped one.
    """
    global OUTPUT_DIR
    if output_dir is not None:
        run_dir = Path(output_dir)
    else:
        run_dir = OUTPUT_BASE / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR = run_dir
    print(f"\nOutput directory: {OUTPUT_DIR}")
    if seed is not None:
        print(f"Seed: {seed} (observed data + estimator)")
    print(f"Experiments to run: {RUN_EXPERIMENTS}")

    results = {}
    t_run_start = time.perf_counter()

    for name in RUN_EXPERIMENTS:
        t_start = time.perf_counter()
        try:
            config = get_experiment_config(name)
            res = run_experiment(
                seed=seed,
                n_steps_list=SAMPLE_SIZES,
                **config,
            )
            res["time_seconds"] = round(time.perf_counter() - t_start, 2)
            results[name] = res
        except KeyError:
            print(f"Unknown experiment '{name}' skipped (not in experiment_configs).")
            results[name] = {"error": f"Unknown experiment: {name}"}
        except Exception as e:
            print(f"Error in {name} experiment: {e}")
            results[name] = {"error": str(e)}

    run_time_seconds = round(time.perf_counter() - t_run_start, 2)
    results["_run_metadata"] = {"run_time_seconds": run_time_seconds, "seed": seed}

    # Save results
    results_file = OUTPUT_DIR / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    
    # Create summary
    summary = []
    summary.append("="*70)
    summary.append("EXPERIMENT SUMMARY")
    summary.append("="*70)
    summary.append(f"\nTotal run time: {run_time_seconds} s")
    
    for name, result in results.items():
        if name == "_run_metadata":
            continue
        summary.append(f"\n{name.upper().replace('_', ' ')}")
        summary.append("-" * 70)
        if 'error' in result:
            summary.append(f"  ERROR: {result['error']}")
        else:
            summary.append(f"  Success: {result['success']}")
            summary.append(f"  RMSE: {result['rmse']:.6f}")
            summary.append(f"  Iterations: {result['n_iterations']}")
            if "time_seconds" in result:
                summary.append(f"  Time: {result['time_seconds']} s")
            # Append full log of printed output for this experiment
            log_text = result.get("log")
            if log_text:
                summary.append("\n  FULL LOG:")
                summary.append(log_text)
    
    summary_text = "\n".join(summary)
    print("\n" + summary_text)
    
    # Save summary
    summary_file = OUTPUT_DIR / "experiment_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    
    print(f"\nSummary saved to: {summary_file}")
    
    return results


if __name__ == "__main__":
    print("Run experiments via: python run_repeated_experiments.py [--base-seed N] [--n-runs M]")
