"""
Experiment configuration: sample sizes, which experiments to run,
and settings for each experiment (bounds, true_theta, simulator, etc.).

Imported by experiments.py and plot_experiments.py.
"""

import numpy as np
from typing import List, Dict, Any

from simulation_models import (
    simulate_iid_gaussian,
    simulate_iid_gaussian_2dim,
    simulate_iid_gaussian_3dim_powers,
    simulate_moving_average,
    simulate_autoregressive_gandk,
    simulate_logistic_map,
    simulate_state_space_model,
    simulate_lorenz63,
    simulate_lorenz63_fixed_param_vary_rho,
    simulate_henon_map,
    simulate_henon_map_fixed_b_sigma,
    simulate_sir,
    simulate_sir_binomial_testing,
    simulate_lotka_volterra,
    simulate_structural_timeseries,
)

SAMPLE_SIZES: List[int] = [100, 1000]

### Experiments for estimation paper
#RUN_EXPERIMENTS: List[str] = [
#    "iid_gaussian",
#    "moving_average",
#    "autoregressive",
#    "logistic_map",
#    "state_space",
#    "sir_binomial_testing",
#    "lotka_volterra",
#    "structural_timeseries",
#]

### Experiments for identification paper
RUN_EXPERIMENTS: List[str] = [
    "lorenz63",
    "henon_map",
]




EXPERIMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "iid_gaussian": {
        "name": "IID Gaussian (unknown mean, known σ)",
        "simulator": simulate_iid_gaussian,
        "true_theta": np.array([0.5]),
        "theta_init": np.array([0.0]),
        "bounds": [(-3.0, 3.0)],
        "param_names": ["μ"],
        "estimator_type": "time_average",
        "plot_prefix": "iid_gaussian",
        "n_lags": 0,
        "n_simulations": 10,
    },
    "iid_gaussian_3dim_powers": {
        "name": "IID Gaussian 3-dim powers (unknown μ, known σ)",
        "simulator": simulate_iid_gaussian_3dim_powers,
        "true_theta": np.array([0.5]),
        "theta_init": np.array([0.0]),
        "bounds": [(-2, 0)],
        "param_names": ["μ"],
        "estimator_type": "time_average",
        "plot_prefix": "iid_gaussian_3dim_powers",
        "n_lags": 1,
        "n_simulations": 10,
    },
    "iid_gaussian_2dim": {
        "name": "IID Gaussian 2-dim (unknown mean, known σ)",
        "simulator": simulate_iid_gaussian_2dim,
        "true_theta": np.array([0.5]),
        "theta_init": np.array([0.0]),
        "bounds": [(-3.0, 3.0)],
        "param_names": ["μ"],
        "estimator_type": "time_average",
        "plot_prefix": "iid_gaussian_2dim",
        "n_lags": 1,
        "n_simulations": 10,
    },
    "moving_average": {
        "name": "MA(1) with g-and-k noise",
        "simulator": simulate_moving_average,
        "true_theta": np.array([0.5, -3.0, 0.2, 0.1]),
        "theta_init": np.array([0.35, -1.0, 0.4, 0.3]),
        "bounds": [(0, 1), (-7, 0), (0, 0.8), (0, 0.5)],
        "param_names": ["ψ", "μ", "σ", "g"],
        "estimator_type": "time_average",
        "plot_prefix": "moving_average",
        "n_lags": 1,
        "n_simulations": 10,
        "fit_options": {"maxiter": 500, "maxfun": 20000},
    },
    "autoregressive": {
        "name": "AR(1) with g-and-k Noise",
        "simulator": simulate_autoregressive_gandk,
        "true_theta": np.array([0.3, 2.0, 0.3, 0.2]),
        "theta_init": np.array([0.5, 1.5, 0.35, 0.2]),
        "bounds": [(0.01, 1.99), (0, 5), (0.1, 0.8), (0, 0.5)],
        "param_names": ["ψ", "μ", "σ", "g"],
        "estimator_type": "time_average",
        "plot_prefix": "autoregressive",
        "n_lags": 1,
        "n_simulations": 10,
        "fit_options": {"maxiter": 500, "maxfun": 20000, "eps": 1e-4},
    },
    "logistic_map": {
        "name": "Logistic Map with iid Noise",
        "simulator": simulate_logistic_map,
        "true_theta": np.array([3.9, 0.1]),
        "theta_init": np.array([3.7, 0.2]),
        "bounds": [(3, 4), (0, 0.3)],
        "param_names": ["ρ", "σ"],
        "estimator_type": "time_average",
        "plot_prefix": "logistic_map",
        "n_lags": 1,
        "n_simulations": 10,
        "fit_options": {"maxiter": 200, "eps": 1e-4, "n_starts": 10},
    },
    "state_space": {
        "name": "State-Space Model (d=25)",
        "simulator": simulate_state_space_model,
        "true_theta": np.array([0.25, 0.08, 2.5, 0.05]),
        "theta_init": np.array([0.3, 0.1, 2, 0.08]),
        "bounds": [(0.01, 0.6), (0.01, 0.2), (0.01, 6), (0.01, 0.15)],
        "param_names": ["ψ", "σ", "μ", "λ"],
        "estimator_type": "time_average",
        "plot_prefix": "state_space",
        "n_lags": 1,
        "n_simulations": 10,
    },
    "lorenz63": {
        "name": "Lorenz-63 System",
        "simulator": simulate_lorenz63,
        "true_theta": np.array([10.0, 100.5, 8/3, 1, 1, 1]),
        "theta_init": np.array([9.5, 102, 2.6, 0.8, 1.2, 1.1]),
        "bounds": [
            (9, 11), (99, 103), (2.5, 2.7),
            (0.5, 1.5), (0.5, 1.5), (0.5, 1.5),
        ],
        "param_names": ["σ", "ρ", "β", "λ₁", "λ₂", "λ₃"],
        "estimator_type": "rolling_window",
        "plot_prefix": "lorenz63",
        "window_size": None,
        "n_lags": 0,
        "initial_offset": None,
        "lag_L": None,
        "n_simulations": 10,
        "fit_options": {"maxiter": 600, "eps": 1e-4, "n_starts": 10},
    },
    "lorenz63_fixed_param_vary_rho": {
        "name": "Lorenz-63 System (fixed σ=10, β=8/3, λ₁=λ₂=λ₃=1)",
        "simulator": simulate_lorenz63_fixed_param_vary_rho,
        "true_theta": np.array([100.5]),
        "theta_init": np.array([100]),
        "bounds": [
            (99, 114),
        ],
        "param_names": ["ρ"],
        "estimator_type": "rolling_window",
        "plot_prefix": "lorenz63_fixed_param_vary_rho",
        "window_size": None,
        "n_lags": 0,
        "initial_offset": None,
        "lag_L": None,
        "n_simulations": 10,
        "fit_options": {"maxiter": 600, "maxfun": 25000},
    },
    "henon_map": {
        "name": "Henon Map with iid Noise",
        "simulator": simulate_henon_map,
        "true_theta": np.array([1.4, 0.3, 0.1]),
        "theta_init": np.array([1.35, 0.28, 0.08]),
        "bounds": [
            (1.25, 1.45), (0.27, 0.33), (0.01, 0.3),
        ],
        "param_names": ["a", "b", "σ"],
        "estimator_type": "time_average",
        "plot_prefix": "henon_map",
        "n_lags": 1,
        "n_simulations": 10,
        "fit_options": {"maxiter": 600, "eps": 1e-4, "n_starts": 10},
    },
    "henon_map_fixed_b_sigma": {
        "name": "Henon Map with iid Noise (fixed b=0.3, σ=0.1)",
        "simulator": simulate_henon_map_fixed_b_sigma,
        "true_theta": np.array([1.4]),
        "theta_init": np.array([1.35]),
        "bounds": [
            (1.25, 1.45),
        ],
        "param_names": ["a"],
        "estimator_type": "time_average",
        "plot_prefix": "henon_map_fixed_b_sigma",
        "n_lags": 1,
        "n_simulations": 10,
        "fit_options": {"maxiter": 600, "eps": 1e-4, "n_starts": 10},
    },
    "sir": {
        "name": "SIR Epidemic Model",
        "simulator": simulate_sir,
        "true_theta": np.array([0.5, 0.2]),
        "theta_init": np.array([0.4, 0.1]),
        "bounds": [(0.01, 1), (0.01, 0.5)],
        "param_names": ["β", "γ"],
        "estimator_type": "rolling_window",
        "plot_prefix": "sir",
        "window_size": None,
        "n_lags": 0,
        "initial_offset": None,
        "lag_L": None,
        "n_simulations": 10,
        "fit_options": {"maxiter": 600, "maxfun": 25000},
    },
    "sir_binomial_testing": {
        "name": "SIR Epidemic Model (Binomial Testing)",
        "simulator": simulate_sir_binomial_testing,
        "true_theta": np.array([0.5, 0.2]),
        "theta_init": np.array([0.4, 0.1]),
        "bounds": [(0.01, 1), (0.01, 0.5)],
        "param_names": ["β", "γ"],
        "estimator_type": "rolling_window",
        "plot_prefix": "sir_binomial_testing",
        "window_size": None,
        "n_lags": 0,
        "initial_offset": None,
        "lag_L": None,
        "n_simulations": 10,
        "fit_options": {"maxiter": 600, "maxfun": 25000},
    },
    "lotka_volterra": {
        "name": "Lotka-Volterra Predator-Prey",
        "simulator": simulate_lotka_volterra,
        "true_theta": np.array([0.8, 0.08, 0.06, 1.2, 0.13, 0.19]),
        "theta_init": np.array([0.65, 0.12, 0.1, 1, 0.17, 0.25]),
        "bounds": [
            (0.01, 1.5), (0.01, 0.2), (0.01, 0.2), (0.1, 2),
            (0.01, 0.5), (0.01, 0.5),
        ],
        "param_names": ["α", "β", "δ", "γ", "σ₁", "σ₂"],
        "estimator_type": "rolling_window",
        "plot_prefix": "lotka_volterra",
        "window_size": None,
        "n_lags": 0,
        "initial_offset": None,
        "lag_L": None,
        "n_simulations": 10,
        "fit_options": {"maxiter": 600, "maxfun": 25000},
    },
    "structural_timeseries": {
        "name": "Structural Time Series with Change-Point",
        "simulator": simulate_structural_timeseries,
        "true_theta": np.array([
            1.8, 0.28, -1.5, 0.4, 2.8, 2.8,
            0.9, 0.9,
        ]),
        "theta_init": np.array([
            2, 0.35, -2, 0.6, 3, 2.5,
            0.5, 0.5,
        ]),
        "bounds": [
            (1, 3), (0, 0.5), (-3.5, 0), (0, 1),
            (1, 4), (1, 4),
            (0, 1), (0, 1),
        ],
        "param_names": [
            "α", "τ", "μ₁", "μ₂", "β₁", "β₂",
            "σ₁", "σ₂",
        ],
        "estimator_type": "rolling_window",
        "plot_prefix": "structural_timeseries",
        "window_size": None,
        "n_lags": 0,
        "initial_offset": None,
        "lag_L": None,
        "n_simulations": 10,
    },
}


def get_experiment_config(key: str) -> Dict[str, Any]:
    """Return a copy of the config for the given experiment key."""
    if key not in EXPERIMENT_CONFIGS:
        raise KeyError(f"Unknown experiment: {key}")
    return dict(EXPERIMENT_CONFIGS[key])
