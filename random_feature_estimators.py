"""
This module implements:
1. Random Fourier features
2. Time-average estimator 
3. Rolling-window estimator 
"""

import os
from functools import partial
import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import (
    minimize,
    differential_evolution,
    dual_annealing,
    basinhopping,
)
from numpy.random import default_rng

_OBJECTIVE_FAIL_PENALTY = 1e30


def _objective_for_optimizer(theta: np.ndarray, estimator: Any, X_obs: np.ndarray) -> float:
    """Module-level objective for optimizers. Required for pickling when differential_evolution uses workers > 1."""
    try:
        return float(estimator.objective(theta, X_obs))
    except Exception:
        return _OBJECTIVE_FAIL_PENALTY


def _run_single_start(args: Tuple) -> Tuple[Any, int]:
    """
    Worker for parallel multi-start. Must be module-level for pickling.
    args: (estimator, X_obs, theta_start, bounds, optimizer, options)
    """
    estimator, X_obs, theta_start, bounds, optimizer, options = args
    obj = partial(_objective_for_optimizer, estimator=estimator, X_obs=X_obs)
    return _run_optimizer(obj, theta_start, bounds, optimizer, options, seed=None)


def _run_optimizer_with_seed(args: Tuple) -> Tuple[Any, int]:
    """
    Worker for K parallel global optimizer runs (e.g. dual_annealing). Must be module-level for pickling.
    args: (obj, theta_init, bounds, optimizer, options, seed)
    """
    obj, theta_init, bounds, optimizer, options, seed = args
    return _run_optimizer(obj, theta_init, bounds, optimizer, options, seed=seed)


def _run_optimizer(
    obj: Callable,
    theta_init: np.ndarray,
    bounds: Optional[list],
    optimizer: str,
    options: Dict[str, Any],
    seed: Optional[int] = None,
) -> Tuple[Any, int]:
    """
    Run a single optimization. Returns (result, n_iterations).
    result has .x, .fun, .success, .message.
    """
    _BOUNDLESS = ('Nelder-Mead', 'Powell')
    if optimizer in _BOUNDLESS and bounds is not None:
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        def obj_bounded(theta):
            theta_clipped = np.clip(theta, lower, upper)
            return obj(theta_clipped)

        obj_use = obj_bounded
    else:
        obj_use = obj

    opts = {k: v for k, v in options.items() if k != 'n_starts'}

    if optimizer == 'L-BFGS-B':
        result = minimize(obj_use, theta_init, method='L-BFGS-B', bounds=bounds, options=opts)
        nit = getattr(result, 'nit', 0)
    elif optimizer == 'Nelder-Mead':
        result = minimize(obj_use, theta_init, method='Nelder-Mead', options=opts)
        nit = getattr(result, 'nit', 0)
    elif optimizer == 'Powell':
        result = minimize(obj_use, theta_init, method='Powell', options=opts)
        nit = getattr(result, 'nit', 0)
    elif optimizer == 'differential_evolution':
        if bounds is None:
            raise ValueError("bounds required for differential_evolution")
        de_opts = {k: v for k, v in opts.items() if k in ('maxiter', 'popsize', 'tol', 'disp', 'polish', 'init', 'workers', 'updating')}
        if de_opts.get('workers', 1) != 1 and 'updating' not in de_opts:
            de_opts['updating'] = 'deferred'  # required by scipy when workers != 1
        result = differential_evolution(obj_use, bounds, seed=seed, x0=theta_init, **de_opts)
        nit = getattr(result, 'nit', getattr(result, 'nfev', 0))
    elif optimizer == 'dual_annealing':
        if bounds is None:
            raise ValueError("bounds required for dual_annealing")
        da_opts = {k: v for k, v in opts.items() if k in ('maxiter', 'maxfun', 'minimizer_kwargs', 'no_local_search')}
        result = dual_annealing(obj_use, bounds, x0=theta_init, seed=seed, **da_opts)
        nit = getattr(result, 'nit', getattr(result, 'nfev', 0))
    elif optimizer == 'basinhopping':
        mkw_opts = {k: v for k, v in opts.items() if k in ('maxiter', 'disp', 'eps', 'gtol')}
        mkw = {'method': 'L-BFGS-B', 'bounds': bounds, 'options': mkw_opts} if bounds else {'method': 'L-BFGS-B', 'options': mkw_opts}
        bh_opts = {k: v for k, v in opts.items() if k in ('niter', 'T', 'stepsize', 'disp', 'niter_success')}
        if 'niter' not in bh_opts and 'maxiter' in opts:
            bh_opts['niter'] = opts['maxiter']  # map maxiter to basinhopping's niter
        result = basinhopping(obj_use, theta_init, seed=seed, minimizer_kwargs=mkw, **bh_opts)
        nit = getattr(result, 'nit', 0)
    else:
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. Options: L-BFGS-B, Nelder-Mead, Powell, "
            "differential_evolution, dual_annealing, basinhopping"
        )

    return result, nit


@dataclass
class RandomFeatures:
    f"""
    Random Fourier features.
    
    φᵢ(x) = cos(∑ⱼ₌₁ᵐ⁺¹ Ωᵢ,ⱼ · xⱼ + αᵢ)
    
    where:
    - Ωᵢ,ⱼ ~ N(0, I_d) iid
    - αᵢ ~ U(-π, π) iid
    """
    k: int  # Number of random features
    m: int  # Number of lags
    d: int  # Dimension of observations
    Omega: np.ndarray  # Shape: (k, m+1, d)
    alpha: np.ndarray  # Shape: (k,)
    
    @classmethod
    def generate(cls, k: int, m: int, d: int, seed: Optional[int] = None) -> 'RandomFeatures':
        """
        Generate k random Fourier features.
        
        Args:
            k: Number of random features (should be 2p+1 for p-dimensional parameter)
            m: Number of lags to use in the features
            d: Dimension of the time series observations
            seed: Random seed for reproducibility
            
        Returns:
            RandomFeatures object
        """
        rng = default_rng(seed)
        
        Omega = rng.standard_normal(size=(k, m + 1, d))
        alpha = rng.uniform(-np.pi, np.pi, size=k)
        
        return cls(k=k, m=m, d=d, Omega=Omega, alpha=alpha)
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate all k random features on a subsequence.
        
        Args:
            X: Array of shape (m+1, d) representing a subsequence X_{(t-m):t}
            
        Returns:
            Array of shape (k,) containing the k feature values
        """
        inner_products = np.sum(self.Omega * X[np.newaxis, :, :], axis=(1, 2))
        features = np.cos(inner_products + self.alpha)
        
        return features
    
    def compute_features_timeseries(self, X: np.ndarray) -> np.ndarray:
        """
        Compute features for all valid time points in a time series.
        
        Args:
            X: Time series of shape (n, d)
            
        Returns:
            Array of shape (n-m, k) where each row contains the k features at time t
        """
        n = len(X)
        features = np.zeros((n - self.m, self.k))
        
        for t in range(self.m, n):
            subsequence = X[t-self.m:t+1]
            features[t - self.m] = self.evaluate(subsequence)
        
        return features


class TimeAverageEstimator:
    """
    Time-average estimator for asymptotically mean stationary processes.
    
    F^obs = (1/(n-m)) ∑ₜ₌ₘ₊₁ⁿ f_t^obs
    F̄^sim(θ) = (1/(n-m)) ∑ₜ₌ₘ₊₁ⁿ f̄_t^sim(θ)
    
    where f̄_t^sim(θ) = (1/s) ∑ᵣ₌₁ˢ f_t^(r)(θ)
    
    Objective: Q̂_n^TA(θ) = ||F^obs - F̄^sim(θ)||
    """
    
    def __init__(
        self,
        simulator: Callable[[np.ndarray, int, int], np.ndarray],
        param_dim: int,
        obs_dim: int,
        n_lags: int = 1,
        n_simulations: int = 10,
        seed: Optional[int] = None,
        n_features: Optional[int] = None
    ):
        """
        Initialize time-average estimator.

        Args:
            simulator: Function that simulates data given parameters.
                       simulator(theta, n_steps, seed, n_simulations) -> (n_simulations, n_steps, obs_dim)
            param_dim: Dimension p of the parameter space
            obs_dim: Dimension d of the observations
            n_lags: Number of lags m to use in random features
            n_simulations: Number s of simulations to average over
            seed: Seed for feature generation
            n_features: Number of random features k. If None, use 2*param_dim+1.
        """
        self.simulator = simulator
        self.param_dim = param_dim
        self.obs_dim = obs_dim
        self.n_lags = n_lags
        self.n_simulations = n_simulations
        self.seed = seed

        k = (n_features if n_features is not None else 2 * param_dim + 1)
        self.random_features = RandomFeatures.generate(
            k=k, m=n_lags, d=obs_dim, seed=seed
        )
    
    def compute_observed_average(self, X_obs: np.ndarray) -> np.ndarray:
        """
        Compute observed average feature vector F^obs.
        
        Args:
            X_obs: Observed time series of shape (n, d)
            
        Returns:
            Average feature vector of shape (k,)
        """
        features = self.random_features.compute_features_timeseries(X_obs)
        return np.mean(features, axis=0)
    
    def compute_simulated_average(
        self, 
        theta: np.ndarray, 
        n_steps: int
    ) -> np.ndarray:
        """
        Compute simulated average feature vector F̄^sim(θ).
        
        Uses self.seed and self.n_simulations. The simulator is called once
        with n_simulations; it uses rng=default_rng(seed) and draws n_simulations
        times for different random trajectories.
        
        Args:
            theta: Parameter vector
            n_steps: Number of time steps to simulate
            
        Returns:
            Average feature vector of shape (k,)
        """
        sim_seed = self.seed if self.seed is not None else 0
        X_all = self.simulator(theta, n_steps, sim_seed, n_simulations=self.n_simulations)
        feature_sums = np.zeros(self.random_features.k)
        for r in range(X_all.shape[0]):
            features = self.random_features.compute_features_timeseries(X_all[r])
            feature_sums += np.mean(features, axis=0)
        return feature_sums / X_all.shape[0]
    
    def objective(
        self, 
        theta: np.ndarray, 
        X_obs: np.ndarray
    ) -> float:
        """
        Compute the time-average objective Q̂_n^TA(θ) = ||F^obs - F̄^sim(θ)||.
        
        Args:
            theta: Parameter vector
            X_obs: Observed time series
            
        Returns:
            Objective value (L2 distance)
        """
        F_obs = self.compute_observed_average(X_obs)
        F_sim = self.compute_simulated_average(theta, len(X_obs))
        
        # Compute L2 norm
        return np.sqrt(np.sum((F_obs - F_sim) ** 2))
    
    def fit(
        self,
        X_obs: np.ndarray,
        theta_init: np.ndarray,
        bounds: Optional[list] = None,
        optimizer: str = 'differential_evolution',
        options: Optional[Dict] = None,
        n_starts: int = 1,
        n_jobs: int = 1,
    ) -> Dict[str, Any]:
        """
        Fit the model by minimizing the objective function.
        
        Args:
            X_obs: Observed time series of shape (n, d)
            theta_init: Initial parameter guess
            bounds: Parameter bounds for optimization
            optimizer: One of 'L-BFGS-B', 'Nelder-Mead', 'Powell', 'differential_evolution',
                'dual_annealing', 'basinhopping'. See module docstring for descriptions.
            options: Options for the optimizer (maxiter, disp, popsize, niter, etc.)
            n_starts: Number of optimization runs from different starting points.
                First start uses theta_init; remaining starts use random points in bounds.
                Requires bounds when > 1. Best result (lowest objective) is returned.
                Ignored for global optimizers (differential_evolution, dual_annealing).
            n_jobs: Number of parallel workers for multi-start. 1 = sequential (default).
                -1 = use all CPUs. Ignored when n_starts <= 1.
            
        Returns:
            Dictionary containing optimization results (from best of n_starts runs)
        """
        if options is None:
            options = {'maxiter': 1000}
        
        obj = partial(_objective_for_optimizer, estimator=self, X_obs=X_obs)
        
        _GLOBAL_OPTIMIZERS = ('differential_evolution', 'dual_annealing')
        use_multistart = n_starts > 1 and optimizer not in _GLOBAL_OPTIMIZERS
        
        if not use_multistart:
            opts = dict(options)
            if optimizer == 'differential_evolution' and n_jobs != 1:
                # Full budget, parallelized via workers
                opts['workers'] = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
                opts.setdefault('updating', 'deferred')
                result, nit = _run_optimizer(
                    obj, theta_init, bounds, optimizer, opts, seed=self.seed
                )
            elif optimizer == 'dual_annealing' and n_jobs != 1:
                # K parallel dual_annealing runs with 1/K budget each, then take best
                K = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
                opts_k = dict(options)
                opts_k['maxfun'] = max(1, opts_k.get('maxfun', 15000) // K)
                opts_k['maxiter'] = max(1, opts_k.get('maxiter', 400) // K)
                opts_k['disp'] = False
                rng = default_rng(self.seed)
                worker_seeds = rng.integers(0, 2**31, size=K) if self.seed is not None else [None] * K
                worker_args = [
                    (obj, theta_init, bounds, optimizer, opts_k, worker_seeds[i])
                    for i in range(K)
                ]
                best_result, best_nit = None, 0
                best_fun = np.inf
                max_workers = None if n_jobs == -1 else n_jobs
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(_run_optimizer_with_seed, a) for a in worker_args]
                    for future in as_completed(futures):
                        result, nit = future.result()
                        if result.fun < best_fun:
                            best_fun = result.fun
                            best_result = result
                            best_nit = nit
                result, nit = best_result, best_nit
            else:
                result, nit = _run_optimizer(
                    obj, theta_init, bounds, optimizer, opts, seed=self.seed
                )
            return {
                'theta_hat': result.x,
                'objective_value': result.fun,
                'success': result.success,
                'message': result.message,
                'n_iterations': nit,
                'full_result': result
            }

        if bounds is None:
            raise ValueError("bounds required when n_starts > 1")
        rng = default_rng(self.seed)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        starts = [theta_init.copy()]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(lower, upper))
        opts_parallel = {**options, 'disp': False} if n_jobs != 1 else options
        worker_args = [
            (self, X_obs, ts, bounds, optimizer, opts_parallel)
            for ts in starts
        ]
        best_result = None
        best_nit = 0
        best_fun = np.inf
        total_iterations = 0
        if n_jobs == 1:
            for args in worker_args:
                result, nit = _run_single_start(args)
                total_iterations += nit
                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
                    best_nit = nit
        else:
            max_workers = None if n_jobs == -1 else n_jobs
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_run_single_start, args) for args in worker_args]
                for future in as_completed(futures):
                    result, nit = future.result()
                    total_iterations += nit
                    if result.fun < best_fun:
                        best_fun = result.fun
                        best_result = result
                        best_nit = nit
        return {
            'theta_hat': best_result.x,
            'objective_value': best_result.fun,
            'success': best_result.success,
            'message': best_result.message,
            'n_iterations': best_nit,
            'full_result': best_result,
            'n_starts': n_starts,
            'total_iterations': total_iterations
        }


class RollingWindowEstimator:
    f"""
    Rolling-window estimator for nonstationary processes.
    
    F_t^obs = (1/(w∧t)) ∑ⱼ₌(t-w)∨(m+1)^t f_j^obs
    F̄_t^sim(θ) = (1/(w∧t)) ∑ⱼ₌(t-w)∨(m+1)^t f̄_j^sim(θ)
    
    Objective: Q̂_n^RW(θ) = (1/(n-m)) ∑_[t=m+τ+L]^n [||F_[t-L]^obs - F̄_[t-L]^sim(θ)||² + K_t(θ)]
    
    where K_t(θ) = 2(F_[t-L]^obs - F̄_[t-L]^sim(θ))ᵀ([f_t^obs - f̄_t^sim(θ)] - [F_[t-L]^obs - F̄_[t-L]^sim(θ)])
    """
    
    def __init__(
        self,
        simulator: Callable[[np.ndarray, int, int], np.ndarray],
        param_dim: int,
        obs_dim: int,
        window_size: Optional[int] = None,
        n_lags: int = 1,
        initial_offset: Optional[int] = None,
        lag_L: Optional[int] = None,
        n_simulations: int = 10,
        seed: Optional[int] = None,
        n_features: Optional[int] = None
    ):
        """
        Initialize rolling-window estimator.

        Args:
            simulator: Function that simulates data given parameters
            param_dim: Dimension p of the parameter space
            obs_dim: Dimension d of the observations
            window_size: Window size w. If None, set in fit() by minimizing SSE on observed features.
            n_lags: Number of lags m to use in random features
            initial_offset: Initial time offset τ. If None, set to window size in fit().
            lag_L: Lag L for the correction term K_t. If None, set from _rolling_lag_L(n) in fit().
            n_simulations: Number s of simulations to average over
            seed: Seed for feature generation
            n_features: Number of random features k. If None, use 2*param_dim+1.
        """
        self.simulator = simulator
        self.param_dim = param_dim
        self.obs_dim = obs_dim
        self.window_size = window_size
        self.n_lags = n_lags
        self.initial_offset = initial_offset
        self.lag_L = lag_L
        self.n_simulations = n_simulations
        self.seed = seed

        k = (n_features if n_features is not None else 2 * param_dim + 1)
        self.random_features = RandomFeatures.generate(
            k=k, m=n_lags, d=obs_dim, seed=seed
        )
    
    def compute_rolling_window_features(
        self, 
        features: np.ndarray
    ) -> np.ndarray:
        """
        Compute rolling window averages of features.
        
        Args:
            features: Array of shape (n-m, k) containing features at each time
            
        Returns:
            Array of shape (n-m, k) containing rolling window averages
        """
        n_times = len(features)
        k = features.shape[1]
        rolling_features = np.zeros((n_times, k))
        
        for t in range(n_times):
            start_idx = max(0, t - self.window_size + 1)
            window = features[start_idx:t+1]
            rolling_features[t] = np.mean(window, axis=0)
        
        return rolling_features
    
    def compute_observed_rolling_features(self, X_obs: np.ndarray) -> np.ndarray:
        """
        Compute rolling-window observed features F_t^obs for all t.
        
        Args:
            X_obs: Observed time series of shape (n, d)
            
        Returns:
            Rolling window features of shape (n-m, k)
        """
        features = self.random_features.compute_features_timeseries(X_obs)
        return self.compute_rolling_window_features(features)
    
    def compute_simulated_rolling_features(
        self,
        theta: np.ndarray,
        n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute rolling-window simulated features F̄_t^sim(θ) for all t.
        
        Uses self.seed and self.n_simulations. The simulator is called once
        with n_simulations; it uses rng=default_rng(seed) and draws n_simulations
        times for different random trajectories.
        
        Args:
            theta: Parameter vector
            n_steps: Number of time steps
            
        Returns:
            Tuple of (rolling_features, instantaneous_features)
            - rolling_features: shape (n-m, k)
            - instantaneous_features: shape (n-m, k) (for K_t computation)
        """
        sim_seed = self.seed if self.seed is not None else 0
        X_all = self.simulator(theta, n_steps, sim_seed, n_simulations=self.n_simulations)
        all_features = []
        for r in range(X_all.shape[0]):
            features = self.random_features.compute_features_timeseries(X_all[r])
            all_features.append(features)
        
        # Average over simulations (instantaneous features)
        instantaneous_features = np.mean(all_features, axis=0)
        
        # Compute rolling window averages
        rolling_features = self.compute_rolling_window_features(instantaneous_features)
        return rolling_features, instantaneous_features
    
    def compute_K_term(
        self,
        F_obs_rolling: np.ndarray,
        F_sim_rolling: np.ndarray,
        f_obs_instant: np.ndarray,
        f_sim_instant: np.ndarray,
        t: int
    ) -> float:
        """
        Compute the correction term K_t(θ):
        K_t(θ) = 2(F_{t-L}^obs - F̄_{t-L}^sim(θ))ᵀ([f_t^obs - f̄_t^sim(θ)] - [F_{t-L}^obs - F̄_{t-L}^sim(θ)])

        
        Args:
            F_obs_rolling: Rolling window features for observed data
            F_sim_rolling: Rolling window features for simulated data
            f_obs_instant: Instantaneous features for observed data
            f_sim_instant: Instantaneous features for simulated data
            t: Current time index
            
        Returns:
            K_t value
        """
        # Check if t-L is valid
        t_lag = t - self.lag_L
        if t_lag < 0:
            return 0.0
        diff_lagged = F_obs_rolling[t_lag] - F_sim_rolling[t_lag]
        diff_current = f_obs_instant[t] - f_sim_instant[t]
        K_t = 2 * np.dot(diff_lagged, diff_current - diff_lagged) 
        
        return K_t
    
    def objective(
        self,
        theta: np.ndarray,
        X_obs: np.ndarray
    ) -> float:
        """
        Compute the rolling-window objective Q̂_n^RW(θ).
        
        Args:
            theta: Parameter vector
            X_obs: Observed time series
            
        Returns:
            Objective value
        """
        # Compute observed features
        f_obs_instant = self.random_features.compute_features_timeseries(X_obs)
        F_obs_rolling = self.compute_rolling_window_features(f_obs_instant)
        
        # Compute simulated features
        F_sim_rolling, f_sim_instant = self.compute_simulated_rolling_features(
            theta, len(X_obs)
        )
        
        # Compute objective
        n = len(X_obs)
        m = self.n_lags
        tau = self.initial_offset
        L = self.lag_L
        objective = 0.0
        for t in range(m + tau + L, n):
            t_lag = t - L
            
            # Main term: ||F_t^obs - F̄_t^sim(θ)||²
            diff = F_obs_rolling[t_lag] - F_sim_rolling[t_lag]
            main_term = np.sum(diff ** 2)
            
            # Correction term K_t(θ)
            K_t = self.compute_K_term(
                F_obs_rolling, F_sim_rolling,
                f_obs_instant, f_sim_instant,
                t
            )
            objective += main_term + K_t

        # Take absolute value and divide by n-m    
        objective = np.abs(objective) / (n-m)

        return objective
    
    def fit(
        self,
        X_obs: np.ndarray,
        theta_init: np.ndarray,
        bounds: Optional[list] = None,
        optimizer: str = 'differential_evolution',
        options: Optional[Dict] = None,
        n_starts: int = 1,
        n_jobs: int = 1,
    ) -> Dict[str, Any]:
        """
        Fit the model by minimizing the objective function.
        
        Args:
            X_obs: Observed time series of shape (n, d)
            theta_init: Initial parameter guess
            bounds: Parameter bounds for optimization
            optimizer: One of 'L-BFGS-B', 'Nelder-Mead', 'Powell', 'differential_evolution',
                'dual_annealing', 'basinhopping'. See module docstring for descriptions.
            options: Options for the optimizer (maxiter, disp, popsize, niter, etc.)
            n_starts: Number of optimization runs from different starting points.
                First start uses theta_init; remaining starts use random points in bounds.
                Requires bounds when > 1. Best result (lowest objective) is returned.
                Ignored for global optimizers (differential_evolution, dual_annealing).
            n_jobs: Number of parallel workers for multi-start. 1 = sequential (default).
                -1 = use all CPUs. Ignored when n_starts <= 1.
            
        Returns:
            Dictionary containing optimization results (from best of n_starts runs)
        """
        if options is None:
            options = {'maxiter': 1000}

        # Set window_size, initial_offset, lag_L from observed data if not yet set
        if self.window_size is None:
            n = len(X_obs)
            self.window_size = select_window_size_sse(X_obs, self.random_features)
            self.initial_offset = self.window_size
            self.lag_L = _rolling_lag_L(n)
        
        obj = partial(_objective_for_optimizer, estimator=self, X_obs=X_obs)
        
        _GLOBAL_OPTIMIZERS = ('differential_evolution', 'dual_annealing')
        use_multistart = n_starts > 1 and optimizer not in _GLOBAL_OPTIMIZERS
        
        if not use_multistart:
            opts = dict(options)
            if optimizer == 'differential_evolution' and n_jobs != 1:
                # Full budget, parallelized via workers 
                opts['workers'] = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
                opts.setdefault('updating', 'deferred')
                result, nit = _run_optimizer(
                    obj, theta_init, bounds, optimizer, opts, seed=self.seed
                )
            elif optimizer == 'dual_annealing' and n_jobs != 1:
                # K parallel dual_annealing runs with 1/K budget each, then take best 
                K = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
                opts_k = dict(options)
                opts_k['maxfun'] = max(1, opts_k.get('maxfun', 15000) // K)
                opts_k['maxiter'] = max(1, opts_k.get('maxiter', 400) // K)
                opts_k['disp'] = False
                rng = default_rng(self.seed)
                worker_seeds = rng.integers(0, 2**31, size=K) if self.seed is not None else [None] * K
                worker_args = [
                    (obj, theta_init, bounds, optimizer, opts_k, worker_seeds[i])
                    for i in range(K)
                ]
                best_result, best_nit = None, 0
                best_fun = np.inf
                max_workers = None if n_jobs == -1 else n_jobs
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(_run_optimizer_with_seed, a) for a in worker_args]
                    for future in as_completed(futures):
                        result, nit = future.result()
                        if result.fun < best_fun:
                            best_fun = result.fun
                            best_result = result
                            best_nit = nit
                result, nit = best_result, best_nit
            else:
                result, nit = _run_optimizer(
                    obj, theta_init, bounds, optimizer, opts, seed=self.seed
                )
            return {
                'theta_hat': result.x,
                'objective_value': result.fun,
                'success': result.success,
                'message': result.message,
                'n_iterations': nit,
                'full_result': result,
                'window_size': self.window_size,
            }
        
        if bounds is None:
            raise ValueError("bounds required when n_starts > 1")
        
        rng = default_rng(self.seed)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        starts = [theta_init.copy()]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(lower, upper))
        
        opts_parallel = {**options, 'disp': False} if n_jobs != 1 else options
        worker_args = [
            (self, X_obs, ts, bounds, optimizer, opts_parallel)
            for ts in starts
        ]
        
        best_result = None
        best_nit = 0
        best_fun = np.inf
        total_iterations = 0
        
        if n_jobs == 1:
            for args in worker_args:
                result, nit = _run_single_start(args)
                total_iterations += nit
                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
                    best_nit = nit
        else:
            max_workers = None if n_jobs == -1 else n_jobs
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_run_single_start, args) for args in worker_args]
                for future in as_completed(futures):
                    result, nit = future.result()
                    total_iterations += nit
                    if result.fun < best_fun:
                        best_fun = result.fun
                        best_result = result
                        best_nit = nit
        
        return {
            'theta_hat': best_result.x,
            'objective_value': best_result.fun,
            'success': best_result.success,
            'message': best_result.message,
            'n_iterations': best_nit,
            'full_result': best_result,
            'n_starts': n_starts,
            'total_iterations': total_iterations,
            'window_size': self.window_size,
        }


def _rolling_lag_L(n: int) -> int:
    """Lag L grows slowly with n at a polylogarithmic rate: (1/10) * log(n)^2."""
    L = int(np.ceil((np.log(n) ** 2) / 10))
    return max(1, L)


def select_window_size_sse(X_obs: np.ndarray, random_features: RandomFeatures) -> int:
    """
    Select window size w by minimizing the sum of squared distances
    Σ_{t=m+1+L}^{n} ||F_{t-L}^{obs} - f_t^{obs}||² on the observed random features, where
    F_{t-L}^{obs} is the rolling average of features with window w ending at time t-L,
    f_t^{obs} is the instantaneous feature at t, m is the number of lags, and L is the lag parameter.
    Search over w ∈ [n^{1/2}, n^{3/4}].
    """
    features = random_features.compute_features_timeseries(X_obs)  # shape (n_f, k), n_f = n - m
    n_f, k = features.shape
    n = len(X_obs)
    m = random_features.m  # number of lags in the random features
    
    if n_f == 1:
        return 1
    L = _rolling_lag_L(n)
    
    # Search range: w ∈ [n^{1/2}, n^{3/4}]
    w_min = max(1, int(np.ceil(n ** 0.5)))
    w_max = int(np.floor(n ** 0.75))
    t_start_feat = 1 + L
    t_end_feat = n_f
    if t_start_feat >= t_end_feat:
        return max(1, w_min)
    
    best_w = w_min
    best_sse = np.inf
    
    for w in range(w_min, w_max + 1):
        sse = 0.0
        for t_feat in range(t_start_feat, t_end_feat):
            t_lag_feat = t_feat - L
            if t_lag_feat < 0:
                continue  # Skip if lagged index is out of bounds
            
            # Compute rolling average F_{t-L} ending at feature index t_lag_feat
            start_j = max(0, t_lag_feat - w + 1)
            F_t_lag = np.mean(features[start_j : t_lag_feat + 1], axis=0)
            
            # f_t^{obs}: instantaneous feature at feature index t_feat
            f_t = features[t_feat]
            sse += float(np.sum((F_t_lag - f_t) ** 2))
        
        if sse < best_sse:
            best_sse = sse
            best_w = w
    
    return best_w
