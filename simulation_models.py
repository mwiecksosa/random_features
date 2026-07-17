"""
Simulation models for the experiments.

This module implements all the data generating processes used in the experiments:
1. IID Gaussian (unknown mean, sd=1)
2. IID Gaussian 2-dimensional (unknown same mean, sd=1)
3. IID Gaussian 3-dimensional (means μ, μ², μ³, sd=1)
4. MA(1) with Gaussian innovations
5. AR(1) with Gaussian innovations
6. MA(1) with g-and-k innovations
7. AR(1) with g-and-k innovations
8. Logistic map model
9. State-space model
10. Lorenz-63 model
11. Lorenz-63 model with fixed σ=10, β=8/3, λ₁=λ₂=λ₃=1
12. Henon map model
13. Henon map model with fixed b=0.3, σ=0.1
14. SIR model
15. SIR model with binomial testing
16. Lotka-Volterra model
17. Structural time series model
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline
from typing import Tuple, Callable, Optional
from numpy.random import default_rng


def simulate_iid_gaussian(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    sigma: float = 1,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate iid Gaussian observations with unknown mean and known sd.

    Model: X_t = μ + σ ε_t, ε_t ~ N(0, 1) iid.

    Args:
        theta: [μ] — unknown mean
        n_steps: Number of time steps
        seed: Random seed
        sigma: Known standard deviation
        n_simulations: Number of trajectories.

    Returns:
        Array of shape (n_simulations, n_steps, 1)
    """
    (mu,) = theta
    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        x = mu + sigma * rng.standard_normal(n_steps)
        X_list.append(x.reshape(-1, 1))
    return np.stack(X_list)


def simulate_iid_gaussian_2dim(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    sigma: float = 1,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate bivariate iid Gaussian observations with unknown mean and known sd.

    Model: X_t^(j) = μ + σ ε_t^(j), ε_t^(j) ~ N(0, 1) iid for j = 1, 2.

    Args:
        theta: [μ] — unknown mean (shared across both dimensions)
        n_steps: Number of time steps
        seed: Random seed
        sigma: Known standard deviation (shared across dimensions)
        n_simulations: Number of trajectories.

    Returns:
        Array of shape (n_simulations, n_steps, 2)
    """
    (mu,) = theta
    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        epsilon = rng.standard_normal((n_steps, 2))
        X_list.append(mu + sigma * epsilon)
    return np.stack(X_list)


def simulate_iid_gaussian_3dim_powers(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    sigma: float = 1,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate trivariate iid Gaussian with power means and known sd.

    Model: X_t^(j) = μ^j + σ ε_t^(j), ε_t^(j) ~ N(0, 1) iid for j = 1, 2, 3.

    Args:
        theta: [μ] — unknown base parameter (dimension j has mean μ^j)
        n_steps: Number of time steps
        seed: Random seed
        sigma: Known standard deviation (shared across dimensions)
        n_simulations: Number of trajectories.

    Returns:
        Array of shape (n_simulations, n_steps, 3)
    """
    (mu,) = theta
    means = np.array([mu, mu ** 2, mu ** 3], dtype=float)
    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        epsilon = rng.standard_normal((n_steps, 3))
        X_list.append(means + sigma * epsilon)
    return np.stack(X_list)


def simulate_ma1_gaussian(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    sigma: float = 0.3,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate MA(1) process with Gaussian innovations (known mean 0, known sd).

    Model: X_t = ψ ε_{t-1} + ε_t, ε_t ~ N(0, σ²) iid with σ fixed.

    Args:
        theta: [ψ] — MA coefficient only (innovation sd fixed)
        n_steps: Number of time steps
        seed: Random seed
        sigma: Innovation standard deviation
        n_simulations: Number of trajectories.

    Returns:
        Array of shape (n_simulations, n_steps, 1)
    """
    (psi,) = theta
    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        n_total = n_steps + 1
        epsilon = sigma * rng.standard_normal(n_total)
        X = np.zeros(n_steps)
        for t in range(n_steps):
            X[t] = psi * epsilon[t] + epsilon[t + 1]
        X_list.append(X.reshape(-1, 1))
    return np.stack(X_list)


def simulate_ar1_gaussian(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    burn_in: int = 100,
    sigma: float = 0.3,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate AR(1) process with Gaussian innovations (known mean 0, known sd).

    Model: X_t = φ X_{t-1} + ε_t, ε_t ~ N(0, σ²) iid with σ fixed.

    Args:
        theta: [φ] — AR coefficient only (innovation sd fixed)
        n_steps: Number of time steps
        seed: Random seed
        burn_in: Burn-in length (discarded)
        sigma: Innovation standard deviation
        n_simulations: Number of trajectories.

    Returns:
        Array of shape (n_simulations, n_steps, 1)
    """
    (phi,) = theta
    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        n_total = n_steps + burn_in
        epsilon = sigma * rng.standard_normal(n_total)
        X_full = np.zeros(n_total)
        X_full[0] = epsilon[0]
        for t in range(1, n_total):
            X_full[t] = phi * X_full[t - 1] + epsilon[t]
        X_list.append(X_full[burn_in:].reshape(-1, 1))
    return np.stack(X_list)


def simulate_moving_average(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    lag_order: int = 1,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate MA(q) process with g-and-k distributed noise.

    Model:
        X_t = μ + ∑_{i=1}^q ψ^i ε_{t-i} + ε_t

    where ε_t ~ g-and-k(0, σ, g, k=0.1) distribution (k fixed).

    Args:
        theta: [ψ, μ, σ, g] where:
            - ψ: MA coefficient
            - μ: location parameter
            - σ: scale parameter
            - g: skewness parameter
        n_steps: Number of time steps
        seed: Random seed
        n_simulations: Number of trajectories.

    Returns:
        Array of shape (n_simulations, n_steps, 1)
    """
    psi, mu, sigma, g = theta
    k = 0.1  # Fix the kurtosis parameter
    c = 0.8  # Standard value for c parameter
    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        n_total = n_steps + lag_order
        z = rng.standard_normal(n_total)
        epsilon = sigma * (1 + c * np.tanh(g * z / 2)) * (1 + z**2)**k * z
        X = np.zeros(n_steps)
        for t in range(n_steps):
            X[t] = mu + epsilon[t + lag_order]
            for i in range(1, lag_order + 1):
                X[t] += psi**i * epsilon[t + lag_order - i]
        X_list.append(X.reshape(-1, 1))
    return np.stack(X_list)


def simulate_autoregressive_gandk(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    initial_value: float = 3.0,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate AR process with g-and-k distributed noise (mean-reverting OU-type process).

    Model:
        X_t = X_{t-1} + ψ(μ - X_{t-1}) + ε_t

    where ε_t ~ g-and-k(0, σ, g, k=0.2) iid (k fixed).

    Args:
        theta: [ψ, μ, σ, g] where:
            - ψ: mean reversion speed
            - μ: long-term mean
            - σ: scale parameter of g-and-k innovations
            - g: skewness parameter
        n_steps: Number of time steps
        seed: Random seed
        n_simulations: Number of trajectories.

    Returns:
        Array of shape (n_simulations, n_steps, 1)
    """
    psi, mu, sigma, g = theta
    k = 0.2  # Fix the kurtosis parameter
    c = 0.8  # Standard value for c parameter
    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        z = rng.standard_normal(n_steps)
        epsilon = sigma * (1 + c * np.tanh(g * z / 2)) * (1 + z**2)**k * z
        X = np.zeros(n_steps)
        X[0] = initial_value
        for t in range(1, n_steps):
            X[t] = X[t - 1] + psi * (mu - X[t - 1]) + epsilon[t]
        X_list.append(X.reshape(-1, 1))
    return np.stack(X_list)


def simulate_logistic_map(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    initial_z: Optional[float] = None,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate logistic map observed through iid noise.

    Model:
        Z_t = ρ Z_{t-1} (1 - Z_{t-1})
        X_t = Z_t + σ ε_t

    where ε_t ~ N(0, 1) iid.

    Args:
        theta: [ρ, σ] where:
            - ρ: logistic map parameter
            - σ: noise scale
        n_steps: Number of time steps
        seed: Random seed
        initial_z: Initial value for logistic map (if None, sample from U[0,1])
        n_simulations: Number of trajectories.

    Returns:
        Array of shape (n_simulations, n_steps, 1)
    """
    rho, sigma = theta
    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        Z = rng.uniform(0, 1) if initial_z is None else initial_z
        epsilon = rng.standard_normal(n_steps)
        X = np.zeros(n_steps)
        for t in range(n_steps):
            X[t] = Z + sigma * epsilon[t]
            Z = rho * Z * (1 - Z)
        X_list.append(X.reshape(-1, 1))
    return np.stack(X_list)




def simulate_state_space_model(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    obs_dim: int = 25,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate high-dimensional state-space model.

    Model:
        Z_t = Z_{t-1} + ψ(μ - Z_{t-1}) + σ ε_t
        X_t^(i) = Z_t + λ ξ_t^(i)  for i = 1, ..., d

    where ε_t, ξ_t^(i) ~ N(0, 1) independently.

    Args:
        theta: [ψ, σ, μ, λ] where:
            - ψ: mean reversion speed
            - σ: state noise scale
            - μ: long-term mean
            - λ: observation noise scale
        n_steps: Number of time steps
        seed: Random seed
        obs_dim: Observation dimension d (default 25)
        n_simulations: Number of trajectories.

    Returns:
        Array of shape (n_simulations, n_steps, obs_dim)
    """
    psi, sigma, mu, lam = theta
    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        burn_in = 100
        n_total = n_steps + burn_in
        epsilon = rng.standard_normal(n_total)
        Z = np.zeros(n_total)
        Z[0] = mu
        for t in range(1, n_total):
            Z[t] = Z[t-1] + psi * (mu - Z[t-1]) + sigma * epsilon[t]
        Z = Z[burn_in:]
        X = np.zeros((n_steps, obs_dim))
        for t in range(n_steps):
            xi = rng.standard_normal(obs_dim)
            X[t] = Z[t] + lam * xi
        X_list.append(X)
    return np.stack(X_list)









def simulate_lorenz63(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    T: float = 1.0,
    Z10: float = 1.0,
    Z20: float = 1.0,
    Z30: float = 1.0,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate Lorenz-63 system observed through iid noise.

    Model (ODEs in physical time v ∈ [0, T]):
        dZ^(1)/dv = σ(Z^(2) - Z^(1))
        dZ^(2)/dv = Z^(1)(ρ - Z^(3)) - Z^(2)
        dZ^(3)/dv = Z^(1)Z^(2) - βZ^(3)

    Observations at times T*t/n for t=0,...,n-1 (iid additive noise):
        X_t^(j) = Z^(j)(T*t/n) + λ^(j) ε_t^(j)

    where ε_t^(j) ~ N(0, 1) iid. Rescaled time u = v/T ∈ [0, 1].
    Initial conditions are known.

    Args:
        theta: [σ, ρ, β, λ₁, λ₂, λ₃]
        n_steps: Number of time steps
        seed: Random seed
        T: Physical time horizon
        Z10, Z20, Z30: Known initial state (default 1, 1, 1)
        n_simulations: Number of trajectories. 
            ODE is solved once; different noise drawn for each trajectory.

    Returns:
        Array of shape (n_simulations, n_steps, 3)
    """
    sigma, rho, beta = theta[0:3]
    lambda_ = theta[3:6]
    z0 = np.array([Z10, Z20, Z30], dtype=float)

    def lorenz_derivatives(v, z):
        dz1 = sigma * (z[1] - z[0])
        dz2 = z[0] * (rho - z[2]) - z[1]
        dz3 = z[0] * z[1] - beta * z[2]
        return [dz1, dz2, dz3]

    t_eval = T * np.arange(n_steps) / n_steps
    sol = solve_ivp(
        lorenz_derivatives,
        (0, T),
        z0,
        t_eval=t_eval,
        method='RK45',
        dense_output=True
    )
    Z = sol.y.T  # Shape: (n_steps, 3)

    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        epsilon = rng.standard_normal((n_steps, 3))
        X_list.append(Z + lambda_ * epsilon)
    return np.stack(X_list)


def simulate_lorenz63_fixed_param_vary_rho(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    T: float = 1.0,
    Z10: float = 1.0,
    Z20: float = 1.0,
    Z30: float = 1.0,
    sigma: float = 10.0,
    beta: float = 8 / 3,
    lambda_: Tuple[float, float, float] = (1, 1, 1),
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate Lorenz-63 with ρ as the only unknown parameter.

    Same model as simulate_lorenz63; σ, β, and observation noise scales λ₁, λ₂, λ₃
    are fixed. Initial conditions are known.

    Args:
        theta: [ρ] — Rayleigh number
        n_steps: Number of time steps
        seed: Random seed
        T: Physical time horizon
        Z10, Z20, Z30: Known initial state (default 1, 1, 1)
        sigma: Fixed Prandtl-related parameter (default 10.0)
        beta: Fixed geometric parameter (default 8/3)
        lambda_: Fixed observation noise scales (λ₁, λ₂, λ₃) (default 1, 1, 1)
        n_simulations: Number of trajectories

    Returns:
        Array of shape (n_simulations, n_steps, 3)
    """
    (rho,) = theta
    full_theta = np.array([sigma, rho, beta, *lambda_], dtype=float)
    return simulate_lorenz63(
        full_theta,
        n_steps,
        seed,
        T=T,
        Z10=Z10,
        Z20=Z20,
        Z30=Z30,
        n_simulations=n_simulations,
    )

### to safely handle parameter values for the henon map that lead to divergence 
_MAX_HENON_STATE = 10.0


def _henon_step(a: float, b: float, z1: float, z2: float) -> Tuple[float, float]:
    """One Henon update; abort before z1**2 overflows if the orbit escapes."""
    if not np.isfinite(z1) or not np.isfinite(z2):
        raise ValueError("Henon map: non-finite state")
    if abs(z1) > _MAX_HENON_STATE or abs(z2) > _MAX_HENON_STATE:
        raise ValueError("Henon map: state magnitude exceeded bound")
    z1_new = 1.0 - a * z1 * z1 + z2
    z2_new = b * z1
    if not np.isfinite(z1_new) or not np.isfinite(z2_new):
        raise ValueError("Henon map: non-finite state update")
    return z1_new, z2_new


def simulate_henon_map(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    Z10: Optional[float] = None,
    Z20: Optional[float] = None,
    burn_in: int = 100,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate Henon map observed through iid noise.

    Model:
        Z^(1)_t = 1 - a (Z^(1)_{t-1})² + Z^(2)_{t-1}
        Z^(2)_t = b Z^(1)_{t-1}
        X_t = Z^(1)_t + σ ε_t

    where ε_t ~ N(0, 1) iid.

    Args:
        theta: [a, b, σ] where:
            - a, b: Henon map parameters
            - σ: observation noise scale
        n_steps: Number of time steps
        seed: Random seed
        burn_in: Burn-in length (discarded)
        n_simulations: Number of trajectories.
        Z10: Initial value for Z^(1) (if None, sample from U[-0.7, 0.7])
        Z20: Initial value for Z^(2) (if None, sample from U[-0.2, 0.2])
    Returns:
        Array of shape (n_simulations, n_steps, 1)
    """
    a, b, sigma = theta[0:3]

    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        z1 = rng.uniform(-0.7, 0.7) if Z10 is None else Z10
        z2 = rng.uniform(-0.2, 0.2) if Z20 is None else Z20
        for _ in range(burn_in):
            z1, z2 = _henon_step(a, b, z1, z2)
        epsilon = rng.standard_normal(n_steps)
        X = np.zeros(n_steps)
        for t in range(n_steps):
            X[t] = z1 + sigma * epsilon[t]
            z1, z2 = _henon_step(a, b, z1, z2)
        X_list.append(X.reshape(-1, 1))
    return np.stack(X_list)





def simulate_henon_map_fixed_b_sigma(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    Z10: Optional[float] = None,
    Z20: Optional[float] = None,
    burn_in: int = 100,
    b: float = 0.3,
    sigma: float = 0.1,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate Henon map observed through iid noise.

    Model:
        Z^(1)_t = 1 - a (Z^(1)_{t-1})² + Z^(2)_{t-1}
        Z^(2)_t = b Z^(1)_{t-1}
        X_t = Z^(1)_t + σ ε_t

    where ε_t ~ N(0, 1) iid.

    Args:
        theta: [a] where:
            - a: Henon map parameter
            - b: Fixed damping parameter (default 0.3)
            - σ: Fixed observation noise scale (default 0.1)
        n_steps: Number of time steps
        seed: Random seed
        burn_in: Burn-in length (discarded)
        n_simulations: Number of trajectories.
        Z10: Initial value for Z^(1) (if None, sample from U[-0.7, 0.7])
        Z20: Initial value for Z^(2) (if None, sample from U[-0.2, 0.2])
    Returns:
        Array of shape (n_simulations, n_steps, 1)
    """
    a = theta[0]

    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        z1 = rng.uniform(-0.7, 0.7) if Z10 is None else Z10
        z2 = rng.uniform(-0.2, 0.2) if Z20 is None else Z20
        for _ in range(burn_in):
            z1, z2 = _henon_step(a, b, z1, z2)
        epsilon = rng.standard_normal(n_steps)
        X = np.zeros(n_steps)
        for t in range(n_steps):
            X[t] = z1 + sigma * epsilon[t]
            z1, z2 = _henon_step(a, b, z1, z2)
        X_list.append(X.reshape(-1, 1))
    return np.stack(X_list)


def simulate_sir(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    population: int = 1_000_000,
    T: float = 50.0,
    S0: float = 0.98,
    I0: float = 0.02,
    R0: float = 0.0,
    rho: float = 0.2,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate SIR epidemic model with Poisson observations.

    Model (ODEs in physical time v ∈ [0, T]):
        dS/dv = -(β/N) S I
        dI/dv = (β/N) S I - γ I
        dR/dv = γ I

    Observations at times T*t/n for t=0,...,n-1:
        X_t ~ Poisson(ρ I(T*t/n))

    ρ (reporting rate) and initial conditions (S₀, I₀, R₀) are known.

    Args:
        theta: [β, γ] where:
            - β: transmission rate (per day per infected)
            - γ: recovery rate (per day)
        n_steps: Number of time steps
        seed: Random seed
        population: Total population N
        T: Physical time horizon (days)
        S0, I0, R0: Known initial proportions (default 0.98, 0.02, 0.0; should sum to 1)
        rho: Known reporting rate (default 0.2)
        n_simulations: Number of trajectories. 
            ODE is solved once; different Poisson draws for each trajectory.

    Returns:
        Array of shape (n_simulations, n_steps, 1)
    """
    beta, gamma = theta[0:2]
    y0 = [S0 * population, I0 * population, R0 * population]

    def sir_derivatives(v, y):
        S, I, R = y
        N = population
        dS = -(beta / N) * S * I
        dI = (beta / N) * S * I - gamma * I
        dR = gamma * I
        return [dS, dI, dR]

    t_eval = T * np.arange(n_steps) / n_steps
    sol = solve_ivp(
        sir_derivatives,
        (0, T),
        y0,
        t_eval=t_eval,
        method='RK45',
        dense_output=True
    )
    I_values = sol.y[1]  # Infected compartment

    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        X = np.zeros(n_steps)
        for t in range(n_steps):
            rate = max(0, rho * I_values[t])
            X[t] = rng.poisson(rate)
        X_list.append(X.reshape(-1, 1))
    return np.stack(X_list)


def simulate_sir_binomial_testing(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    population: int = 1_000_000,
    T: float = 50.0,
    S0: float = 0.98,
    I0: float = 0.02,
    R0: float = 0.0,
    sensitivity: float = 0.95,
    specificity: float = 0.99,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate SIR epidemic model with binomial observation scheme.

    Same SIR dynamics as simulate_sir. At each observation time t:
        n_tests = N / 100
        prob_pos_t = sensitivity * (I[t]/N) + (1 - specificity) * (1 - I[t]/N)
        Y_t ~ Binomial(n_tests, prob_pos_t)
        X_t = Y_t / n_tests

    sensitivity and specificity are known (default 0.95, 0.99).

    Args:
        theta: [β, γ] — same as simulate_sir
        n_steps: Number of time steps
        seed: Random seed
        population: Total population N
        T: Physical time horizon (days)
        S0, I0, R0: Known initial proportions (default 0.98, 0.02, 0.0)
        sensitivity: Test sensitivity (default 0.95)
        specificity: Test specificity (default 0.99)
        n_simulations: Number of trajectories.

    Returns:
        Array of shape (n_simulations, n_steps, 1)
    """
    beta, gamma = theta[0:2]
    y0 = [S0 * population, I0 * population, R0 * population]
    N = population
    n_tests = N // 100

    def sir_derivatives(v, y):
        S, I, R = y
        dS = -(beta / N) * S * I
        dI = (beta / N) * S * I - gamma * I
        dR = gamma * I
        return [dS, dI, dR]

    t_eval = T * np.arange(n_steps) / n_steps
    sol = solve_ivp(
        sir_derivatives,
        (0, T),
        y0,
        t_eval=t_eval,
        method='RK45',
        dense_output=True
    )
    I_values = sol.y[1]  # Infected compartment

    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        X = np.zeros(n_steps)
        for t in range(n_steps):
            i_prop = I_values[t] / N # proportion of infected individuals
            prob_pos_t = sensitivity * i_prop + (1 - specificity) * (1 - i_prop)
            prob_pos_t = np.clip(prob_pos_t, 0.0, 1.0)
            Y_t = rng.binomial(n_tests, prob_pos_t)
            X[t] = Y_t / n_tests
        X_list.append(X.reshape(-1, 1))
    return np.stack(X_list)


def simulate_lotka_volterra(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    T: float = 20.0,
    Z10: float = 21.0,
    Z20: float = 9.0,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate Lotka-Volterra predator-prey model with iid additive noise.

    Model (ODEs in physical time v ∈ [0, T]):
        dZ^(1)/dv = α Z^(1) - β Z^(1) Z^(2)
        dZ^(2)/dv = δ Z^(1) Z^(2) - γ Z^(2)

    Observations at times T*t/n for t=0,...,n-1 (iid additive noise):
        X_t^(j) = Z^(j)(T*t/n) + σ^(j) ε_t^(j)

    Initial conditions are known.

    Args:
        theta: [α, β, δ, γ, σ₁, σ₂] where:
            - α: prey growth rate
            - β: predation rate
            - δ: predator growth rate from predation
            - γ: predator death rate
            - σ₁, σ₂: noise std devs
        n_steps: Number of time steps
        seed: Random seed
        T: Physical time horizon
        Z10, Z20: Known initial prey and predator populations (default 21, 9)
        n_simulations: Number of trajectories. 
            ODE is solved once; different noise drawn for each trajectory.

    Returns:
        Array of shape (n_simulations, n_steps, 2)
    """
    alpha, beta, delta, gamma = theta[0:4]
    sigma = np.asarray(theta[4:6], dtype=float)
    z0 = np.array([Z10, Z20], dtype=float)

    if not np.all(np.isfinite(theta)):
        raise ValueError("Lotka-Volterra: theta must be finite")
    if not np.all(np.isfinite(z0)) or np.any(z0 <= 0):
        raise ValueError("Lotka-Volterra: initial states must be finite and strictly positive")
    if not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError("Lotka-Volterra: sigma must be finite and non-negative")

    def lotka_volterra_derivatives(v, z):
        z1, z2 = z
        dz1 = alpha * z1 - beta * z1 * z2
        dz2 = delta * z1 * z2 - gamma * z2
        return [dz1, dz2]

    t_eval = T * np.arange(n_steps) / n_steps
    sol = solve_ivp(
        lotka_volterra_derivatives,
        (0, T),
        z0,
        t_eval=t_eval,
        method="RK45",
        dense_output=True,
    )

    if not sol.success:
        raise ValueError("Lotka-Volterra: ODE integration failed")

    Z = sol.y.T
    if Z.shape[0] != n_steps or Z.shape[1] != 2:
        raise ValueError("Lotka-Volterra: integration returned wrong shape")
    if not np.all(np.isfinite(Z)):
        raise ValueError("Lotka-Volterra: integration produced non-finite state")

    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        epsilon = rng.standard_normal((n_steps, 2))
        X_list.append(Z + sigma * epsilon)
    return np.stack(X_list)


def simulate_structural_timeseries(
    theta: np.ndarray,
    n_steps: int,
    seed: int,
    n_simulations: int = 10
) -> np.ndarray:
    """
    Simulate structural time series with trends, cycles, and change-point.

    Model:
        Z^(1)(u) = μ₁ u + α 1_{u≥τ} + cos(2π β₁ u)
        Z^(2)(u) = μ₂ u - α 1_{u≥τ} + sin(2π β₂ u)

        X_t^(j) = Z^(j)(t/n) + σ^(j) ε_t^(j),  ε_t^(j) ~ N(0, 1) iid

    Args:
        theta: [α, τ, μ₁, μ₂, β₁, β₂, σ₁, σ₂] where:
            - α: change-point magnitude
            - τ: change-point time
            - μ₁, μ₂: trend slopes
            - β₁, β₂: cycle frequencies
            - σ₁, σ₂: observation noise scales
        n_steps: Number of time steps
        seed: Random seed
        n_simulations: Number of trajectories. 

    Returns:
        Array of shape (n_simulations, n_steps, 2)
    """
    alpha, tau = theta[0:2]
    mu = theta[2:4]
    beta = theta[4:6]
    sigma = theta[6:8]

    u = np.linspace(0, 1, n_steps)
    Z = np.zeros((n_steps, 2))
    for t in range(n_steps):
        changepoint_indicator = 1.0 if u[t] >= tau else 0.0
        Z[t, 0] = mu[0] * u[t] + alpha * changepoint_indicator + np.cos(2 * np.pi * beta[0] * u[t])
        Z[t, 1] = mu[1] * u[t] - alpha * changepoint_indicator + np.sin(2 * np.pi * beta[1] * u[t])

    rng = default_rng(seed)
    X_list = []
    for _ in range(n_simulations):
        epsilon = rng.standard_normal((n_steps, 2))
        X_list.append(Z + sigma * epsilon)
    return np.stack(X_list)
