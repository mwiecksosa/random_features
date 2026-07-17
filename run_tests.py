"""
Tests.

Usage:
    pytest run_tests.py -v
"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from random_feature_estimators import (
    RandomFeatures,
    TimeAverageEstimator,
    RollingWindowEstimator
)
from simulation_models import (
    simulate_moving_average,
    simulate_logistic_map,
    simulate_autoregressive_gandk,
    simulate_state_space_model,
    simulate_lorenz63,
    simulate_henon_map,
    simulate_iid_gaussian_3dim_powers,
    simulate_lotka_volterra,
    simulate_structural_timeseries,
    simulate_sir,
    simulate_sir_binomial_testing,
)


def test_random_features_generation():
    """Test random feature generation."""
    
    k, m, d = 5, 2, 3
    rf = RandomFeatures.generate(k=k, m=m, d=d, seed=42)
    
    assert rf.k == k
    assert rf.m == m
    assert rf.d == d
    assert rf.Omega.shape == (k, m+1, d)
    assert rf.alpha.shape == (k,)
    assert np.all(rf.alpha >= -np.pi)
    assert np.all(rf.alpha <= np.pi)
    


def test_random_features_reproducibility():
    """Test reproducibility with same seed."""
    
    rf1 = RandomFeatures.generate(k=5, m=2, d=3, seed=42)
    rf2 = RandomFeatures.generate(k=5, m=2, d=3, seed=42)
    
    assert_array_equal(rf1.Omega, rf2.Omega)
    assert_array_equal(rf1.alpha, rf2.alpha)
    


def test_random_features_evaluation():
    """Test feature evaluation."""
    
    rf = RandomFeatures.generate(k=3, m=1, d=2, seed=42)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    features = rf.evaluate(X)
    
    assert features.shape == (3,)
    assert np.all(features >= -1.0)
    assert np.all(features <= 1.0)
    


def test_time_average_estimator_init():
    """Test TimeAverageEstimator initialization."""
    
    def dummy_sim(theta, n, seed):
        return np.random.randn(n, 2)
    
    est = TimeAverageEstimator(
        simulator=dummy_sim,
        param_dim=3,
        obs_dim=2,
        n_lags=1,
        n_simulations=5,
        seed=42
    )
    
    assert est.param_dim == 3
    assert est.obs_dim == 2
    assert est.n_lags == 1
    assert est.n_simulations == 5
    assert est.random_features.k == 7


def test_rolling_window_estimator_init():
    """Test RollingWindowEstimator initialization."""
    
    def dummy_sim(theta, n, seed):
        return np.random.randn(n, 2)
    
    est = RollingWindowEstimator(
        simulator=dummy_sim,
        param_dim=3,
        obs_dim=2,
        window_size=20,
        n_lags=1,
        seed=42
    )
    
    assert est.param_dim == 3
    assert est.window_size == 20
    assert est.random_features.k == 7




def test_simulation_models():
    """Test simulation model outputs."""
    theta = np.array([0.5])
    X1 = simulate_iid_gaussian_3dim_powers(theta, 50, seed=42)[0]
    assert X1.shape == (50, 3)
    X2 = simulate_iid_gaussian_3dim_powers(theta, 50, seed=42)[0]
    assert_array_equal(X1, X2)
    assert_allclose(np.mean(X1, axis=0), [0.5, 0.25, 0.125], atol=0.5)
    theta = np.array([3.8, 0.1])
    X1 = simulate_logistic_map(theta, 50, seed=42)[0]
    assert X1.shape == (50, 1)
    X2 = simulate_logistic_map(theta, 50, seed=42)[0]
    assert_array_equal(X1, X2)
    theta = np.array([0.35, 2.0, 0.3, 0.2])
    X1 = simulate_autoregressive_gandk(theta, 50, seed=42)[0]
    assert X1.shape == (50, 1)
    X2 = simulate_autoregressive_gandk(theta, 50, seed=42)[0]
    assert_array_equal(X1, X2)
    theta = np.array([0.4, -3.0, 0.3, 0.2])
    X1 = simulate_moving_average(theta, 50, seed=42)[0]
    assert X1.shape == (50, 1)
    X2 = simulate_moving_average(theta, 50, seed=42)[0]
    assert_array_equal(X1, X2)
    theta = np.array([0.2, 0.2, 1.2, 0.5])
    X1 = simulate_state_space_model(theta, 50, seed=42)[0]
    assert X1.shape == (50, 25)
    X2 = simulate_state_space_model(theta, 50, seed=42)[0]
    assert_array_equal(X1, X2)
    theta = np.array([1.0, 0.5, 0.1, 0.2, 0.1, 0.1])
    X1 = simulate_lotka_volterra(theta, 50, seed=42)[0]
    assert X1.shape == (50, 2)
    X2 = simulate_lotka_volterra(theta, 50, seed=42)[0]
    assert_array_equal(X1, X2)
    assert X2.shape == (50, 2)
    theta = np.array([1.0, 0.5, 0.1, 0.2, 2.0, 2.0, 0.5, 0.5])
    X1 = simulate_structural_timeseries(theta, 50, seed=42)[0]
    assert X1.shape == (50, 2)
    X2 = simulate_structural_timeseries(theta, 50, seed=42)[0]
    assert_array_equal(X1, X2)
    theta = np.array([0.2, 0.1])
    X1 = simulate_sir(theta, 50, seed=42)[0]
    assert X1.shape == (50, 1)
    X2 = simulate_sir(theta, 50, seed=42)[0]
    assert_array_equal(X1, X2)
    theta = np.array([0.5, 0.2])
    X1 = simulate_sir_binomial_testing(theta, 50, seed=42)[0]
    assert X1.shape == (50, 1)
    assert np.all(X1 >= 0) and np.all(X1 <= 1)
    X2 = simulate_sir_binomial_testing(theta, 50, seed=42)[0]
    assert_array_equal(X1, X2)
    theta = np.array([10.0, 15.0, 8/3, 0.1, 0.1, 0.1])
    X1 = simulate_lorenz63(theta, 50, seed=42)[0]
    assert X1.shape == (50, 3)
    X2 = simulate_lorenz63(theta, 50, seed=42)[0]
    assert_array_equal(X1, X2)
    theta = np.array([1.4, 0.3, 0.1])
    X1 = simulate_henon_map(theta, 50, seed=42)[0]
    assert X1.shape == (50, 1)
    X2 = simulate_henon_map(theta, 50, seed=42)[0]
    assert_array_equal(X1, X2)



def run_all_tests():
    """Run all tests."""
    
    tests = [
        test_random_features_generation,
        test_random_features_reproducibility,
        test_random_features_evaluation,
        test_time_average_estimator_init,
        test_rolling_window_estimator_init,
        test_simulation_models,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"Error: {e}")
            failed += 1
    
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
