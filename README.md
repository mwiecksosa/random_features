## Repository Contents

- **`random_feature_estimators.py`** - Implements the random feature estimators
- **`simulation_models.py`** - Contains the data generating processes for the experiments
- **`experiment_configs.py`** - Contains the settings for each experiment 
- **`experiments.py`** - Contains the experiments
- **`run_repeated_experiments.py`** - Runs the experiments
- **`combine_repeated_experiments.py`** - Merges results from multiple directories and plots the estimates
- **`plot_experiments.py`** - Creates plots of the time series, objectives, and random features
- **`run_tests.py`** - Runs tests for the code



## Requirements

```bash
pip install numpy scipy matplotlib pytest
```



## Examples


Below are examples for a Gaussian with unknown mean and a structural time series model with trends, cycles, and a change-point, see `simulation_models.py` for the implementations.

We compare the results using `L-BFGS-B` and `differential_evolution` as the optimizers. Also, we compare with the maximum likelihood estimator for the IID Gaussian model.

The following optimizers are implemented: `L-BFGS-B`, `Nelder-Mead`, `Powell`, `differential_evolution`, `dual_annealing`, `basinhopping`.

### IID Gaussian model using time-average estimator


We observe $$X_t=\mu+\sigma \epsilon_t,$$ where $\epsilon_{t} \overset{\mathrm{iid}}{\sim} N(0,1)$.

We use the time-average estimator to estimate the mean $\mu$.



```python
import numpy as np
import time
from random_feature_estimators import TimeAverageEstimator
from simulation_models import simulate_iid_gaussian

param_name = "μ"
theta_true = np.array([1]) # unknown mean
n_steps = 1000
my_seed = 2026
X_obs = simulate_iid_gaussian(theta_true, n_steps=n_steps, seed=my_seed)[0]  

param_dim = len(theta_true)
obs_dim = X_obs.shape[-1]
est = TimeAverageEstimator(
    simulator=simulate_iid_gaussian,
    param_dim=param_dim,
    obs_dim=obs_dim,
    n_lags=0,
    n_simulations=100,
    seed=my_seed,
)

# Initial guess and bounds for the parameter
theta_init = np.array([0.5])
bounds = [(-2.0, 2.0)]  

# Differential Evolution optimizer
start_de = time.time()
result_de = est.fit(
    X_obs=X_obs,
    theta_init=theta_init,
    bounds=bounds,
    optimizer="differential_evolution",
    n_jobs=10,
)
time_de = time.time() - start_de

# L-BFGS-B optimizer
start_lbfgs = time.time()
result_lbfgs = est.fit(
    X_obs=X_obs,
    theta_init=theta_init,
    bounds=bounds,
    optimizer="L-BFGS-B",
    n_starts=30,
    n_jobs=10,
)
time_lbfgs = time.time() - start_lbfgs

# Maximum Likelihood Estimator
start_ml = time.time()
ml_est = np.mean(X_obs)
time_ml = time.time() - start_ml

formatted_results = (
    f"{'':<25} {param_name:<12} {'Time (s)'}\n"
    f"{'True':<25} {theta_true[0]:<12.6f}\n"
    f"{'Differential Evolution':<25} {result_de['theta_hat'][0]:<12.6f} {time_de:<10.3f}\n"
    f"{'L-BFGS-B':<25} {result_lbfgs['theta_hat'][0]:<12.6f} {time_lbfgs:<10.3f}\n"
    f"{'Maximum Likelihood':<25} {ml_est:<12.6f} {time_ml:<10.3f}"
)
print(formatted_results)
```



### Structural time series model using rolling-window estimator

We consider a two-dimensional structural time series model with cycles, trends, and a change-point. For $u\in [0,1]$, define $Z_u^{(1)}=\mu^{(1)}u+\alpha 1_{u \geq \tau}+\mathrm{cos}(2 \pi \beta^{(1)} u)$ and $Z_u^{(2)}=\mu^{(2)}u-\alpha 1_{u \geq \tau}+\mathrm{sin}(2 \pi \beta^{(2)} u)$. 

For $t=1,\ldots,n$ and $j= 1,2$, we observe $$X_{t}^{(j)} = Z_{t/n}^{(j)} + \sigma^{(j)}  \epsilon_{t}^{(j)},$$ where $\epsilon_{t}^{(j)} \overset{\mathrm{iid}}{\sim} N(0,1)$.

We use the rolling-window estimator to estimate the parameters $\alpha$, $\tau$, $\mu^{(1)}$, $\mu^{(2)}$, $\beta^{(1)}$, $\beta^{(2)}$, $\sigma^{(1)}$, $\sigma^{(2)}$.




```python
import time
import numpy as np
from random_feature_estimators import RollingWindowEstimator
from simulation_models import simulate_structural_timeseries

# True parameter for the structural time series
param_names = ["α", "τ", "μ1", "μ2", "β1", "β2", "σ1", "σ2"]
theta_true = np.array([
    1.0,   # α: change-point magnitude
    0.5,   # τ: change-point time in [0,1]
    0.5,   # μ1: trend slope for component 1
    -0.3,  # μ2: trend slope for component 2
    1.0,   # β1: cycle frequency for component 1
    1.5,   # β2: cycle frequency for component 2
    0.2,   # σ1: noise scale for component 1
    0.2,   # σ2: noise scale for component 2
])

n_steps = 1000
my_seed = 2026 
X_obs = simulate_structural_timeseries(theta_true, n_steps=n_steps, seed=my_seed)[0]  
param_dim = len(theta_true)
obs_dim = X_obs.shape[-1]

# Let the estimator choose window size, initial offset, and L from the data
est = RollingWindowEstimator(
    simulator=simulate_structural_timeseries,
    param_dim=param_dim,
    obs_dim=obs_dim,
    window_size=None, 
    n_lags=0,
    n_simulations=10,
    seed=my_seed,
)

# Initial guess and bounds for each parameter
theta_init = theta_true + np.array([0.2, 0.1, 0.1, 0.1, 0.0, 0.0, 0.05, 0.05])
bounds = [
    (0.0, 2.0),    # α
    (0.0, 1.0),    # τ
    (-1.0, 1.0),   # μ1
    (-1.0, 1.0),   # μ2
    (0.0, 3.0),    # β1
    (0.0, 3.0),    # β2
    (0.01, 1.0),   # σ1
    (0.01, 1.0),   # σ2
]

start_de = time.time()
result_de = est.fit(
    X_obs=X_obs,
    theta_init=theta_init,
    bounds=bounds,
    optimizer="differential_evolution",
    n_jobs=10,
)
end_de = time.time()

start_bfgs = time.time()
result_bfgs = est.fit(
    X_obs=X_obs,
    theta_init=theta_init,
    bounds=bounds,
    optimizer="L-BFGS-B",
    n_starts=100,
    n_jobs=10,
)
end_bfgs = time.time()


formatted_results = (
    f"{' ':<25} " + " ".join([f"{name:<12}" for name in param_names]) + "Time (s)\n" +
    f"{'True':<25} " + " ".join([f"{v:<12.6f}" for v in theta_true]) + "\n" +
    f"{'Differential Evolution':<25} " + " ".join([f"{v:<12.6f}" for v in result_de['theta_hat']]) + f"{end_de - start_de:<10.3f}\n" +
    f"{'L-BFGS-B':<25} " + " ".join([f"{v:<12.6f}" for v in result_bfgs['theta_hat']]) + f"{end_bfgs - start_bfgs:<10.3f}"
)
print(formatted_results)
```


### Experiments

```bash
python run_repeated_experiments.py --base-seed MYSEEDNUMBER --n-runs 1000
```

Results are saved in a time-stamped directory under `outputs/`.

Then merge all results and create plots of estimates:

```bash
python combine_repeated_experiments.py outputs
```

To run all experiments one time:

```bash
# Run the experiments once
python run_repeated_experiments.py --base-seed MYSEEDNUMBER --n-runs 1

# Default: 100 runs with base_seed=42
python run_repeated_experiments.py
```
### Generate Plots

To generate the plots of the time series, objectives, and random features:

```bash
python plot_experiments.py
```

Plots are written to the `plots/` directory (or to `--output-dir` if given). 


### Run Tests

To run the tests: 

```bash
pytest run_tests.py -v
```




