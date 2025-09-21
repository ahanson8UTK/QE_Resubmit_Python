# hmc-gibbs

## How to run smoke

Execute the smoke-test Gibbs sweep with:

```bash
python -m scripts.run_gibbs --smoke --chains 4 --seed 0
```

A reproducible, typed JAX scaffold for the Hamiltonian Monte Carlo (HMC) in Gibbs sampler
outlined in Creal & Wu (2017, *International Economic Review*). The package emphasises
square-root Kalman filtering, Durbin–Koopman simulation smoothing, and ChEES-tuned dynamic
HMC kernels for each conditional block of the sampler. All targets and gradients are written
for JAX with 64-bit precision enabled for numerical stability.

## Getting started

```bash
pip install -e .
```

The repository ships with configuration files under [`configs/`](configs/) that describe data
paths, priors, and warm-up policies. To run a smoke-test Gibbs sweep with synthetic data
placeholders, execute:

```bash
python -m scripts.run_gibbs --config configs/defaults.yaml --chains 2 --seed 0
```

The command produces a timestamped folder under [`results/`](results/) with summary metrics
and HTML diagnostics. Dynamic HMC kernels are adapted with the ChEES procedure described in
the [BlackJAX documentation](https://blackjax-devs.github.io/blackjax/generated/blackjax.adaptation.chees_adaptation.html).

### Design highlights

- **Float64 throughout** to stabilise Kalman filtering and HMC dynamics.
- **SR-Kalman filter** and Durbin–Koopman simulation smoother stubs ready for completion.
- Modular block definitions with clear extension points for equity pricing constraints and
  pseudo-marginal bubble components.
- Instrumentation hooks for effective sample size per second, work units, and warm-up status.

### Repository layout

The project follows a ``src`` layout so editable installs and tooling target the
package directly without referring to the legacy ``src/cw2017`` path. The major
directories are:

```
.
├── configs/        # YAML configuration files for experiments
├── scripts/        # Command-line utilities and helpers
├── src/
│   └── hmc_gibbs/  # Package modules
└── tests/          # Smoke-test suites exercising key components
```

Refer to the extensive TODO lists in each math-heavy module for the next implementation steps.
