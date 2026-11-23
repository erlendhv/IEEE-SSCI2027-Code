# CMO-BPSO for Tieback Portfolio Optimization

Code for the paper "Constrained Multi-Objective Binary PSO for Project Portfolio Optimization under Uncertainty" (IEEE SSCI 2027).

CMO-BPSO selects and schedules oil field tiebacks on the Norwegian Continental Shelf. For each field it chooses a host platform, a development year and a well count, and it trades expected real option value (ROV) against conditional value at risk (CVaR). Portfolios are valued with Schwartz-Smith price scenarios and least-squares Monte Carlo, under host capacity constraints.

## Setup

Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests
```

## Running one optimization

From the repository root:

```python
from src.orchestrator import run_optimization

result = run_optimization(
    config_path='config/default.yaml',
    overrides={'pso.n_particles': 50, 'pso.max_iterations': 200},
)
```

`result` contains `config`, `archive`, `pareto_front` (re-evaluated on the final scenario set) and `history`. All parameters are in `config/default.yaml` and can be overridden with dot notation.

## Reproducing the experiments

Each experiment ID runs a set of configurations over seeds 1 to 20 and writes to `results/<ID>/`:

```bash
python run_experiment.py E1.1 --seeds 1-20
```

The paper's runs used the Slurm scripts in `hpc/` (one array task per configuration and seed; set `--account` first). Each run used 20 parallel evaluation workers.

| Paper | Experiment IDs | Analysis notebook |
|---|---|---|
| Exp.1: Encoding | E1.1 | `13_E7_sensitivity_nsga2.ipynb` (section 6), `07_E1_encoding.ipynb` |
| Exp.2: Constraint handling | E2.1 | `13_E7_sensitivity_nsga2.ipynb` (section 6), `08_E2_constraints.ipynb` |
| Exp.3: Stickiness | E4.1 | `13_E7_sensitivity_nsga2.ipynb` (section 6), `10_E4_stickiness.ipynb` |
| Exp.4: Sensitivity of CMO-BPSO and BPSO | E7.1, E7.2, E7.3, E7.4, E7.5, E7.7, E7.8 | `13_E7_sensitivity_nsga2.ipynb` |
| Exp.5: Benchmark against NSGA-II | E7.3, E7.6 | `13_E7_sensitivity_nsga2.ipynb` |
| Portfolio structure along the Pareto front | E4.1 (seed 20) | `11_E5_portfolio_analysis.ipynb` |

The paper reports HVI and IGD with references built per seed (`src/experiments/metrics.py`, `per_instance_metrics`), as in notebook 13. Notebooks 07 to 10 use references pooled over all seeds.

E3.1 (ROV vs NPV objective) and E5.2 (sensitivity to problem parameters) are additional experiments, analysed in notebooks 09 and 12.

## Notebooks

- `01` to `05`: data generation, price scenarios, valuation, a single optimization run and portfolio analytics
- `06`: a larger stylized case with 15 fields and 5 hosts
- `07` to `12`: analysis of E1.1 to E5.2
- `13`: analysis of the E7 experiments and the per-seed metrics for E1.1, E2.1 and E4.1

The experiment notebooks read from `results/`, so they need the corresponding runs first.

## Structure

```
├── config/default.yaml      # All parameters
├── run_experiment.py        # Run an experiment over seeds
├── hpc/                     # Slurm scripts
├── src/
│   ├── config.py            # YAML loader and seed derivation
│   ├── models.py            # Shared dataclasses
│   ├── orchestrator.py      # End-to-end pipeline
│   ├── data_generation/     # Synthetic field, host and cost data
│   ├── price/               # Schwartz-Smith two-factor model
│   ├── valuation/           # Capacity capping, NPV and LSMC
│   ├── optimizer/           # Encoding, CMO-BPSO (DSBPSO update), BPSO, NSGA-II, evaluation
│   └── experiments/         # Experiment configurations, runner, HVI and IGD
├── tests/
└── notebooks/
```
