# Quantum Calibration of Local-Volatility Surfaces from Sparse Option Quotes

Replication of selected experiments from *Quantum Calibration of Local-Volatility
Surfaces from Sparse Option Quotes* (paper indexed for this replication; see
`results/RESULTS.md` for the exact sections and tables reproduced).

## Layout

- `src/` — Dupire forward-PDE discretisation on a log-strike grid, the positive
  affine local-variance basis, the three propagators (exact matrix exponential,
  backward Euler, LCHS with the "simple" and "beta" kernels), sequential bounded
  least-squares calibration, Black-formula implied-volatility inversion, an
  Andreasen-Huge arbitrage-free interpolation comparator, a physics-informed
  neural-network comparator, and the quantum residual-circuit construction used
  for the small measured-calibration instance and the resource sweep.
- `exp/` — one script per experiment, run as `python -m exp.exp1` ... `exp.exp5`.
- `data/` — synthetic data generation only (`data/generate_synthetic.py`); no
  market data is used anywhere in this replication.
- `tests/` — unit tests of `src/` on synthetic fixtures.
- `results/` — every table, plot and metric produced by the experiments, plus
  `results/RESULTS.md` with the comparison against the paper.

## Setup

```bash
pip install -r requirements.txt
```

No API keys are required: every experiment uses internally generated synthetic
option quotes, as specified by the paper's experimental design.

## Running the experiments

```bash
python -m exp.exp1   # Synthetic surface reconstruction (smile and skew)
python -m exp.exp2   # Classical solver comparison benchmark
python -m exp.exp3   # Small-circuit measured calibration
python -m exp.exp4   # Financial-consistency stress test
python -m exp.exp5   # Quantum circuit resource estimation
```

`exp2` accepts a reduced configuration by default (see "Deviations from the
paper" in `results/RESULTS.md`); `python -m exp.exp2 --full` runs the paper's
full grid/method/noise-dataset sweep.

## Tests

```bash
python -m pytest
```

All tests run on synthetic fixtures; none depend on a live market-data call.

## Reproduce everything

```bash
bash reproduce.sh
```
