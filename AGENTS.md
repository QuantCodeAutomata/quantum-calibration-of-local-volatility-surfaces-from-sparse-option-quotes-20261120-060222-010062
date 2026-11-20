# Repository guide

This repository reproduces selected experiments from *Quantum Calibration of
Local-Volatility Surfaces from Sparse Option Quotes*.

## Layout

- `src/` — flat library modules: grid/operator construction (`grid.py`,
  `local_variance.py`), propagators (`propagators.py`), calibration
  (`calibration.py`), pricing/IV inversion (`pricing.py`), comparators
  (`andreasen_huge.py`, `pinn.py`), quantum circuit construction
  (`quantum_circuits.py`), and `config.py` holding every experiment parameter as
  a module-level constant with a source comment.
- `exp/` — `exp1.py` .. `exp5.py`, each runnable as `python -m exp.<name>` from
  the repository root.
- `data/` — `generate_synthetic.py` builds the synthetic option-quote surfaces
  used by every experiment; no market data is used in this replication.
- `tests/` — pytest unit tests of `src/` on synthetic fixtures.
- `results/` — every artifact (`.csv`, `.json`, `.png`) produced by the
  experiments, plus `results/RESULTS.md` with the paper comparison.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -m "not market_data"   # unit tests (no market data is used at all)
python -m exp.exp1
python -m exp.exp2            # reduced configuration; python -m exp.exp2 --full for the paper's full sweep
python -m exp.exp3
python -m exp.exp4
python -m exp.exp5
bash reproduce.sh
```

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  `UPPER_CASE` variables with a comment giving their meaning, unit, and source
  (a paper section, or "not specified in the paper").
- Data preparation lives only in `data/`; `src/` and `exp/` never generate or
  fetch data themselves.
- Experiments import only the public functions of `src/`; they do not
  reimplement propagators, calibration, or pricing.
- No market-data API is used anywhere in this repository: every experiment
  consumes synthetic option quotes generated internally, as specified by the
  paper's experimental design.
- Running an experiment overwrites its files in `results/`. The numbers and
  verdicts written by hand in `results/RESULTS.md` must be checked against the
  freshly written files after any rerun and updated if they changed.
