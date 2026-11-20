"""Experiment 5: quantum circuit resource estimation (Sec. V.D "Computational resources").

Builds the residual circuit of Sec. III.B for every (N, J, n_intervals)
configuration in EXP5_SWEEP, transpiles each to a {cx, u3} basis (dense,
all-to-all synthesis), and reports the resulting qubit and CNOT counts.

The Hamiltonian-simulation and block-encoding unitaries are implemented as
deterministic pseudo-random placeholders (see src/resource_count.py
docstring); the reported qubit counts follow the paper's formula exactly,
but the CNOT counts are illustrative only and will not match the paper's
Table III (which relies on an unspecified gate synthesis).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import EXP5_SWEEP
from src.resource_count import run_resource_sweep

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CSV_PATH = os.path.join(RESULTS_DIR, "exp5_resource_counts.csv")
PLOT_PATH = os.path.join(RESULTS_DIR, "exp5_resource_scaling.png")


def _check_qubit_formula(df) -> None:
    import numpy as np

    n = np.log2(df["N"]).astype(int)
    aux = np.ceil(np.log2(df["J"])).astype(int)
    expected = n + df["n_intervals"] * aux + 2
    mismatches = df[df["qubits"] != expected]
    if len(mismatches) > 0:
        print("MISMATCHES in qubit formula n + s*ceil(log2(J)) + 2:")
        print(mismatches)
        raise AssertionError("qubit count formula failed for at least one row")
    print("Qubit formula n + s*ceil(log2(J)) + 2 holds for all rows.")


def _plot_resource_scaling(df) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for (N, s), group in df.groupby(["N", "n_intervals"]):
        group = group.sort_values("J")
        x = [str(j) for j in group["J"]]
        ax.plot(
            x,
            group["cx_count"],
            marker="o",
            label=f"N={N}, s={s}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("LCHS terms J")
    ax.set_ylabel("CNOT count (log scale)")
    ax.set_title("Residual circuit CNOT count scaling (Sec. V.D)")
    ax.legend(title="N, n_intervals (s)")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = run_resource_sweep(list(EXP5_SWEEP))
    print(df.to_string(index=False))

    _check_qubit_formula(df)

    df.to_csv(CSV_PATH, index=False)
    _plot_resource_scaling(df)

    print(f"Saved table to {CSV_PATH}")
    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
