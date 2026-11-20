"""Quantum resource estimation for the residual circuit (Sec. III.B, V.D).

Builds, for a given (N, J, n_intervals) configuration, the "residual circuit"
described in Sec. III.B ("Residual Measurement") and Sec. V.D ("Computational
resources"): an n = log2(N)-qubit price register, one auxiliary LCHS register
per propagated maturity interval (prepared with amplitudes sqrt(|w_l|),
selecting J controlled Hamiltonian-simulation unitaries, then uncomputed), an
observation-map block encoding, and a reference-state block encoding, each
using one extra ancilla/flag qubit (Sec. III.B).

The paper does not specify the exact gate synthesis of the Hamiltonian-
simulation unitaries e^{-i*Delta*(H_s+eta_l*L_s)} nor of the observation/
reference block encodings (flagged uncertainty). This module is a custom
implementation choice: each such unitary is replaced by a deterministic
pseudo-random unitary of the right dimension (`qiskit.quantum_info.
random_unitary`, seeded), which has the same qubit count and a representative
two-qubit gate cost after dense (all-to-all) transpilation, but does not
reproduce the paper's specific CNOT counts (e.g. Table III).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import random_unitary

from src.propagators import lchs_kernel_nodes_weights

Config = tuple[int, int, int]


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _aux_amplitudes(J: int, aux: int, K_c: float, beta: float = 0.8) -> np.ndarray:
    """sqrt(|w_l|)/sqrt(sum|w_l|) amplitudes for the J LCHS terms, zero-padded to 2**aux."""
    _, w = lchs_kernel_nodes_weights(J, K_c=K_c, beta=beta, kernel="beta")
    amps = np.sqrt(np.abs(w))
    amps = amps / np.sqrt(np.sum(np.abs(w)))
    padded = np.zeros(2 ** aux, dtype=float)
    padded[:J] = amps
    return padded


def build_residual_circuit(
    N: int, J: int, n_intervals: int, seed: int = 0, K_c: float | None = None
) -> QuantumCircuit:
    """Assemble the per-maturity residual circuit for a given configuration.

    Qubit layout (must satisfy n + n_intervals*aux + 2 exactly, see module
    docstring): one "price" register of n = log2(N) qubits, one "aux{i}"
    register of aux = ceil(log2(J)) qubits per propagated interval, one
    "obs_flag" qubit (observation-map block encoding, step (c)) and one
    "ref_flag" qubit (reference-state block encoding, step (d)).
    """
    if not isinstance(N, int) or N <= 0 or not _is_power_of_two(N):
        raise ValueError(f"N must be a positive power of two, got {N}")
    if not isinstance(J, int) or J <= 0:
        raise ValueError(f"J must be a positive integer, got {J}")
    if n_intervals <= 0:
        raise ValueError(f"n_intervals must be positive, got {n_intervals}")

    if K_c is None:
        K_c = J / 8.0

    n = int(np.log2(N))
    aux = int(np.ceil(np.log2(J))) if J > 1 else 1

    price = QuantumRegister(n, "price")
    aux_regs = [QuantumRegister(aux, f"aux{i}") for i in range(n_intervals)]
    obs_flag = QuantumRegister(1, "obs_flag")
    ref_flag = QuantumRegister(1, "ref_flag")

    circuit = QuantumCircuit(price, *aux_regs, obs_flag, ref_flag)

    amplitudes = _aux_amplitudes(J, aux, K_c=K_c, beta=0.8)
    prep = StatePreparation(amplitudes, label="LCHS-prep")
    unprep = prep.inverse()

    for i, aux_reg in enumerate(aux_regs):
        circuit.append(prep, aux_reg)
        for l in range(J):
            gate = random_unitary(2 ** n, seed=seed + l).to_instruction()
            gate.name = f"H_sim_l{l}"
            controlled = gate.control(aux, ctrl_state=l)
            circuit.append(controlled, list(aux_reg) + list(price))
        circuit.append(unprep, aux_reg)

    obs_gate = random_unitary(2 ** (n + 1), seed=seed + 10_000).to_instruction()
    obs_gate.name = "obs_map"
    circuit.append(obs_gate, list(price) + list(obs_flag))

    ref_gate = random_unitary(2 ** (n + 1), seed=seed + 20_000).to_instruction()
    ref_gate.name = "ref_prep"
    circuit.append(ref_gate, list(price) + list(ref_flag))

    expected_qubits = n + n_intervals * aux + 2
    assert circuit.num_qubits == expected_qubits, (
        f"qubit count mismatch: got {circuit.num_qubits}, expected {expected_qubits}"
    )
    return circuit


def count_resources(N: int, J: int, n_intervals: int, seed: int = 0) -> dict:
    """Build the residual circuit, transpile to {cx, u3}, and report resource counts."""
    start = time.monotonic()
    circuit = build_residual_circuit(N, J, n_intervals, seed=seed)
    transpiled = transpile(circuit, basis_gates=["cx", "u3"], optimization_level=0)
    elapsed = time.monotonic() - start
    if elapsed > 60.0:
        print(
            f"WARNING: count_resources(N={N}, J={J}, n_intervals={n_intervals}) "
            f"took {elapsed:.1f}s (> 60s)"
        )
    ops = transpiled.count_ops()
    return {
        "N": N,
        "J": J,
        "n_intervals": n_intervals,
        "qubits": transpiled.num_qubits,
        "cx_count": ops.get("cx", 0),
    }


def run_resource_sweep(configs: list[Config], seed: int = 0) -> pd.DataFrame:
    """Run count_resources over every (N, J, n_intervals) config, sorted by N, J, n_intervals."""
    rows = [count_resources(N, J, n_intervals, seed=seed) for N, J, n_intervals in configs]
    df = pd.DataFrame(rows, columns=["N", "J", "n_intervals", "qubits", "cx_count"])
    df = df.sort_values(["N", "J", "n_intervals"]).reset_index(drop=True)
    return df
