"""Experiment 3: small-circuit measured calibration (Sec. III.B, III.C, IV.C, Algorithm 1).

Small N=4 interior-node, 2-maturity, 2-coefficient-per-interval instance used
by the paper to demonstrate the full measured-circuit calibration pipeline
(Sec. IV.C). This module:

1. Builds the small instance (grid, two-function basis, operator family) and
   generates calibration/held-out target prices from known generating
   coefficients via the exact propagator (Sec. IV.C).
2. Validates, via explicit Qiskit statevector simulation, that a literal LCU
   ("linear combination of unitaries") quantum circuit implementing the
   LCHS-propagated residual measurement reproduces the classical matrix-LCHS
   objective to a documented, honestly-reported tolerance (Sec. III.A, III.B).
   The paper does not give the exact matrix/circuit details of the LCHS
   decomposition, the observation-map block encoding or the reference-state
   block encoding (flagged uncertainty throughout); this module implements a
   concrete, self-consistent convention, verifies it numerically against the
   classical propagators, and reports the real achieved tolerance rather than
   assuming the paper's idealised ~1e-17.
3. Runs the bounded pattern search of Algorithm 1 (Sec. III.C) from a fixed
   start, once per objective ("exact", "finite_difference", "matrix_lchs",
   and "measured_lchs" if the statevector check passes to < 1e-4), and
   reports fitted coefficients, coefficient RMSE and held-out price RMSE (bp).

Run as `python -m exp.exp3` from the repository root.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from typing import Callable

import numpy as np
import pandas as pd
import scipy.linalg as sla
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.stats import beta as beta_dist

from src.config import (
    EXP3_CI_LEVEL,
    EXP3_GENERATING_COEFFICIENTS,
    EXP3_GRID_NODES,
    EXP3_HELDOUT_STRIKES,
    EXP3_LCHS_BETA,
    EXP3_LCHS_J,
    EXP3_LCHS_KC,
    EXP3_MATURITIES,
    EXP3_N_COEFFS_PER_INTERVAL,
    EXP3_PATTERN_SEARCH_START,
    EXP3_PATTERN_SEARCH_STEP,
    EXP3_PATTERN_SEARCH_SWEEPS,
    EXP3_QUOTE_STRIKES,
    EXP3_SAMPLER_SEEDS,
    EXP3_SHOT_BUDGETS,
    EXP3_STRIKE_RANGE,
)
from src.calibration import price_surface
from src.grid import LogStrikeGrid, boundary_lift, payoff
from src.local_variance import build_operator_family, two_function_basis
from src.pricing import price_at_strikes, rmse_bp, strike_interpolation_weights
from src.propagators import (
    assemble_A,
    coefficient_independent_theta,
    lchs_kernel_nodes_weights,
    propagate_backward_euler,
    propagate_exact,
    propagate_lchs,
)

Array = np.ndarray
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

BACKWARD_EULER_STEPS_PER_INTERVAL = 64  # not specified in the paper; chosen for a tiny N=4 instance
GENERATING_FLAT = np.array(
    [c for interval in EXP3_GENERATING_COEFFICIENTS for c in interval]
)  # (0.75, 0.25, 0.25, 0.75), Sec. IV.C


# ---------------------------------------------------------------------------
# Small-instance setup
# ---------------------------------------------------------------------------
def build_instance() -> tuple[LogStrikeGrid, Array, list[Array], float]:
    """Grid, operator family (A0, A_list) and coefficient-independent theta."""
    grid = LogStrikeGrid(EXP3_GRID_NODES)
    A0, A_list = build_operator_family(grid, two_function_basis())
    theta = coefficient_independent_theta(A0, A_list)
    return grid, A0, A_list, theta


def quote_strikes_array() -> Array:
    return np.array(EXP3_QUOTE_STRIKES, dtype=float)


def held_out_strikes() -> Array:
    """81 equally spaced strikes on EXP3_STRIKE_RANGE, excluding the 4 quote strikes."""
    full = np.linspace(EXP3_STRIKE_RANGE[0], EXP3_STRIKE_RANGE[1], EXP3_HELDOUT_STRIKES)
    keep = np.ones(len(full), dtype=bool)
    for q in EXP3_QUOTE_STRIKES:
        keep &= np.abs(full - q) > 1e-9
    return full[keep]


def generating_coeffs_list() -> list[Array]:
    return [np.array(c, dtype=float) for c in EXP3_GENERATING_COEFFICIENTS]


def xi_flat_to_list(xi_flat: Array) -> list[Array]:
    """Reshape a flat (4,) coefficient vector into [xi_interval1 (2,), xi_interval2 (2,)]."""
    n = EXP3_N_COEFFS_PER_INTERVAL
    return [xi_flat[:n], xi_flat[n:]]


# ---------------------------------------------------------------------------
# Sequential propagation and objective (Sec. II.A Eq. 5, III.B)
# ---------------------------------------------------------------------------
def sequential_states(
    grid: LogStrikeGrid,
    A0: Array,
    A_list: list[Array],
    coefficients: list[Array],
    propagate: Callable[[Array, Array, float], Array],
) -> list[Array]:
    """v_s for s=1..S, propagating sequentially from v(0) = payoff - boundary_lift."""
    G = boundary_lift(grid)
    v = payoff(grid) - G
    t_prev = 0.0
    states = []
    for s, T_s in enumerate(EXP3_MATURITIES):
        delta = T_s - t_prev
        A = assemble_A(A0, A_list, coefficients[s])
        v = propagate(v, A, delta)
        states.append(v)
        t_prev = T_s
    return states


def model_adjusted_prices(grid: LogStrikeGrid, v: Array, strikes: Array) -> Array:
    """W_s v: interpolate v (not v+G) onto strikes (observation map applied to the residual)."""
    G = boundary_lift(grid)
    full = price_at_strikes(v, G, grid.x, strikes)
    boundary_at_strikes = np.interp(np.log(strikes), grid.x, G)
    return full - boundary_at_strikes


def objective_value(
    grid: LogStrikeGrid,
    A0: Array,
    A_list: list[Array],
    coefficients: list[Array],
    propagate: Callable[[Array, Array, float], Array],
    quote_strikes: Array,
    y_targets: Array,
) -> float:
    """F(xi) = (1/S) sum_s (1/q_s) || W_s v_s - y_s ||^2, q_s = 1 (Eq. 5).

    q_s = 1 is used throughout: the paper does not specify a per-maturity
    normaliser beyond the generic symbol q_s (Sec. II.A, Eq. 5), and all
    maturities here share the same number (4) of quotes, so there is no
    natural unequal weighting to apply.
    """
    states = sequential_states(grid, A0, A_list, coefficients, propagate)
    total = 0.0
    S = len(EXP3_MATURITIES)
    for s in range(S):
        model = model_adjusted_prices(grid, states[s], quote_strikes)
        total += np.sum((model - y_targets[s]) ** 2)
    return total / S


# ---------------------------------------------------------------------------
# Bounded pattern search (Algorithm 1, Sec. III.C)
# ---------------------------------------------------------------------------
def _pattern_directions(n_coeffs_total: int, n_per_interval: int) -> list[Array]:
    """12 candidate directions per sweep point (documented convention, Sec. III.C).

    For each of the n_coeffs_total coefficients: +1 and -1 unit steps (2*n
    directions). Additionally, "opposing within-interval directions": for
    each interval, the direction that increases one coefficient and decreases
    the other coefficient of the SAME interval by one step, and its reverse
    (2 extra directions per interval). With n_coeffs_total=4 (2 intervals x 2
    coefficients) this gives 2*4 + 2*2 = 12 directions, matching the
    instructions. This is a custom, documented interpretation: the paper
    states only that "opposing within-interval directions" are included in
    the 12-direction pattern (Algorithm 1, Sec. III.C) without enumerating
    them.
    """
    directions = []
    for j in range(n_coeffs_total):
        e = np.zeros(n_coeffs_total)
        e[j] = 1.0
        directions.append(e.copy())
        directions.append(-e.copy())
    n_intervals = n_coeffs_total // n_per_interval
    for interval in range(n_intervals):
        base = interval * n_per_interval
        for i in range(n_per_interval):
            for k in range(i + 1, n_per_interval):
                d = np.zeros(n_coeffs_total)
                d[base + i] = 1.0
                d[base + k] = -1.0
                directions.append(d.copy())
                directions.append(-d.copy())
    return directions


def pattern_search(
    objective: Callable[[Array], float],
    start: Array,
    step: float,
    n_sweeps: int,
    n_per_interval: int,
) -> tuple[Array, float]:
    """Algorithm 1: bounded coordinate-and-diagonal pattern search (Sec. III.C).

    At every sweep, evaluate the objective at the incumbent and at every
    candidate xi + step*d (clipped to [0,1]) for the 12 documented
    directions; keep the best (ties keep the incumbent).
    """
    n = len(start)
    directions = _pattern_directions(n, n_per_interval)
    xi = start.copy()
    best_val = objective(xi)
    for _ in range(n_sweeps):
        candidates = [xi] + [np.clip(xi + step * d, 0.0, 1.0) for d in directions]
        values = [objective(c) for c in candidates]
        best_idx = int(np.argmin(values))
        if values[best_idx] < best_val - 1e-15:
            xi = candidates[best_idx]
            best_val = values[best_idx]
        # ties keep the incumbent (do nothing if best_idx==0 or value not strictly better)
    return xi, best_val


# ---------------------------------------------------------------------------
# Targets (Sec. IV.C): quote prices + held-out prices from the generating
# coefficients, priced exactly on the N=4 small-instance grid itself (this is
# the small-circuit validation instance, not a convergence study against a
# finer reference grid).
# ---------------------------------------------------------------------------
def build_targets(grid: LogStrikeGrid, A0: Array, A_list: list[Array]) -> tuple[Array, Array, Array, Array]:
    """Return (quote_strikes, quote_prices, heldout_strikes, heldout_prices)."""
    quote_strikes = quote_strikes_array()
    heldout = held_out_strikes()
    coeffs = generating_coeffs_list()
    quote_prices = price_surface(grid, A0, A_list, EXP3_MATURITIES, coeffs, propagate_exact, quote_strikes)
    heldout_prices = price_surface(grid, A0, A_list, EXP3_MATURITIES, coeffs, propagate_exact, heldout)
    return quote_strikes, quote_prices, heldout, heldout_prices


def y_targets_from_quotes(grid: LogStrikeGrid, quote_strikes: Array, quote_prices: Array) -> Array:
    """y_s = quote_prices[s] - boundary lift at the quote strikes (Sec. II.A, III.B)."""
    G = boundary_lift(grid)
    boundary_at_strikes = np.interp(np.log(quote_strikes), grid.x, G)
    return quote_prices - boundary_at_strikes[None, :]


# ---------------------------------------------------------------------------
# Statevector circuit validation (Sec. III.A, III.B; Algorithm 1 pre-check)
#
# Self-consistent convention adopted here (the paper does not give the exact
# matrix/circuit details, flagged uncertainty throughout Sec. III):
#
# * LCHS decomposition: A_s + theta*I = L_s + i*H_s (L_s symmetric part, H_s
#   = -i * antisymmetric part), exactly the convention already documented in
#   `src/propagators.py::_hermitian_split`, re-implemented here (that helper
#   is private and not in the task's allowed src API). Verified below to
#   reproduce `propagate_lchs` to ~1e-14 on random test vectors before any
#   circuit is built.
# * LCU aux register: `ceil(log2(J))` qubits prepared with amplitudes
#   sqrt(w_l) (all beta-kernel weights are non-negative and already sum to
#   one, so the LCU 1-norm alpha = sum|w_l| = 1 exactly and J=8=2**3 needs no
#   zero-padding here), controlled unitaries exp(-i*delta*(H_s+eta_l*L_s))
#   selected on computational basis state l, then uncomputed.
# * Observation map: W (interpolation matrix from the 4 grid nodes onto the 4
#   quote strikes, identical at both maturities since the same quote strikes
#   are used) normalised by its operator 2-norm gamma, dilated to a unitary
#   via the standard 2x2 block trick (M, sqrt(I-MM^+); sqrt(I-M^+M), -M^+)
#   with one extra "obs_flag" qubit (valid because M is square, N=4=n_quotes).
# * Residual/reference combination: a single extra "comb" qubit implements a
#   two-term LCU subtraction of (a) the model branch -- state-prep v(0)/||v0||,
#   apply the aux-LCU blocks for every interval up to s, apply the obs
#   dilation -- controlled on comb=0, and (b) a direct `StatePreparation` of
#   -y_s/||y_s|| controlled on comb=1, with RY(angle) comb-register state
#   amplitudes beta_1 = sqrt(gamma*||v0||*exp(theta*T_s) / zeta_s), beta_2 =
#   sqrt(||y_s|| / zeta_s), zeta_s = gamma*||v0||*exp(theta*T_s) + ||y_s||
#   (using alpha=1). Uncomputing the comb register (RY(-angle)) and measuring
#   (comb=0, every aux register=0, obs_flag=0) gives a probability that is a
#   provably exact LCU identity for ||W v_s - y_s||^2 / zeta_s^2, verified
#   below both algebraically (see module notes) and numerically against the
#   classical matrix-LCHS objective.
# ---------------------------------------------------------------------------
def hermitian_split(A: Array, theta: float) -> tuple[Array, Array]:
    """A + theta*I = L + i*H; see module docstring for the sign convention."""
    B = A + theta * np.eye(A.shape[0])
    L = 0.5 * (B + B.T)
    B_a = 0.5 * (B - B.T)
    H = -1j * B_a
    return L, H


def verify_lchs_matrix_convention(
    A: Array, delta: float, theta: float, J: int, K_c: float, beta_param: float, rng: np.random.Generator
) -> float:
    """Confirm the hand-built LCU matrix sum matches `propagate_lchs` (Sec. III.A).

    Returns the max abs error over a few random real test vectors; must be
    < 1e-6 before the sign/assembly convention is used in any circuit.
    """
    L, H = hermitian_split(A, theta)
    eta, w = lchs_kernel_nodes_weights(J, K_c, beta_param, kernel="beta")
    errors = []
    for _ in range(5):
        v0 = rng.standard_normal(A.shape[0])
        acc = np.zeros_like(v0, dtype=complex)
        for l in range(J):
            Ml = H + eta[l] * L
            Ul = sla.expm(-1j * delta * Ml)
            acc += w[l] * (Ul @ v0.astype(complex))
        acc *= np.exp(theta * delta)
        ref = propagate_lchs(v0, A, delta, theta, J, K_c, beta_param, kernel="beta")
        errors.append(np.max(np.abs(acc.real - ref)))
    return float(max(errors))


def _aux_lcu_block(circuit: QuantumCircuit, price_reg, aux_reg, A: Array, delta: float, theta: float,
                    J: int, K_c: float, beta_param: float) -> None:
    """Append prep -> controlled e^{-i*delta*(H+eta_l*L)} -> unprep onto (price_reg, aux_reg)."""
    L, H = hermitian_split(A, theta)
    eta, w = lchs_kernel_nodes_weights(J, K_c, beta_param, kernel="beta")
    aux_qubits = len(aux_reg)
    amps = np.zeros(2 ** aux_qubits)
    amps[:J] = np.sqrt(np.abs(w))  # zero-padded if J < 2**aux_qubits (not the case here, J=8=2**3)
    prep = StatePreparation(amps)
    circuit.append(prep, list(aux_reg))
    for l in range(J):
        Ml = H + eta[l] * L
        Ul = sla.expm(-1j * delta * Ml)
        if w[l] < 0:
            Ul = -Ul  # global phase encoding sign(w_l); beta-kernel weights are all >=0 here
        gate = UnitaryGate(Ul).control(aux_qubits, ctrl_state=l)
        circuit.append(gate, list(aux_reg) + list(price_reg))
    circuit.append(prep.inverse(), list(aux_reg))


def _obs_dilation_unitary(W: Array, gamma: float) -> Array:
    """2x2 block dilation U = [[M, sqrt(I-MM^+)], [sqrt(I-M^+M), -M^+]], M = W/gamma (Sec. III.B)."""
    M = W / gamma
    n = M.shape[0]
    I = np.eye(n)
    top_right = sla.sqrtm(I - M @ M.conj().T).real
    bot_left = sla.sqrtm(I - M.conj().T @ M).real
    U = np.block([[M, top_right], [bot_left, -M.conj().T]])
    assert np.allclose(U.conj().T @ U, np.eye(2 * n), atol=1e-7), "observation-map dilation is not unitary"
    return U


def build_model_pipeline_gate(
    grid: LogStrikeGrid, A0: Array, A_list: list[Array], coeffs_upto_s: list[Array],
    theta: float, J: int, K_c: float, beta_param: float, W: Array, gamma: float,
) -> tuple["Gate", int, list[int]]:
    """Build the (price, aux_1, [aux_2], obs_flag) model pipeline as a single Gate.

    Returns (gate, n_qubits, aux_register_sizes) so the caller can lay out
    the combined circuit's registers consistently.
    """
    n = int(np.log2(grid.n_interior))
    aux_size = int(np.ceil(np.log2(J))) if J > 1 else 1
    price_reg = QuantumRegister(n, "price")
    aux_regs = [QuantumRegister(aux_size, f"aux{i+1}") for i in range(len(coeffs_upto_s))]
    obs_flag = QuantumRegister(1, "obs_flag")
    circuit = QuantumCircuit(price_reg, *aux_regs, obs_flag)

    G = boundary_lift(grid)
    v0 = payoff(grid) - G
    v0_hat = v0 / np.linalg.norm(v0)
    circuit.append(StatePreparation(v0_hat), list(price_reg))

    t_prev = 0.0
    for s_idx, (T_s, xi_s) in enumerate(zip(EXP3_MATURITIES[: len(coeffs_upto_s)], coeffs_upto_s)):
        delta = T_s - t_prev
        A = assemble_A(A0, A_list, xi_s)
        _aux_lcu_block(circuit, price_reg, aux_regs[s_idx], A, delta, theta, J, K_c, beta_param)
        t_prev = T_s

    U_obs = _obs_dilation_unitary(W, gamma)
    circuit.append(UnitaryGate(U_obs), list(price_reg) + list(obs_flag))

    gate = circuit.to_gate(label="model_pipeline")
    return gate, circuit.num_qubits, [aux_size] * len(coeffs_upto_s)


def circuit_residual_probability(
    grid: LogStrikeGrid, A0: Array, A_list: list[Array], coeffs_upto_s: list[Array],
    theta: float, J: int, K_c: float, beta_param: float, W: Array, gamma: float, y_s: Array,
) -> tuple[float, float]:
    """Build the full residual circuit for maturity s and return (p_circuit, zeta_s).

    p_circuit is the exact Born probability (from `Statevector`) of the
    combined event {comb=0, every aux register=0, obs_flag=0}.
    """
    n_price = int(np.log2(grid.n_interior))
    model_gate, n_model_qubits, aux_sizes = build_model_pipeline_gate(
        grid, A0, A_list, coeffs_upto_s, theta, J, K_c, beta_param, W, gamma
    )

    price_reg = QuantumRegister(n_price, "price")
    aux_regs = [QuantumRegister(sz, f"aux{i+1}") for i, sz in enumerate(aux_sizes)]
    obs_flag = QuantumRegister(1, "obs_flag")
    comb = QuantumRegister(1, "comb")
    circuit = QuantumCircuit(price_reg, *aux_regs, obs_flag, comb)

    G = boundary_lift(grid)
    v0 = payoff(grid) - G
    v0_norm = np.linalg.norm(v0)
    T_s = EXP3_MATURITIES[len(coeffs_upto_s) - 1]
    zeta_s = gamma * v0_norm * float(np.exp(theta * T_s)) + float(np.linalg.norm(y_s))
    beta1_sq = (gamma * v0_norm * float(np.exp(theta * T_s))) / zeta_s
    angle = 2.0 * np.arccos(np.sqrt(beta1_sq))

    circuit.ry(angle, comb)
    model_qargs = list(price_reg)
    for aux_reg in aux_regs:
        model_qargs += list(aux_reg)
    model_qargs += list(obs_flag)
    model_ctrl = model_gate.control(1, ctrl_state=0)
    circuit.append(model_ctrl, list(comb) + model_qargs)

    y_hat = -y_s / np.linalg.norm(y_s)
    ref_prep = StatePreparation(y_hat)
    ref_ctrl = ref_prep.control(1, ctrl_state=1)
    circuit.append(ref_ctrl, list(comb) + list(price_reg))
    circuit.ry(-angle, comb)

    sv = Statevector(circuit).data
    n_aux_total = sum(aux_sizes)
    full = sv.reshape(2, 2, 2 ** n_aux_total, 2 ** n_price)  # (comb, obs_flag, aux(all), price)
    sub = full[0, 0, 0, :]
    p_circuit = float(np.sum(np.abs(sub) ** 2))
    return p_circuit, zeta_s


def classical_residual_value(
    grid: LogStrikeGrid, A0: Array, A_list: list[Array], coeffs_upto_s: list[Array],
    quote_strikes: Array, y_s: Array,
) -> float:
    """||W_s v_s(xi) - y_s||^2 computed from the classical matrix-LCHS propagator."""
    theta = coefficient_independent_theta(A0, A_list)
    G = boundary_lift(grid)
    v = payoff(grid) - G
    t_prev = 0.0
    for T_s, xi_s in zip(EXP3_MATURITIES[: len(coeffs_upto_s)], coeffs_upto_s):
        delta = T_s - t_prev
        A = assemble_A(A0, A_list, xi_s)
        v = propagate_lchs(v, A, delta, theta, EXP3_LCHS_J, EXP3_LCHS_KC, EXP3_LCHS_BETA, kernel="beta")
        t_prev = T_s
    model = price_at_strikes(v, G, grid.x, quote_strikes) - np.interp(np.log(quote_strikes), grid.x, G)
    return float(np.sum((model - y_s) ** 2))


def run_statevector_check(
    grid: LogStrikeGrid, A0: Array, A_list: list[Array], theta: float, W: Array, gamma: float,
    quote_strikes: Array, y_targets: Array,
) -> dict:
    """Validate the circuit's residual probability against the classical objective (Sec. III.B).

    Runs the sign-convention check first (Sec. III.A), then the full
    statevector residual check for two test coefficient vectors and both
    maturities, and returns a dict with the maximum absolute error achieved.
    """
    rng = np.random.default_rng(0)
    A_test = assemble_A(A0, A_list, np.array([0.6, 0.4]))
    sign_check_error = verify_lchs_matrix_convention(
        A_test, 0.5, theta, EXP3_LCHS_J, EXP3_LCHS_KC, EXP3_LCHS_BETA, rng
    )
    print(f"[exp3] LCHS sign-convention check: max abs error = {sign_check_error:.3e} (must be < 1e-6)")
    assert sign_check_error < 1e-6, "LCHS matrix sign convention does not match propagate_lchs"

    test_coeffs_sets = {
        "generating": generating_coeffs_list(),
        "arbitrary": [np.array([0.4, 0.8]), np.array([0.9, 0.1])],
    }
    errors = []
    details = []
    for label, coeffs in test_coeffs_sets.items():
        for s in (1, 2):
            coeffs_upto_s = coeffs[:s]
            y_s = y_targets[s - 1]
            p_circuit, zeta_s = circuit_residual_probability(
                grid, A0, A_list, coeffs_upto_s, theta, EXP3_LCHS_J, EXP3_LCHS_KC, EXP3_LCHS_BETA,
                W, gamma, y_s,
            )
            residual_sq = classical_residual_value(grid, A0, A_list, coeffs_upto_s, quote_strikes, y_s)
            p_classical = residual_sq / zeta_s ** 2
            err = abs(p_circuit - p_classical)
            errors.append(err)
            details.append(
                {"coeffs": label, "s": s, "p_circuit": p_circuit, "p_classical": p_classical, "abs_error": err}
            )
    max_abs_error = float(max(errors))
    print(f"[exp3] statevector residual check: max abs error over {len(errors)} (xi, s) combos = {max_abs_error:.3e}")
    return {
        "sign_convention_max_abs_error": sign_check_error,
        "max_abs_error": max_abs_error,
        "details": details,
    }


def build_observation_matrix(grid: LogStrikeGrid, quote_strikes: Array) -> tuple[Array, float]:
    """W (4x4 interpolation matrix, grid nodes -> quote strikes) and its operator 2-norm gamma."""
    rows = [strike_interpolation_weights(np.log(k), grid.x) for k in quote_strikes]
    W = np.array(rows)
    gamma = float(np.linalg.norm(W, 2))
    return W, gamma


# ---------------------------------------------------------------------------
# measured_lchs objective (Sec. III.B, III.C, IV.C) -- only run if the
# statevector check above passes to < 1e-4. The exact circuit Born
# probability has just been validated to match
# zeta_s^{-2} * ||W_s v_s(xi) - y_s||^2 computed from the classical
# matrix-LCHS propagator; re-building and re-simulating full Qiskit circuits
# for every pattern-search candidate would be unnecessarily slow for no
# extra fidelity, so the (validated) classical formula is used as the exact
# per-shot success probability p_s(xi), and finite-shot measurement is
# simulated with `rng.binomial(B_s, p_s(xi))` (Sec. IV.C: "measured-circuit
# objective"). This matches the task's own expectation that the shot
# dimension is O(1) per draw rather than O(shots).
# ---------------------------------------------------------------------------
def measured_objective_exact_probability(
    grid: LogStrikeGrid, A0: Array, A_list: list[Array], xi_flat: Array,
    quote_strikes: Array, y_targets: Array, theta: float, gamma: float, v0_norm: float,
) -> tuple[list[float], list[float]]:
    """Per-maturity (p_s, zeta_s) at the given flat coefficient vector."""
    coeffs = xi_flat_to_list(xi_flat)
    p_list, zeta_list = [], []
    for s in range(1, len(EXP3_MATURITIES) + 1):
        y_s = y_targets[s - 1]
        T_s = EXP3_MATURITIES[s - 1]
        zeta_s = gamma * v0_norm * float(np.exp(theta * T_s)) + float(np.linalg.norm(y_s))
        residual_sq = classical_residual_value(grid, A0, A_list, coeffs[:s], quote_strikes, y_s)
        p_list.append(residual_sq / zeta_s ** 2)
        zeta_list.append(zeta_s)
    return p_list, zeta_list


def measured_objective_value(
    grid: LogStrikeGrid, A0: Array, A_list: list[Array], xi_flat: Array, quote_strikes: Array,
    y_targets: Array, theta: float, gamma: float, v0_norm: float, shots: int, rng: np.random.Generator,
) -> float:
    """F_hat(xi) = (1/S) sum_s (zeta_s^2/q_s) * (h_s/B_s), h_s ~ Binomial(B_s, p_s(xi))."""
    p_list, zeta_list = measured_objective_exact_probability(
        grid, A0, A_list, xi_flat, quote_strikes, y_targets, theta, gamma, v0_norm
    )
    S = len(EXP3_MATURITIES)
    total = 0.0
    for p_s, zeta_s in zip(p_list, zeta_list):
        p_s_clipped = float(np.clip(p_s, 0.0, 1.0))
        h_s = rng.binomial(shots, p_s_clipped)
        total += (zeta_s ** 2) * (h_s / shots)
    return total / S


def measured_lchs_confidence_interval(
    grid: LogStrikeGrid, A0: Array, A_list: list[Array], xi_flat: Array, quote_strikes: Array,
    y_targets: Array, theta: float, gamma: float, v0_norm: float, shots: int, rng: np.random.Generator,
    ci_level: float,
) -> dict:
    """95% conservative binomial CI (Clopper-Pearson via scipy.stats.beta) at a fixed candidate (Sec. IV.C)."""
    p_list, zeta_list = measured_objective_exact_probability(
        grid, A0, A_list, xi_flat, quote_strikes, y_targets, theta, gamma, v0_norm
    )
    alpha_level = 1.0 - ci_level
    lo_total, hi_total, point_total = 0.0, 0.0, 0.0
    per_maturity = []
    for p_s, zeta_s in zip(p_list, zeta_list):
        p_s_clipped = float(np.clip(p_s, 0.0, 1.0))
        h_s = int(rng.binomial(shots, p_s_clipped))
        lo = beta_dist.ppf(alpha_level / 2, h_s, shots - h_s + 1) if h_s > 0 else 0.0
        hi = beta_dist.ppf(1 - alpha_level / 2, h_s + 1, shots - h_s) if h_s < shots else 1.0
        lo = 0.0 if np.isnan(lo) else lo
        hi = 1.0 if np.isnan(hi) else hi
        scale = zeta_s ** 2
        per_maturity.append({"h_s": h_s, "B_s": shots, "lo": scale * lo, "hi": scale * hi, "point": scale * h_s / shots})
        lo_total += scale * lo
        hi_total += scale * hi
        point_total += scale * h_s / shots
    S = len(EXP3_MATURITIES)
    return {
        "per_maturity": per_maturity,
        "total_lo": lo_total / S,
        "total_hi": hi_total / S,
        "total_point": point_total / S,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    grid, A0, A_list, theta = build_instance()
    quote_strikes, quote_prices, heldout_strikes, heldout_prices = build_targets(grid, A0, A_list)
    y_targets = y_targets_from_quotes(grid, quote_strikes, quote_prices)
    W, gamma = build_observation_matrix(grid, quote_strikes)
    G = boundary_lift(grid)
    v0_norm = float(np.linalg.norm(payoff(grid) - G))

    print(f"[exp3] grid N={grid.n_interior}, maturities={EXP3_MATURITIES}, quote strikes={quote_strikes}")

    # --- Step 3: statevector validation (before the pattern search) ---
    check = run_statevector_check(grid, A0, A_list, theta, W, gamma, quote_strikes, y_targets)
    measured_lchs_run = check["max_abs_error"] < 1e-4
    if not measured_lchs_run:
        print(
            "[exp3] measured_lchs objective skipped: statevector check did not pass to required tolerance"
        )
    statevector_summary = {
        "max_abs_error": check["max_abs_error"],
        "sign_convention_max_abs_error": check["sign_convention_max_abs_error"],
        "tolerance_achieved": check["max_abs_error"],
        "measured_lchs_run": bool(measured_lchs_run),
    }
    with open(os.path.join(RESULTS_DIR, "exp3_statevector_check.json"), "w") as f:
        json.dump({**statevector_summary, "details": check["details"]}, f, indent=2, default=float)

    # --- Step 4-6: pattern search for each classical objective ---
    start = np.full(4, EXP3_PATTERN_SEARCH_START)

    def make_objective(propagate):
        return lambda xi_flat: objective_value(
            grid, A0, A_list, xi_flat_to_list(xi_flat), propagate, quote_strikes, y_targets
        )

    propagators = {
        "exact": propagate_exact,
        "finite_difference": lambda v0, A, d: propagate_backward_euler(
            v0, A, d, BACKWARD_EULER_STEPS_PER_INTERVAL
        ),
        "matrix_lchs": lambda v0, A, d: propagate_lchs(
            v0, A, d, theta, EXP3_LCHS_J, EXP3_LCHS_KC, EXP3_LCHS_BETA, kernel="beta"
        ),
    }

    summary_rows = []
    for name, propagate in propagators.items():
        obj = make_objective(propagate)
        xi_fit, _ = pattern_search(
            obj, start, EXP3_PATTERN_SEARCH_STEP, EXP3_PATTERN_SEARCH_SWEEPS, EXP3_N_COEFFS_PER_INTERVAL
        )
        coeff_rmse = float(np.sqrt(np.mean((xi_fit - GENERATING_FLAT) ** 2)))
        fitted_coeffs_list = xi_flat_to_list(xi_fit)
        heldout_model = price_surface(
            grid, A0, A_list, EXP3_MATURITIES, fitted_coeffs_list, propagate_exact, heldout_strikes
        )
        heldout_rmse = rmse_bp(heldout_model, heldout_prices)
        print(f"[exp3] objective={name}: xi={xi_fit}, coeff_rmse={coeff_rmse:.4f}, heldout_rmse_bp={heldout_rmse:.4f}")
        summary_rows.append(
            {
                "objective": name,
                "shots": "",
                "seed": "",
                "fitted_coefficients": str([round(float(x), 6) for x in xi_fit]),
                "coefficient_rmse": coeff_rmse,
                "held_out_rmse_bp": heldout_rmse,
            }
        )

    # --- Step 5/7: measured_lchs (only if the statevector check passed) ---
    ci_results = []
    if measured_lchs_run:
        for shots in EXP3_SHOT_BUDGETS:
            for seed in EXP3_SAMPLER_SEEDS:
                rng = np.random.default_rng(seed)

                def meas_obj(xi_flat, shots=shots, rng=rng):
                    return measured_objective_value(
                        grid, A0, A_list, xi_flat, quote_strikes, y_targets, theta, gamma, v0_norm, shots, rng
                    )

                xi_fit, _ = pattern_search(
                    meas_obj, start, EXP3_PATTERN_SEARCH_STEP, EXP3_PATTERN_SEARCH_SWEEPS,
                    EXP3_N_COEFFS_PER_INTERVAL,
                )
                coeff_rmse = float(np.sqrt(np.mean((xi_fit - GENERATING_FLAT) ** 2)))
                fitted_coeffs_list = xi_flat_to_list(xi_fit)
                heldout_model = price_surface(
                    grid, A0, A_list, EXP3_MATURITIES, fitted_coeffs_list, propagate_exact, heldout_strikes
                )
                heldout_rmse = rmse_bp(heldout_model, heldout_prices)
                print(
                    f"[exp3] objective=measured_lchs shots={shots} seed={seed}: xi={xi_fit}, "
                    f"coeff_rmse={coeff_rmse:.4f}, heldout_rmse_bp={heldout_rmse:.4f}"
                )
                summary_rows.append(
                    {
                        "objective": "measured_lchs",
                        "shots": shots,
                        "seed": seed,
                        "fitted_coefficients": str([round(float(x), 6) for x in xi_fit]),
                        "coefficient_rmse": coeff_rmse,
                        "held_out_rmse_bp": heldout_rmse,
                    }
                )

                ci_rng = np.random.default_rng(seed + 1_000_000)
                ci = measured_lchs_confidence_interval(
                    grid, A0, A_list, xi_fit, quote_strikes, y_targets, theta, gamma, v0_norm,
                    shots, ci_rng, EXP3_CI_LEVEL,
                )
                ci_results.append({"shots": shots, "seed": seed, **ci})

        with open(os.path.join(RESULTS_DIR, "exp3_confidence_intervals.json"), "w") as f:
            json.dump(ci_results, f, indent=2, default=float)

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp3_summary.csv"), index=False)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    elapsed = time.perf_counter() - t_start
    print(f"[exp3] total wall-clock time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
