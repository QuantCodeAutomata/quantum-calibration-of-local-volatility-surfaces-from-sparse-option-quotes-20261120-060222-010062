"""Unit tests for the shared Dupire discretisation and propagators (Sec. II.A, III.A)."""

import numpy as np
import pytest

from src.grid import LogStrikeGrid, boundary_lift, payoff
from src.local_variance import build_operator_family, smile_basis, local_variance_surface
from src.propagators import (
    assemble_A,
    coefficient_independent_theta,
    propagate_backward_euler,
    propagate_exact,
    propagate_lchs,
)


def test_fitted_derivative_annihilates_constants_and_linear():
    """Sec. II.A: the fitted 3-point strike-derivative stencil (including the
    ghost boundary point) reproduces df/dk=0, d2f/dk2=0 exactly on a constant
    function, and d2f/dk2=0 on a strike-linear function. This is the weight
    property that lets the boundary lift G cancel exactly in the residual
    variable v = c - G, so the truncated interior operator needs no forcing
    term (the actual property exercised by build_base_operators, since G's
    ghost contribution analytically cancels rather than being dropped)."""
    grid = LogStrikeGrid(32)
    k_full = np.concatenate(([grid.k_lo], grid.k, [grid.k_hi]))
    for i in range(grid.n_interior):
        km1, ki, kp1 = k_full[i], k_full[i + 1], k_full[i + 2]
        h_m, h_p = ki - km1, kp1 - ki
        w = np.array([-h_p / (h_m * (h_m + h_p)), (h_p - h_m) / (h_m * h_p), h_m / (h_p * (h_m + h_p))])
        w2 = np.array([2.0 / (h_m * (h_m + h_p)), -2.0 / (h_m * h_p), 2.0 / (h_p * (h_m + h_p))])
        nodes = np.array([km1, ki, kp1])
        assert abs(w @ np.ones(3)) < 1e-10        # D1 annihilates constants
        assert abs(w2 @ np.ones(3)) < 1e-10        # D2 annihilates constants
        assert abs(w2 @ nodes) < 1e-8              # D2 annihilates strike-linear
        assert abs(w @ nodes - 1.0) < 1e-8         # D1 is exact on strike-linear


def test_local_variance_positive_and_bounded_by_floor():
    grid = LogStrikeGrid(16)
    basis = smile_basis()
    xi = np.array([0.6, 0.5, 0.5])
    a = local_variance_surface(grid.x, basis, xi)
    assert np.all(a >= 0.01 - 1e-12)


def test_propagate_exact_matches_backward_euler_closely():
    """Exact and backward-Euler propagation of the same linear dynamics should
    agree closely over a short interval on a modest grid (Sec. II.A, Eq. 4)."""
    grid = LogStrikeGrid(32)
    basis = smile_basis()
    A0, A_list = build_operator_family(grid, basis)
    xi = np.array([0.6, 0.5, 0.5])
    A = assemble_A(A0, A_list, xi)
    v0 = payoff(grid) - boundary_lift(grid)
    v_exact = propagate_exact(v0, A, 0.25)
    v_euler = propagate_backward_euler(v0, A, 0.25, 512)
    rmse = np.sqrt(np.mean((v_exact - v_euler) ** 2))
    assert rmse < 1e-4


def test_propagate_lchs_matches_exact_on_small_grid():
    """beta-LCHS at ample J, K_c should closely track exact propagation
    (Sec. III.A, Eq. 7); verifies the congruence-based diagonalisation."""
    grid = LogStrikeGrid(16)
    basis = smile_basis()
    A0, A_list = build_operator_family(grid, basis)
    xi = np.array([0.6, 0.5, 0.5])
    A = assemble_A(A0, A_list, xi)
    theta = coefficient_independent_theta(A0, A_list)
    v0 = payoff(grid) - boundary_lift(grid)
    v_exact = propagate_exact(v0, A, 0.25)
    v_lchs = propagate_lchs(v0, A, 0.25, theta, J=4096, K_c=512.0, beta=0.8, kernel="beta")
    rmse = np.sqrt(np.mean((v_exact - v_lchs) ** 2))
    assert rmse < 1e-3


def test_theta_makes_symmetric_part_psd():
    grid = LogStrikeGrid(16)
    basis = smile_basis()
    A0, A_list = build_operator_family(grid, basis)
    theta = coefficient_independent_theta(A0, A_list)
    for xi in (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.array([0.3, 0.7, 0.2])):
        A = assemble_A(A0, A_list, xi)
        sym = 0.5 * (A + A.T) + theta * np.eye(A.shape[0])
        assert np.linalg.eigvalsh(sym).min() > 0


def test_invalid_grid_size_rejected():
    with pytest.raises(ValueError):
        LogStrikeGrid(2)
