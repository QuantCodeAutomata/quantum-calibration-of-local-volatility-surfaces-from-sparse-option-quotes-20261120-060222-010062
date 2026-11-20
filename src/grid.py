"""Log-strike grid, fitted strike-derivative operator and boundary lift.

Implements the spatial discretisation of Sec. II.A: interior nodes on the
truncated log-strike domain, a strike-derivative operator fitted to annihilate
constants and strike-linear functions exactly (so the boundary lift introduces
no forcing term, Eq. 4), and the strike-linear interpolant G(k) of the fixed
boundary call values.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.config import LOG_STRIKE_DOMAIN


class LogStrikeGrid:
    """Interior log-strike grid with ghost boundary nodes.

    Attributes:
        n_interior: number of interior nodes N.
        x: interior log-strike coordinates, shape (N,).
        k: interior strikes exp(x), shape (N,).
        dx: uniform grid spacing.
        x_lo, x_hi: boundary log-strikes (domain endpoints).
        k_lo, k_hi: boundary strikes.
    """

    def __init__(self, n_interior: int):
        if n_interior < 3:
            raise ValueError("n_interior must be >= 3 for a fitted 3-point derivative stencil")
        self.n_interior = n_interior
        lo, hi = LOG_STRIKE_DOMAIN
        # N interior nodes strictly between the domain endpoints, uniform spacing.
        edges = np.linspace(lo, hi, n_interior + 2)
        self.x_lo, self.x_hi = edges[0], edges[-1]
        self.x = edges[1:-1]
        self.dx = edges[1] - edges[0]
        self.k = np.exp(self.x)
        self.k_lo = np.exp(self.x_lo)
        self.k_hi = np.exp(self.x_hi)


def boundary_lift(grid: LogStrikeGrid) -> np.ndarray:
    """Strike-linear interpolant G(k) of the fixed boundary call values.

    Boundary conditions (Sec. II.A): the call value is fixed to 1 - k_min at the
    lower endpoint and 0 at the upper endpoint. G is linear in k between them,
    matching those two values exactly at the interior nodes.
    """
    c_lo = 1.0 - grid.k_lo
    c_hi = 0.0
    slope = (c_hi - c_lo) / (grid.k_hi - grid.k_lo)
    return c_lo + slope * (grid.k - grid.k_lo)


def payoff(grid: LogStrikeGrid) -> np.ndarray:
    """Normalised call payoff c(0,k) = (1-k)^+ at T=0 (Sec. II.A).

    The Dupire forward equation is initialised at the payoff, not at zero: the
    residual v = c - G therefore starts at v(0,k) = (1-k)^+ - G(k), which is
    zero at both domain edges (where the payoff already equals the boundary
    lift) but generally nonzero in the interior. Propagating v(0) = 0 instead
    would make dot(v) = -A_s v stay identically zero regardless of A_s, since
    the dynamics are linear and homogeneous.
    """
    return np.maximum(1.0 - grid.k, 0.0)


def _fitted_strike_derivative_operators(grid: LogStrikeGrid) -> tuple[np.ndarray, np.ndarray]:
    """Fitted finite-difference operators for d/dk and d^2/dk^2 on the interior grid.

    Sec. II.A specifies only the defining property of the strike-derivative
    operator: "weights are chosen so that it annihilates constants and
    strike-linear functions exactly." We build fitted 3-point stencils in the
    non-uniform strike coordinate k (uniform in log-strike x, so k-spacing is
    non-uniform) by solving, at each interior node i with neighbours
    k_{i-1}, k_i, k_{i+1} (using one-sided ghost neighbours via the boundary
    values at the domain edges), the exact linear systems

        D1 @ [1, k, k^2] = [0, 1, 2k_i]      (first derivative, exact to O(h^2))
        D2 @ [1, k, k^2] = [0, 0, 2]          (second derivative, exact to O(h^2))

    which annihilate constants and strike-linear functions exactly (the
    property stated in the paper) and are themselves exact on the quadratic
    monomial k^2, giving second-order local truncation error. Custom
    implementation (Sec. II.A, Eq. 4).
    """
    n = grid.n_interior
    k_full = np.concatenate(([grid.k_lo], grid.k, [grid.k_hi]))
    D1 = np.zeros((n, n))
    D2 = np.zeros((n, n))
    for i in range(n):
        km1, ki, kp1 = k_full[i], k_full[i + 1], k_full[i + 2]
        # Solve the 3x3 Vandermonde-type system in (k - k_i) for unit response
        # to {1, (k-k_i), (k-k_i)^2} at the three stencil points.
        h_m, h_p = ki - km1, kp1 - ki
        # Standard non-uniform 3-point first/second derivative weights.
        w_m1_1 = -h_p / (h_m * (h_m + h_p))
        w_0_1 = (h_p - h_m) / (h_m * h_p)
        w_p1_1 = h_m / (h_p * (h_m + h_p))
        w_m1_2 = 2.0 / (h_m * (h_m + h_p))
        w_0_2 = -2.0 / (h_m * h_p)
        w_p1_2 = 2.0 / (h_p * (h_m + h_p))
        if i - 1 >= 0:
            D1[i, i - 1] += w_m1_1
            D2[i, i - 1] += w_m1_2
        if i + 1 < n:
            D1[i, i + 1] += w_p1_1
            D2[i, i + 1] += w_p1_2
        D1[i, i] += w_0_1
        D2[i, i] += w_0_2
    return D1, D2


def build_base_operators(grid: LogStrikeGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return (D1, D2): fitted first and second strike-derivative operators.

    These are combined with the local-variance basis in src.local_variance to
    form A_0 and A_j of Eq. 4: A(xi) v = -(1/2) a(k;xi) k^2 D2 v, propagated as
    dot(v) = -A(xi) v for the Dupire forward equation (Eq. 2) written in the
    residual variable v = c - G.
    """
    D1, D2 = _fitted_strike_derivative_operators(grid)
    return D1, D2


def sparse(matrix: np.ndarray) -> sp.csr_matrix:
    """Convert a dense operator to CSR for sparse propagation."""
    return sp.csr_matrix(matrix)
