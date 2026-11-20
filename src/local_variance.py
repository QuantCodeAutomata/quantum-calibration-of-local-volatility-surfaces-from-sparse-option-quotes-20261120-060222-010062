"""Positive affine local-variance representations and Dupire operators.

Implements the local-variance basis a_s(k; xi) = a_min + sum_j xi_j phi_j(x),
x = log(k), of Eq. 3, and assembles the Dupire forward operator A_s(xi) =
A_0 + sum_j xi_j A_j of Eq. 4, where A_0 is the a_min floor contribution and
A_j the contribution of basis function phi_j.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.config import A_MIN
from src.grid import LogStrikeGrid, build_base_operators


def smile_basis() -> list[Callable[[np.ndarray], np.ndarray]]:
    """Three-function positive quadratic basis used in Sec. IV.A, Eq. 13.

    phi_1(x) = 0.08 (constant), phi_2(x) = 0.1*(x+0.45)^2 (left wing),
    phi_3(x) = 0.1*(x-0.45)^2 (right wing).
    """
    return [
        lambda x: 0.08 * np.ones_like(x),
        lambda x: 0.1 * (x + 0.45) ** 2,
        lambda x: 0.1 * (x - 0.45) ** 2,
    ]


def four_cosine_basis() -> list[Callable[[np.ndarray], np.ndarray]]:
    """Four overlapping cosine basis functions for the benchmark model (Sec. IV.B).

    The paper states only that the model uses "four overlapping cosine
    functions of x = log(k)" without listing them (Sec. IV.B, flagged as not
    specified). Custom implementation: four raised-cosine bumps of equal width
    0.6 centred at -0.6, -0.2, 0.2, 0.6 that overlap across the domain
    [-1.2, 1.2] and are non-negative by construction, each capped at zero
    outside its support:

        phi_j(x) = 0.5 * (1 + cos(pi*(x-c_j)/w)) for |x-c_j| <= w, else 0.
    """
    centres = (-0.6, -0.2, 0.2, 0.6)
    width = 0.6

    def make(c):
        def phi(x):
            z = (x - c) / width
            out = np.where(np.abs(z) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * z)), 0.0)
            return out
        return phi

    return [make(c) for c in centres]


def two_function_basis() -> list[Callable[[np.ndarray], np.ndarray]]:
    """Two basis functions for the small-circuit instance (Sec. IV.C).

    The paper states only "two functions" without defining them (flagged as
    not specified). Custom implementation: the first two members of the
    smile basis (Eq. 13), phi_1 (constant floor bump) and phi_2 (left wing),
    reused here to keep the local-variance family consistent across
    experiments.
    """
    basis = smile_basis()
    return [basis[0], basis[1]]


def alternative_target_local_variance(x: np.ndarray) -> np.ndarray:
    """Alternative target local variance a(T,k), constant across maturities (Sec. IV.B).

    a(T,k) = 0.01 + 0.055*exp(-((x+0.1)/0.3)^2) + 0.035/(1+exp(4x)).
    """
    return 0.01 + 0.055 * np.exp(-((x + 0.1) / 0.3) ** 2) + 0.035 / (1.0 + np.exp(4.0 * x))


def build_operator_family(
    grid: LogStrikeGrid, basis: list[Callable[[np.ndarray], np.ndarray]]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Assemble A_0 (a_min floor) and {A_j} (one per basis function), Eq. 4.

    The Dupire forward equation for the normalised call price c(T,k) reads
    partial_T c = (1/2) a(T,k) k^2 partial_kk c (Eq. 1-2). Writing c = G + v
    with G the strike-linear boundary lift, v satisfies dot(v) = -A(xi) v with
    A(xi) = -(1/2) a(k;xi) k^2 D2, since D2 annihilates the strike-linear G
    exactly (Sec. II.A), leaving no forcing term.
    """
    _, D2 = build_base_operators(grid)
    k2 = grid.k ** 2
    A0 = -0.5 * (A_MIN * k2)[:, None] * D2
    A_list = [-0.5 * (phi(grid.x) * k2)[:, None] * D2 for phi in basis]
    return A0, A_list


def local_variance_surface(
    x: np.ndarray, basis: list[Callable[[np.ndarray], np.ndarray]], xi: np.ndarray
) -> np.ndarray:
    """Evaluate a(x; xi) = a_min + sum_j xi_j phi_j(x) on an array of log-strikes."""
    a = A_MIN * np.ones_like(x)
    for coeff, phi in zip(xi, basis):
        a = a + coeff * phi(x)
    return a
