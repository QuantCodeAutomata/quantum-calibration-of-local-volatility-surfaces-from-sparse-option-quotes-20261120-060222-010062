"""Synthetic option-quote generation shared by every experiment.

All experiments in this repository calibrate to internally generated option
quotes rather than external market data (Sec. IV.A-C): a positive local-
variance surface with known generating coefficients is propagated forward on
a fine reference grid to produce "true" prices, from which a handful of
irregular quoted strikes are extracted as calibration targets, optionally
perturbed by bounded uniform noise (Sec. IV.B).
"""

from __future__ import annotations

import numpy as np

from src.calibration import price_surface
from src.grid import LogStrikeGrid
from src.local_variance import build_operator_family
from src.propagators import propagate_exact

Array = np.ndarray


def irregular_strikes(strike_range: tuple[float, float], n_quotes: int, seed: int = 0) -> Array:
    """Nine (or n_quotes) irregular strikes between strike_range, denser near the money.

    The paper specifies the count and the overall range [0.7, 1.4] (Sec.
    IV.A) but not the exact locations (flagged uncertainty). We space them by
    a cosine (Chebyshev-like) rule in log-strike, which concentrates more
    strikes near the money as is standard for option quote grids, and is
    fully deterministic given n_quotes and the range (seed kept for API
    symmetry with noise generation, unused here).
    """
    del seed
    lo, hi = np.log(strike_range[0]), np.log(strike_range[1])
    j = np.arange(n_quotes)
    # Chebyshev nodes on [lo, hi]: cluster near the ends; reflect to cluster
    # near the centre instead (near-the-money) by using cos over a half-period.
    u = 0.5 * (1.0 - np.cos(np.pi * j / (n_quotes - 1)))
    x = lo + u * (hi - lo)
    return np.exp(np.sort(x))


def generate_reference_surface(
    n_reference_nodes: int,
    basis,
    maturities: tuple[float, ...],
    coefficients_per_interval: Array,
) -> tuple[LogStrikeGrid, Array]:
    """Exact propagation of the generating coefficients on a fine reference grid.

    Returns the reference grid and the fitted A0/A_list-propagated call
    prices (interior nodes, shape (S, n_reference_nodes)) used both as
    calibration targets (at quoted strikes) and as the noise-free target for
    held-out RMSE evaluation.
    """
    grid = LogStrikeGrid(n_reference_nodes)
    A0, A_list = build_operator_family(grid, basis)
    G_prices = price_surface(
        grid, A0, A_list, maturities, list(coefficients_per_interval), propagate_exact, grid.k
    )
    return grid, G_prices


def quotes_from_reference(
    grid: LogStrikeGrid,
    reference_prices: Array,
    quote_strikes: Array,
) -> Array:
    """Extract clean quote prices at quote_strikes by interpolating the reference surface."""
    out = np.empty((reference_prices.shape[0], len(quote_strikes)))
    for s in range(reference_prices.shape[0]):
        out[s] = np.interp(np.log(quote_strikes), grid.x, reference_prices[s])
    return out


def add_uniform_noise(prices: Array, bound_bp: float, seed: int) -> Array:
    """Uniform price perturbation bounded by bound_bp basis points (Sec. IV.B).

    Each quote is perturbed independently by U(-bound_bp, bound_bp) * 1e-4 in
    normalised call value, then clipped to stay non-negative.
    """
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-bound_bp, bound_bp, size=prices.shape) * 1e-4
    return np.clip(prices + noise, 0.0, 1.0)
