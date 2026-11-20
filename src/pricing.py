"""Black-formula implied volatility inversion and pricing utilities.

Normalised units throughout: forward F(T)=1, strike k = K/F(T), call prices
c(T,k) are forward (undiscounted) Black call prices divided by F(T), so that
c in [0, 1]. Price errors are reported in basis points (bp) of this
normalised call value, 1 bp = 1e-4 in c (Sec. II.A, IV.A).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

Array = np.ndarray


def black_call(k: Array, total_var: Array) -> Array:
    """Normalised forward Black call price, forward = 1, total variance w = sigma^2 * T.

    c(k, w) = N(d1) - k*N(d2), d1 = (-log(k) + w/2)/sqrt(w), d2 = d1 - sqrt(w).
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(total_var, dtype=float)
    w = np.maximum(w, 1e-14)
    sqrt_w = np.sqrt(w)
    d1 = (-np.log(k) + 0.5 * w) / sqrt_w
    d2 = d1 - sqrt_w
    return norm.cdf(d1) - k * norm.cdf(d2)


def implied_vol_from_price(k: float, T: float, price: float) -> float:
    """Invert the normalised Black formula for implied volatility via bisection.

    Returns NaN if the price is outside the arbitrage-free bounds
    [max(1-k,0), 1], flagged rather than silently clipped.
    """
    lower = max(1.0 - k, 0.0)
    if not (lower - 1e-10 <= price <= 1.0 + 1e-10):
        return float("nan")
    price = min(max(price, lower + 1e-12), 1.0 - 1e-12)

    def obj(sigma):
        return black_call(np.array([k]), np.array([sigma ** 2 * T]))[0] - price

    try:
        return brentq(obj, 1e-6, 5.0, xtol=1e-10)
    except ValueError:
        return float("nan")


def strike_interpolation_weights(x_eval: float, x_grid: Array) -> Array:
    """Linear interpolation weights on the interior log-strike grid (observation map W_s).

    Returns a sparse row of weights w such that w @ f approximates f(x_eval)
    for any f sampled on x_grid, used to build the observation map W_s/gamma_s
    mapping grid prices onto quoted strikes (Sec. II.A, III.B).
    """
    n = len(x_grid)
    w = np.zeros(n)
    if x_eval <= x_grid[0]:
        w[0] = 1.0
        return w
    if x_eval >= x_grid[-1]:
        w[-1] = 1.0
        return w
    idx = np.searchsorted(x_grid, x_eval) - 1
    idx = min(max(idx, 0), n - 2)
    x0, x1 = x_grid[idx], x_grid[idx + 1]
    t = (x_eval - x0) / (x1 - x0)
    w[idx] = 1.0 - t
    w[idx + 1] = t
    return w


def price_at_strikes(v: Array, G: Array, x_grid: Array, strikes_k: Array) -> Array:
    """Interpolate grid call prices c = v + G onto arbitrary strikes (observation map)."""
    c_grid = v + G
    x_eval = np.log(strikes_k)
    out = np.empty_like(x_eval)
    for i, xe in enumerate(x_eval):
        w = strike_interpolation_weights(xe, x_grid)
        out[i] = w @ c_grid
    return out


def rmse_bp(prices_a: Array, prices_b: Array) -> float:
    """Root-mean-square price error in basis points of normalised call value."""
    return float(np.sqrt(np.mean((prices_a - prices_b) ** 2)) * 1e4)
