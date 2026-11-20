"""Sequential bounded least-squares calibration (Sec. II.A, Eq. 5).

Fits the coefficients xi_s of one maturity interval at a time, propagating the
fitted state onward to the next interval, minimising the joint criterion

    F(xi) = (1/S) * sum_s (1/q_s) * || W_s v_s - y_s ||^2            (Eq. 5)

restricted here to one interval s at a time (sequential calibration, Sec.
II.A). W_s is the observation map (interpolation onto quoted strikes), y_s the
adjusted market observations (quote prices minus the boundary lift G at the
quoted strikes), and q_s a per-maturity normaliser.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from src.config import COEFFICIENT_LOWER_BOUND, COEFFICIENT_UPPER_BOUND
from src.grid import LogStrikeGrid, boundary_lift, payoff
from src.pricing import price_at_strikes

Array = np.ndarray
Propagator = Callable[[Array, Array, float], Array]


def calibrate_sequential(
    grid: LogStrikeGrid,
    A0: Array,
    A_list: list[Array],
    maturities: tuple[float, ...],
    quote_strikes: Array,
    quote_prices: Array,
    propagate: Propagator,
    n_coeffs: int,
    x0: Array | None = None,
) -> tuple[list[Array], list[Array]]:
    """Fit coefficients xi_s interval by interval using bounded least squares.

    Args:
        grid: log-strike grid.
        A0, A_list: operator family (Eq. 4).
        maturities: increasing maturities (T_1, ..., T_S); interval s spans
            (T_{s-1}, T_s) with T_0 = 0.
        quote_strikes: shape (S, n_quotes), calibration strikes per maturity.
        quote_prices: shape (S, n_quotes), target (noisy or clean) call prices.
        propagate: callable (v0, A, delta) -> v(delta), one of the three
            forward-propagation methods (exact, backward Euler, LCHS).
        n_coeffs: number of basis coefficients per interval.
        x0: optional initial guess per interval, shape (S, n_coeffs); defaults
            to the midpoint 0.5 for every coefficient.

    Returns:
        (fitted_coefficients, fitted_states): fitted_coefficients[s] is the
        xi vector of interval s; fitted_states[s] is v at T_s after
        propagating with the fitted coefficients (used as the initial state
        of the next interval).
    """
    from src.propagators import assemble_A

    G = boundary_lift(grid)
    v = payoff(grid) - G  # v(0) = c(0,k) - G(k), c(0,k) = (1-k)^+ (Sec. II.A)
    t_prev = 0.0
    fitted_coeffs: list[Array] = []
    fitted_states: list[Array] = []
    bounds = (
        np.full(n_coeffs, COEFFICIENT_LOWER_BOUND),
        np.full(n_coeffs, COEFFICIENT_UPPER_BOUND),
    )
    for s, T_s in enumerate(maturities):
        delta = T_s - t_prev
        y_s = quote_prices[s] - np.interp(np.log(quote_strikes[s]), grid.x, G) * 0.0
        y_s = quote_prices[s] - _boundary_at_strikes(grid, G, quote_strikes[s])
        guess = x0[s] if x0 is not None else np.full(n_coeffs, 0.5)

        def residual(xi, v0=v, delta=delta, y_s=y_s, strikes=quote_strikes[s]):
            A = assemble_A(A0, A_list, xi)
            v1 = propagate(v0, A, delta)
            model_prices = price_at_strikes(v1, G, grid.x, strikes)
            model_adjusted = model_prices - _boundary_at_strikes(grid, G, strikes)
            return model_adjusted - y_s

        result = least_squares(residual, guess, bounds=bounds, method="trf", xtol=1e-10, ftol=1e-10)
        xi_hat = result.x
        A_hat = assemble_A(A0, A_list, xi_hat)
        v = propagate(v, A_hat, delta)
        fitted_coeffs.append(xi_hat)
        fitted_states.append(v)
        t_prev = T_s
    return fitted_coeffs, fitted_states


def _boundary_at_strikes(grid: LogStrikeGrid, G: Array, strikes: Array) -> Array:
    """Boundary lift G(k) evaluated at arbitrary strikes (linear interpolation)."""
    return np.interp(np.log(strikes), grid.x, G)


def price_surface(
    grid: LogStrikeGrid,
    A0: Array,
    A_list: list[Array],
    maturities: tuple[float, ...],
    coefficients: list[Array],
    propagate: Propagator,
    eval_strikes: Array,
) -> Array:
    """Prices at eval_strikes for every maturity, given fitted (or generating) coefficients.

    Propagates sequentially from v(0) = c(0,k) - G(k), c(0,k) = (1-k)^+, using
    the supplied propagator, and evaluates at eval_strikes (same strikes for
    every maturity) via the observation map.
    """
    from src.propagators import assemble_A

    G = boundary_lift(grid)
    v = payoff(grid) - G
    t_prev = 0.0
    prices = np.empty((len(maturities), len(eval_strikes)))
    for s, T_s in enumerate(maturities):
        delta = T_s - t_prev
        A = assemble_A(A0, A_list, coefficients[s])
        v = propagate(v, A, delta)
        prices[s] = price_at_strikes(v, G, grid.x, eval_strikes)
        t_prev = T_s
    return prices
