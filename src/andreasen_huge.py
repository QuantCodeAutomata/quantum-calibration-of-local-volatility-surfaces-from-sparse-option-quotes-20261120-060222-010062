"""Andreasen-Huge arbitrage-free price interpolation (paper ref. [2]); shape-preserving
PCHIP interpolation used as the practical arbitrage-respecting proxy since the paper
does not give the authors' exact spline construction.

For each maturity, the quoted call prices are interpolated directly in strike space
(rather than through a basis-function local-variance model) using
`scipy.interpolate.PchipInterpolator`: a shape-preserving cubic Hermite interpolant
that does not overshoot and preserves the monotonicity of the input data. This is a
reasonable, fully self-contained proxy for an arbitrage-free spline when no
closed-form Andreasen-Huge solver is available. Outputs are additionally clipped
into the model-free arbitrage band c(T,k) in [max(1-k,0), 1], and extrapolation
beyond the quoted strike range is held flat at the boundary quote rather than
extrapolating the spline. If the PCHIP construction fails for a maturity (quotes
not strictly increasing in strike, or quotes that violate the arbitrage band), a
plain linear (`numpy.interp`) fallback is used for that maturity only.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator

Array = np.ndarray


def _arbitrage_lower_bound(k: Array) -> Array:
    return np.maximum(1.0 - k, 0.0)


def _make_pchip_price_fn(k_sorted: Array, c_sorted: Array) -> Callable[[Array], Array]:
    interpolant = PchipInterpolator(k_sorted, c_sorted, extrapolate=False)
    k_min, k_max = k_sorted[0], k_sorted[-1]
    c_min_boundary, c_max_boundary = c_sorted[0], c_sorted[-1]

    def price_fn(strikes: Array) -> Array:
        strikes = np.asarray(strikes, dtype=float)
        clipped_k = np.clip(strikes, k_min, k_max)
        prices = np.asarray(interpolant(clipped_k), dtype=float)
        # Hold flat at the boundary quote for any extrapolated strike.
        prices = np.where(strikes < k_min, c_min_boundary, prices)
        prices = np.where(strikes > k_max, c_max_boundary, prices)
        # Guard against stray NaNs from evaluating exactly at the domain edge
        # (extrapolate=False) due to floating-point rounding of the clip above.
        prices = np.where(np.isnan(prices), np.where(strikes <= k_min, c_min_boundary, c_max_boundary), prices)
        lower = _arbitrage_lower_bound(strikes)
        return np.clip(prices, lower, 1.0)

    return price_fn


def _make_linear_fallback_fn(k_sorted: Array, c_sorted: Array) -> Callable[[Array], Array]:
    def price_fn(strikes: Array) -> Array:
        strikes = np.asarray(strikes, dtype=float)
        prices = np.interp(strikes, k_sorted, c_sorted)
        lower = _arbitrage_lower_bound(strikes)
        return np.clip(prices, lower, 1.0)

    return price_fn


def calibrate_andreasen_huge_surface(
    maturities: tuple[float, ...],
    quote_strikes: Array,
    quote_prices: Array,
) -> dict:
    """Build a per-maturity arbitrage-respecting price interpolant from quotes.

    For each maturity `s`, a shape-preserving PCHIP interpolant is fit to
    `(quote_strikes[s], quote_prices[s])`. If this fails (strikes not strictly
    increasing once sorted, or quotes violating the arbitrage band
    `[max(1-k,0), 1]`), a linear (`numpy.interp`) interpolant is used instead for
    that maturity. In both cases, predictions are clipped to the arbitrage band
    and extrapolation beyond the quoted strike range is held flat.

    Args:
        maturities: Strictly increasing maturities `T_1 < ... < T_S`.
        quote_strikes: Quoted strikes, shape `(S, n_quotes)`.
        quote_prices: Quoted normalised call prices, shape `(S, n_quotes)`.

    Returns:
        Dict with keys:
          - 'price_fn': Callable[[int, np.ndarray], np.ndarray], `price_fn(s, strikes)`
            returns predicted call prices for maturity index `s` at arbitrary strikes.
          - 'success': list[bool] of length `S`, True iff the PCHIP interpolant was
            built successfully for maturity `s` (False means the linear fallback
            was used).

    Raises:
        ValueError: if `quote_strikes.shape != quote_prices.shape` or if
            `len(maturities) != quote_strikes.shape[0]`.
    """
    quote_strikes = np.asarray(quote_strikes, dtype=float)
    quote_prices = np.asarray(quote_prices, dtype=float)
    if quote_strikes.shape != quote_prices.shape:
        raise ValueError(
            f"quote_strikes.shape {quote_strikes.shape} != quote_prices.shape {quote_prices.shape}"
        )
    if len(maturities) != quote_strikes.shape[0]:
        raise ValueError(
            f"len(maturities)={len(maturities)} != quote_strikes.shape[0]={quote_strikes.shape[0]}"
        )

    n_maturities = quote_strikes.shape[0]
    per_maturity_fn: list[Callable[[Array], Array]] = []
    success: list[bool] = []

    for s in range(n_maturities):
        k = quote_strikes[s]
        c = quote_prices[s]
        order = np.argsort(k)
        k_sorted = k[order]
        c_sorted = c[order]

        try:
            if np.any(np.diff(k_sorted) <= 0):
                raise ValueError("quoted strikes are not strictly increasing")
            lower_bound = _arbitrage_lower_bound(k_sorted)
            if np.any(c_sorted < lower_bound - 1e-8) or np.any(c_sorted > 1.0 + 1e-8):
                raise ValueError("quoted prices violate the arbitrage band")
            per_maturity_fn.append(_make_pchip_price_fn(k_sorted, c_sorted))
            success.append(True)
        except Exception:
            per_maturity_fn.append(_make_linear_fallback_fn(k_sorted, c_sorted))
            success.append(False)

    def price_fn(s: int, strikes: Array) -> Array:
        return per_maturity_fn[s](strikes)

    return {"price_fn": price_fn, "success": success}
