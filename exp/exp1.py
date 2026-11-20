"""Experiment 1: synthetic surface reconstruction, smile and skew (Sec. IV.A).

For each of the two synthetic surfaces (smile, skew):

1. Generate reference ("true") prices on a 1024-node fine grid using exact
   matrix propagation of the generating coefficients (Eq. 13).
2. Extract nine irregular calibration strikes per maturity.
3. Calibrate three coefficients sequentially (one maturity interval at a
   time) on a 128-node grid using bounded least squares with exact matrix
   propagation (Sec. II.A, Eq. 5).
4. At the fitted coefficients, evaluate held-out prices with exact
   propagation, backward Euler, and beta-LCHS (J=8192, K_c=96), and compute
   RMSE in basis points against noise-free reference prices on 141 held-out
   strikes excluding the calibration strikes (Sec. IV.A).
5. Check grid convergence: reference quote prices at 1024 vs. 2048 nodes.
6. Convert fitted surfaces to implied volatility for visualisation.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.generate_synthetic import (
    generate_reference_surface,
    irregular_strikes,
    quotes_from_reference,
)
from src.calibration import calibrate_sequential, price_surface
from src.config import (
    A_MIN,
    EXP1_BACKWARD_EULER_STEPS,
    EXP1_CALIBRATION_NODES,
    EXP1_HELDOUT_STRIKES,
    EXP1_LCHS_BETA,
    EXP1_LCHS_J,
    EXP1_LCHS_KC,
    EXP1_MATURITIES,
    EXP1_N_QUOTE_STRIKES,
    EXP1_REFERENCE_NODES,
    EXP1_REFERENCE_NODES_CHECK,
    EXP1_SKEW_COEFFICIENTS,
    EXP1_SMILE_COEFFICIENTS,
    EXP1_STRIKE_RANGE,
)
from src.grid import LogStrikeGrid
from src.local_variance import build_operator_family, local_variance_surface, smile_basis
from src.pricing import implied_vol_from_price, rmse_bp
from src.propagators import (
    coefficient_independent_theta,
    propagate_backward_euler,
    propagate_exact,
    propagate_lchs,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

CASES = {
    "smile": EXP1_SMILE_COEFFICIENTS,
    "skew": EXP1_SKEW_COEFFICIENTS,
}


def _held_out_strikes(quote_strikes: np.ndarray) -> np.ndarray:
    """141 equally spaced strikes on the quoted range, excluding calibration strikes."""
    full = np.linspace(EXP1_STRIKE_RANGE[0], EXP1_STRIKE_RANGE[1], EXP1_HELDOUT_STRIKES)
    mask = np.array([np.min(np.abs(full - q)) > 1e-9 for q in quote_strikes.reshape(-1)])
    keep = np.ones(len(full), dtype=bool)
    for q in quote_strikes.reshape(-1):
        keep &= np.abs(full - q) > 1e-9
    del mask
    return full[keep]


def run_case(name: str, generating_coeffs: tuple[float, ...]) -> dict:
    basis = smile_basis()
    n_coeffs = len(basis)
    maturities = EXP1_MATURITIES
    coeffs_per_interval = np.tile(np.array(generating_coeffs), (len(maturities), 1))

    # Step 1: reference prices on the 1024-node fine grid (exact propagation).
    ref_grid, ref_prices = generate_reference_surface(
        EXP1_REFERENCE_NODES, basis, maturities, coeffs_per_interval
    )

    # Step 2: nine irregular calibration strikes per maturity (same strikes at every maturity).
    quote_strikes_1d = irregular_strikes(EXP1_STRIKE_RANGE, EXP1_N_QUOTE_STRIKES)
    quote_strikes = np.tile(quote_strikes_1d, (len(maturities), 1))
    quote_prices = quotes_from_reference(ref_grid, ref_prices, quote_strikes_1d)

    # Step 3: sequential calibration on the 128-node grid, exact propagation.
    cal_grid = LogStrikeGrid(EXP1_CALIBRATION_NODES)
    A0, A_list = build_operator_family(cal_grid, basis)
    fitted_coeffs, _ = calibrate_sequential(
        cal_grid, A0, A_list, maturities, quote_strikes, quote_prices, propagate_exact, n_coeffs
    )

    # Step 4: held-out strikes (141, excluding calibration strikes).
    held_out = _held_out_strikes(quote_strikes)
    target_prices = price_surface(
        ref_grid, *build_operator_family(ref_grid, basis), maturities,
        list(coeffs_per_interval), propagate_exact, held_out,
    )

    theta = coefficient_independent_theta(A0, A_list)
    propagators = {
        "exact": lambda v0, A, delta: propagate_exact(v0, A, delta),
        "backward_euler": lambda v0, A, delta: propagate_backward_euler(
            v0, A, delta, EXP1_BACKWARD_EULER_STEPS
        ),
        "beta_lchs": lambda v0, A, delta: propagate_lchs(
            v0, A, delta, theta, EXP1_LCHS_J, EXP1_LCHS_KC, EXP1_LCHS_BETA, kernel="beta"
        ),
    }
    held_out_rmse = {}
    fitted_prices_by_method = {}
    for method, propagate in propagators.items():
        prices = price_surface(
            cal_grid, A0, A_list, maturities, fitted_coeffs, propagate, held_out
        )
        held_out_rmse[method] = rmse_bp(prices, target_prices)
        fitted_prices_by_method[method] = prices

    # Step 5: grid convergence check (1024 vs 2048 reference nodes).
    ref_grid2, ref_prices2 = generate_reference_surface(
        EXP1_REFERENCE_NODES_CHECK, basis, maturities, coeffs_per_interval
    )
    quote_prices_2 = quotes_from_reference(ref_grid2, ref_prices2, quote_strikes_1d)
    grid_convergence_rmse_bp = rmse_bp(quote_prices, quote_prices_2)

    # Step 6: implied-volatility surface (from the exact-propagation fit).
    exact_prices = fitted_prices_by_method["exact"]
    iv_surface = np.full((len(maturities), len(held_out)), np.nan)
    for s, T_s in enumerate(maturities):
        for i, k in enumerate(held_out):
            iv_surface[s, i] = implied_vol_from_price(float(k), float(T_s), float(exact_prices[s, i]))

    return {
        "name": name,
        "fitted_coeffs": [c.tolist() for c in fitted_coeffs],
        "generating_coeffs": list(generating_coeffs),
        "held_out_strikes": held_out,
        "held_out_rmse_bp": held_out_rmse,
        "grid_convergence_rmse_bp": grid_convergence_rmse_bp,
        "iv_surface": iv_surface,
        "maturities": maturities,
    }


def _plot_surface(result: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for s, T_s in enumerate(result["maturities"]):
        ax.plot(result["held_out_strikes"], result["iv_surface"][s] * 100.0, label=f"T={T_s}")
    ax.set_xlabel("Strike k = K/F(T)")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"Calibrated implied-volatility surface: {result['name']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows = []
    for name, generating_coeffs in CASES.items():
        result = run_case(name, generating_coeffs)
        print(
            f"[{name}] generating={result['generating_coeffs']} "
            f"fitted(interval0)={result['fitted_coeffs'][0]} "
            f"held_out_rmse_bp={result['held_out_rmse_bp']} "
            f"grid_convergence_rmse_bp={result['grid_convergence_rmse_bp']:.5f}"
        )
        _plot_surface(result, os.path.join(RESULTS_DIR, f"exp1_{name}_iv_surface.png"))
        for method, rmse in result["held_out_rmse_bp"].items():
            summary_rows.append(
                {
                    "surface": name,
                    "propagator": method,
                    "held_out_rmse_bp": rmse,
                    "grid_convergence_rmse_bp": result["grid_convergence_rmse_bp"],
                }
            )
        with open(os.path.join(RESULTS_DIR, f"exp1_{name}_fitted_coefficients.json"), "w") as f:
            json.dump(
                {
                    "generating_coeffs": result["generating_coeffs"],
                    "fitted_coeffs": result["fitted_coeffs"],
                },
                f,
                indent=2,
            )

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp1_summary.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
