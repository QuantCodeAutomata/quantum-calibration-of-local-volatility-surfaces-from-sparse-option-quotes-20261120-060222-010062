"""Physics-informed neural network (PINN) price-surface comparators (paper ref. [3]).

Two small comparators are provided, "fixed" and "adaptive", that both directly
model the normalised forward call-price surface C_theta(T, x) with a small
feed-forward MLP, where x = log(k) is the log-strike. Training minimises a
composite loss combining:

  1. Data misfit against the quoted call prices.
  2. A Dupire PDE residual `dC/dT - 0.5 * a_min * k^2 * d2C/dk2`, evaluated at
     random collocation points, using the constant local-variance floor
     `a_min` as a simple proxy for the full local-variance surface. The paper
     does not specify a PINN architecture or loss design for this comparator,
     so the following choices are simplifications made here:
       - Only the floor `a_min` of the local variance is enforced through the
         PDE residual; the nonparametric local-variance surface a(T,k;xi) of
         the paper's own model is not separately learned or embedded.
       - The loss weights (1.0 data, 0.1 PDE residual, 0.1 shape penalties)
         are modest, fixed values chosen for numerical balance, not values
         taken from the paper.
       - Collocation points are redrawn uniformly at every epoch (rather than
         a single fixed batch reused throughout training) to avoid overfitting
         to a particular sample; this does not change the per-epoch cost.
  3. Shape (no-arbitrage) penalties on the collocation points: strike
     monotonicity (call price non-increasing in strike, slope >= -1), strike
     convexity, and calendar monotonicity (call price non-decreasing in
     maturity at fixed strike), each penalised only in the direction of
     violation via a squared ReLU hinge.

The "fixed" variant uses a small, fixed architecture (2 hidden layers of 32
units) trained for a fixed 300 epochs of full-batch Adam at a constant
learning rate. The "adaptive" variant realises "adaptive" as a change to both
the training schedule and the architecture (the paper does not specify PINN
hyperparameters so this is a documented choice, not a literature value): a
slightly larger network (3 hidden layers of 48 units), up to 600 epochs,
`ReduceLROnPlateau` halving the learning rate when the loss has not improved
by more than 1% over the last 50 epochs, and early stopping once the
epoch-over-epoch loss change drops below 1e-7.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn

Array = np.ndarray

_N_COLLOCATION = 200
_DATA_LOSS_WEIGHT = 1.0
_PDE_LOSS_WEIGHT = 0.1
_SHAPE_LOSS_WEIGHT = 0.1
_LEARNING_RATE = 1e-3


class _PriceNet(nn.Module):
    """Small MLP for C_theta(T, x), x = log(k), with smooth (tanh) activations
    so that first and second derivatives w.r.t. the inputs exist everywhere."""

    def __init__(self, n_hidden_layers: int, hidden_width: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden_width), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
        layers.append(nn.Linear(hidden_width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, T: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        inp = torch.stack([T, x], dim=-1)
        return self.net(inp).squeeze(-1)


def _variant_hyperparameters(variant: str) -> tuple[int, int, int]:
    if variant == "fixed":
        return 2, 32, 300
    if variant == "adaptive":
        return 3, 48, 600
    raise ValueError(f"variant must be 'fixed' or 'adaptive', got {variant!r}")


def calibrate_pinn(
    maturities: tuple[float, ...],
    quote_strikes: Array,
    quote_prices: Array,
    a_min: float,
    log_strike_domain: tuple[float, float],
    variant: str = "fixed",
    seed: int = 0,
) -> dict:
    """Train a small PINN comparator for the normalised call-price surface.

    Args:
        maturities: Strictly increasing maturities `T_1 < ... < T_S`.
        quote_strikes: Quoted strikes, shape `(S, n_quotes)`.
        quote_prices: Quoted normalised call prices, shape `(S, n_quotes)`.
        a_min: Local-variance floor used in the Dupire PDE residual.
        log_strike_domain: `(x_lo, x_hi)` bounds on log-strike `x = log(k)`
            used to sample PDE/shape collocation points.
        variant: "fixed" (small network, 300 epochs, constant LR) or
            "adaptive" (larger network, up to 600 epochs, LR scheduling and
            early stopping).
        seed: Seed for `torch.manual_seed`, for reproducibility.

    Returns:
        Dict with keys:
          - 'price_fn': Callable[[int, np.ndarray], np.ndarray], `price_fn(s, strikes)`
            returns predicted call prices for maturity index `s` at arbitrary
            strikes, clipped into `[0, 1]`.
          - 'final_loss': float, the final total training loss.

    Raises:
        ValueError: if `variant` is not "fixed"/"adaptive", or if
            `quote_strikes.shape != quote_prices.shape`, or if
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
    n_hidden_layers, hidden_width, n_epochs = _variant_hyperparameters(variant)

    torch.manual_seed(seed)
    net = _PriceNet(n_hidden_layers, hidden_width)

    n_maturities = quote_strikes.shape[0]
    T_data = torch.tensor(
        np.concatenate(
            [np.full(quote_strikes.shape[1], maturities[s], dtype=float) for s in range(n_maturities)]
        ),
        dtype=torch.float32,
    )
    x_data = torch.tensor(np.log(quote_strikes).reshape(-1), dtype=torch.float32)
    c_data = torch.tensor(quote_prices.reshape(-1), dtype=torch.float32)

    max_T = float(max(maturities))
    x_lo, x_hi = float(log_strike_domain[0]), float(log_strike_domain[1])

    optimizer = torch.optim.Adam(net.parameters(), lr=_LEARNING_RATE)
    scheduler = None
    if variant == "adaptive":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=50, threshold=0.01, threshold_mode="rel"
        )

    loss_history: list[float] = []
    final_loss = float("nan")

    for _epoch in range(n_epochs):
        optimizer.zero_grad()

        pred_data = net(T_data, x_data)
        data_loss = torch.mean((pred_data - c_data) ** 2)

        T_c = torch.rand(_N_COLLOCATION) * max_T
        x_c = x_lo + torch.rand(_N_COLLOCATION) * (x_hi - x_lo)
        T_c.requires_grad_(True)
        x_c.requires_grad_(True)
        k_c = torch.exp(x_c)

        C_c = net(T_c, x_c)
        ones = torch.ones_like(C_c)
        dC_dT = torch.autograd.grad(C_c, T_c, grad_outputs=ones, create_graph=True)[0]
        dC_dx = torch.autograd.grad(C_c, x_c, grad_outputs=ones, create_graph=True)[0]
        d2C_dx2 = torch.autograd.grad(dC_dx, x_c, grad_outputs=torch.ones_like(dC_dx), create_graph=True)[0]

        # x = log(k): dC/dk = (dC/dx) / k,  d2C/dk2 = (d2C/dx2 - dC/dx) / k^2.
        dC_dk = dC_dx / k_c
        d2C_dk2 = (d2C_dx2 - dC_dx) / (k_c ** 2)

        pde_residual = dC_dT - 0.5 * a_min * (k_c ** 2) * d2C_dk2
        pde_loss = torch.mean(pde_residual ** 2)

        mono_loss = torch.mean(torch.relu(dC_dk) ** 2 + torch.relu(-1.0 - dC_dk) ** 2)
        convex_loss = torch.mean(torch.relu(-d2C_dk2) ** 2)

        T1 = torch.rand(_N_COLLOCATION) * max_T
        T2 = torch.rand(_N_COLLOCATION) * max_T
        T_lo = torch.minimum(T1, T2)
        T_hi = torch.maximum(T1, T2)
        x_cal = x_lo + torch.rand(_N_COLLOCATION) * (x_hi - x_lo)
        C_lo = net(T_lo, x_cal)
        C_hi = net(T_hi, x_cal)
        calendar_loss = torch.mean(torch.relu(C_lo - C_hi) ** 2)

        shape_loss = mono_loss + convex_loss + calendar_loss

        total_loss = (
            _DATA_LOSS_WEIGHT * data_loss + _PDE_LOSS_WEIGHT * pde_loss + _SHAPE_LOSS_WEIGHT * shape_loss
        )

        total_loss.backward()
        optimizer.step()

        loss_val = float(total_loss.item())
        loss_history.append(loss_val)
        final_loss = loss_val

        if variant == "adaptive":
            assert scheduler is not None
            scheduler.step(loss_val)
            if len(loss_history) >= 2 and abs(loss_history[-1] - loss_history[-2]) < 1e-7:
                break

    def price_fn(s: int, strikes: Array) -> Array:
        strikes = np.asarray(strikes, dtype=float)
        shape = strikes.shape
        flat = strikes.reshape(-1)
        T_val = float(maturities[s])
        with torch.no_grad():
            T_t = torch.full((flat.shape[0],), T_val, dtype=torch.float32)
            x_t = torch.tensor(np.log(flat), dtype=torch.float32)
            out = net(T_t, x_t).numpy()
        out = np.clip(out, 0.0, 1.0)
        return out.reshape(shape)

    return {"price_fn": price_fn, "final_loss": final_loss}
