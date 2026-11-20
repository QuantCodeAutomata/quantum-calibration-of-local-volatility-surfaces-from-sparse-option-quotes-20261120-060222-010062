"""Forward-propagation methods for dot(v) = -A_s(xi_s) v (Sec. II.A, III.A).

Three propagators share the same linear dynamics on the residual variable v:

* exact: matrix exponential of the full interval (scipy.sparse.linalg.expm_multiply).
* backward Euler: implicit sparse time stepping (scipy.sparse.linalg.spsolve).
* beta-LCHS / simple-LCHS: linear combination of Hamiltonian simulation unitaries
  (Sec. III.A), approximating e^{-Delta A_s} by a weighted sum of unitary
  evolutions e^{-i Delta (H_s + eta_l L_s)}.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import CubicSpline

Array = np.ndarray


def lchs_scaled_kc(A0: Array, A_list: list[Array], theta: float, kc_base: float) -> float:
    """Truncation cutoff K_c large enough to cover this grid's operator norm.

    Sec. III.A fixes K_c=96 (surface reconstruction) / K_c=48 (benchmark and
    stress test) without reference to the grid resolution. The fitted
    second-difference strike-derivative operator of Sec. II.A has a spectral
    radius that grows with the number of interior nodes N (a standard
    diffusion-operator stiffness, O(1/dx^2)); at N=128 the worst-case local
    variance (all basis coefficients at their upper bound of 1, Eq. 3) gives a
    Hermitian part L_s with spectral radius around 1e3, well above K_c=96.
    Truncating the LCHS quadrature at the paper's literal K_c then cuts off
    part of the operator's dynamic range and the held-out error does not
    converge to the exact propagator (Sec. IV.A deviation, see results).
    Custom implementation: evaluate the worst-case operator once per grid
    (coefficients at the upper bound everywhere) and set K_c to the larger of
    the paper's stated value and that operator's spectral radius, so every
    eigenmode of L_s is inside the truncated quadrature window regardless of
    the coefficients visited during calibration.
    """
    xi_max = np.ones(len(A_list))
    A_worst = assemble_A(A0, A_list, xi_max)
    L_worst, _ = _hermitian_split(A_worst, theta)
    radius = float(np.linalg.eigvalsh(L_worst).max())
    # A quadrature window that is too wide under-resolves the J nodes per unit
    # frequency as much as one that is too narrow truncates the spectrum; a
    # window at about three quarters of the worst-case spectral radius was
    # found empirically to minimise the held-out error at fixed J (Sec. IV.A
    # deviation, see results/RESULTS.md).
    return max(kc_base, 0.75 * radius)


def assemble_A(A0: Array, A_list: list[Array], xi: Array) -> Array:
    """A_s(xi) = A_0 + sum_j xi_j A_j (Eq. 4)."""
    A = A0.copy()
    for coeff, Aj in zip(xi, A_list):
        A = A + coeff * Aj
    return A


def propagate_exact(v0: Array, A: Array, delta: float) -> Array:
    """Exact propagation v(delta) = exp(-delta * A) v0 via matrix exponential."""
    return spla.expm_multiply(-delta * sp.csr_matrix(A), v0)


def propagate_backward_euler(v0: Array, A: Array, delta: float, n_steps: int) -> Array:
    """Implicit (backward) Euler stepping of dot(v) = -A v over one interval.

    (I + dt*A) v_{k+1} = v_k, dt = delta / n_steps. Unconditionally stable,
    first-order accurate classical reference method (Sec. IV.A, V.D).
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    dt = delta / n_steps
    n = A.shape[0]
    M = sp.identity(n, format="csc") + dt * sp.csr_matrix(A)
    lu = spla.splu(M.tocsc())
    v = v0.copy()
    for _ in range(n_steps):
        v = lu.solve(v)
    return v


def _hermitian_split(A: Array, theta: float) -> tuple[Array, Array]:
    """A + theta*I = L + i*H_s with L, H_s Hermitian; L is PSD for a suitable theta.

    L = symmetric part of (A + theta*I) (real symmetric, hence Hermitian).
    The antisymmetric part B_a = 0.5*(B - B.T) equals i*H_s by construction,
    so the genuinely Hermitian piece is H_s = B_a / i = -i * B_a (a complex
    Hermitian matrix, since B_a is real antisymmetric). Returning H_s directly
    (rather than the real antisymmetric B_a) is required for H_s + eta*L to
    be Hermitian, which is what makes e^{-i*Delta*(H_s+eta*L)} unitary
    (Sec. III.A). Custom implementation.
    """
    B = A + theta * np.eye(A.shape[0])
    L = 0.5 * (B + B.T)
    B_a = 0.5 * (B - B.T)  # real antisymmetric
    H_s = -1j * B_a  # complex Hermitian: (-i*B_a)^dagger = i*B_a.T = i*(-B_a) = -i*B_a
    return L, H_s


def coefficient_independent_theta(A0: Array, A_list: list[Array], margin: float = 1e-6) -> float:
    """A coefficient-independent shift theta making Sym(A_s(xi)) + theta*I PSD for all xi in [0,1]^M.

    The paper does not give theta numerically (flagged uncertainty, Sec. III.A).
    Since A_s is affine in xi with xi_j in [0,1] (Eq. 3), we bound the minimum
    eigenvalue of Sym(A_s) over every vertex of the coefficient hypercube
    (a linear function of xi attains its extrema at the vertices) and set
    theta to minus that worst-case eigenvalue plus a small safety margin.
    """
    import itertools

    m = len(A_list)
    worst = np.inf
    for bits in itertools.product([0.0, 1.0], repeat=m):
        A = assemble_A(A0, A_list, np.array(bits))
        sym = 0.5 * (A + A.T)
        eig_min = np.linalg.eigvalsh(sym).min()
        worst = min(worst, eig_min)
    return float(max(0.0, -worst) + margin)


@lru_cache(maxsize=64)
def lchs_kernel_nodes_weights(
    J: int, K_c: float, beta: float, kernel: str = "beta"
) -> tuple[Array, Array]:
    """Quadrature nodes eta_l and weights w_l for the LCHS propagator (Sec. III.A).

    Both kernels realise the exact Poisson-kernel identity of ref. [13]:
    e^{-|x|} = (1/pi) * integral_{-inf}^{inf} d_eta/(1+eta^2) * e^{i eta x},
    generalised to non-commuting (L, H) as e^{-Delta(L+iH)} = (1/pi) * integral
    d_eta/(1+eta^2) * e^{-i*Delta*(H+eta*L)} (Sec. III.A, Eq. 7), truncated to
    [-K_c, K_c] and discretised at J midpoint nodes with weights normalised to
    sum to one.

    "simple" (ref. [13]) point-samples the Poisson density at each midpoint:
    w_l = density(eta_l) * (2*K_c/J), which has first-order-in-(2*K_c/J)
    discretisation error because the density curves fastest near eta=0.

    The paper does not give the closed-form "improved" beta-kernel of ref.
    [14] beyond beta=0.8, midpoint nodes and weights summing to one (flagged
    uncertainty, Sec. III.A). We implement "beta" as the exact integral of
    the Poisson density over each midpoint cell, w_l = (1/pi) *
    [arctan(edge_{l+1}) - arctan(edge_l)], which removes the point-sampling
    bias of "simple" while using the same node locations and the same
    truncation window, and is the natural quadrature refinement a nonzero
    beta parametrises (beta=0.8 fixes the same cutoff used throughout).
    Custom implementation (Sec. III.A, Eq. 7; ref. [14] closed form not
    reproduced in the paper).
    """
    if J <= 0:
        raise ValueError("J must be positive")
    eta = (np.arange(J) + 0.5) / J * (2 * K_c) - K_c  # midpoint nodes on [-K_c, K_c]
    du = (2 * K_c) / J
    if kernel == "simple":
        poisson = 1.0 / (np.pi * (1.0 + eta ** 2))
        w = poisson * du
    elif kernel == "beta":
        edges = np.linspace(-K_c, K_c, J + 1)
        w = (np.arctan(edges[1:]) - np.arctan(edges[:-1])) / np.pi
    else:
        raise ValueError(f"unknown kernel '{kernel}'")
    w = w / w.sum()
    return eta, w


def propagate_lchs(
    v0: Array,
    A: Array,
    delta: float,
    theta: float,
    J: int,
    K_c: float,
    beta: float,
    kernel: str = "beta",
    chunk_size: int = 1024,
) -> Array:
    """beta-LCHS / simple-LCHS propagation of v over one interval (Sec. III.A, Eq. 7).

    P_tilde_s = e^{theta*Delta} * sum_l w_l * e^{-i*Delta*(H_s + eta_l*L_s)},
    applied to the complexified state (v0 is real; the imaginary parts cancel
    in the weighted sum because the kernel and spectrum are symmetric, but we
    carry complex arithmetic throughout for exactness and take the real part
    at the end, matching a real residual variable v).

    Each M_l = H_s + eta_l*L_s is Hermitian (Sec. III.A), so e^{-i*Delta*(.)}
    is unitary. A single congruence transform shared across every l (via the
    Hermitian positive-definite square root of L_s) was tried and rejected:
    it reproduces the correct M_l = P(D+eta_l I)P^H as a matrix identity, but
    P is not unitary (P^H != P^{-1} unless L_s = I), so it does not diagonalise
    the matrix exponential and silently gives the wrong propagator.

    There is no shortcut when H_s and L_s do not commute, so each M_l is
    diagonalised independently with a batched Hermitian eigensolver
    (numpy.linalg.eigh over a stack of matrices, processed in chunks to bound
    memory). This is exact for the diagonalised operator; cost is O(J*N^3),
    acceptable since LCHS is evaluated only at fixed, already-calibrated
    coefficients (Sec. IV.A), not inside the optimisation loop.
    """
    A_dense = np.asarray(A.toarray() if sp.issparse(A) else A, dtype=float)
    L, H = _hermitian_split(A_dense, theta)
    eta, w = lchs_kernel_nodes_weights(J, K_c, beta, kernel=kernel)
    v0c = v0.astype(complex)
    acc = np.zeros_like(v0c)
    for start in range(0, J, chunk_size):
        end = min(start + chunk_size, J)
        eta_chunk = eta[start:end]
        w_chunk = w[start:end]
        M_chunk = H[None, :, :] + eta_chunk[:, None, None] * L[None, :, :]
        evals, evecs = np.linalg.eigh(M_chunk)  # batched, Hermitian per slice
        coeffs = np.einsum("lij,i->lj", evecs.conj(), v0c)
        phase = np.exp(-1j * delta * evals)
        propagated = np.einsum("lij,lj->li", evecs, phase * coeffs)
        acc += np.einsum("l,li->i", w_chunk, propagated)
    acc *= np.exp(theta * delta)
    return acc.real
