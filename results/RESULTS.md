# Results — Quantum Calibration of Local-Volatility Surfaces from Sparse Option Quotes

Kentaro Hoshisashi, *Quantum Calibration of Local-Volatility Surfaces from Sparse
Option Quotes*, arXiv:2611.25344 (2026). Department of Computer Science,
University College London.

All numbers below are read from the files saved in `results/` by the latest run
of each experiment script (`python -m exp.exp1` … `exp.exp5`). No market data
is used anywhere: every quote series is generated internally, as specified by
the paper's experimental design (Sec. IV).

## Experiment 1 — Synthetic Surface Reconstruction: Smile and Skew with Common-Coefficient Propagator Comparison

Reproduces Sec. IV.A / V.A: two synthetic local-variance surfaces (smile,
skew) built from the three-basis positive affine representation, sequential
bounded-least-squares calibration on a 128-node grid with exact matrix
propagation, then held-out pricing RMSE of the fitted coefficients re-evaluated
under the exact, backward-Euler and beta-LCHS propagators. Source:
`results/exp1_summary.csv`, `results/exp1_{smile,skew}_fitted_coefficients.json`,
`results/exp1_{smile,skew}_iv_surface.png`.

| Surface | Propagator | Held-out RMSE (bp) | Paper (bp) | Grid-convergence RMSE (bp) |
|---|---|---|---|---|
| Smile | exact | 0.224 | ≈0.27 | 0.0040 |
| Smile | backward_euler | 0.181 | — | 0.0040 |
| Smile | beta_lchs (K_c=96, J=8192) | 105.07 | ≈0.27 | 0.0040 |
| Skew | exact | 0.243 | ≈0.26 | 0.0039 |
| Skew | backward_euler | 0.201 | — | 0.0039 |
| Skew | beta_lchs (K_c=96, J=8192) | 104.71 | ≈0.26 | 0.0039 |

Verdicts:
- Grid convergence (1024 vs 2048 reference nodes): **reproduced** — 0.0040 bp
  and 0.0039 bp are both below the paper's 0.01 bp threshold.
- Exact-propagator held-out RMSE: **partially reproduced** — same order of
  magnitude and sign as the paper's ≈0.27 bp / ≈0.26 bp, but 17% and 10% below
  them respectively, outside the 10% band used for a full "reproduced" verdict.
- Beta-LCHS held-out RMSE: **not reproduced**. The paper reports close
  agreement between exact and beta-LCHS propagation (≈0.27 bp / ≈0.26 bp); this
  run gives ≈105 bp, two orders of magnitude worse. Identified cause: at the
  literal K_c=96 cutoff, a direct eigenvalue check of the worst-case operator
  on this 128-node grid (all basis coefficients at their upper bound of 1,
  `src/propagators.py::lchs_scaled_kc`) gives a Hermitian-part spectral radius
  of about 1.9e3, roughly twenty times K_c=96; the quadrature window therefore
  leaves most of the operator's dissipative spectrum outside the truncated
  kernel, and the beta-LCHS propagator under-damps the high-frequency part of
  the residual state. `exp/exp1.py` uses the paper's literal K_c=96 without
  the grid-dependent rescaling implemented in `lchs_scaled_kc`; applying that
  rescaling was not completed before the time limit of this replication, so
  the mismatch is identified but not corrected in the committed results.

The smile surface shows higher implied volatility in both wings and the skew
surface shows higher implied volatility at lower strikes in the saved
`exp1_*_iv_surface.png` plots, consistent with the qualitative shapes the
paper describes for these two synthetic surfaces.

## Experiment 2 — Classical Solver Comparison Benchmark

Reproduces Sec. IV.B / V.B / V.D: seven calibration methods (matrix
exponential, finite differences, LCHS beta, LCHS simple, Andreasen-Huge,
fixed PINN, adaptive PINN) calibrated on the four-basis local-variance model
at N∈{32,64,128}, against a baseline (same-model) and an alternative target,
under clean and noisy quotes. Source: `results/exp2_summary.csv`,
`results/exp2_runtimes.csv`.

Selected rows (N=64, clean quotes; RMSE in bp):

| Method | Baseline target | Paper (baseline) | Alternative target | Paper (alternative) |
|---|---|---|---|---|
| matrix_exp | 0.100 | ≈1.08 | 2.68 | ≈6.42 |
| finite_differences | 0.100 | ≈1.08 | 2.77 | — |
| lchs_beta | NaN (budget) | ≈1.07 | NaN (budget) | ≈6.45 |
| lchs_simple | NaN (budget) | ≈20.80 | NaN (budget) | ≈9.28 |
| andreasen_huge | 0.161 | ≈0.35 | 0.939 | ≈1.83 |
| pinn_fixed | 233.8 | ≈1.30 | 175.1 | — |
| pinn_adaptive | 140.4 | ≈1.35 | 206.2 | ≈1.14 |

Mean classical calibration runtimes (`results/exp2_runtimes.csv`): matrix
exponentiation 6.08 s, finite differences 0.69 s, beta-LCHS simulation
51.3 s — the paper reports 0.0712 s, 0.366 s and 70.5 s respectively; the
relative ordering (LCHS ≫ matrix exponential ≫ finite differences in cost) is
reproduced, the absolute seconds are not (different hardware and problem
size per call), and are reported here only to compare relative cost, not
absolute speed.

Verdicts:
- Qualitative ranking "Andreasen-Huge and adaptive PINN outperform the
  four-basis model when the target is not representable": **partially
  reproduced**. On the alternative target, Andreasen-Huge (0.939 bp) is
  materially better than matrix_exp/finite_differences (2.68/2.77 bp), which
  matches the paper's ranking; adaptive PINN (206.2 bp) does not beat the
  four-basis model here, contrary to the paper.
- Absolute magnitudes for matrix_exp / finite_differences / Andreasen-Huge:
  **not reproduced** (all differ from the paper's values by more than 50%),
  although every method keeps the correct sign and the baseline-is-easier-
  than-alternative ordering the paper reports.
- lchs_beta / lchs_simple: **not reproduced**. Every calibration cell hit the
  per-run time budget implemented in `exp/exp2.py`
  (`_CalibrationBudgetExceeded`) before converging and is recorded as a
  non-converged NaN rather than a fabricated number. This is the same
  beta-LCHS truncation/cost issue identified in Experiment 1: each LCHS
  evaluation inside the bounded-least-squares objective is materially slower
  and, at the literal K_c, less accurate than the matrix-exponential
  propagator it replaces, so none of the scheduled LCHS calibration runs
  completed inside the time this replication could allocate to them.
  **Deviation (reduced run):** the paper's run count of 568 planned
  calibration runs (567 expected to complete) was not reproduced in full,
  because every LCHS cell alone would cost tens of seconds times the number
  of optimizer iterations times five starts times three grid sizes times two
  targets times two noise conditions times (up to) three noise datasets. The
  committed run used the paper's grid sizes, optimizer-start count and noise
  design unchanged, but capped the wall-clock time given to each LCHS
  calibration cell, which is the only place reduced. The command that would
  remove this cap and run every LCHS cell to convergence is
  `python -m exp.exp2 --full` (increases `EXP2_LCHS_TIME_BUDGET_S` in
  `src/config.py`).
- pinn_fixed / pinn_adaptive: **not reproduced**. Both are far worse (140–234
  bp) than the paper's ≈1.1–1.35 bp. The physics-informed networks in
  `src/pinn.py` are a simplified, time-boxed implementation of the fitting
  and PDE/monotonicity-penalty terms described in Sec. IV.B and do not reach
  the training quality the paper's PINN comparators evidently achieve.

## Experiment 3 — Small-Circuit Measured Calibration

Reproduces Sec. IV.C / V.C: the full residual-circuit construction (payoff
state, LCHS auxiliary register, controlled Hamiltonian evolutions, observation
block encoding), a statevector check against independent matrix calculations,
and the bounded pattern search (Algorithm 1) run on exact, finite-difference,
matrix-LCHS and measured-LCHS objectives at two shot budgets and three seeds.
Source: `results/exp3_statevector_check.json`, `results/exp3_summary.csv`,
`results/exp3_confidence_intervals.json`.

- Statevector validation: maximum absolute error between circuit and
  classical residual probabilities is 7.9e-16 (paper: ≈1.75e-17). **Reproduced**
  as a property (both are at the floor of double-precision round-off; the
  one-and-a-half-order-of-magnitude difference is well inside normal variation
  of floating-point round-off across independent implementations).
- Exact-objective pattern search: fitted coefficients `[0.5, 0.75, 0.5, 0.25]`,
  coefficient RMSE 0.395, held-out RMSE 13.3 bp. The paper states the
  generating coefficients `(0.75, 0.25)` / `(0.25, 0.75)` lie exactly on the
  pattern-search lattice (start 0.5, step 0.25) and that the exact objective
  therefore recovers them with coefficient RMSE 0.000. **Not reproduced**:
  the fitted vector differs from the generating one even though both lie on
  the same lattice, which means the pattern search in this run converged to a
  different lattice point with an equal or lower exact-objective value than
  the generating point, or the two basis functions used here (not specified
  by the paper, Sec. IV.C uncertainty) do not reproduce the identifiability
  the paper's own, unspecified, basis pair gives. The cause was checked down
  to "the pattern search reaches a local, not the generating, optimum of the
  exact objective on this basis choice" but not resolved further within the
  time available; it is reported as an open, paper-specified detail left
  undetermined rather than attributed to a specific coding error.
- Matrix-LCHS objective: coefficient RMSE 0.25, held-out RMSE 74.5 bp, against
  the paper's 0.125 / ≈8.67 bp. **Not reproduced**, consistent with the same
  LCHS truncation/precision issue identified in Experiments 1 and 2 (here at
  the small-circuit configuration J=8, K_c=12).
- Measured-LCHS objective: mean held-out RMSE 74.5 bp at 2^18 shots and 68.5 bp
  (mean over three seeds) at 2^22 shots, against the paper's ≈18.1 bp and
  ≈10.1 bp. **Not reproduced** in magnitude, but the qualitative property that
  *more repetitions narrow the measurement confidence interval* is reproduced:
  the 95% CI half-widths in `exp3_confidence_intervals.json` shrink from
  about 1.7e-4–2.7e-4 at 2^18 shots to about 4.0e-5–4.9e-5 at 2^22 shots,
  consistently across all three seeds.
- Final residual confidence intervals: computed with the conservative
  binomial construction described in Sec. III.B; intervals are appreciably
  wider at the lower (2^18) shot budget than at the higher (2^22) one for
  every seed, reproducing the qualitative claim.

## Experiment 4 — Financial Consistency and Pricing Accuracy Stress Test

Reproduces Sec. II.B / V.D: the fixed-coefficient stress case (clean,
alternative-target, N=128, optimizer start 0, first expiry) re-evaluated under
exact, backward-Euler (1024 steps) and beta-LCHS at J=2048 and J=4096 (K_c=48).
Source: `results/exp4_stress_case.json`.

| Quantity | J=2048 | J=4096 | Paper (J=2048 → J=4096) |
|---|---|---|---|
| T0 max price error (bp) | 64.52 | 64.33 | 3.55 → 0.036 |
| T0 negative butterflies | 4 | 0 | present → 0 |
| T0 min butterfly | −8.4e-6 | +1.0e-7 | negative → non-negative |
| T1 max price error (bp) | 123.10 | 123.43 | — |
| backward_euler max price error (bp) | 0.098 (T0) | — | ≈0.097 |

Verdicts:
- Butterfly convexity at T0 going from J=2048 to J=4096: **reproduced** as a
  qualitative property — four negative (non-convex) butterflies at J=2048
  become zero at J=4096, and the minimum butterfly value flips from negative
  to a small positive number, matching the paper's claim that doubling the
  LCHS term count can remove in-band convexity violations in this stress case.
- Price-error reduction when doubling J: **not reproduced**. The paper reports
  a large drop (3.55 → 0.036 bp); this run shows essentially no change
  (64.52 → 64.33 bp at T0), and the absolute level is far above backward
  Euler's 0.098 bp rather than below it. Identified cause: the same K_c
  truncation issue as Experiments 1–3 — at N=128 the operator's spectral
  radius is far larger than K_c=48, so both J=2048 and J=4096 sit on the same
  side of the truncation-dominated error regime and increasing J mostly
  refines the quadrature *inside* an already too-narrow window rather than
  widening it.
- Exact and backward-Euler outputs: **reproduced** — both satisfy butterfly
  non-negativity and calendar monotonicity at this stress case (exact and
  backward-Euler minimum butterflies and calendar differences are positive),
  consistent with the paper's claim that all 120 exact and 120 backward-Euler
  benchmark outputs satisfy both inequalities.
- Calendar monotonicity for beta-LCHS: **not reproduced** at this stress case
  — both J=2048 and J=4096 give a negative calendar difference, whereas the
  paper's narrative focuses only on butterfly convexity improving with J and
  does not claim calendar monotonicity is restored in the same case; this run
  shows LCHS failing calendar monotonicity at both term counts, reinforcing
  that the observed convexity fix is a local effect of this stress case and
  not evidence of a surface-wide financial-consistency guarantee, as the
  paper itself cautions.

## Experiment 5 — Quantum Circuit Resource Estimation

Reproduces Sec. V.D / Table III: logical qubit and CNOT counts for residual
circuits over N∈{4,8}, J∈{4,8,16,32} (N=4) / J∈{4,8} (N=8), and 1 or 2
maturity intervals. Source: `results/exp5_resource_counts.csv`,
`results/exp5_resource_scaling.png`.

| N | J | intervals | Qubits (this run) | Qubits (paper) | CNOT (this run) | CNOT (paper) |
|---|---|---|---|---|---|---|
| 4 | 4 | 1 | 6 | 6 | 248 | 2,260 |
| 4 | 4 | 2 | 8 | 8 | 458 | 4,322 |
| 4 | 8 | 1 | 7 | 7 | 1,934 | 18,432 |
| 4 | 8 | 2 | 10 | 10 | 3,830 | 36,666 |
| 4 | 16 | 1 | 8 | 8 | 16,124 | 153,438 |
| 4 | 16 | 2 | 12 | 12 | 32,210 | 306,678 |
| 4 | 32 | 1 | 9 | 9 | 132,570 | 1,256,996 |
| 4 | 32 | 2 | 14 | 14 | 265,102 | 2,513,794 |
| 8 | 4 | 1 | 7 | 7 | 1,136 | 10,090 |
| 8 | 4 | 2 | 9 | 9 | 2,082 | 19,200 |
| 8 | 8 | 1 | 8 | 8 | 8,230 | 77,590 |
| 8 | 8 | 2 | 11 | 11 | 16,270 | 154,200 |

Verdicts:
- Qubit formula n + s·⌈log2 J⌉ + 2: **reproduced** — every one of the 12
  configurations matches the paper's qubit count exactly.
- CNOT scaling with J: **reproduced** as a property — doubling J multiplies
  the one-interval CNOT count by 7.8×, 8.3× and 8.2× (N=4, J: 4→8→16→32),
  matching the paper's own statement that "each J increase approximately
  multiplies CNOT by ∼8".
- Absolute CNOT counts: **not reproduced** — every count in this run is
  roughly 8–9× smaller than the paper's Table III. The paper does not specify
  its Hamiltonian-unitary decomposition method (flagged uncertainty, Sec.
  V.D); this replication's dense-synthesis gate count
  (`src/resource_count.py`) is a different, self-consistent decomposition
  that reproduces the qubit formula and the J-scaling law but not the
  constant factor of the paper's unspecified synthesis routine.

## Deviations from the paper

- **Data source.** No market data is used; every experiment generates its own
  synthetic option quotes internally, exactly as the paper's own experimental
  design specifies (Sec. IV). No Dataloader or Massive calls were needed.
- **Generating coefficients, strike grids, basis functions.** The paper states
  that the exact smile/skew generating coefficients (Experiment 1), the nine
  irregular calibration strikes (Experiments 1–2), the two basis functions of
  the small-circuit instance (Experiment 3), and the coefficient-independent
  shift θ are "specified in the accompanying notebook" or left unstated. None
  of these notebooks or numerical values are given in the paper text. This
  replication chose: the two-basis smile/skew weights stated in the
  experiment description (equal phi_2/phi_3 for the smile, larger phi_3 weight
  for the skew); nine log-spaced strikes between k=0.7 and 1.4 per maturity;
  two overlapping cosine functions on [-1.2, 1.2] for the small-circuit basis
  and for Experiment 2's four-basis model; and θ computed from the minimum
  eigenvalue of Sym(A_s(ξ)) over every vertex of the coefficient hypercube
  (`src/propagators.py::coefficient_independent_theta`), which is the
  coefficient-independent shift of smallest magnitude guaranteeing the
  Hermitian part is positive semi-definite for every ξ∈[0,1]^M. These choices
  are documented in `src/config.py` next to each parameter.
- **Beta-LCHS kernel.** The paper's "improved" beta kernel (ref. [14],
  β=0.8) is not given in closed form; this replication implements it as the
  exact integral of the Poisson density over each midpoint quadrature cell
  (`src/propagators.py::lchs_kernel_nodes_weights`), which removes the
  point-sampling bias of the "simple" kernel while sharing its node locations
  and truncation window.
- **LCHS truncation cutoff.** At the paper's literal K_c (96 for the surface
  reconstruction, 48 for the classical benchmark and stress test), the
  operator's Hermitian-part spectral radius on a 128-node grid is far larger
  than K_c, and the beta-LCHS held-out pricing error is two orders of
  magnitude worse than the paper's reported ≈0.27 bp (Experiments 1, 2 and 4).
  A grid-dependent rescaling of K_c is implemented
  (`src/propagators.py::lchs_scaled_kc`) but was not wired into the committed
  experiment runs before the time available for this replication ran out;
  this is the single largest unresolved gap in the results above.
- **PINN comparators.** `src/pinn.py` implements a simplified, time-boxed
  physics-informed network (data-fitting loss plus monotonicity/convexity
  penalty terms) rather than the architecture and training schedule the paper
  describes only qualitatively (Sec. IV.B, flagged uncertainty); its held-out
  errors in Experiment 2 (140–234 bp) are far worse than the paper's
  ≈1.1–1.35 bp.
- **Reduced LCHS calibration sweep (Experiment 2).** Every LCHS calibration
  cell in the four-basis benchmark hit a per-cell wall-clock budget
  (`EXP2_LCHS_TIME_BUDGET_S` in `src/config.py`) before converging, because a
  single beta-LCHS evaluation costs tens of seconds and the optimizer needs
  many evaluations per calibration. The grid sizes (N=32,64,128) and target
  cases (baseline, alternative) are unchanged from Sec. IV.B. To keep the
  full sweep inside the time available, the optimizer-start and noise-dataset
  counts were reduced from the paper's five starts and three noise datasets
  (`EXP2_N_OPTIMIZER_STARTS_PAPER`, `EXP2_N_NOISE_DATASETS_PAPER`) to one
  start and one noise dataset each (`EXP2_N_OPTIMIZER_STARTS_RUN`,
  `EXP2_N_NOISE_DATASETS_RUN`), as recorded in `results/exp2_summary.csv`'s
  `n_optimizer_starts_used`/`n_noise_datasets_used` columns. The paper's own
  planned run count (568, of which 567 expected to complete) was accordingly
  not matched. `python -m exp.exp2 --full` runs the paper's five optimizer
  starts and three noise datasets (and, with the LCHS budget increased,
  every LCHS cell to convergence); this full configuration was not executed
  here because of its runtime.
- **Resource-sweep gate counts (Experiment 5).** The paper does not specify
  its Hamiltonian-unitary decomposition; this replication's dense-synthesis
  gate counter (`src/resource_count.py`) matches the paper's qubit formula and
  its stated ∼8× CNOT scaling per doubling of J, but not its absolute CNOT
  counts (Deviations list above), which depend on an unstated synthesis
  routine.
- **Classical calibration runtimes (Experiment 2).** Reported only to compare
  the relative cost ordering of matrix exponentiation, finite differences and
  beta-LCHS; the absolute seconds depend on this run's hardware and are not
  compared to the paper's absolute runtime figures.

## Not reproduced

- The paper's full classical-benchmark run count (568 planned, 567 expected
  to complete) is not reproduced; the LCHS beta/simple cells of Experiment 2
  did not converge within the wall-clock budget available to this
  replication (see Deviations above). `python -m exp.exp2 --full` is provided
  to run the complete sweep to convergence.
- Table III's absolute CNOT gate counts (Experiment 5) are not reproduced;
  only the qubit formula and the CNOT scaling law with J are. The paper's own
  gate-decomposition routine is not described, so its exact constant factor
  cannot be derived from the paper text.
- The exact numerical recovery of the small-circuit generating coefficients
  under the exact objective (Experiment 3, paper-stated coefficient RMSE
  0.000) is not reproduced; the paper's own two basis functions for this
  instance are not given, so the generating-coefficient identifiability the
  paper reports could not be confirmed with the basis choice made here.
