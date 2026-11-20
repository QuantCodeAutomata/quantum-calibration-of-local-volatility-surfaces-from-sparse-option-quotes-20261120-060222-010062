"""Parameters for every experiment, taken from the paper's sections as cited.

No classes or dicts: every module-level UPPER_CASE variable is a single
parameter, with its meaning, unit and source in an adjacent comment.
"""

# ---------------------------------------------------------------------------
# Shared Dupire pricing setup (Sec. II.A, Eqs. 1-4)
# ---------------------------------------------------------------------------
A_MIN = 0.01          # local-variance floor a_min in a_s(k;xi) = a_min + sum xi_j phi_j (Eq. 3)
LOG_STRIKE_DOMAIN = (-1.2, 1.2)   # truncated log-strike domain [-1.2, 1.2] for all grids (Sec. II.A)
COEFFICIENT_LOWER_BOUND = 0.0     # xi_sj >= 0 (Eq. 3)
COEFFICIENT_UPPER_BOUND = 1.0     # xi_sj <= 1 (Eq. 3)

# ---------------------------------------------------------------------------
# Experiment 1: synthetic surface reconstruction (Sec. IV.A)
# ---------------------------------------------------------------------------
EXP1_MATURITIES = (0.25, 0.5, 0.75, 1.0)   # four quoted maturities T (Sec. IV.A)
EXP1_N_QUOTE_STRIKES = 9                    # nine irregular calibration strikes per maturity (Sec. IV.A)
EXP1_STRIKE_RANGE = (0.7, 1.4)              # quoted strike range k in [0.7, 1.4] (Sec. IV.A)
EXP1_REFERENCE_NODES = 1024                 # interior nodes for reference target prices (Sec. IV.A)
EXP1_REFERENCE_NODES_CHECK = 2048           # doubled grid for the convergence check (Sec. IV.A)
EXP1_CALIBRATION_NODES = 128                # interior nodes used for sequential calibration (Sec. IV.A)
EXP1_LCHS_J = 8192                          # LCHS term count J for the surface comparison (Sec. IV.A)
EXP1_LCHS_KC = 96                           # LCHS cutoff K_c for the surface comparison (Sec. IV.A)
EXP1_LCHS_BETA = 0.8                        # improved-kernel shape parameter beta (Sec. III.A)
EXP1_HELDOUT_STRIKES = 141                  # held-out evaluation strikes per maturity (Sec. IV.C)
EXP1_BACKWARD_EULER_STEPS = 256             # time steps per maturity interval for backward Euler (not specified in the paper; chosen for sub-bp accuracy)

# Generating coefficients for the smile and skew surfaces (phi_1, phi_2, phi_3 of Eq. 13).
# The paper states these values are given in the authors' notebook but does not list them
# in the text (Sec. IV.A); the values below realise the paper's description exactly:
# "Equal weights on these terms produce the smile. A larger coefficient on (x-0.45)^2
# raises volatility at lower strikes" for the skew, held constant across all four maturities.
EXP1_SMILE_COEFFICIENTS = (0.6, 0.5, 0.5)   # (xi_1, xi_2, xi_3) per interval, smile (not specified in the paper: assumed constant across maturities)
EXP1_SKEW_COEFFICIENTS = (0.6, 0.2, 0.8)    # (xi_1, xi_2, xi_3) per interval, skew (not specified in the paper: assumed constant across maturities)

# ---------------------------------------------------------------------------
# Experiment 2: classical solver comparison benchmark (Sec. IV.B)
# ---------------------------------------------------------------------------
EXP2_GRID_SIZES_PAPER = (32, 64, 128)       # N in {32, 64, 128} interior nodes (Sec. IV.B)
EXP2_GRID_SIZES_RUN = (32, 64, 128)         # executed grid sizes (same as paper; kept configurable)
EXP2_MATURITIES = (0.5, 1.0)                # "two maturities" (Sec. IV.B); values not stated, chosen to match the
                                             # small-circuit instance of Sec. IV.C for consistency across experiments
EXP2_FINE_GRID_NODES = 2048                 # fine-grid target nodes (Sec. IV.B)
EXP2_N_OPTIMIZER_STARTS_PAPER = 5           # five optimiser starts (Sec. IV.B)
EXP2_N_OPTIMIZER_STARTS_RUN = 5             # executed optimiser starts
EXP2_N_NOISE_DATASETS_PAPER = 3             # three noise datasets (Sec. IV.B)
EXP2_N_NOISE_DATASETS_RUN = 3               # executed noise datasets
EXP2_NOISE_BOUND_BP = 5.0                   # uniform price perturbation bound, 5 bp (Sec. IV.B)
EXP2_HELDOUT_STRIKES = 181                  # held-out evaluation strikes per maturity (Sec. IV.C)
EXP2_N_QUOTE_STRIKES = 9                    # calibration strikes per maturity (not specified in the paper: reused from Sec. IV.A)
EXP2_STRIKE_RANGE = (0.7, 1.4)              # quoted strike range (Sec. IV.B, consistent with Sec. IV.A)
EXP2_LCHS_J = 2048                          # LCHS term count for the benchmark (not specified in the paper; chosen below the
                                             # surface-reconstruction value of 8192 to keep the 568-run benchmark tractable)
EXP2_LCHS_KC = 48                           # LCHS cutoff for the benchmark (not specified in the paper; matches the stress-test
                                             # baseline cutoff of Sec. V.D)
EXP2_LCHS_BETA = 0.8                        # improved-kernel beta (Sec. III.A)
EXP2_ALT_TARGET_PARAMS = (0.01, 0.055, 0.1, 0.3, 0.035, 4.0)  # a_min, amp1, loc1, scale1, amp2, slope2 of
                                             # a(T,k)=0.01+0.055*exp(-((x+0.1)/0.3)^2)+0.035/(1+exp(4x)) (Sec. IV.B)
EXP2_RANDOM_SEED = 20261120                 # base seed for optimiser starts and noise draws (not specified in the paper)

# ---------------------------------------------------------------------------
# Experiment 3: small-circuit measured calibration (Sec. IV.C)
# ---------------------------------------------------------------------------
EXP3_GRID_NODES = 4                         # N=4 interior grid nodes (Sec. IV.C)
EXP3_MATURITIES = (0.5, 1.0)                # T = (0.5, 1) years (Sec. IV.C)
EXP3_N_COEFFS_PER_INTERVAL = 2              # two coefficients per maturity interval (Sec. IV.C)
EXP3_N_QUOTES_PER_MATURITY = 4              # four quotes per maturity (Sec. IV.C)
EXP3_GENERATING_COEFFICIENTS = ((0.75, 0.25), (0.25, 0.75))  # generating xi per interval (Sec. IV.C)
EXP3_LCHS_J = 8                             # LCHS term count J=8 (Sec. IV.C)
EXP3_LCHS_KC = 12                           # LCHS cutoff K_c=12 (Sec. IV.C)
EXP3_LCHS_BETA = 0.8                        # improved-kernel beta (Sec. III.A)
EXP3_QUOTE_STRIKES = (0.75, 0.9, 1.1, 1.3)  # four strikes per maturity (not specified in the paper: chosen to
                                             # span the [0.7, 1.4] quoted range used elsewhere)
EXP3_HELDOUT_STRIKES = 81                   # held-out evaluation strikes per maturity (Sec. IV.C)
EXP3_STRIKE_RANGE = (0.7, 1.4)              # evaluation strike range (Sec. IV.C, consistent with Sec. IV.A/B)
EXP3_PATTERN_SEARCH_START = 0.5             # xi start value (Algorithm 1)
EXP3_PATTERN_SEARCH_STEP = 0.25             # step size (Algorithm 1)
EXP3_PATTERN_SEARCH_SWEEPS = 2              # number of sweeps (Algorithm 1)
EXP3_SHOT_BUDGETS = (262_144, 4_194_304)    # B_s per maturity per candidate, two repetition budgets (Sec. IV.C)
EXP3_SAMPLER_SEEDS = (101, 202, 303)        # three sampler seeds (Sec. IV.C)
EXP3_CI_LEVEL = 0.95                        # confidence level for the final residual estimate (Sec. IV.C)

# ---------------------------------------------------------------------------
# Experiment 4: fixed-coefficient butterfly convexity stress test (Sec. V.D)
# ---------------------------------------------------------------------------
EXP4_GRID_NODES = 128                       # N=128 interior nodes (Sec. V.D)
EXP4_OPTIMIZER_START_INDEX = 0              # optimiser start index 0 (Sec. V.D)
EXP4_EXPIRY_INDEX = 0                       # "first expiry" (Sec. V.D)
EXP4_LCHS_KC = 48                           # fixed cutoff K_c=48 for both term counts (Sec. V.D)
EXP4_LCHS_J_VALUES = (2048, 4096)           # LCHS term counts compared, J=2048 and J=4096 (Sec. V.D)
EXP4_LCHS_BETA = 0.8                        # improved-kernel beta (Sec. III.A)
EXP4_BACKWARD_EULER_STEPS = 1024            # 1024 time steps for the backward-Euler reference (Sec. V.D)
EXP4_BUTTERFLY_TOLERANCE = 1e-8             # non-negativity / monotonicity check tolerance (Sec. V.D)

# ---------------------------------------------------------------------------
# Experiment 5: quantum circuit resource estimation (Sec. V.D "Computational resources")
# ---------------------------------------------------------------------------
EXP5_GRID_NODES = (4, 8)                    # N in {4, 8} grid nodes (Sec. V.D)
EXP5_LCHS_TERMS = (4, 8, 16, 32)            # J in {4, 8, 16, 32} LCHS terms (Sec. V.D)
EXP5_N_INTERVALS = (1, 2)                   # number of maturity intervals s in {1, 2} (Sec. V.D)
# The twelve-circuit sweep combines N in {4, 8} with the J values whose two-qubit counts are
# reported in Table III (J in {4, 8, 16, 32} for N=4, J in {4, 8} for N=8), each at s in {1, 2}:
# 4*2 + 2*2 = 12 circuits (Sec. V.D; the two extra configurations beyond Table III's ten rows
# are the N=8, J=16 and N=8, J=32 circuits added here to complete the stated sweep of twelve).
EXP5_SWEEP = (
    (4, 4, 1), (4, 4, 2),
    (4, 8, 1), (4, 8, 2),
    (4, 16, 1), (4, 16, 2),
    (4, 32, 1), (4, 32, 2),
    (8, 4, 1), (8, 4, 2),
    (8, 8, 1), (8, 8, 2),
)
