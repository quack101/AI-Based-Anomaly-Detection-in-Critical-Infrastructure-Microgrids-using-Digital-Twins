"""
RES.md D10 Phase 7 -- the 3-phase AC state estimator: nonlinear
weighted least squares (Gauss-Newton) over the measurement model
Phase 4 built, with the zero-injection constraints enforced as HARD
LAGRANGIAN EQUALITY CONSTRAINTS (Sha25 Sec II eq. 1-6) -- never folded
in as high-weight pseudo-measurements.

This is not a style choice. Phase 4's own sweep
(observability/sweep.py, reports/phase4_observability_report.txt)
found cond(H^T W H) = infinite at EVERY configuration tested, because
zero-injection was correctly excluded from the weighted block there.
The alternative Sha25 rejects -- folding zero-injection in as high-
weight (near-zero-sigma) pseudo-measurements -- creates an UNBOUNDED
weight ratio between a real 1%-sigma AMI meter and an "exact" row
whose sigma is deliberately driven toward 0; that is exactly the
"conditioning damage" Sha25's Lagrangian form exists to avoid.
demonstrate_plain_wls_failure() below shows this empirically (a real
cond() number, not an assertion) for the report to cite directly.

## The equality-constrained problem

Minimise      J(x) = (z - h(x))^T W (z - h(x))
subject to               c(x) = 0                (zero-injection, EXACT)

Linearise at iterate x_k (H_k = dh/dx, C_k = dc/dx, both from
observability.jacobian.build_jacobian, evaluated AT x_k -- not just
flat start, via that module's `at_v` parameter, new this phase):

  dz = z - h(x_k)      (measurement residual)
  dc = -c(x_k)         (constraint residual)

## Solved via the NULL-SPACE method, not the dense KKT block matrix

An earlier version of this module solved the textbook Lagrangian KKT
block system directly:

  [ H_k^T W H_k   C_k^T ] [ dx     ]   [ H_k^T W dz ]
  [ C_k           0     ] [ -lambda] = [ dc         ]

This is mathematically the same equality-constrained least-squares
problem Sha25 Sec II eq. 1-6 specifies, but proved numerically
UNUSABLE on this specific feeder: forming H_k^T W H_k SQUARES
H_k's own condition number (a well-known pitfall of normal-equations
methods), and this feeder's near-zero-impedance regulator/switch
branches (Y ~10,000+ Siemens, vs. ~1-1000 for ordinary lines) already
give H_k itself a huge dynamic range -- squaring it pushed cond(KKT)
past 1e20, far beyond float64's ~1e16 usable precision. No amount of
trust-region step-capping or Levenberg-Marquardt diagonal damping (both
tried, both logged in reports/phase7_state_estimator_report.txt) fixed
this, because the SOLVE ITSELF was returning numerically meaningless
directions, not just imprecise magnitudes.

The fix: solve the SAME linearised equality-constrained least-squares
problem via the NULL-SPACE method instead, which never forms H^T W H:

  1. Find ONE particular solution to C_k dx = dc (minimum-norm, via
     np.linalg.lstsq -- avoids forming C^T C too).
  2. Compute an orthonormal basis Z for the null space of C_k via SVD
     (Z's columns span every direction that leaves the constraint
     satisfied to first order).
  3. Substitute dx = dx_particular + Z w and solve the now-UNCONSTRAINED,
     REDUCED weighted least squares min_w ||sqrt(W)(H_k(dx_p + Z w) - dz)||^2
     directly on the matrix sqrt(W) H_k Z via lstsq -- condition number
     cond(H_k), not cond(H_k)^2, and a much smaller system (dim = n -
     rank(C_k) instead of n + p).

This is the SAME Lagrangian-constrained formulation RES.md/Sha25
specify (zero-injection is still an EXACT equality constraint, still
never a weighted pseudo-measurement) -- only the LINEAR ALGEBRA used to
solve the resulting system changed, from normal equations to an
orthogonal (SVD-based) decomposition. Sha25 eq. (6)'s alpha =
1/max(diag(W)) scaling is still applied to W for the same reason as
before (bounding the weighted block's own internal dynamic range).

A diagonal state-scale preconditioner (state_scale, below) is applied
throughout: the state vector mixes ACTUAL VOLTS (~2000-2500) and
RADIANS (~0-2), a column-scale disparity that (even without ever
squaring it into normal equations) still needs handling for a well-
posed SVD/lstsq -- scaling each V-column by that node's own V_base
(so the solve works in per-unit-like units internally) is standard
practice, not a new invention.

A trust-region step cap is kept (see MAX_STEP_V_PU/MAX_STEP_DELTA_RAD)
as a final safeguard against a single iteration's linearisation being
locally poor -- with the null-space method the RAW steps are far more
reasonable than before, but the cap costs nothing to keep and guards
against a residual bad iterate.

## REMEDIATION (Phase 7 review, Block 2.1): the reduced system needs
## its own damping -- the null-space method is not automatically safe

Requiring the CONSTRAINT residual (not just step size) below tolerance
to declare convergence (DEFAULT_TOL_CONSTRAINT_KW) exposed a real
problem the step-size-only criterion had been masking: at bus 83 (a
fixed shunt capacitor bus), the solve got permanently STUCK -- not
slow, STUCK, identically across 30 iterations, constraint_residual_inf
frozen at ~200 kvar. Traced (not guessed) to the REDUCED system itself:
A_reduced = sqrt(W) H_meas Z has cond() ~2e22, with singular values
spanning from ~6e7 down to ~5e-6 that plain np.linalg.lstsq's default
rcond retains (only the two genuinely-zero, ~1e-15-scale singular
values get truncated) -- dividing by singular values as small as 5e-6
amplifies whatever noise/inconsistency sits in b_reduced by a factor up
to ~1e13, producing a `w` with norm ~600 (confirmed directly) that the
trust-region cap can shrink in MAGNITUDE (a 2000x-oversized raw step)
but not correct in DIRECTION -- every tested step fraction from 1.0
down to 1e-6 made bus 83's constraint WORSE, because the direction
itself was dominated by amplified numerical noise, not signal.

Fix: Levenberg-Marquardt damping on the reduced system specifically --
solve the augmented system [A_reduced; sqrt(lambda) I] w = [b_reduced; 0]
(equivalent to Tikhonov-regularised normal equations A^T A + lambda I,
via lstsq on the augmented stack so A^T A is still never formed
explicitly) instead of plain lstsq on A_reduced alone, with the
STANDARD adaptive lambda schedule (accept + decrease lambda on a step
that improves the merit function; reject + increase lambda and retry
otherwise). This is NOT the same experiment the paragraph above
describes failing -- THAT was LM damping applied to the dense KKT
block matrix (H^T W H, the SQUARED, ~1e21-conditioned system), which
damping legitimately cannot rescue once precision is already lost to
squaring. This damping is applied to A_reduced, a DIFFERENT, smaller
(226x225) matrix that the null-space method was specifically built to
avoid ever squaring -- LM here regularises the leftover ill-
conditioning WITHIN that already-improved system, not the same failure
recurring. Replaces the step-FRACTION backtracking line search (which
could only shrink a bad direction, never fix it) as the primary step-
quality control; the trust-region cap is kept as a final safeguard.

## Pseudo-measurements stay a WEIGHTED row, not a constraint

The Phase-5 pseudo-measurements (sigma_pseudo estimated per node,
27.7-42.1% of mean, from held-out IDEAL-corpus variance) enter the
WEIGHTED block (H_k, W) exactly like a real meter -- they are ordinary,
uncertain measurements of REAL, nonzero consumption, not exact facts
about the network. Kept structurally distinct from zero-injection (a
separate id namespace, PSEUDO_* vs ZI_*, and a separate code path --
observability.pseudo_measurements vs. observability.zero_injection) --
Phase 5's own explicit design constraint, re-verified in code here
rather than merely re-asserted (tests/test_state_estimator.py).

## A third category, found in Phase 7: regulator-adjacent WEIGHTED rows

Some genuinely load-free nodes sit adjacent to a voltage regulator,
whose admittance depends on a discrete tap position that drifts, over
the corpus horizon, away from the single nominal-load snapshot this
module's Y-bus is built from (observability/zero_injection.py's
_regulator_buses docstring has the full empirical account). Excluding
them from the hard-constraint set entirely (as an earlier version of
this phase did) reopened exactly the rank Phase 4/5's full-rank finding
relied on and produced a systematic, non-noise-like bias in x_hat
(unobserved directions land wherever the null-space minimum-norm
solution puts them, not at x_true). Hard-constraining them, instead,
made the Gauss-Newton solve diverge for late-corpus timestamps, since a
tap-drifted node's true injection is NOT actually 0.

The fix consistent with Sha25's own distinction (exact facts get a
hard constraint; anything with real uncertainty is a weighted
measurement) is a THIRD category: model.reg_pseudo_entries --
REGZI_*-namespaced, with BOTH target (z_value) and sigma estimated
empirically per node (observability.estimator_model.
_estimate_regulator_mean_sigma, held-out-train-split, never hand-picked
-- same discipline as Phase 5's sigma_pseudo). NOT a z_value=0.0
assumption: direct inspection of the per-node ground-truth distribution
found some regulator-adjacent nodes read a small but PERSISTENTLY
nonzero injection, not "0 plus occasional drift noise" -- so these are
modelled exactly like a Phase-5 load pseudo-measurement (empirical mean
AND spread), not a zero-injection node with extra uncertainty bolted
on. This keeps them IN the weighted block (restoring dof/rank) while
never asserting a target the data itself contradicts. Structurally
distinct from BOTH other categories: three id namespaces (ZI_ /
PSEUDO_ / REGZI_), three reasons for uncertainty (none, load
variability, regulator-adjacency), three code paths.

## Layer boundary

This module NEVER imports opendssdirect or simulation.opendss_utils,
and never reads a live circuit at estimation time -- the model it
operates on (Y, H-structure, entry lists, pseudo-measurement tables) is
built ONCE, outside this module, by observability.estimator_model
(which is permitted to touch a compiled circuit -- a FIXED engineering
input any state estimator needs, not a live measurement) and handed in
as a plain EstimatorModel. z itself is a plain dict the caller supplies
(Phase 6's corpus files, in this phase's own validation) --
tests/test_import_boundaries.py enforces the direct-import half of this
boundary; this module's own docstring states the design intent.
"""

from dataclasses import dataclass

import numpy as np

from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from observability.pseudo_measurements import build_pseudo_entries_for_hour

DEFAULT_MAX_ITER = 30
DEFAULT_TOL_V_PU = 1e-4       # relative to each node's own V_base
DEFAULT_TOL_DELTA_RAD = 1e-4  # ~0.0057 deg

# RES.md D10 Phase 7 REMEDIATION (Block 2.1): a hard equality constraint
# that is not satisfied is not a constraint. An earlier version of this
# module declared "converged" on step-size alone -- found (reviewing
# Phase 7) to let 2 of 328 zero-injection rows (bus 83's shunt-capacitor
# Q rows) sit ~200 kvar off target at the declared solution. ROOT CAUSE
# (see the module docstring's REMEDIATION section): the reduced null-
# space system's ill-conditioning (cond~2e22), not a mere line-search-
# step artifact -- fixed at its source via LM damping on that system,
# below; this tolerance is the independent CHECK that the fix actually
# worked, not the fix itself. 1.0 kW/kvar is tight relative to real load
# magnitudes on this feeder (tens-hundreds of kW) and to the 200 kvar
# failure this is meant to catch, while staying loose enough that
# ordinary floating-point granularity (94.8% of the 328 rows already
# sat under 0.01 kW/kvar even before the LM fix) never
# forces an extra iteration for no reason.
DEFAULT_TOL_CONSTRAINT_KW = 1.0

# Trust-region step cap -- final safeguard, see module docstring.
MAX_STEP_V_PU = 0.10
MAX_STEP_DELTA_RAD = 0.10

# RES.md D10 Phase 7 REMEDIATION (Block 2.1): Levenberg-Marquardt
# damping on the REDUCED null-space system (module docstring's
# REMEDIATION section has the full root-cause trace). Standard adaptive
# LM schedule: a step that improves the merit function is accepted and
# lambda is decreased (trust the linearisation more next time); a step
# that does not is rejected and lambda is increased (trust it less, get
# closer to plain gradient descent on the reduced system) and retried
# AT THE SAME ITERATE, up to LM_MAX_TRIES times. LM_LAMBDA_INIT=1.0 was
# chosen empirically: at bus 83's ill-conditioned reduced system,
# lambda=1.0 already keeps the raw (pre-trust-region-cap) step within
# the trust region without any capping needed (max_dx_v_pu=0.026 pu, vs
# 2077x-over-cap at lambda=0).
#
# LM_LAMBDA_MAX BUG FOUND AND FIXED HERE: a first attempt at
# LM_LAMBDA_MAX=1e8 left several real timestamps unable to find ANY
# improving step at ANY tested damping level -- the "no step" branch
# (added by the ratchet-divergence fix above) then correctly refused to
# move, but for 30 straight iterations, burning ~140s/timestamp for
# zero progress. Root cause: A_reduced's largest singular value is
# ~6.4e7 (bus 83's timestamp; varies by timestamp but is consistently
# in this range on this feeder's near-zero-impedance-branch-adjacent
# rows) -- LM only enters its GUARANTEED-monotonic-improvement regime
# (damping >> sigma_max^2, where the augmented system is dominated by
# the identity block and the step becomes a small, provably-improving
# steepest-descent step) once lambda is on the order of sigma_max^2,
# i.e. ~4e15, not 1e8 -- 1e8 was still ~1.5x sigma_max itself, nowhere
# near dominant. Raised by 10 orders of magnitude so LM_MAX_TRIES=25
# factor-of-10 steps from LM_LAMBDA_INIT=1.0 can actually reach it.
LM_LAMBDA_INIT = 1.0
LM_LAMBDA_MIN = 1e-6
LM_LAMBDA_MAX = 1e18
LM_INCREASE_FACTOR = 10.0
LM_DECREASE_FACTOR = 3.0
LM_MAX_TRIES = 15

# Fractional multiplier tried on the FULL step (y_particular + damped
# Z@w together) at each lambda level -- see the "SECOND BUG" note at
# this constant's call site: LM damping alone only regularises the
# null-space part, not y_particular, which can itself overshoot a
# strongly nonlinear constraint (bus 83) regardless of lambda.
STEP_FRACTIONS = (1.0, 0.1, 0.01)

LINESEARCH_CONSTRAINT_PENALTY = 1e4

# Null-space rank tolerance -- singular values of C_k (scaled) below
# this fraction of the largest are treated as "in the row space", i.e.
# not contributing an independent constraint direction, per the
# standard numerical-rank convention.
NULLSPACE_RANK_TOL = 1e-8


@dataclass
class EstimationResult:
    x_v: np.ndarray
    x_delta: np.ndarray
    converged: bool
    iterations: int
    max_dx_v_pu: float
    max_dx_delta_rad: float
    J: float             # weighted residual sum of squares at x_hat, weighted block only
    dof: int              # m_weighted - (dim(x) - p_constraints)
    m_weighted: int
    p_constraints: int
    constraint_residual_inf: float  # max|c(x_hat)| -- should be ~0


class StateEstimator:
    """Wraps a pre-built observability.estimator_model.EstimatorModel
    (constructed ONCE, via a live circuit compile THAT MODULE performs,
    never this one) and does the actual Gauss-Newton solve -- pure
    numpy, no circuit access, per this module's own docstring."""

    def __init__(self, model):
        self.model = model
        state = model.state

        self.m_weighted = (
            len(model.meas_entries)
            + len(model.reg_pseudo_entries)
            + 2 * len(model.bus_phase_by_load)
        )
        self.p_constraints = len(model.zi_entries)
        self.dof = self.m_weighted - (state.dim - self.p_constraints)

        v_base = np.array([state.v_base[name] for name in state.all_node_names])
        self._state_scale = np.concatenate([v_base, np.ones(state.dim - state.n_phase_nodes)])

    def _weighted_entries_for_hour(self, hour):
        model = self.model
        pseudo_entries = build_pseudo_entries_for_hour(
            model.bus_phase_by_load, model.per_hour_by_load, hour,
        )
        return model.meas_entries + model.reg_pseudo_entries + pseudo_entries

    def estimate(self, z_real, real_timestamp,
                 max_iter=DEFAULT_MAX_ITER,
                 tol_v_pu=DEFAULT_TOL_V_PU,
                 tol_delta_rad=DEFAULT_TOL_DELTA_RAD,
                 tol_constraint_kw=DEFAULT_TOL_CONSTRAINT_KW):
        """z_real: {registry_id: value} for the real (meter+head)
        entries -- e.g. one row of Phase 6's {split}_telemetry.csv.
        real_timestamp: 'YYYY-MM-DD HH:MM:SS' -- selects which static
        hour-of-day pseudo-measurement bucket to use (Phase 5's design;
        see observability/pseudo_measurements.py)."""

        state = self.model.state
        hour = int(real_timestamp[11:13])
        weighted_entries = self._weighted_entries_for_hour(hour)

        z = np.array([
            e["z_value"] if "z_value" in e else z_real[e["id"]]
            for e in weighted_entries
        ])
        sigma = np.array([e["sigma"] for e in weighted_entries])
        W = 1.0 / (sigma ** 2)
        alpha = 1.0 / W.max()
        sqrt_Ws = np.sqrt(alpha * W)

        all_entries = weighted_entries + self.model.zi_entries
        m = len(weighted_entries)
        n = state.dim
        p = self.p_constraints
        s = self._state_scale  # diagonal preconditioner: dx = s * y
        v_base = s[:state.n_phase_nodes]

        x_v = np.array(state.x0_v, dtype=float)
        x_delta = np.array(state.x0_delta, dtype=float)

        def _merit(xv, xd):
            at_v = complex_v_from_state(state, xv, xd)
            H_c, h0_c = J.build_jacobian(all_entries, self.model.Y, self.model.node_index, state, at_v=at_v)
            dz_c = z - h0_c[:m]
            dc_c = -h0_c[m:]
            merit_val = float(np.sum(W * dz_c ** 2) + LINESEARCH_CONSTRAINT_PENALTY * np.sum(dc_c ** 2))
            return merit_val, H_c, h0_c

        converged = False
        iterations = 0
        max_dx_v_pu = float("nan")
        max_dx_delta_rad = float("nan")
        lm_lambda = LM_LAMBDA_INIT

        current_merit, H_all, h0_all = _merit(x_v, x_delta)

        for iterations in range(1, max_iter + 1):
            H_meas, h0_meas = H_all[:m], h0_all[:m]
            C, h0_zi = H_all[m:], h0_all[m:]

            dz = z - h0_meas
            dc = -h0_zi

            Hs_meas = H_meas * s[None, :]  # scaled: dx = s*y => H_meas @ dx = (H_meas*s) @ y
            Cs = C * s[None, :]

            # 1. particular solution to Cs @ y = dc (minimum-norm)
            y_particular, *_ = np.linalg.lstsq(Cs, dc, rcond=None)

            # 2. null space of Cs via SVD -- Z's columns are the
            # directions that leave the (linearised) constraint
            # satisfied.
            _, Sigma, Vt = np.linalg.svd(Cs, full_matrices=True)
            rank_c = int(np.sum(Sigma > NULLSPACE_RANK_TOL * Sigma[0])) if Sigma.size else 0
            Z = Vt[rank_c:].T  # n x (n - rank_c)

            # 3. reduced least squares in w -- LEVENBERG-MARQUARDT damped
            # (module docstring's REMEDIATION section): A_reduced alone
            # is ill-conditioned enough (cond~2e22 observed) that plain
            # lstsq's default rcond retains singular values as small as
            # ~5e-6 relative to the largest, amplifying measurement
            # noise into a numerically-meaningless `w`. Solving the
            # augmented [A_reduced; sqrt(lambda) I] system instead is
            # equivalent to Tikhonov-regularised normal equations
            # (A^T A + lambda I) w = A^T b WITHOUT ever forming A^T A --
            # still routed through lstsq/SVD, consistent with the null-
            # space method's whole reason for existing (Part 2 of this
            # module's docstring).
            A_reduced = (Hs_meas * sqrt_Ws[:, None]) @ Z
            b_reduced = sqrt_Ws * (dz - Hs_meas @ y_particular)
            n_w = A_reduced.shape[1]

            # ADAPTIVE LM: a step that improves the merit function is
            # accepted (lambda decreases -- trust the linearisation more
            # next time); one that doesn't is REJECTED OUTRIGHT, lambda
            # increases (trust it less), and the SAME iterate is retried
            # with the new lambda, up to LM_MAX_TRIES.
            #
            # BUG FOUND AND FIXED HERE (found running this exact fix,
            # first version): an earlier version of this loop fell back
            # to the LOWEST-merit candidate tried even when NO try
            # actually improved on current_merit, and unconditionally
            # applied it as the next iterate. That silently accepts a
            # merit-INCREASING step as the new baseline, so the NEXT
            # iteration's own "improves on current_merit" test compares
            # against an already-degraded baseline -- a ratchet that
            # only ever gets easier to satisfy with a worse point. On
            # the exact bus-83 case DEFAULT_TOL_CONSTRAINT_KW was added
            # to catch, this ratchet drove the solve from a ~200 kvar
            # localised residual (the OLD line-search code's failure
            # mode -- stuck, but not diverging) to constraint_residual_
            # inf=16.2 MILLION kvar and 1.03 pu V error over 30
            # iterations -- confirmed empirically before this fix, not
            # theorised. Correct LM behaviour: if NO try improves the
            # merit even at LM_LAMBDA_MAX, take NO step this iteration
            # (x_v/x_delta/current_merit stay exactly as they were) --
            # lambda stays maxed into the next outer iteration, so a
            # persistently-unsatisfiable direction costs wasted
            # iterations, never a wrong move.
            # SECOND BUG FOUND AND FIXED HERE: LM damping alone still was
            # not enough -- it only regularises `w` (the null-space,
            # measurement-fit part of the step); y_particular (the
            # constraint-CORRECTING part) is UNREGULARISED and, by
            # construction, satisfies only the LINEARISED constraint
            # exactly -- on bus 83's strongly nonlinear (quadratic-in-V)
            # self-susceptance term, y_particular ALONE was already
            # confirmed to overshoot the true root (-200 kvar -> +106.5
            # kvar in one uncapped step, tested in isolation with w=0).
            # No amount of damping `w` can fix a problem that lives
            # entirely inside y_particular, which is why several real
            # timestamps found NO improving (lambda, w) pair at all --
            # every candidate carried the same bad y_particular-driven
            # overshoot regardless of w. Fix: ALSO search a step-
            # fraction on the FULL step (particular + damped null-space
            # part together), nested inside the lambda loop -- this is
            # the one lever that can shrink y_particular's own
            # contribution, which pure LM damping cannot reach.
            improved = False
            for _ in range(LM_MAX_TRIES):
                A_aug = np.vstack([A_reduced, np.sqrt(lm_lambda) * np.eye(n_w)])
                b_aug = np.concatenate([b_reduced, np.zeros(n_w)])
                w, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
                y_full = y_particular + Z @ w

                for frac in STEP_FRACTIONS:
                    y = frac * y_full
                    dx = s * y
                    dx_v0, dx_delta0 = dx[:state.n_phase_nodes], dx[state.n_phase_nodes:]

                    # Trust-region step cap -- final safeguard (see
                    # module docstring), kept even with LM damping +
                    # fractional search since it guards against a single
                    # iteration's linearisation being locally poor in a
                    # DIFFERENT direction than the one this remediation
                    # found and fixed.
                    step_ratio = max(
                        np.max(np.abs(dx_v0) / v_base) / MAX_STEP_V_PU,
                        np.max(np.abs(dx_delta0)) / MAX_STEP_DELTA_RAD,
                        1.0,
                    )
                    cand_dx_v, cand_dx_delta = dx_v0 / step_ratio, dx_delta0 / step_ratio

                    cand_v = x_v + cand_dx_v
                    cand_delta = x_delta + cand_dx_delta
                    cand_merit, cand_H, cand_h0 = _merit(cand_v, cand_delta)

                    if cand_merit < current_merit:
                        best = (cand_merit, cand_v, cand_delta, cand_dx_v, cand_dx_delta, cand_H, cand_h0)
                        lm_lambda = max(lm_lambda / LM_DECREASE_FACTOR, LM_LAMBDA_MIN)
                        improved = True
                        break

                if improved:
                    break

                lm_lambda = min(lm_lambda * LM_INCREASE_FACTOR, LM_LAMBDA_MAX)

            if not improved:
                # no damping level found an improving step -- take NO
                # step this iteration (see BUG note above). dx_v/dx_delta
                # report as exactly 0 so the step-size convergence check
                # below correctly reads this as "not converged" (a null
                # step is not a small-and-satisfied step) rather than
                # spuriously passing.
                dx_v = np.zeros(state.n_phase_nodes)
                dx_delta = np.zeros(state.dim - state.n_phase_nodes)
            else:
                current_merit, x_v, x_delta, dx_v, dx_delta, H_all, h0_all = best

            max_dx_v_pu = float(np.max(np.abs(dx_v) / v_base))
            max_dx_delta_rad = float(np.max(np.abs(dx_delta)))
            constraint_residual_iter = float(np.max(np.abs(h0_all[m:]))) if p else 0.0

            if (max_dx_v_pu < tol_v_pu and max_dx_delta_rad < tol_delta_rad
                    and constraint_residual_iter < tol_constraint_kw):
                converged = True
                break

        h0_meas_final, h0_zi_final = h0_all[:m], h0_all[m:]
        residual = z - h0_meas_final
        Jval = float(np.sum(W * residual ** 2))
        constraint_residual_inf = float(np.max(np.abs(h0_zi_final))) if p else 0.0

        return EstimationResult(
            x_v=x_v, x_delta=x_delta, converged=converged, iterations=iterations,
            max_dx_v_pu=max_dx_v_pu, max_dx_delta_rad=max_dx_delta_rad,
            J=Jval, dof=self.dof, m_weighted=m, p_constraints=p,
            constraint_residual_inf=constraint_residual_inf,
        )


def demonstrate_plain_wls_failure(estimator, hour=12, pseudo_sigma_for_zi=1e-6):
    """RES.md D10 Phase 7's own instruction: 'show it would fail the
    plain-WLS way if attempted.' Builds the ALTERNATIVE Sha25 rejects
    -- zero-injection folded in as HIGH-WEIGHT (near-zero-sigma)
    pseudo-measurements inside one plain (unconstrained) weighted
    normal-equations matrix H^T W H, sigma=pseudo_sigma_for_zi kW for
    every zero-injection row (an "almost exact" weight, the natural
    choice if someone tried to fake a hard constraint this way) beside
    real measurements at 1% sigma and Phase-5 pseudo-measurements at
    27-42% sigma -- and reports cond() for that matrix, with the SAME
    diagonal state-scaling this module always applies, so the failure
    is attributable to the WEIGHT-RATIO REGIME itself, not to omitting
    the (separately necessary) state preconditioning."""

    model = estimator.model
    state = model.state
    weighted_entries = estimator._weighted_entries_for_hour(hour)
    all_entries_plain = weighted_entries + model.zi_entries

    at_v = complex_v_from_state(state, state.x0_v, state.x0_delta)
    H_all, h0_all = J.build_jacobian(all_entries_plain, model.Y, model.node_index, state, at_v=at_v)

    s = estimator._state_scale
    H_scaled = H_all * s[None, :]

    sigma_meas = np.array([e["sigma"] for e in weighted_entries])
    sigma_zi = np.full(estimator.p_constraints, pseudo_sigma_for_zi)
    sigma_plain = np.concatenate([sigma_meas, sigma_zi])
    W_plain = 1.0 / (sigma_plain ** 2)
    alpha_plain = 1.0 / W_plain.max()

    HtWH_plain = H_scaled.T @ (H_scaled * (alpha_plain * W_plain)[:, None])
    cond_plain = float(np.linalg.cond(HtWH_plain))

    # this module's own weighted-block Jacobian, same iterate, same
    # state scaling -- its condition number (NOT squared into normal
    # equations, since this module solves via the null-space method)
    m = len(weighted_entries)
    cond_H_meas = float(np.linalg.cond(H_scaled[:m]))

    return cond_plain, cond_H_meas
