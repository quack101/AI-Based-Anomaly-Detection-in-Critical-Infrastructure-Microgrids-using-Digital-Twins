"""
RES.md D10 Phase 7 SCIPY TRUST-CONSTR TEST -- a SEPARATE code path.

Six prior reports (reports/phase7_convergence_remediation_report.txt,
phase7_solver_instability_fix_report.txt, phase7_accept_criterion_fix_
report.txt, phase7_iteration_cap_test_report.txt, phase7_node_
demotion_test_report.txt) all assumed estimation/state_estimator.py's
hand-rolled null-space/LM-damping/trust-region-clip Gauss-Newton solve
was the right architecture and tried to patch it -- five independent
patches, all failed or made things worse. This module does NOT patch
that solver -- it does not import from or modify estimation/
state_estimator.py in any way, and is not called from anywhere in the
shipped estimation pipeline. It replaces the SOLVE METHOD only, for
the SAME problem (same WLS objective, same hard zero-injection
equality constraints, same analytic Jacobians from observability.
jacobian -- reused, not rederived), via scipy.optimize.minimize(
method='trust-constr'), a mature, independently-implemented trust-
region SQP solver.

## Problem (identical to estimation/state_estimator.py's own docstring)

Minimise      J(x) = (z - h(x))^T W (z - h(x))      [measurement rows]
subject to               c(x) = 0                    [zero-injection rows, EXACT]

Same diagonal state-scale preconditioner (x = s*y, s = state._state_scale
-- V-columns scaled by each node's own V_base, delta-columns by 1.0) is
applied here too, for the SAME reason estimation/state_estimator.py
applies it: the state vector mixes actual volts (~2000-2500) and
radians (~0-2), and scipy's trust-constr -- like any gradient-based
method -- benefits from a well-scaled parameterisation.

Run: python -m validation.phase7_scipy_trustconstr_pilot
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from observability.pseudo_measurements import build_pseudo_entries_for_hour

DEFAULT_MAX_ITER = 1000
DEFAULT_TOL_V_PU = 1e-4
DEFAULT_TOL_DELTA_RAD = 1e-4
DEFAULT_TOL_CONSTRAINT_KW = 1.0


@dataclass
class ScipyEstimationResult:
    x_v: np.ndarray
    x_delta: np.ndarray
    converged: bool
    iterations: int
    nfev: int
    constraint_residual_inf: float
    J: float
    status: int
    message: str


class ScipyTrustConstrEstimator:
    """scipy.optimize.minimize(method='trust-constr') solve over the
    SAME EstimatorModel estimation.state_estimator.StateEstimator
    consumes -- separate solve path, model construction (observability.
    estimator_model.build_estimator_model) is unchanged and shared."""

    def __init__(self, model):
        self.model = model
        state = model.state
        v_base = np.array([state.v_base[name] for name in state.all_node_names])
        self._state_scale = np.concatenate([v_base, np.ones(state.dim - state.n_phase_nodes)])

    def _weighted_entries_for_hour(self, hour):
        model = self.model
        pseudo_entries = build_pseudo_entries_for_hour(
            model.bus_phase_by_load, model.per_hour_by_load, hour)
        return model.meas_entries + model.reg_pseudo_entries + pseudo_entries

    def estimate(self, z_real, real_timestamp, max_iter=DEFAULT_MAX_ITER,
                 tol_v_pu=DEFAULT_TOL_V_PU, tol_delta_rad=DEFAULT_TOL_DELTA_RAD,
                 tol_constraint_kw=DEFAULT_TOL_CONSTRAINT_KW):
        state = self.model.state
        hour = int(real_timestamp[11:13])
        weighted_entries = self._weighted_entries_for_hour(hour)

        z = np.array([
            e["z_value"] if "z_value" in e else z_real[e["id"]]
            for e in weighted_entries
        ])
        sigma = np.array([e["sigma"] for e in weighted_entries])
        W = 1.0 / (sigma ** 2)

        all_entries = weighted_entries + self.model.zi_entries
        m = len(weighted_entries)
        p = len(self.model.zi_entries)
        s = self._state_scale
        v_base = s[:state.n_phase_nodes]

        # Cache the LAST Jacobian build -- scipy calls fun/jac/constraint-
        # jac separately and often re-queries the SAME point; one
        # analytic Jacobian build (observability.jacobian.build_jacobian,
        # REUSED unchanged, not rederived) covers measurement AND
        # constraint rows in a single pass, exactly as estimation.
        # state_estimator.StateEstimator's own _merit() does.
        cache = {"y": None, "H": None, "h0": None}

        def _ensure(y):
            if cache["y"] is not None and np.array_equal(cache["y"], y):
                return
            x = s * y
            x_v, x_delta = x[:state.n_phase_nodes], x[state.n_phase_nodes:]
            at_v = complex_v_from_state(state, x_v, x_delta)
            H, h0 = J.build_jacobian(all_entries, self.model.Y, self.model.node_index, state, at_v=at_v)
            cache["y"] = y.copy()
            cache["H"] = H
            cache["h0"] = h0

        def fun(y):
            _ensure(y)
            h0_meas = cache["h0"][:m]
            r = z - h0_meas
            return float(np.sum(W * r ** 2))

        def grad(y):
            _ensure(y)
            H_meas = cache["H"][:m]
            h0_meas = cache["h0"][:m]
            r = z - h0_meas
            # d/dx sum(W*(z-h)^2) = -2 H^T (W*r); d/dy = s * d/dx (chain
            # rule through x = s*y).
            grad_x = -2.0 * (H_meas.T @ (W * r))
            return s * grad_x

        def cons_fun(y):
            _ensure(y)
            return cache["h0"][m:]  # h_zi(x) -- target 0

        def cons_jac(y):
            _ensure(y)
            C = cache["H"][m:]
            return C * s[None, :]  # dg/dy = C * diag(s), same "Cs" as the existing solver

        nonlinear_constraint = NonlinearConstraint(cons_fun, 0.0, 0.0, jac=cons_jac)

        x0 = np.concatenate([np.array(state.x0_v, dtype=float), np.array(state.x0_delta, dtype=float)])
        y0 = x0 / s

        result = minimize(
            fun, y0, jac=grad, method="trust-constr",
            constraints=[nonlinear_constraint],
            options={
                "maxiter": max_iter,
                "gtol": 1e-10,
                "xtol": 1e-12,
                "verbose": 0,
            },
        )

        x_final = s * result.x
        x_v_final = x_final[:state.n_phase_nodes]
        x_delta_final = x_final[state.n_phase_nodes:]

        _ensure(result.x)
        h0_zi_final = cache["h0"][m:]
        constraint_residual_inf = float(np.max(np.abs(h0_zi_final))) if p else 0.0

        h0_meas_final = cache["h0"][:m]
        Jval = float(np.sum(W * (z - h0_meas_final) ** 2))

        # RES.md D10 Phase 7 HIGH-LOAD INVESTIGATION fix: "converged"
        # is judged by THIS problem's own constraint-residual bar
        # (the SAME tol_constraint_kw estimation.state_estimator uses),
        # not by scipy's OWN internal result.success/gtol self-
        # assessment -- trust-constr's gtol termination test is
        # calibrated to its own (tighter) notion of stationarity, and
        # can report status=4/"Constraint violation exceeds 'gtol'"
        # even when constraint_residual_inf is already far below the
        # ~1 kvar bar that actually matters for this problem (observed
        # directly: cres=0.028 kvar, essentially exact, reported
        # alongside success=False before this fix).
        converged = constraint_residual_inf < tol_constraint_kw

        return ScipyEstimationResult(
            x_v=x_v_final, x_delta=x_delta_final, converged=converged,
            iterations=int(result.niter) if hasattr(result, "niter") else int(result.nit),
            nfev=int(result.nfev), constraint_residual_inf=constraint_residual_inf,
            J=Jval, status=int(result.status), message=str(result.message),
        )
