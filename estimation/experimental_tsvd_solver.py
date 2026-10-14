"""
RES.md D10 Phase 7 fix-jacobian-bugs branch -- EXPERIMENTAL ONLY,
default OFF, never imported by the default solver path
(estimation/state_estimator.py is untouched). Solves each reduced
Gauss-Newton step via a TRUNCATED SVD of A_reduced = sqrt(W)*H*Z
(dropping singular values below rcond*largest) INSTEAD of
Levenberg-Marquardt damping -- everything else (null-space
construction, backtracking line search STEP_FRACTIONS, trust-region
clip MAX_STEP_V_PU/MAX_STEP_DELTA_RAD) is verbatim from the real
solver. Stopping rule: relative cost decrease < rel_tol for
`patience` consecutive iterations, or max_iter.

Run standalone for a self-test: python -m estimation.experimental_tsvd_solver
"""

import time

import numpy as np

from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import (
    StateEstimator, STEP_FRACTIONS, MAX_STEP_V_PU, MAX_STEP_DELTA_RAD,
    NULLSPACE_RANK_TOL, LINESEARCH_CONSTRAINT_PENALTY,
)


def estimate_tsvd(est, z_real, real_timestamp, x0_v, x0_delta, rcond=1e-8,
                   max_iter=40, rel_tol=1e-4, patience=3):
    """Experimental: truncated-SVD step instead of LM damping. Returns
    a dict (not the real EstimationResult type, to keep this fully
    separate from the production return type)."""

    model = est.model
    state = model.state

    hour = int(real_timestamp[11:13])
    weighted_entries = est._weighted_entries_for_hour(hour)
    z = np.array([e["z_value"] if "z_value" in e else z_real[e["id"]] for e in weighted_entries])
    sigma = np.array([e["sigma"] for e in weighted_entries])
    W = 1.0 / sigma ** 2
    alpha = 1.0 / W.max()
    sqrt_Ws = np.sqrt(alpha * W)

    all_entries = weighted_entries + model.zi_entries
    m = len(weighted_entries)
    p = est.p_constraints
    s = est._state_scale
    v_base_scale = s[:state.n_phase_nodes]

    x_v = np.array(x0_v, dtype=float)
    x_delta = np.array(x0_delta, dtype=float)

    def _merit(xv, xd):
        at_v = complex_v_from_state(state, xv, xd)
        H_c, h0_c = J.build_jacobian(all_entries, model.Y, model.node_index, state, at_v=at_v)
        dz_c = z - h0_c[:m]
        dc_c = -h0_c[m:]
        merit_val = float(np.sum(W * dz_c ** 2) + LINESEARCH_CONSTRAINT_PENALTY * np.sum(dc_c ** 2))
        return merit_val, H_c, h0_c

    current_merit, H_all, h0_all = _merit(x_v, x_delta)
    cost_history = [current_merit]
    stall_count = 0
    converged_by_cost_rule = False
    t_start = time.time()
    iterations_used = 0

    for iterations in range(1, max_iter + 1):
        iterations_used = iterations
        H_meas, h0_meas = H_all[:m], h0_all[:m]
        C, h0_zi = H_all[m:], h0_all[m:]
        dz = z - h0_meas
        dc = -h0_zi
        Hs_meas = H_meas * s[None, :]
        Cs = C * s[None, :]

        y_particular, *_ = np.linalg.lstsq(Cs, dc, rcond=None)
        _, Sigma, Vt = np.linalg.svd(Cs, full_matrices=True)
        rank_c = int(np.sum(Sigma > NULLSPACE_RANK_TOL * Sigma[0])) if Sigma.size else 0
        Z = Vt[rank_c:].T

        A_reduced = (Hs_meas * sqrt_Ws[:, None]) @ Z
        b_reduced = sqrt_Ws * (dz - Hs_meas @ y_particular)

        # TRUNCATED SVD solve, replacing LM damping entirely
        U, sv, Vt2 = np.linalg.svd(A_reduced, full_matrices=False)
        keep = sv > rcond * sv[0]
        sv_inv = np.zeros_like(sv)
        sv_inv[keep] = 1.0 / sv[keep]
        w = (Vt2.T * sv_inv) @ (U.T @ b_reduced)
        y_full = y_particular + Z @ w

        improved = False
        for frac in STEP_FRACTIONS:
            y = frac * y_full
            dx = s * y
            dx_v0, dx_delta0 = dx[:state.n_phase_nodes], dx[state.n_phase_nodes:]
            step_ratio = max(
                np.max(np.abs(dx_v0) / v_base_scale) / MAX_STEP_V_PU,
                np.max(np.abs(dx_delta0)) / MAX_STEP_DELTA_RAD,
                1.0,
            )
            cand_dx_v, cand_dx_delta = dx_v0 / step_ratio, dx_delta0 / step_ratio
            cand_v = x_v + cand_dx_v
            cand_delta = x_delta + cand_dx_delta
            cand_merit, cand_H, cand_h0 = _merit(cand_v, cand_delta)

            if cand_merit < current_merit:
                current_merit, x_v, x_delta, H_all, h0_all = (
                    cand_merit, cand_v, cand_delta, cand_H, cand_h0)
                improved = True
                break

        cost_history.append(current_merit)
        rel_decrease = (cost_history[-2] - cost_history[-1]) / max(abs(cost_history[-2]), 1e-12)
        if 0 <= rel_decrease < rel_tol:
            stall_count += 1
        else:
            stall_count = 0
        if stall_count >= patience:
            converged_by_cost_rule = True
            break

    constraint_residual = float(np.max(np.abs(h0_all[m:]))) if p else 0.0
    elapsed = time.time() - t_start

    return {
        "x_v": x_v, "x_delta": x_delta, "converged": converged_by_cost_rule,
        "iterations": iterations_used, "J": float(np.sum(W * (z - h0_all[:m]) ** 2)),
        "constraint_residual_inf": constraint_residual, "elapsed_s": elapsed,
        "cost_history": cost_history,
    }


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    from observability.estimator_model import build_estimator_model

    BASE_DIR = Path(__file__).resolve().parent.parent
    CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=50)

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    row = df.iloc[46]
    real_ts = str(row["real_timestamp"])
    z_real = {c: float(row[c]) for c in z_cols}

    result = estimate_tsvd(est, z_real, real_ts, state.x0_v, state.x0_delta, rcond=1e-8)
    print(f"self-test: converged={result['converged']} iters={result['iterations']} "
          f"J={result['J']:.4e} elapsed={result['elapsed_s']:.2f}s")
