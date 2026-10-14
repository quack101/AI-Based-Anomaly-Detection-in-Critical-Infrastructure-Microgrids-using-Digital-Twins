"""
RES.md D10 Phase 7 fix-jacobian-bugs branch -- DIAGNOSTIC ONLY,
steps 3-5. Per-iteration trace (cost, gradient norm, scaled step
norm, LM lambda, accept/reject) for 3 points, run out to 100
iterations so step-30-vs-100 can be read directly off one trace.
Does not modify estimation/state_estimator.py.

Points: i=46 (A_flat), i=94 (A_flat), i=95 (B_warm -- warm-started
from i=94's converged/final x_v,x_delta, matching the real sweep's
own A/B pairing convention).

Run: python -m validation.phase_convergence_trace
"""

import numpy as np
import pandas as pd
from pathlib import Path

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import (
    StateEstimator, DEFAULT_MAX_ITER, DEFAULT_TOL_V_PU, DEFAULT_TOL_DELTA_RAD,
    DEFAULT_TOL_CONSTRAINT_KW, LM_LAMBDA_INIT, LM_MAX_TRIES, LM_DECREASE_FACTOR,
    LM_INCREASE_FACTOR, LM_LAMBDA_MAX, LM_LAMBDA_MIN, STEP_FRACTIONS,
    MAX_STEP_V_PU, MAX_STEP_DELTA_RAD, NULLSPACE_RANK_TOL, LINESEARCH_CONSTRAINT_PENALTY,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
FORCE_MAX_ITER = 100


def traced_solve(est, model, state, z_real, real_ts, x0_v, x0_delta, force_max_iter, v_base_full, x_v_true):
    hour = int(real_ts[11:13])
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

    converged = False
    convergence_iter = None
    lm_lambda = LM_LAMBDA_INIT
    current_merit, H_all, h0_all = _merit(x_v, x_delta)

    trace = []
    snapshot_30 = None

    for iterations in range(1, force_max_iter + 1):
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
        n_w = A_reduced.shape[1]
        grad_w = A_reduced.T @ (sqrt_Ws * (Hs_meas @ y_particular - dz))  # = -A_reduced^T b_reduced, gradient of 0.5||A w - b||^2

        improved = False
        n_lm_tries = 0
        for _ in range(LM_MAX_TRIES):
            n_lm_tries += 1
            A_aug = np.vstack([A_reduced, np.sqrt(lm_lambda) * np.eye(n_w)])
            b_aug = np.concatenate([b_reduced, np.zeros(n_w)])
            w, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
            y_full = y_particular + Z @ w

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
                    best = (cand_merit, cand_v, cand_delta, cand_dx_v, cand_dx_delta, cand_H, cand_h0)
                    lm_lambda = max(lm_lambda / LM_DECREASE_FACTOR, LM_LAMBDA_MIN)
                    improved = True
                    break
            if improved:
                break
            lm_lambda = min(lm_lambda * LM_INCREASE_FACTOR, LM_LAMBDA_MAX)

        if not improved:
            dx_v = np.zeros(state.n_phase_nodes)
            dx_delta = np.zeros(state.dim - state.n_phase_nodes)
        else:
            current_merit, x_v, x_delta, dx_v, dx_delta, H_all, h0_all = best

        max_dx_v_pu = float(np.max(np.abs(dx_v) / v_base_scale))
        max_dx_delta_rad = float(np.max(np.abs(dx_delta)))
        constraint_residual_iter = float(np.max(np.abs(h0_all[m:]))) if p else 0.0
        step_norm = float(np.hypot(np.linalg.norm(dx_v / v_base_scale), np.linalg.norm(dx_delta)))
        v_err_pu = float(np.max(np.abs(x_v - x_v_true) / v_base_full))

        trace.append({
            "iter": iterations, "cost": current_merit, "grad_norm": float(np.linalg.norm(grad_w)),
            "step_norm": step_norm, "max_dx_v_pu": max_dx_v_pu, "max_dx_delta_rad": max_dx_delta_rad,
            "cres": constraint_residual_iter, "lm_lambda_end": lm_lambda, "n_lm_tries": n_lm_tries,
            "accepted": improved, "v_err_pu": v_err_pu,
        })

        would_converge = (max_dx_v_pu < DEFAULT_TOL_V_PU and max_dx_delta_rad < DEFAULT_TOL_DELTA_RAD
                           and constraint_residual_iter < DEFAULT_TOL_CONSTRAINT_KW)
        if would_converge and not converged:
            converged = True
            convergence_iter = iterations

        if iterations == 30:
            snapshot_30 = {"cost": current_merit, "v_err_pu": v_err_pu, "converged_by_30": converged}

    return trace, converged, convergence_iter, snapshot_30


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_true_all = npz["x_v"]

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    v_base_full = np.array([state.v_base[n] for n in state.all_node_names])
    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)

    print(f"solver stopping criteria (from estimation/state_estimator.py): "
          f"max_dx_v_pu < DEFAULT_TOL_V_PU={DEFAULT_TOL_V_PU}, "
          f"max_dx_delta_rad < DEFAULT_TOL_DELTA_RAD={DEFAULT_TOL_DELTA_RAD}, "
          f"constraint_residual_iter < DEFAULT_TOL_CONSTRAINT_KW={DEFAULT_TOL_CONSTRAINT_KW} "
          f"-- ALL THREE must hold simultaneously.")
    print()

    row46 = df.iloc[46]

    row94 = df.iloc[94]
    z94 = {c: float(row94[c]) for c in z_cols}
    traceA94, convA94, citA94, snapA94 = traced_solve(
        est, model, state, z94, str(row94["real_timestamp"]), list(flat_v), list(flat_delta),
        FORCE_MAX_ITER, v_base_full, x_v_true_all[94],
    )
    print("=== i=94 (A_flat) trace (selected iterations) ===")
    for row in traceA94:
        if row["iter"] <= 10 or row["iter"] % 10 == 0:
            print(f"  iter={row['iter']:<4d} cost={row['cost']:.4e} grad_norm={row['grad_norm']:.4e} "
                  f"step_norm={row['step_norm']:.6f} lambda={row['lm_lambda_end']:.3e} "
                  f"tries={row['n_lm_tries']:<3d} accepted={row['accepted']} v_err_pu={row['v_err_pu']:.5f}")
    print(f"converged={convA94} at iter={citA94}  snapshot@30={snapA94}")
    print(f"final@100: cost={traceA94[-1]['cost']:.4e} v_err_pu={traceA94[-1]['v_err_pu']:.5f}")
    print()

    row46dict = {c: float(row46[c]) for c in z_cols}
    traceA46, convA46, citA46, snapA46 = traced_solve(
        est, model, state, row46dict, str(row46["real_timestamp"]), list(flat_v), list(flat_delta),
        FORCE_MAX_ITER, v_base_full, x_v_true_all[46],
    )
    print("=== i=46 (A_flat) trace (selected iterations) ===")
    for row in traceA46:
        if row["iter"] <= 10 or row["iter"] % 10 == 0:
            print(f"  iter={row['iter']:<4d} cost={row['cost']:.4e} grad_norm={row['grad_norm']:.4e} "
                  f"step_norm={row['step_norm']:.6f} lambda={row['lm_lambda_end']:.3e} "
                  f"tries={row['n_lm_tries']:<3d} accepted={row['accepted']} v_err_pu={row['v_err_pu']:.5f}")
    print(f"converged={convA46} at iter={citA46}  snapshot@30={snapA46}")
    print(f"final@100: cost={traceA46[-1]['cost']:.4e} v_err_pu={traceA46[-1]['v_err_pu']:.5f}")
    print()

    # B_warm: need i=94's actual FINAL x_v/x_delta to warm-start i=95.
    # Re-run i=94 with the real (non-traced) estimator quickly to get its x_v/x_delta at iter=30
    # (matching what the real sweep script actually does -- warm start from the NORMAL,
    # 30-iter-capped estimate() call, not from a forced 100-iter run).
    state.x0_v, state.x0_delta = list(flat_v), list(flat_delta)
    result94 = est.estimate(z94, str(row94["real_timestamp"]))

    row95 = df.iloc[95]
    z95 = {c: float(row95[c]) for c in z_cols}
    traceB95, convB95, citB95, snapB95 = traced_solve(
        est, model, state, z95, str(row95["real_timestamp"]), list(result94.x_v), list(result94.x_delta),
        FORCE_MAX_ITER, v_base_full, x_v_true_all[95],
    )
    print("=== i=95 (B_warm, warm-started from i=94's 30-iter result) trace (selected iterations) ===")
    for row in traceB95:
        if row["iter"] <= 10 or row["iter"] % 10 == 0:
            print(f"  iter={row['iter']:<4d} cost={row['cost']:.4e} grad_norm={row['grad_norm']:.4e} "
                  f"step_norm={row['step_norm']:.6f} lambda={row['lm_lambda_end']:.3e} "
                  f"tries={row['n_lm_tries']:<3d} accepted={row['accepted']} v_err_pu={row['v_err_pu']:.5f}")
    print(f"converged={convB95} at iter={citB95}  snapshot@30={snapB95}")
    print(f"final@100: cost={traceB95[-1]['cost']:.4e} v_err_pu={traceB95[-1]['v_err_pu']:.5f}")


if __name__ == "__main__":
    main()
