"""
RES.md D10 Phase 7 OVERRUN TEST -- is ~4.5% error a stopping-too-
early bug or a true solver minimum?

For 5 points (mix of low/mid/high load stratum), runs the real
StateEstimator.estimate() normally (flat start) to find where it
actually declares converged (iteration count, final step size), then
replicates the exact same solve loop (verbatim logic, no changes to
estimation/state_estimator.py) forced to run 30 MORE iterations past
that point with no early-stop, logging max_v_err_pu (over all 278
nodes, vs x_true) at every iteration.

Run: python -m validation.phase7_overrun_test
"""

from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import (
    StateEstimator, DEFAULT_TOL_V_PU, DEFAULT_TOL_DELTA_RAD,
    DEFAULT_TOL_CONSTRAINT_KW, LM_LAMBDA_INIT, LM_MAX_TRIES, LM_DECREASE_FACTOR,
    LM_INCREASE_FACTOR, LM_LAMBDA_MAX, LM_LAMBDA_MIN, STEP_FRACTIONS,
    MAX_STEP_V_PU, MAX_STEP_DELTA_RAD, NULLSPACE_RANK_TOL, LINESEARCH_CONSTRAINT_PENALTY,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
EXTRA_ITERS = 30

TEST_POINTS = [46, 844, 824, 1016, 1104]  # low, low, mid, high, high


def run_forced(est, model, state, z_dict, real_timestamp, x_v_true, v_base, force_max_iter):
    hour = int(real_timestamp[11:13])
    weighted_entries = est._weighted_entries_for_hour(hour)
    z = np.array([e["z_value"] if "z_value" in e else z_dict[e["id"]] for e in weighted_entries])
    sigma = np.array([e["sigma"] for e in weighted_entries])
    W = 1.0 / (sigma ** 2)
    alpha = 1.0 / W.max()
    sqrt_Ws = np.sqrt(alpha * W)

    all_entries = weighted_entries + model.zi_entries
    m = len(weighted_entries)
    p = est.p_constraints
    s = est._state_scale
    v_base_scale = s[:state.n_phase_nodes]

    x_v = np.array(state.x0_v, dtype=float)
    x_delta = np.array(state.x0_delta, dtype=float)

    def _merit(xv, xd):
        at_v = complex_v_from_state(state, xv, xd)
        H_c, h0_c = J.build_jacobian(all_entries, model.Y, model.node_index, state, at_v=at_v)
        dz_c = z - h0_c[:m]
        dc_c = -h0_c[m:]
        merit_val = float(np.sum(W * dz_c ** 2) + LINESEARCH_CONSTRAINT_PENALTY * np.sum(dc_c ** 2))
        return merit_val, H_c, h0_c

    converged = False
    convergence_iter = None
    convergence_err = None
    lm_lambda = LM_LAMBDA_INIT
    current_merit, H_all, h0_all = _merit(x_v, x_delta)

    log = []
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

        max_v_err_pu = float(np.max(np.abs(x_v - x_v_true) / v_base))
        log.append((iterations, max_dx_v_pu, constraint_residual_iter, max_v_err_pu))

        would_converge = (max_dx_v_pu < DEFAULT_TOL_V_PU and max_dx_delta_rad < DEFAULT_TOL_DELTA_RAD
                           and constraint_residual_iter < DEFAULT_TOL_CONSTRAINT_KW)
        if would_converge and not converged:
            converged = True
            convergence_iter = iterations
            convergence_err = max_v_err_pu
        # no break -- forced to keep going

    return log, convergence_iter, convergence_err


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_all = npz["x_v"][:1440]

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    v_base = np.array([state.v_base[n] for n in state.all_node_names])

    print(f"{'i':<6s}{'normal_iters':<14s}{'normal_err_pu':<16s}{'forced_total_iters':<20s}"
          f"{'err_at_declared_conv':<24s}{'err_after_+30':<16s}{'delta':<14s}")

    for i in TEST_POINTS:
        row = df.iloc[i]
        real_ts = row["real_timestamp"]
        z_dict = {c: float(row[c]) for c in z_cols}

        state.x0_v = [state.v_base[n] for n in state.all_node_names]
        # reset to true flat start (matches StateEstimator's own x0 construction)
        model2 = model  # reuse

        result_normal = est.estimate(z_dict, real_ts)
        normal_err = float(np.max(np.abs(result_normal.x_v - x_v_all[i]) / v_base))

        force_total = result_normal.iterations + EXTRA_ITERS
        log, conv_iter, conv_err = run_forced(est, model, state, z_dict, real_ts, x_v_all[i], v_base, force_total)

        err_at_declared = conv_err if conv_err is not None else float("nan")
        err_after_30_more = log[-1][3]

        print(f"{i:<6d}{result_normal.iterations:<14d}{normal_err:<16.6f}{force_total:<20d}"
              f"{err_at_declared:<24.6f}{err_after_30_more:<16.6f}{err_after_30_more-err_at_declared:<+14.6f}")

        print(f"   full per-iteration log for i={i}:")
        for it, dxv, cres, everr in log:
            marker = " <-- declared converged" if it == conv_iter else ""
            print(f"     iter={it:<4d} max_dx_v_pu={dxv:.6f} cres={cres:.4f} max_v_err_pu={everr:.6f}{marker}")
        print()


if __name__ == "__main__":
    main()
