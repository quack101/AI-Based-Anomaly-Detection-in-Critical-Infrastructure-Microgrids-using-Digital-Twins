"""
RES.md D10 Phase 7 BUS-83 INVESTIGATION -- LM lambda trajectory.

Read-only diagnostic: same verbatim solve-loop replica as
phase7_bus83_iteration_trace.py, but logging the LM lambda value at
the START of every iteration (before that iteration's LM try-loop
runs), 1 through 50, at i=46. Does not modify estimation/state_estimator.py.

Run: python -m validation.phase7_bus83_lambda_trace
"""

import numpy as np
import pandas as pd
from pathlib import Path

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
TARGET_I = 46
FORCE_MAX_ITER = 50


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)

    row = df.iloc[TARGET_I]
    real_timestamp = row["real_timestamp"]
    z_real = {c: float(row[c]) for c in z_cols}

    hour = int(real_timestamp[11:13])
    weighted_entries = est._weighted_entries_for_hour(hour)
    z = np.array([e["z_value"] if "z_value" in e else z_real[e["id"]] for e in weighted_entries])
    sigma = np.array([e["sigma"] for e in weighted_entries])
    W = 1.0 / (sigma ** 2)
    alpha = 1.0 / W.max()
    sqrt_Ws = np.sqrt(alpha * W)

    all_entries = weighted_entries + model.zi_entries
    m = len(weighted_entries)
    p = est.p_constraints
    s = est._state_scale
    v_base = s[:state.n_phase_nodes]

    x_v = np.array(state.x0_v, dtype=float)
    x_delta = np.array(state.x0_delta, dtype=float)

    def _merit(xv, xd):
        at_v = complex_v_from_state(state, xv, xd)
        H_c, h0_c = J.build_jacobian(all_entries, model.Y, model.node_index, state, at_v=at_v)
        dz_c = z - h0_c[:m]
        dc_c = -h0_c[m:]
        merit_val = float(np.sum(W * dz_c ** 2) + LINESEARCH_CONSTRAINT_PENALTY * np.sum(dc_c ** 2))
        return merit_val, H_c, h0_c

    lm_lambda = LM_LAMBDA_INIT
    current_merit, H_all, h0_all = _merit(x_v, x_delta)

    print(f"=== i={TARGET_I}, lambda trajectory, iterations 1..{FORCE_MAX_ITER} ===")
    print(f"LM_LAMBDA_INIT={LM_LAMBDA_INIT}  LM_LAMBDA_MAX={LM_LAMBDA_MAX}  LM_LAMBDA_MIN={LM_LAMBDA_MIN}  "
          f"INCREASE_FACTOR={LM_INCREASE_FACTOR}  DECREASE_FACTOR={LM_DECREASE_FACTOR}")
    print()

    for iterations in range(1, FORCE_MAX_ITER + 1):
        lambda_start_of_iter = lm_lambda

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
        lm_tries_used = 0
        for _ in range(LM_MAX_TRIES):
            lm_tries_used += 1
            A_aug = np.vstack([A_reduced, np.sqrt(lm_lambda) * np.eye(n_w)])
            b_aug = np.concatenate([b_reduced, np.zeros(n_w)])
            w, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
            y_full = y_particular + Z @ w

            for frac in STEP_FRACTIONS:
                y = frac * y_full
                dx = s * y
                dx_v0, dx_delta0 = dx[:state.n_phase_nodes], dx[state.n_phase_nodes:]
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
            dx_v = np.zeros(state.n_phase_nodes)
            dx_delta = np.zeros(state.dim - state.n_phase_nodes)
        else:
            current_merit, x_v, x_delta, dx_v, dx_delta, H_all, h0_all = best

        print(f"iter={iterations:<3d} lambda_start={lambda_start_of_iter:.6e}  "
              f"lm_tries_used={lm_tries_used:<3d} improved={str(improved):<5s} "
              f"lambda_end={lm_lambda:.6e}  merit={current_merit:.6e}")

    print()
    print(f"final lambda value: {lm_lambda:.6e}  (LM_LAMBDA_MAX={LM_LAMBDA_MAX})")


if __name__ == "__main__":
    main()
