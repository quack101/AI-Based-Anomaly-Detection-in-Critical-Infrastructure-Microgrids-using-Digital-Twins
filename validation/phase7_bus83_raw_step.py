"""
RES.md D10 Phase 7 BUS-83 INVESTIGATION -- raw Newton step, iteration 1.

Read-only diagnostic: replicates estimation.state_estimator.
StateEstimator.estimate()'s iteration-1 logic verbatim (no changes to
the real file), capturing the RAW proposed Newton step for bus 83
phase C's voltage coordinate -- y_full = y_particular + Z@w, converted
to actual-volt dx -- BEFORE any STEP_FRACTIONS multiplier or trust-
region (step_ratio) clipping is applied. Also checks, for each tried
(lambda, frac) combination, whether that combination was even tried
and/or accepted, and what dx_833 ends up being at each.

Run: python -m validation.phase7_bus83_raw_step
"""

import numpy as np
import pandas as pd
from pathlib import Path

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import (
    StateEstimator, LM_LAMBDA_INIT, LM_MAX_TRIES, LM_DECREASE_FACTOR,
    LM_INCREASE_FACTOR, LM_LAMBDA_MAX, LM_LAMBDA_MIN, STEP_FRACTIONS,
    MAX_STEP_V_PU, MAX_STEP_DELTA_RAD, NULLSPACE_RANK_TOL, LINESEARCH_CONSTRAINT_PENALTY,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
TARGET_I = 46


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_true_all, x_delta_true_all = npz["x_v"], npz["x_delta"]

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    v_base_833 = state.v_base["83.3"]
    v_idx_833 = state.v_index["83.3"]
    true_v_833 = x_v_true_all[TARGET_I][v_idx_833]

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

    current_merit, H_all, h0_all = _merit(x_v, x_delta)

    orig_err_833 = abs(x_v[v_idx_833] - true_v_833) / v_base_833
    print(f"=== i={TARGET_I}, iteration 1, raw Newton step for bus 83.3 ===")
    print(f"flat-start V_83.3 error: {orig_err_833:.6f} pu")
    print()

    # ---- iteration 1 setup (verbatim) ----
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

    lm_lambda = LM_LAMBDA_INIT
    found_accept = False

    for try_num in range(LM_MAX_TRIES):
        A_aug = np.vstack([A_reduced, np.sqrt(lm_lambda) * np.eye(n_w)])
        b_aug = np.concatenate([b_reduced, np.zeros(n_w)])
        w, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
        y_full = y_particular + Z @ w

        # RAW proposed step (before ANY frac multiplier or trust-region clip)
        dx_full = s * y_full
        dx_833_raw = dx_full[v_idx_833]

        if try_num == 0:
            print(f"--- LM try 0 (lambda={lm_lambda}), RAW y_full (no frac, no clip) ---")
            print(f"1. raw proposed dx for bus 83.3 (actual volts): {dx_833_raw:+.6f} V")
            print(f"   raw proposed dx for bus 83.3 (pu of v_base): {dx_833_raw/v_base_833:+.6f} pu")
            v_if_only_833 = x_v[v_idx_833] + dx_833_raw
            new_err_833_only = abs(v_if_only_833 - true_v_833) / v_base_833
            print()
            print(f"2. IF ONLY this one coordinate's raw step were applied:")
            print(f"   x_v_833 before: {x_v[v_idx_833]:.4f} V  ({x_v[v_idx_833]/v_base_833:.6f} pu), err={orig_err_833:.6f} pu")
            print(f"   x_v_833 after:  {v_if_only_833:.4f} V  ({v_if_only_833/v_base_833:.6f} pu), err={new_err_833_only:.6f} pu")
            print(f"   error got {'SMALLER' if new_err_833_only < orig_err_833 else 'BIGGER'} "
                  f"(delta={new_err_833_only-orig_err_833:+.6f} pu)")
            print()
            print("   (direction check: true V_833 is HIGHER than flat start by "
                  f"{(true_v_833-x_v[v_idx_833])/v_base_833:+.6f} pu -- the raw step needs to be "
                  f"POSITIVE and roughly that size to be 'the right way'.)")
            print()

        print(f"--- LM try {try_num} (lambda={lm_lambda:.3e}) ---")
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
            dx_833_applied = cand_dx_v[v_idx_833]

            cand_v = x_v + cand_dx_v
            cand_delta = x_delta + cand_dx_delta
            cand_merit, cand_H, cand_h0 = _merit(cand_v, cand_delta)

            accepted = cand_merit < current_merit
            print(f"  frac={frac:<5}: global step_ratio={step_ratio:.4e} (clip {'ENGAGED' if step_ratio>1.000001 else 'no-op'})  "
                  f"dx_833_RAW(this frac, pre-clip)={dx_v0[v_idx_833]:+.6f}V  "
                  f"dx_833_APPLIED(post-clip)={dx_833_applied:+.6f}V ({dx_833_applied/v_base_833:+.6e} pu)  "
                  f"merit={cand_merit:.6e}  accepted={accepted}")

            if accepted:
                found_accept = True
                break
        if found_accept:
            break
        lm_lambda = min(lm_lambda * LM_INCREASE_FACTOR, LM_LAMBDA_MAX)

    print()
    print(f"3. Did the line search try anything close to the raw step size for 83.3?")
    print(f"   raw (unclipped) dx_833 at try 0, frac=1.0: {dx_full[v_idx_833]:+.6f} V")
    print(f"   -> compare to the 'dx_833_APPLIED' values printed per frac above.")


if __name__ == "__main__":
    main()
