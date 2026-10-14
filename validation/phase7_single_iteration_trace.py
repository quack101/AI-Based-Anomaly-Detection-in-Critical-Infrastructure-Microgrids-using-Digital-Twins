"""
RES.md D10 Phase 7 SOLVER-INSTABILITY FIX -- deep single-iteration trace.

Per explicit review request: instrument the ORIGINAL (unfixed) solver's
FIRST Gauss-Newton iteration end-to-end at the exact failing repro point
(i=46, delta_frac=0.001, the SAME point validation/
phase7_circularity_diagnostic.py uses, reproducing its EXACT random draw
sequence so x0 matches bit-for-bit), rather than guessing at another
fix. Reports, in order:

  1. y_particular's raw (UNCLIPPED, UNDAMPED) magnitude -- and whether a
     Byrd-Omojokun-style trust-region clip on it alone would even
     ENGAGE (ratio > 1) at this point, so a prior "the clip made things
     worse" result isn't silently re-testing a no-op clip under a new
     label.
  2. The tangential step Z@w alone (first LM try, lambda=LM_LAMBDA_INIT).
  3. The combined step actually applied this iteration (y_particular +
     Z@w, after the existing step-fraction search and global
     trust-region clip -- whichever (fraction, outcome) the CURRENT
     unfixed code would actually pick).
  4. g(x) [h_zi] re-evaluated TWO ways at the candidate point:
       (a) LINEAR PREDICTION: h0_zi(x0) + C @ dx_applied (first-order
           model, using the Jacobian evaluated AT x0)
       (b) ACTUAL (nonlinear): h_zi evaluated via a fresh Jacobian
           build AT the candidate point (what _merit() uses to accept/
           reject)
     If (a) is already huge, the step itself (as sized from Cs/
     y_particular) is the problem. If (a) is modest but (b) is huge,
     the step carries the state into a region where the LINEARISATION
     badly mispredicts the TRUE nonlinear constraint -- a model-
     validity problem, not a step-SIZE problem.

Run: python -m validation.phase7_single_iteration_trace
"""

import numpy as np
import pandas as pd
from pathlib import Path

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from observability.pseudo_measurements import build_pseudo_entries_for_hour
from estimation.state_estimator import (
    NULLSPACE_RANK_TOL, MAX_STEP_V_PU, MAX_STEP_DELTA_RAD,
    STEP_FRACTIONS, LM_LAMBDA_INIT, LINESEARCH_CONSTRAINT_PENALTY,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"

SEED = 12345
PICKS = [46, 368]
DELTA_FRACS = [0.0, 0.001, 0.01, 0.05, 0.10]
TARGET_I = 46
TARGET_FRAC = 0.001


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_all, x_delta_all = npz["x_v"], npz["x_delta"]

    model = build_estimator_model("multiphase")
    state = model.state
    v_base = np.array([state.v_base[name] for name in state.all_node_names])
    delta_ref = 2.0 * np.pi / 3.0

    # --- reproduce the EXACT random draw sequence of
    # phase7_circularity_diagnostic.py up to (i=TARGET_I, frac=TARGET_FRAC) ---
    rng = np.random.default_rng(SEED)
    x0_v = x0_delta = None
    for i in PICKS:
        xt_v, xt_delta = x_v_all[i], x_delta_all[i]
        for frac in DELTA_FRACS:
            unit_v = rng.uniform(-1.0, 1.0, size=v_base.shape)
            unit_delta = rng.uniform(-1.0, 1.0, size=xt_delta.shape)
            if i == TARGET_I and frac == TARGET_FRAC:
                x0_v = xt_v + frac * unit_v * v_base
                x0_delta = xt_delta + frac * unit_delta * delta_ref
                break
        if x0_v is not None:
            break
    assert x0_v is not None

    row = df.iloc[TARGET_I]
    real_ts = row["real_timestamp"]
    z_real = {c: float(row[c]) for c in z_cols}
    hour = int(real_ts[11:13])

    weighted_entries = model.meas_entries + model.reg_pseudo_entries + build_pseudo_entries_for_hour(
        model.bus_phase_by_load, model.per_hour_by_load, hour)
    all_entries = weighted_entries + model.zi_entries
    m = len(weighted_entries)
    p = len(model.zi_entries)
    s = np.concatenate([v_base, np.ones(state.dim - state.n_phase_nodes)])

    z = np.array([e["z_value"] if "z_value" in e else z_real[e["id"]] for e in weighted_entries])
    sigma = np.array([e["sigma"] for e in weighted_entries])
    W = 1.0 / (sigma ** 2)
    alpha = 1.0 / W.max()
    sqrt_Ws = np.sqrt(alpha * W)

    def merit(xv, xd):
        at_v = complex_v_from_state(state, xv, xd)
        H_c, h0_c = J.build_jacobian(all_entries, model.Y, model.node_index, state, at_v=at_v)
        dz_c = z - h0_c[:m]
        dc_c = -h0_c[m:]
        return float(np.sum(W * dz_c ** 2) + LINESEARCH_CONSTRAINT_PENALTY * np.sum(dc_c ** 2)), H_c, h0_c

    x_v, x_delta = x0_v, x0_delta
    current_merit, H_all, h0_all = merit(x_v, x_delta)

    H_meas, h0_meas = H_all[:m], h0_all[:m]
    C, h0_zi = H_all[m:], h0_all[m:]
    dz = z - h0_meas
    dc = -h0_zi
    Hs_meas = H_meas * s[None, :]
    Cs = C * s[None, :]

    print("=== STEP 0: starting point (x0 = x_true + 0.1% perturbation, i=46) ===")
    print("max|dc| (constraint violation at x0):", np.max(np.abs(dc)))
    print("current_merit at x0:", current_merit)
    print()

    # --- 1. y_particular: raw, unclipped, undamped ---
    y_particular, *_ = np.linalg.lstsq(Cs, dc, rcond=None)
    dx_particular = s * y_particular
    dx_pv = dx_particular[:state.n_phase_nodes]
    dx_pdelta = dx_particular[state.n_phase_nodes:]
    particular_ratio = max(
        np.max(np.abs(dx_pv) / v_base) / MAX_STEP_V_PU,
        np.max(np.abs(dx_pdelta)) / MAX_STEP_DELTA_RAD,
        1.0,
    )
    print("=== 1. y_particular (raw, UNCLIPPED) ===")
    print("||y_particular||_2:", np.linalg.norm(y_particular))
    print("max|dx_particular_V| (volts):", np.max(np.abs(dx_pv)), " as pu-equiv (/v_base/MAX_STEP_V_PU):",
          np.max(np.abs(dx_pv) / v_base) / MAX_STEP_V_PU)
    print("max|dx_particular_delta| (rad):", np.max(np.abs(dx_pdelta)), " as frac of cap (/MAX_STEP_DELTA_RAD):",
          np.max(np.abs(dx_pdelta)) / MAX_STEP_DELTA_RAD)
    print("particular_ratio (would a trust-region clip on y_particular ALONE engage? ratio>1 means YES):",
          particular_ratio)
    y_particular_clipped = y_particular / particular_ratio
    print("||y_particular_clipped||_2 (if clip applied):", np.linalg.norm(y_particular_clipped),
          " -- shrink factor:", particular_ratio)
    print()

    # --- 2. null space + tangential step w (first LM try, lambda=LM_LAMBDA_INIT) ---
    _, Sigma, Vt = np.linalg.svd(Cs, full_matrices=True)
    rank_c = int(np.sum(Sigma > NULLSPACE_RANK_TOL * Sigma[0])) if Sigma.size else 0
    Z = Vt[rank_c:].T
    A_reduced = (Hs_meas * sqrt_Ws[:, None]) @ Z
    n_w = A_reduced.shape[1]
    b_reduced = sqrt_Ws * (dz - Hs_meas @ y_particular)  # UNCLIPPED y_particular, matching current shipped code
    lm_lambda = LM_LAMBDA_INIT
    A_aug = np.vstack([A_reduced, np.sqrt(lm_lambda) * np.eye(n_w)])
    b_aug = np.concatenate([b_reduced, np.zeros(n_w)])
    w, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    tangential = Z @ w
    dx_tangential = s * tangential
    print("=== 2. tangential step Z@w alone (first LM try, lambda=1.0, y_particular UNCLIPPED) ===")
    print("||w||_2:", np.linalg.norm(w), " ||Z@w||_2:", np.linalg.norm(tangential))
    print("max|dx_tangential_V| (volts):", np.max(np.abs(dx_tangential[:state.n_phase_nodes])))
    print("max|dx_tangential_delta| (rad):", np.max(np.abs(dx_tangential[state.n_phase_nodes:])))
    print()

    # --- 3. combined step, as the CURRENT (unfixed) code actually applies it:
    # y_full = y_particular (UNCLIPPED) + Z@w, step-fraction search + global clip ---
    y_full = y_particular + tangential
    print("=== 3. combined step (y_particular UNCLIPPED + Z@w), as shipped code applies it ===")
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

        # (a) LINEAR PREDICTION using Jacobian AT x0
        dx_applied = np.concatenate([cand_dx_v, cand_dx_delta])
        h_zi_linear_pred = h0_zi + C @ dx_applied

        # (b) ACTUAL nonlinear re-evaluation
        cand_merit, cand_H, cand_h0 = merit(cand_v, cand_delta)
        h_zi_actual = cand_h0[m:]

        accepted = cand_merit < current_merit
        print(f"-- frac={frac}: step_ratio={step_ratio:.3e} (global clip {'ENGAGED' if step_ratio>1.000001 else 'no-op'})")
        print(f"   max|dx_v applied| (V): {np.max(np.abs(cand_dx_v)):.4f}   max|dx_delta applied| (rad): {np.max(np.abs(cand_dx_delta)):.6f}")
        print(f"   LINEAR-PREDICTED max|h_zi| after step: {np.max(np.abs(h_zi_linear_pred)):.4f}")
        print(f"   ACTUAL (nonlinear) max|h_zi| after step: {np.max(np.abs(h_zi_actual)):.4f}")
        print(f"   current_merit={current_merit:.6e}  candidate_merit={cand_merit:.6e}  ACCEPTED={accepted}")
        print()


if __name__ == "__main__":
    main()
