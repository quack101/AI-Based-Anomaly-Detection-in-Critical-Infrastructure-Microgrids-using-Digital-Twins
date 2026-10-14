"""
RES.md D10 Phase 7/8 Q-ANGLE JACOBIAN BUG -- full verification, step 1.

Compares the analytic H from observability/jacobian.py against a
finite-difference Jacobian of the estimator's own h(x), at the clean
estimate of point i=46. Covers EVERY row the real solver actually
uses (meas_entries + reg_pseudo_entries + pseudo_entries +
zi_entries) against EVERY state column, bucketed by
(quantity-type x column-type).

Run: python -m validation.phase_q_angle_bug_verification
"""

import numpy as np
import pandas as pd
from pathlib import Path

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability.pseudo_measurements import build_pseudo_entries_for_hour
from observability import jacobian as J
from estimation.state_estimator import StateEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
TARGET_I = 46
EPS = 1e-5
MIN_MAG = 1e-9


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=50)

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)

    row = df.iloc[TARGET_I]
    real_ts = str(row["real_timestamp"])
    z_real = {c: float(row[c]) for c in z_cols}
    result_clean = est.estimate(z_real, real_ts)
    x_hat_v, x_hat_delta = result_clean.x_v, result_clean.x_delta

    hour = int(real_ts[11:13])
    pseudo_entries = build_pseudo_entries_for_hour(model.bus_phase_by_load, model.per_hour_by_load, hour)
    all_entries = model.meas_entries + model.reg_pseudo_entries + pseudo_entries + model.zi_entries
    quantities = [e["quantity"] for e in all_entries]
    print(f"total rows under test: {len(all_entries)}  (quantities: "
          f"{ {q: quantities.count(q) for q in set(quantities)} })")

    at_v = complex_v_from_state(state, x_hat_v, x_hat_delta)
    H_an, h0_an = J.build_jacobian(all_entries, model.Y, model.node_index, state, at_v=at_v)

    n = H_an.shape[1]
    nV = state.n_phase_nodes
    H_fd = np.zeros_like(H_an)
    for col in range(n):
        xv_p, xd_p = x_hat_v.copy(), x_hat_delta.copy()
        xv_m, xd_m = x_hat_v.copy(), x_hat_delta.copy()
        if col < nV:
            xv_p[col] += EPS; xv_m[col] -= EPS
        else:
            d = col - nV
            xd_p[d] += EPS; xd_m[d] -= EPS
        _, h0_p = J.build_jacobian(all_entries, model.Y, model.node_index, state,
                                    at_v=complex_v_from_state(state, xv_p, xd_p))
        _, h0_m = J.build_jacobian(all_entries, model.Y, model.node_index, state,
                                    at_v=complex_v_from_state(state, xv_m, xd_m))
        H_fd[:, col] = (h0_p - h0_m) / (2 * EPS)

    # bucket by (quantity-group, column-group)
    def qgroup(q):
        if q in ("P_injection", "P_head", "P_flow"):
            return "P"
        if q in ("Q_injection", "Q_head", "Q_flow"):
            return "Q"
        if q == "V_magnitude":
            return "V_magnitude"
        return "other:" + q

    row_groups = np.array([qgroup(q) for q in quantities])
    col_groups = np.array(["V"] * nV + ["angle"] * (n - nV))

    results = []
    for rg in sorted(set(row_groups)):
        for cg in ["V", "angle"]:
            mask = np.outer(row_groups == rg, col_groups == cg)
            analytic_vals = H_an[mask]
            fd_vals = H_fd[mask]
            keep = np.abs(analytic_vals) > MIN_MAG
            n_total = int(mask.sum())
            n_compared = int(keep.sum())
            if n_compared == 0:
                results.append({"row_block": rg, "col_block": cg, "n_total": n_total,
                                 "n_compared": 0, "frac_relerr_gt_1e-3": None, "frac_sign_flip": None})
                continue
            a = analytic_vals[keep]
            f = fd_vals[keep]
            relerr = np.abs(a - f) / np.abs(a)
            sign_flip = (np.sign(a) != np.sign(f)) & (np.abs(f) > MIN_MAG)
            results.append({
                "row_block": rg, "col_block": cg, "n_total": n_total, "n_compared": n_compared,
                "frac_relerr_gt_1e-3": float(np.mean(relerr > 1e-3)),
                "frac_sign_flip": float(np.mean(sign_flip)),
            })

    out = pd.DataFrame(results)
    pd.set_option("display.width", 160)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
