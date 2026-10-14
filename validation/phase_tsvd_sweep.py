"""
RES.md D10 Phase 7 fix-jacobian-bugs branch -- DIAGNOSTIC ONLY, Part B.
Tests the experimental truncated-SVD step solver
(estimation/experimental_tsvd_solver.py, default-off, not the real
solver) on 3 points x 3 rcond values. Does not modify the default
solver path.

Run: python -m validation.phase_tsvd_sweep
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator
from estimation.experimental_tsvd_solver import estimate_tsvd

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"

RCONDS = [1e-6, 1e-8, 1e-10]
POINTS = [("i=46 (A_flat)", 46, "flat"), ("i=94 (A_flat)", 94, "flat"), ("i=95 (B_warm)", 95, "warm_from_94")]


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

    # warm-start source for i=95: real 30-iter estimate() on i=94
    row94 = df.iloc[94]
    z94 = {c: float(row94[c]) for c in z_cols}
    state.x0_v, state.x0_delta = list(flat_v), list(flat_delta)
    result94 = est.estimate(z94, str(row94["real_timestamp"]))

    rows = []
    for label, i, start_kind in POINTS:
        row = df.iloc[i]
        real_ts = str(row["real_timestamp"])
        z_real = {c: float(row[c]) for c in z_cols}
        x0_v = flat_v if start_kind == "flat" else list(result94.x_v)
        x0_delta = flat_delta if start_kind == "flat" else list(result94.x_delta)

        for rcond in RCONDS:
            t0 = time.time()
            res = estimate_tsvd(est, z_real, real_ts, x0_v, x0_delta, rcond=rcond, max_iter=40)
            dt = time.time() - t0
            v_err = float(np.max(np.abs(res["x_v"] - x_v_true_all[i]) / v_base_full))
            rows.append({
                "point": label, "i": i, "rcond": rcond, "iterations": res["iterations"],
                "converged": res["converged"], "seconds": dt, "cost_final": res["cost_history"][-1],
                "J": res["J"], "v_err_pu": v_err, "cres": res["constraint_residual_inf"],
            })
            print(f"[{label}] rcond={rcond:.0e}: iters={res['iterations']:<3d} converged={res['converged']} "
                  f"time={dt:.2f}s  J={res['J']:.4e}  v_err_pu={v_err:.5f}  cres={res['constraint_residual_inf']:.4f}",
                  flush=True)

    out = pd.DataFrame(rows)
    print()
    print("=== FULL TABLE: rcond vs iterations/seconds/J/v_err/cres ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
