"""
RES.md D10 Phase 7 -- validation step 1: "confirm the estimator
recovers x_true on a handful of Phase-6 normal timestamps to within a
small, stated tolerance. This is the basic sanity check -- if it fails
here, nothing downstream can be trusted."

10 timestamps spread across the full 9-day test split (different days
AND different hours-of-day, so different pseudo-measurement buckets are
exercised), each estimated from a FLAT start (no warm-start head start).
Run: python validation/phase7_sanity_check.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_all, x_delta_all = npz["x_v"], npz["x_delta"]

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    v_base = np.array([state.v_base[n] for n in state.all_node_names])

    print("m_weighted:", est.m_weighted, "p_constraints:", est.p_constraints, "dof:", est.dof)
    print()

    n = len(df)
    picks = [int(n * f) for f in [0.02, 0.13, 0.24, 0.35, 0.46, 0.57, 0.68, 0.79, 0.90, 0.99]]

    header = f"{'real_timestamp':20s} {'converged':10s} {'iters':6s} {'time_s':8s} {'maxVerr_pu':11s} {'medVerr_pu':11s} {'maxDerr_rad':12s}"
    print(header)
    for i in picks:
        row = df.iloc[i]
        real_ts = row["real_timestamp"]
        z_real = {c: float(row[c]) for c in z_cols}

        t0 = time.time()
        result = est.estimate(z_real, real_ts)
        dt = time.time() - t0

        err_v_pu = np.abs(result.x_v - x_v_all[i]) / v_base
        err_delta = np.abs(result.x_delta - x_delta_all[i])

        print(f"{real_ts:20s} {str(result.converged):10s} {result.iterations:<6d} {dt:<8.2f} "
              f"{err_v_pu.max():<11.5f} {np.median(err_v_pu):<11.5f} {err_delta.max():<12.5f}")


if __name__ == "__main__":
    main()
