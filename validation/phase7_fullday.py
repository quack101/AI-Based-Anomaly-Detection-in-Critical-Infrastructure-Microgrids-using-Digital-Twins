"""
RES.md D10 Phase 7 -- validation steps 2 and 3: run the estimator across
a full day of the Phase-6 TEST split and report estimation-error
statistics plus convergence-rate/iteration statistics across the whole
run (not a single cherry-picked timestamp).

Streaming-style operation: each timestamp's converged solution seeds
the next timestamp's starting point (only the FIRST timestamp of the
day is a flat start) -- this is how a continuously-running state
estimator actually operates, and is cheaper than a flat start every
minute since the true state moves only slightly minute to minute.

Run: python validation/phase7_fullday.py
Output: validation/phase7_fullday_results.csv (per-timestamp records)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase7_fullday_results.csv"

N_DAY = 1440  # rows 0..1439 == 2018-05-25, the first test-split day


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=N_DAY)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_all, x_delta_all = npz["x_v"][:N_DAY], npz["x_delta"][:N_DAY]

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    v_base = np.array([state.v_base[n] for n in state.all_node_names])

    print("m_weighted:", est.m_weighted, "p_constraints:", est.p_constraints, "dof:", est.dof, flush=True)

    records = []
    t_start = time.time()
    warm_v, warm_delta = None, None

    for i in range(N_DAY):
        row = df.iloc[i]
        real_ts = row["real_timestamp"]
        z_real = {c: float(row[c]) for c in z_cols}

        if warm_v is not None:
            state.x0_v = list(warm_v)
            state.x0_delta = list(warm_delta)

        t0 = time.time()
        result = est.estimate(z_real, real_ts)
        dt = time.time() - t0

        warm_v, warm_delta = result.x_v, result.x_delta

        err_v_pu = np.abs(result.x_v - x_v_all[i]) / v_base
        err_delta = np.abs(result.x_delta - x_delta_all[i])

        records.append({
            "i": i, "real_timestamp": real_ts, "converged": result.converged,
            "iterations": result.iterations, "time_s": dt,
            "max_v_err_pu": err_v_pu.max(), "median_v_err_pu": float(np.median(err_v_pu)),
            "max_delta_err_rad": err_delta.max(), "median_delta_err_rad": float(np.median(err_delta)),
            "constraint_residual_inf": result.constraint_residual_inf,
        })

        if (i + 1) % 120 == 0:
            elapsed = time.time() - t_start
            print(f"  {i + 1}/{N_DAY} done, elapsed={elapsed:.0f}s, last real_ts={real_ts}", flush=True)

    out = pd.DataFrame(records)
    out.to_csv(OUT_PATH, index=False)

    print()
    print("=== SUMMARY ===")
    print("total timestamps:", len(out))
    print("convergence rate:", out["converged"].mean())
    print("iterations: mean=%.2f median=%.1f max=%d" % (
        out["iterations"].mean(), out["iterations"].median(), out["iterations"].max()))
    print("time_s: mean=%.2f median=%.2f max=%.2f total=%.0f" % (
        out["time_s"].mean(), out["time_s"].median(), out["time_s"].max(), out["time_s"].sum()))
    print("max_v_err_pu: mean=%.5f median=%.5f p95=%.5f max=%.5f" % (
        out["max_v_err_pu"].mean(), out["max_v_err_pu"].median(),
        out["max_v_err_pu"].quantile(0.95), out["max_v_err_pu"].max()))
    print("median_v_err_pu: mean=%.5f max=%.5f" % (out["median_v_err_pu"].mean(), out["median_v_err_pu"].max()))
    print("max_delta_err_rad: mean=%.5f median=%.5f p95=%.5f max=%.5f" % (
        out["max_delta_err_rad"].mean(), out["max_delta_err_rad"].median(),
        out["max_delta_err_rad"].quantile(0.95), out["max_delta_err_rad"].max()))
    print("constraint_residual_inf: mean=%.2f median=%.2f max=%.2f" % (
        out["constraint_residual_inf"].mean(), out["constraint_residual_inf"].median(),
        out["constraint_residual_inf"].max()))
    print()
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
