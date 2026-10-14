"""
RES.md D10 Phase 7 REMEDIATION -- Step 2 condition A, stratified sample.

A full 1440-timestep flat-start run (every timestep reset to the
nominal flat start, no warm start) was found computationally infeasible
with the current shipped estimator under the new combined (step-size
AND constraint-residual) convergence criterion: a 4-timestamp probe at
consecutive off-peak minutes (2018-05-25 00:00-00:03) took 68-129s each
and ALL FOUR hit the 30-iteration cap without satisfying
constraint_residual_inf < DEFAULT_TOL_CONSTRAINT_KW -- extrapolating,
a full 1440-point flat-start run would take on the order of 30-40+
hours. This script instead runs a STRATIFIED sample of 30 timestamps
(10 each from the low/mid/high tertile of total feeder P across the
1440-timestep TEST day, by total metered P -- not a uniform-random
draw, since Gauss-Newton basin size is plausibly load-dependent and a
naive random sample could cluster in one regime), fixed seed 777 for
reproducibility.

Run: python validation/phase7_stratified_flatstart.py
Output: validation/phase7_stratified_flatstart_results.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase7_stratified_flatstart_results.csv"

SEED = 777
N_PER_STRATUM = 10


def stratified_picks(df, p_cols):
    total_p = df[p_cols].sum(axis=1)
    q1, q2 = total_p.quantile([1 / 3, 2 / 3])
    low = df.index[total_p <= q1]
    mid = df.index[(total_p > q1) & (total_p <= q2)]
    high = df.index[total_p > q2]
    rng = np.random.default_rng(SEED)

    def pick(idx, k):
        return sorted(rng.choice(idx, size=k, replace=False).tolist())

    picks_low = pick(low, N_PER_STRATUM)
    picks_mid = pick(mid, N_PER_STRATUM)
    picks_high = pick(high, N_PER_STRATUM)
    return {"low": picks_low, "mid": picks_mid, "high": picks_high}


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_all, x_delta_all = npz["x_v"][:1440], npz["x_delta"][:1440]

    p_cols = [c for c in df.columns if c.endswith("_P")]
    strata = stratified_picks(df, p_cols)
    print("stratum picks:", strata, flush=True)

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    v_base = np.array([state.v_base[n] for n in state.all_node_names])
    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)

    records = []
    for stratum_name, idxs in strata.items():
        for i in idxs:
            row = df.iloc[i]
            real_ts = row["real_timestamp"]
            z_real = {c: float(row[c]) for c in z_cols}

            state.x0_v = list(flat_v)
            state.x0_delta = list(flat_delta)

            t0 = time.time()
            result = est.estimate(z_real, real_ts)
            dt = time.time() - t0

            err_v_pu = np.abs(result.x_v - x_v_all[i]) / v_base
            err_delta = np.abs(result.x_delta - x_delta_all[i])

            rec = {
                "stratum": stratum_name, "i": i, "real_timestamp": real_ts,
                "converged": result.converged, "iterations": result.iterations,
                "time_s": dt, "max_v_err_pu": err_v_pu.max(),
                "median_v_err_pu": float(np.median(err_v_pu)),
                "max_delta_err_rad": err_delta.max(),
                "constraint_residual_inf": result.constraint_residual_inf,
            }
            records.append(rec)
            print(f"[{stratum_name}] i={i} real_ts={real_ts} converged={result.converged} "
                  f"iters={result.iterations:<3d} time_s={dt:.2f} maxVerr={err_v_pu.max():.5f} "
                  f"cres={result.constraint_residual_inf:.3f}", flush=True)

            pd.DataFrame(records).to_csv(OUT_PATH, index=False)

    out = pd.DataFrame(records)
    print()
    print("=== STRATIFIED FLAT-START SUMMARY (n=%d) ===" % len(out))
    print("convergence rate:", out["converged"].mean())
    print("iterations: mean=%.2f median=%.1f p95=%.1f max=%d" % (
        out["iterations"].mean(), out["iterations"].median(),
        out["iterations"].quantile(0.95), out["iterations"].max()))
    print("time_s: mean=%.2f median=%.2f max=%.2f" % (
        out["time_s"].mean(), out["time_s"].median(), out["time_s"].max()))
    print("max_v_err_pu: mean=%.5f max=%.5f" % (out["max_v_err_pu"].mean(), out["max_v_err_pu"].max()))
    print("constraint_residual_inf: mean=%.3f median=%.3f max=%.3f" % (
        out["constraint_residual_inf"].mean(), out["constraint_residual_inf"].median(),
        out["constraint_residual_inf"].max()))
    print()
    print("by stratum:")
    print(out.groupby("stratum")[["converged", "iterations", "time_s", "constraint_residual_inf"]].mean())
    print()
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
