"""
RES.md D10 Phase 7 REMEDIATION -- Step 2, conditions A and B, stratified.

A full 1440-timestep run of either condition was found computationally
infeasible with the current shipped estimator (LM-damping + combined
step-size/constraint-residual criterion): a 4-point flat-start probe
(2018-05-25 00:00-00:03) took 68-129s/timestep, ALL FOUR hitting the
30-iteration cap without satisfying constraint_residual_inf < 1.0; a
2-point causal-warm-start probe at the SAME minutes was no faster
(79s, 161s) and got stuck at an IDENTICAL constraint_residual_inf
(100.467) at both points -- direct evidence the stall is a property of
this regime's reduced system, not of the starting point. Per the bail
condition set before this run, B is capped at the same stratified
sample as A rather than extended to the full 1440.

Design: 30 timestamps stratified by total feeder P tertile (10 each
low/mid/high, seed 777 -- same picks as phase7_stratified_flatstart.py)
across the 1440-point TEST day. For EACH pick i:
  A. flat start at i (state.x0_v/x0_delta = the model's own nominal
     flat start, fresh every time).
  B. causal warm start at i+1, SEEDED FROM A's x_hat at i (whether or
     not A actually converged -- this is what a real streaming
     estimator does: it warm-starts from whatever the previous solve
     produced, converged or not). This gives a genuine one-step-causal
     comparison at all 30 locations without requiring a continuous
     1440-step run.

Run: python validation/phase7_stratified_AB.py
Output: validation/phase7_stratified_AB_results.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase7_stratified_AB_results.csv"

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

    return {"low": pick(low, N_PER_STRATUM), "mid": pick(mid, N_PER_STRATUM), "high": pick(high, N_PER_STRATUM)}


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
            # --- condition A: flat start at i ---
            row_a = df.iloc[i]
            real_ts_a = row_a["real_timestamp"]
            z_a = {c: float(row_a[c]) for c in z_cols}

            state.x0_v = list(flat_v)
            state.x0_delta = list(flat_delta)

            t0 = time.time()
            result_a = est.estimate(z_a, real_ts_a)
            dt_a = time.time() - t0

            err_v_a = np.abs(result_a.x_v - x_v_all[i]) / v_base
            err_delta_a = np.abs(result_a.x_delta - x_delta_all[i])

            rec_a = {
                "condition": "A_flat", "stratum": stratum_name, "i": i, "real_timestamp": real_ts_a,
                "converged": result_a.converged, "iterations": result_a.iterations, "time_s": dt_a,
                "max_v_err_pu": err_v_a.max(), "median_v_err_pu": float(np.median(err_v_a)),
                "max_delta_err_rad": err_delta_a.max(),
                "constraint_residual_inf": result_a.constraint_residual_inf,
            }
            records.append(rec_a)
            print(f"[A_flat/{stratum_name}] i={i} real_ts={real_ts_a} converged={result_a.converged} "
                  f"iters={result_a.iterations:<3d} time_s={dt_a:.2f} maxVerr={err_v_a.max():.5f} "
                  f"cres={result_a.constraint_residual_inf:.3f}", flush=True)
            pd.DataFrame(records).to_csv(OUT_PATH, index=False)

            # --- condition B: causal warm start at i+1, seeded from A's x_hat at i ---
            j = i + 1
            if j >= len(df):
                continue
            row_b = df.iloc[j]
            real_ts_b = row_b["real_timestamp"]
            z_b = {c: float(row_b[c]) for c in z_cols}

            state.x0_v = list(result_a.x_v)
            state.x0_delta = list(result_a.x_delta)

            t0 = time.time()
            result_b = est.estimate(z_b, real_ts_b)
            dt_b = time.time() - t0

            err_v_b = np.abs(result_b.x_v - x_v_all[j]) / v_base
            err_delta_b = np.abs(result_b.x_delta - x_delta_all[j])

            rec_b = {
                "condition": "B_warm", "stratum": stratum_name, "i": j, "real_timestamp": real_ts_b,
                "converged": result_b.converged, "iterations": result_b.iterations, "time_s": dt_b,
                "max_v_err_pu": err_v_b.max(), "median_v_err_pu": float(np.median(err_v_b)),
                "max_delta_err_rad": err_delta_b.max(),
                "constraint_residual_inf": result_b.constraint_residual_inf,
            }
            records.append(rec_b)
            print(f"[B_warm/{stratum_name}] i={j} real_ts={real_ts_b} converged={result_b.converged} "
                  f"iters={result_b.iterations:<3d} time_s={dt_b:.2f} maxVerr={err_v_b.max():.5f} "
                  f"cres={result_b.constraint_residual_inf:.3f}", flush=True)
            pd.DataFrame(records).to_csv(OUT_PATH, index=False)

    out = pd.DataFrame(records)
    print()
    print("=== SUMMARY BY CONDITION ===")
    print(out.groupby("condition")[["converged", "iterations", "time_s", "max_v_err_pu", "constraint_residual_inf"]].agg(
        ["mean", "median", "max"]))
    print()
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
