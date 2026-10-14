"""
RES.md D10 Phase 7 REMEDIATION -- Step 4 controlled ablation.

Isolates JUST the convergence-criterion change (step-size-only vs
step-size AND constraint-residual) from the OTHER fixes already baked
into estimation/state_estimator.py (LM damping on the reduced system,
the ratchet-divergence fix, the step-fraction search on the full step).
Calls the CURRENT, already-fixed estimator twice per sampled timestamp
-- once with tol_constraint_kw=inf (stops exactly as the pre-
remediation code did: step-size alone) and once with the shipped
default (DEFAULT_TOL_CONSTRAINT_KW=1.0) -- so the only difference
between the two columns is the stopping RULE itself, not the solver
internals.

Same 30-point stratified sample as phase7_stratified_AB.py (seed 777,
10 each low/mid/high feeder-P tertile) -- a full 1440-point run was
found computationally infeasible (see that script's docstring for the
timing evidence). Flat start at every sampled i (condition A's start
rule) -- the OLD criterion's own historical validation
(phase7_fullday_results.csv, 2026-08-30) used causal warm start, but
isolating the criterion effect cleanly requires holding the START RULE
fixed too, so flat start is used for both columns here.

Run: python -m validation.phase7_criterion_ablation
Output: validation/phase7_criterion_ablation_results.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase7_criterion_ablation_results.csv"

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
            row = df.iloc[i]
            real_ts = row["real_timestamp"]
            z_real = {c: float(row[c]) for c in z_cols}

            for label, tol_c in [("OLD_stepsize_only", float("inf")), ("NEW_plus_constraint", 1.0)]:
                state.x0_v = list(flat_v)
                state.x0_delta = list(flat_delta)

                t0 = time.time()
                result = est.estimate(z_real, real_ts, tol_constraint_kw=tol_c)
                dt = time.time() - t0

                err_v_pu = np.abs(result.x_v - x_v_all[i]) / v_base
                err_delta = np.abs(result.x_delta - x_delta_all[i])

                rec = {
                    "criterion": label, "stratum": stratum_name, "i": i, "real_timestamp": real_ts,
                    "converged": result.converged, "iterations": result.iterations, "time_s": dt,
                    "max_v_err_pu": err_v_pu.max(), "median_v_err_pu": float(np.median(err_v_pu)),
                    "max_delta_err_rad": err_delta.max(),
                    "constraint_residual_inf": result.constraint_residual_inf,
                }
                records.append(rec)
                print(f"[{label}/{stratum_name}] i={i} real_ts={real_ts} converged={result.converged} "
                      f"iters={result.iterations:<3d} time_s={dt:.2f} maxVerr={err_v_pu.max():.5f} "
                      f"cres={result.constraint_residual_inf:.3f}", flush=True)
                pd.DataFrame(records).to_csv(OUT_PATH, index=False)

    out = pd.DataFrame(records)
    print()
    print("=== SUMMARY BY CRITERION ===")
    print(out.groupby("criterion")[["converged", "iterations", "time_s", "max_v_err_pu", "constraint_residual_inf"]].agg(
        ["mean", "median", "max"]))
    print()
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
