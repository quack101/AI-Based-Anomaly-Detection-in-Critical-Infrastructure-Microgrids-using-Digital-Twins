"""
RES.md D10 Phase 7 SCIPY TRUST-CONSTR TEST -- Step 2 pilot.

Same 6 points (i=46,94,359,504,368,412) as the last two reports, flat
start. Uses estimation/state_estimator_scipy.ScipyTrustConstrEstimator
-- a SEPARATE code path, estimation/state_estimator.py is untouched.

Run: python -m validation.phase7_scipy_trustconstr_pilot
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator_scipy import ScipyTrustConstrEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"

SEED = 777
N_PER_STRATUM = 10
PILOT_PER_STRATUM = 2

PRIOR = {
    46: 107.473957, 94: 110.507183, 359: 109.354523,
    504: 99.349377, 368: 91.494258, 412: 84.696546,
}


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
    pilot_picks = {name: idxs[:PILOT_PER_STRATUM] for name, idxs in strata.items()}
    print("pilot picks (SAME as prior two pilots):", pilot_picks, flush=True)

    model = build_estimator_model("multiphase")
    state = model.state
    est = ScipyTrustConstrEstimator(model)
    v_base = np.array([state.v_base[n] for n in state.all_node_names])

    records = []
    for stratum_name, idxs in pilot_picks.items():
        for i in idxs:
            row = df.iloc[i]
            real_ts = row["real_timestamp"]
            z_real = {c: float(row[c]) for c in z_cols}

            t0 = time.time()
            result = est.estimate(z_real, real_ts, max_iter=500)
            dt = time.time() - t0

            err_v_pu = np.abs(result.x_v - x_v_all[i]) / v_base
            prior_val = PRIOR.get(i, float("nan"))

            rec = {
                "stratum": stratum_name, "i": i, "real_timestamp": real_ts,
                "converged": result.converged, "iterations": result.iterations,
                "nfev": result.nfev, "time_s": dt, "max_v_err_pu": err_v_pu.max(),
                "constraint_residual_inf": result.constraint_residual_inf,
                "status": result.status, "message": result.message,
                "prior_cres": prior_val,
            }
            records.append(rec)
            print(f"[{stratum_name}] i={i} real_ts={real_ts} converged={result.converged} "
                  f"iters={result.iterations:<4d} nfev={result.nfev:<4d} time_s={dt:.2f} "
                  f"maxVerr={err_v_pu.max():.5f} cres={result.constraint_residual_inf:.4f} "
                  f"(prior={prior_val:.3f}) status={result.status} msg={result.message}", flush=True)

    out = pd.DataFrame(records)
    out.to_csv(BASE_DIR / "validation" / "phase7_scipy_trustconstr_pilot_results.csv", index=False)
    print()
    print("=== PILOT SUMMARY (n=%d) ===" % len(out))
    print(out[["i", "converged", "iterations", "nfev", "time_s", "constraint_residual_inf", "prior_cres"]])
    print()
    print("under 1.0 kvar threshold:", (out["constraint_residual_inf"] < 1.0).sum(), "/", len(out))
    print("converged flag True:", out["converged"].sum(), "/", len(out))
    print("dramatic improvement (>50% lower than prior):",
          (out["constraint_residual_inf"] < 0.5 * out["prior_cres"]).sum(), "/", len(out))
    print("mean time_s:", out["time_s"].mean())


if __name__ == "__main__":
    main()
