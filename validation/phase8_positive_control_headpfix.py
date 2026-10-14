"""
RES.md D10 Phase 8 -- positive control, re-run after the HEAD_P
measurement-model fix (branch fix-head-p-model) with the new
thresholds (config/bdd_thresholds_headpfix.json). Adds a 20-sigma
bias to one ordinary (non-zero-redundancy) real P meter at i=46 and
10 other stratified scorecard points, re-solves with the frozen
estimator, and reports how often it is flagged by J99/LNR99.

Run: python -m validation.phase8_positive_control_headpfix
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
THRESHOLDS_PATH = BASE_DIR / "config" / "bdd_thresholds_headpfix.json"
OUT_PATH = BASE_DIR / "validation" / "phase8_positive_control_headpfix_results.csv"

SEED = 777
N_PER_STRATUM = 10
BIAS_SIGMAS = 20.0
ZERO_REDUNDANCY_REAL_METER_IDS = {"M004_P", "M004_Q", "M040_P", "HEAD_Q_A", "HEAD_Q_B", "HEAD_Q_C"}


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
    with open(THRESHOLDS_PATH) as f:
        th = json.load(f)
    J99, LNR99 = th["J99"], th["LNR99"]

    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    meas_ids = [e["id"] for e in model.meas_entries]
    sigma_by_id = {e["id"]: e["sigma"] for e in model.meas_entries}

    candidates = [mid for mid in meas_ids if mid.startswith("M0") and mid.endswith("_P")
                  and mid not in ZERO_REDUNDANCY_REAL_METER_IDS]
    target = candidates[len(candidates) // 2]
    print("target channel:", target, "sigma=", sigma_by_id[target], flush=True)

    p_cols = [c for c in df.columns if c.endswith("_P")]
    strata = stratified_picks(df, p_cols)
    sweep30 = sorted(set(strata["low"] + strata["mid"] + strata["high"]))
    points = sweep30[:11]
    print("points:", points, flush=True)

    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)
    rows = []
    for i in points:
        row = df.iloc[i]
        real_ts = str(row["real_timestamp"])
        z_real = {c: float(row[c]) for c in z_cols}
        state.x0_v, state.x0_delta = list(flat_v), list(flat_delta)
        result_clean = est.estimate(z_real, real_ts)

        z_bad = dict(z_real)
        bias = BIAS_SIGMAS * sigma_by_id[target]
        z_bad[target] = z_real[target] + bias
        state.x0_v, state.x0_delta = list(result_clean.x_v), list(result_clean.x_delta)
        result_bad = est.estimate(z_bad, real_ts)
        stats_bad = compute_chi2_and_lnr(est, result_bad, z_bad, real_ts)

        flagged_j = bool(stats_bad["J"] > J99)
        flagged_lnr = bool(stats_bad["lnr"] > LNR99)
        rows.append({"i": i, "converged": result_bad.converged, "J": stats_bad["J"], "lnr": stats_bad["lnr"],
                     "flagged_by_J99": flagged_j, "flagged_by_LNR99": flagged_lnr})
        print(f"i={i}: converged={result_bad.converged} J={stats_bad['J']:.4f} lnr={stats_bad['lnr']:.4f} "
              f"flag_J99={flagged_j} flag_LNR99={flagged_lnr}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print()
    print(f"flagged by J99: {out['flagged_by_J99'].sum()}/{len(out)}")
    print(f"flagged by LNR99: {out['flagged_by_LNR99'].sum()}/{len(out)}")
    print(f"converged: {out['converged'].sum()}/{len(out)}")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
