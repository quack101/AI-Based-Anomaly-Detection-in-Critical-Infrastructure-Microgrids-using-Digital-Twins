"""
RES.md D10 Phase 8 -- re-run of the clean-data baseline (same 90
points, same seeds) on branch est-pf-start, with the power-flow start
(est.estimate default init="pseudo_pf", observability/pf_start.py) on
top of the HEAD_P-excluded estimator. Writes the ACTIVE thresholds
(validation/bdd_thresholds.py -> config/bdd_thresholds_pfstart.json);
the older JSON files and the headp-excluded results are left
untouched. Also computes head_p_balance_residual (observability/jacobian.py,
branch headp-excluded) at every point -- NOT wired into J/LNR, a
standalone balance check reported separately.

Run: python -m validation.phase8_bdd_baseline_pfstart
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

from observability.estimator_model import build_estimator_model
from observability.jacobian import head_p_balance_residual
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr
from validation.bdd_thresholds import ACTIVE_THRESHOLDS_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase8_bdd_baseline_pfstart_results.csv"
OUT_BALANCE = BASE_DIR / "validation" / "phase8_head_balance_pfstart_results.csv"
OUT_THRESHOLDS = ACTIVE_THRESHOLDS_PATH

SEED = 777
N_PER_STRATUM = 10
N_EXTRA_CLEAN = 60


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
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols)
    n_rows = len(df)
    print(f"test split has {n_rows} rows total", flush=True)

    p_cols = [c for c in df.columns if c.endswith("_P")]
    strata = stratified_picks(df.iloc[:1440], p_cols)
    sweep_idxs = sorted(set(strata["low"] + strata["mid"] + strata["high"]))
    print("stratified sweep picks:", sweep_idxs, flush=True)

    rng2 = np.random.default_rng(SEED + 1)
    remaining = [i for i in range(n_rows) if i not in sweep_idxs]
    extra_idxs = sorted(rng2.choice(remaining, size=min(N_EXTRA_CLEAN, len(remaining)), replace=False).tolist())
    print(f"extra clean rows: {len(extra_idxs)}", flush=True)

    model = build_estimator_model("multiphase")  # exclude_dead_head_p=True (default)
    state = model.state
    est = StateEstimator(model)
    print(f"dof={est.dof}  m_weighted={est.m_weighted}", flush=True)
    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)

    records = []
    balance_records = []
    all_idxs = [(i, "stratified") for i in sweep_idxs] + [(i, "extra_clean") for i in extra_idxs]

    for idx, (i, src) in enumerate(all_idxs):
        state.x0_v = list(flat_v)
        state.x0_delta = list(flat_delta)
        row = df.iloc[i]
        real_ts = str(row["real_timestamp"])
        z_dict = {c: float(row[c]) for c in z_cols}
        result = est.estimate(z_dict, real_ts)
        stats_i = compute_chi2_and_lnr(est, result, z_dict, real_ts)

        ids = stats_i["ids"]
        nr = np.asarray(stats_i["normalized_residuals"])
        finite = np.isfinite(nr)
        idx_dom = int(np.argmax(np.abs(np.where(finite, nr, 0))))

        records.append({
            "i": i, "source": src, "converged": result.converged,
            "J": stats_i["J"], "lnr": stats_i["lnr"], "lnr_all": stats_i["lnr_all"],
            "n_excluded": stats_i["n_excluded"], "cond_G": stats_i["cond_G"],
            "dominant_channel": ids[idx_dom],
        })

        bal = head_p_balance_residual(model, result, z_dict)
        for phase, d in bal.items():
            balance_records.append({"i": i, "source": src, "phase": phase, **d})

        if idx % 50 == 0:
            print(f"  [{idx+1}/{len(all_idxs)}] i={i} src={src} J={stats_i['J']:.3e} "
                  f"lnr={stats_i['lnr']:.3f}", flush=True)

    out = pd.DataFrame(records)
    out.to_csv(OUT_PATH, index=False)
    bal_out = pd.DataFrame(balance_records)
    bal_out.to_csv(OUT_BALANCE, index=False)

    n = len(out)
    j99 = out["J"].quantile(0.99)
    lnr99 = out["lnr"].quantile(0.99)
    print(f"EMPIRICAL 99th percentile J = {j99:.4f}")
    print(f"EMPIRICAL 99th percentile LNR = {lnr99:.4f}")
    print(f"LNR: min={out['lnr'].min():.4f} median={out['lnr'].median():.4f} max={out['lnr'].max():.4f}")
    print("dominant channel frequency:", out["dominant_channel"].value_counts().to_dict())
    print(f"converged: {int(out['converged'].sum())}/{n}")

    print()
    print("=== head_p_balance_residual, per phase, over the 90 points ===")
    for phase in ["A", "B", "C"]:
        sub = bal_out[bal_out["phase"] == phase]
        print(f"  phase {phase}: mean|r_kw|={sub['r_kw'].abs().mean():.4f}  std|r_kw|={sub['r_kw'].abs().std():.4f}  "
              f"p99|r_kw|={sub['r_kw'].abs().quantile(0.99):.4f}  "
              f"mean|r_sigma|={sub['r_sigma'].abs().mean():.4f}  std|r_sigma|={sub['r_sigma'].abs().std():.4f}  "
              f"p99|r_sigma|={sub['r_sigma'].abs().quantile(0.99):.4f}")

    with open(OUT_THRESHOLDS, "w") as f:
        json.dump({
            "J99": float(j99), "LNR99": float(lnr99), "n": n,
            "source_file": "validation/phase8_bdd_baseline_pfstart_results.csv",
            "note": "Same protocol as the other bdd_thresholds_*.json files, re-measured on branch "
                    "est-pf-start with the power-flow start (init=pseudo_pf) as the estimator default. "
                    "The older JSON files are left untouched.",
        }, f, indent=2)

    print()
    print("saved:", OUT_PATH)
    print("saved balance:", OUT_BALANCE)
    print("saved thresholds:", OUT_THRESHOLDS)


if __name__ == "__main__":
    main()
