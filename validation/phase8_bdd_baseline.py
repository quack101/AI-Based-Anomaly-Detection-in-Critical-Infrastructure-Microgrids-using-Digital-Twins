"""
RES.md D10 Phase 8 pre-work -- bad-data-detection baseline on CLEAN
data, before any attack class exists. Uses estimation/bad_data_detection.py
(read-only) on:
  1. the same 60-point stratified sweep (seed=777) used throughout
     Phase 7's investigation.
  2. ~300 additional clean rows from the test split, A_flat only
     (cheap -- single flat-start calls, not warm-started pairs).

Reports:
  - false-alarm rate on clean data at textbook thresholds
    (chi2 95% one-sided for dof=57; LNR > 3)
  - empirical 99th-percentile thresholds for J and LNR
  - the unattackable-subspace nullspace dimension (same point)

Run: python -m validation.phase8_bdd_baseline
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr, unattackable_nullspace_dimension

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase8_bdd_baseline_results.csv"

SEED = 777
N_PER_STRATUM = 10
N_EXTRA_CLEAN = 60  # reduced from 300 for this run to fit the 30-min ETA budget
DOF = 57
CHI2_THRESH = sstats.chi2.ppf(0.95, DOF)
LNR_THRESH = 3.0


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

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)

    records = []
    all_idxs = [(i, "stratified") for i in sweep_idxs] + [(i, "extra_clean") for i in extra_idxs]

    for idx, (i, src) in enumerate(all_idxs):
        state.x0_v = list(flat_v)
        state.x0_delta = list(flat_delta)
        row = df.iloc[i]
        real_ts = str(row["real_timestamp"])
        z_dict = {c: float(row[c]) for c in z_cols}
        result = est.estimate(z_dict, real_ts)
        stats_i = compute_chi2_and_lnr(est, result, z_dict, real_ts)

        records.append({
            "i": i, "source": src, "converged": result.converged,
            "J": stats_i["J"], "lnr": stats_i["lnr"], "lnr_all": stats_i["lnr_all"],
            "n_excluded": stats_i["n_excluded"], "cond_G": stats_i["cond_G"],
        })
        if idx % 50 == 0:
            print(f"  [{idx+1}/{len(all_idxs)}] i={i} src={src} J={stats_i['J']:.3e} "
                  f"lnr={stats_i['lnr']:.3f}", flush=True)

    out = pd.DataFrame(records)
    out.to_csv(OUT_PATH, index=False)

    print()
    print(f"CHI2_THRESH (95%, dof={DOF}) = {CHI2_THRESH:.4f}")
    print(f"LNR_THRESH = {LNR_THRESH}")
    print()
    n = len(out)
    chi2_alarm = (out["J"] > CHI2_THRESH).sum()
    lnr_alarm = (out["lnr"] > LNR_THRESH).sum()
    either_alarm = ((out["J"] > CHI2_THRESH) | (out["lnr"] > LNR_THRESH)).sum()
    print(f"n_clean_points={n}")
    print(f"chi2 false-alarm rate (J > {CHI2_THRESH:.2f}): {chi2_alarm}/{n} = {100*chi2_alarm/n:.2f}%")
    print(f"LNR false-alarm rate (lnr > 3): {lnr_alarm}/{n} = {100*lnr_alarm/n:.2f}%")
    print(f"either-test false-alarm rate: {either_alarm}/{n} = {100*either_alarm/n:.2f}%")
    print()
    print("J distribution:", out["J"].describe())
    print()
    print("lnr distribution:", out["lnr"].describe())
    print()
    j99 = out["J"].quantile(0.99)
    lnr99 = out["lnr"].quantile(0.99)
    print(f"EMPIRICAL 99th percentile J = {j99:.4f}")
    print(f"EMPIRICAL 99th percentile LNR = {lnr99:.4f}")
    print()

    # unattackable nullspace -- compute at a representative point and
    # a couple of others to check stability
    for i in [sweep_idxs[0], sweep_idxs[len(sweep_idxs)//2], sweep_idxs[-1]]:
        state.x0_v = list(flat_v)
        state.x0_delta = list(flat_delta)
        row = df.iloc[i]
        real_ts = str(row["real_timestamp"])
        z_dict = {c: float(row[c]) for c in z_cols}
        result = est.estimate(z_dict, real_ts)
        ns = unattackable_nullspace_dimension(est, result, real_ts)
        print(f"i={i}: {ns}")

    print()
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
