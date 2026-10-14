"""
RES.md D10 Phase 8 -- does holding the 42 unmetered loads flat at DSS
nominal in the ground truth (generate_phase6_corpus.py's P0 physics)
explain the accuracy/J gap seen in the Phase 7 scorecard (n=60) and
Phase 8 baseline (n=90)? Builds a SECOND ground truth, P1, where all
91 loads are driven from dataset/node_loads_91 at the same timestamps,
and compares the FROZEN estimator (StateEstimator.estimate(), default
truncated-SVD) on both.

Scope: the UNION of the scorecard's 60 points (phase7_scorecard_
tsvd_default.py: 30 stratified picks, seed=777, + their i+1 "B_warm"
successors) and the baseline's 90 points (phase8_bdd_baseline.py: the
SAME 30 stratified picks + 60 extra clean rows drawn from the full
test split, seed=778) -- NOT all 86,400 corpus rows. This is a stated
methodology choice, not a bug: the real corpus's noise stream is ONE
continuous random.Random() draw across all 86,400 steps (generate_
phase6_corpus.py's RNG_SEED=42 comment), and the real OpenDSS
regulator-tap trajectory is path-dependent on the full step-by-step
history (confirmed earlier this investigation: validation/
phase_tap_wiring_check.py). Replaying only the sampled subset means
this experiment's tap trajectory and noise realization do NOT bit-
for-bit match the committed corpus's -- an earlier check (phase_tap_
wiring_check.py) found nominal-vs-true-tap accuracy statistically
identical for this estimator, which is the basis for accepting this
simplification rather than a 86,400-step double replay.

Timestamp mapping (confirmed by direct inspection, not assumed): a
test-split row's own "timestamp" INTEGER column is the row index into
dataset/node_loads_91/node_<name>.csv directly -- e.g. test_telemetry
row 0 has timestamp=73440, and node_s1a.csv.iloc[73440]'s own
"timestamp" string matches that row's "real_timestamp" exactly.

Paired noise: for a given point i, P0's and P1's measurement dicts
each get their OWN fresh random.Random(NOISE_SEED_BASE + i) instance
(same seed, not a shared, order-dependent stream) -- so the per-
channel Gaussian draws are identical between physics versions at the
same i, isolating the truth-physics difference from the noise
realization. sensors/sensor_layer.py's build_measurement_dict (the
same function the real corpus uses) is called unmodified.

Does NOT change the estimator, solver, Jacobian, bad-data code,
pseudo-measurement code, or any registry file. Does NOT write into
simulation/phase6_corpus/. All OpenDSS truth-building solves run in
ONE process, sequentially (Part 1). Estimator solves (Part 2) may use
up to 4 worker processes.

Run: python -m validation.exp_all91_physics
"""

import json
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

import opendssdirect as dss

from simulation.opendss_utils import compile_feeder
from observability.estimator_model import build_estimator_model
from observability.state_vector import build_state_vector, extract_solved_state
from sensors.sensor_layer import build_measurement_dict
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
DATASET_DIR = BASE_DIR / "dataset" / "node_loads_91"
REPRESENTATIVE_METERS = BASE_DIR / "simulation" / "representative_meters.json"
OUT_CSV = BASE_DIR / "validation" / "exp_all91_physics_results.csv"
OUT_TRUTH_CACHE = BASE_DIR / "validation" / "exp_all91_physics_truth_cache.pkl"
OUT_REPORT = BASE_DIR / "reports" / "exp_all91_physics.md"
THIS_SCRIPT = "validation/exp_all91_physics.py"

SEED = 777
N_PER_STRATUM = 10
N_EXTRA_CLEAN = 60
NOISE_SEED_BASE = 900000
DOF = 57
MAX_WORKERS = 4
HOUR_BUCKETS = [(0, 6), (6, 12), (12, 18), (18, 24)]


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


def determine_points(df_full):
    """Union of the scorecard's 60 and the baseline's 90, same seeds
    and logic as validation/phase7_scorecard_tsvd_default.py and
    validation/phase8_bdd_baseline.py."""

    n_rows = len(df_full)
    df1440 = df_full.iloc[:1440]
    p_cols = [c for c in df_full.columns if c.endswith("_P")]
    strata = stratified_picks(df1440, p_cols)
    sweep30 = sorted(set(strata["low"] + strata["mid"] + strata["high"]))

    b_warm = sorted({i + 1 for i in sweep30 if i + 1 < 1440})
    scorecard60 = set(sweep30) | set(b_warm)

    rng2 = np.random.default_rng(SEED + 1)
    remaining = [i for i in range(n_rows) if i not in sweep30]
    extra60 = sorted(rng2.choice(remaining, size=min(N_EXTRA_CLEAN, len(remaining)), replace=False).tolist())
    baseline90 = set(sweep30) | set(extra60)

    union_points = sorted(scorecard60 | baseline90)
    provenance = {}
    for i in union_points:
        tags = []
        if i in sweep30:
            tags.append("stratified")
        if i in b_warm:
            tags.append("b_warm")
        if i in extra60:
            tags.append("extra_clean")
        provenance[i] = "+".join(tags)

    return union_points, sweep30, b_warm, extra60, scorecard60, baseline90


def build_truth_single_process(points, df_full, z_cols):
    """Part 1 -- ONE OpenDSS process, sequential. Returns per-point dict
    with P0/P1 ground-truth state and measurement dicts, plus the raw
    per-load P values needed for Step 5's mechanism checks."""

    with open(REPRESENTATIVE_METERS) as f:
        representative_meters = json.load(f)
    metered_names = {m["load_name"] for m in representative_meters}

    available = {p.stem[len("node_"):].lower(): p for p in DATASET_DIR.glob("node_*.csv")}
    all_load_names = sorted(available.keys())
    unmetered_names = sorted(n for n in all_load_names if n not in {m.lower() for m in metered_names})
    assert len(unmetered_names) == 42, f"expected 42 unmetered loads, got {len(unmetered_names)}"

    datasets = {name: pd.read_csv(available[name]) for name in all_load_names}
    # map representative_meters's exact-cased load_name -> lowercase dataset key
    meter_to_key = {m["load_name"]: m["load_name"].lower() for m in representative_meters}

    model = build_estimator_model("multiphase")
    state = model.state

    nominal_p = {}
    nominal_q = {}
    for name in dss.Loads.AllNames():
        dss.Loads.Name(name)
        nominal_p[name.lower()] = dss.Loads.kW()
        nominal_q[name.lower()] = dss.Loads.kvar()

    results = {}
    t0 = time.time()
    for idx, i in enumerate(points):
        row = df_full.iloc[i]
        real_ts = str(row["real_timestamp"])
        t_global = int(row["timestamp"])

        # ---- P0: 49 metered driven, 42 unmetered RESET to nominal ----
        for meter in representative_meters:
            key = meter_to_key[meter["load_name"]]
            dss_row = datasets[key].iloc[t_global]
            dss.Loads.Name(meter["load_name"])
            dss.Loads.kW(float(dss_row["P_kW"]))
            dss.Loads.kvar(float(dss_row["Q_kvar"]))
        for name in unmetered_names:
            dss.Loads.Name(name)
            dss.Loads.kW(nominal_p[name])
            dss.Loads.kvar(nominal_q[name])

        dss.Solution.Solve()
        x_v_p0, x_delta_p0 = extract_solved_state(state)
        z_p0 = build_measurement_dict(model.meas_entries, add_noise=True, rng=random.Random(NOISE_SEED_BASE + i))

        # ---- P1: all 91 driven, same 49 unchanged, 42 now from their own series ----
        unmetered_p1_kw = {}
        for name in unmetered_names:
            dss_row = datasets[name].iloc[t_global]
            p_kw, q_kvar = float(dss_row["P_kW"]), float(dss_row["Q_kvar"])
            unmetered_p1_kw[name] = p_kw
            dss.Loads.Name(name)
            dss.Loads.kW(p_kw)
            dss.Loads.kvar(q_kvar)

        dss.Solution.Solve()
        x_v_p1, x_delta_p1 = extract_solved_state(state)
        z_p1 = build_measurement_dict(model.meas_entries, add_noise=True, rng=random.Random(NOISE_SEED_BASE + i))

        results[i] = {
            "real_ts": real_ts, "t_global": t_global, "hour": int(real_ts[11:13]),
            "x_v_p0": np.asarray(x_v_p0), "x_delta_p0": np.asarray(x_delta_p0), "z_p0": z_p0,
            "x_v_p1": np.asarray(x_v_p1), "x_delta_p1": np.asarray(x_delta_p1), "z_p1": z_p1,
            "unmetered_p1_kw": unmetered_p1_kw,
        }
        if (idx + 1) % 20 == 0 or idx == len(points) - 1:
            elapsed = time.time() - t0
            print(f"  truth-build [{idx+1}/{len(points)}] i={i} elapsed={elapsed:.1f}s", flush=True)

    meta = {
        "unmetered_names": unmetered_names,
        "nominal_p": {n: nominal_p[n] for n in unmetered_names},
        "nominal_q": {n: nominal_q[n] for n in unmetered_names},
        "v_base": {n: state.v_base[n] for n in state.all_node_names},
        "all_node_names": list(state.all_node_names),
    }
    return results, meta


def estimate_case(args):
    """Part 2 -- one worker process: fresh model+estimator, one
    est.estimate() call."""

    i, physics, z_dict, real_ts = args
    model = build_estimator_model("multiphase")
    est = StateEstimator(model)
    result = est.estimate(z_dict, real_ts)
    stats = compute_chi2_and_lnr(est, result, z_dict, real_ts)
    return {
        "i": i, "physics": physics, "converged": result.converged, "iterations": result.iterations,
        "J": stats["J"], "lnr": stats["lnr"],
        "ids": stats["ids"], "normalized_residuals": stats["normalized_residuals"].tolist(),
        "x_v": result.x_v.tolist(), "x_delta": result.x_delta.tolist(),
    }


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df_full = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols)

    points, sweep30, b_warm, extra60, scorecard60, baseline90 = determine_points(df_full)
    print(f"union points: {len(points)}  (scorecard60={len(scorecard60)}, baseline90={len(baseline90)}, "
          f"overlap={len(scorecard60 & baseline90)})", flush=True)

    t_truth_start = time.time()
    truth, meta = build_truth_single_process(points, df_full, z_cols)
    t_truth = time.time() - t_truth_start
    print(f"truth-build total: {t_truth:.1f}s", flush=True)

    with open(OUT_TRUTH_CACHE, "wb") as f:
        pickle.dump({"truth": truth, "meta": meta, "points": points}, f)

    jobs = []
    for i in points:
        jobs.append((i, "P0", truth[i]["z_p0"], truth[i]["real_ts"]))
        jobs.append((i, "P1", truth[i]["z_p1"], truth[i]["real_ts"]))

    t_est_start = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(estimate_case, job): job for job in jobs}
        n_done = 0
        for fut in as_completed(futures):
            rows.append(fut.result())
            n_done += 1
            if n_done % 40 == 0 or n_done == len(jobs):
                print(f"  estimator [{n_done}/{len(jobs)}] elapsed={time.time()-t_est_start:.1f}s", flush=True)
    t_est = time.time() - t_est_start

    v_base_full = np.array([meta["v_base"][n] for n in meta["all_node_names"]])

    out_rows = []
    pooled_err = {"P0": [], "P1": []}
    pooled_err_tagged = {"P0": [], "P1": []}  # (i, hour, err_array) -- for hour-bucket slicing
    for r in rows:
        i, physics = r["i"], r["physics"]
        x_v_truth = truth[i]["x_v_p0"] if physics == "P0" else truth[i]["x_v_p1"]
        x_v_est = np.asarray(r["x_v"])
        err = np.abs(x_v_est - x_v_truth) / v_base_full
        pooled_err[physics].append(err)
        pooled_err_tagged[physics].append((i, truth[i]["hour"], err))
        ids = r["ids"]
        nr = np.asarray(r["normalized_residuals"])
        finite = np.isfinite(nr)
        idx_dom = int(np.argmax(np.abs(np.where(finite, nr, 0))))
        head_p = {}
        for hid in ("HEAD_P_A", "HEAD_P_B", "HEAD_P_C"):
            if hid in ids:
                head_p[hid] = float(nr[ids.index(hid)])
        out_rows.append({
            "i": i, "physics": physics, "real_ts": truth[i]["real_ts"], "hour": truth[i]["hour"],
            "converged": r["converged"], "iterations": r["iterations"], "J": r["J"], "lnr": r["lnr"],
            "dominant_channel": ids[idx_dom], "dominant_value": float(np.abs(nr[idx_dom])),
            "max_v_err_pu": float(err.max()), "median_v_err_pu": float(np.median(err)),
            "head_p_a_nr": head_p.get("HEAD_P_A"), "head_p_b_nr": head_p.get("HEAD_P_B"),
            "head_p_c_nr": head_p.get("HEAD_P_C"),
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"saved {len(out_df)} rows to {OUT_CSV}", flush=True)

    write_report(out_df, pooled_err, pooled_err_tagged, points, sweep30, b_warm, extra60, scorecard60, baseline90,
                 truth, meta, t_truth, t_est, v_base_full)


def pooled_stats(err_list):
    pooled = np.concatenate(err_list)
    return {
        "n": len(pooled), "median": float(np.median(pooled)), "p95": float(np.percentile(pooled, 95)),
        "max": float(pooled.max()), "mean": float(pooled.mean()),
    }


def write_report(out_df, pooled_err, pooled_err_tagged, points, sweep30, b_warm, extra60, scorecard60, baseline90,
                  truth, meta, t_truth, t_est, v_base_full):
    lines = []
    lines.append("# All-91 vs 49-load physics experiment (frozen estimator)\n")
    lines.append(f"Script: `{THIS_SCRIPT}`. Every number below comes from this one run unless noted.\n")

    lines.append("## Points\n")
    lines.append(f"- 30 stratified (seed=777): {sweep30}  [{THIS_SCRIPT}::determine_points]")
    lines.append(f"- scorecard B_warm (i+1), 30: {b_warm}")
    lines.append(f"- baseline extra_clean, 60: {extra60}")
    lines.append(f"- scorecard60 distinct: {len(scorecard60)}, baseline90 distinct: {len(baseline90)}, "
                 f"overlap: {len(scorecard60 & baseline90)} (exactly the 30 stratified)")
    lines.append(f"- UNION used for this experiment: {len(points)} points\n")

    lines.append("## Step 4 -- P0 vs P1 under the frozen estimator\n")
    chi2_lo, chi2_hi = sstats.chi2.ppf(0.025, DOF), sstats.chi2.ppf(0.975, DOF)
    for physics in ["P0", "P1"]:
        sub = out_df[out_df["physics"] == physics]
        ps = pooled_stats(pooled_err[physics])
        n_in_ci = int(((sub["J"] >= chi2_lo) & (sub["J"] <= chi2_hi)).sum())
        worst = sub.reindex(sub.groupby("i")["max_v_err_pu"].idxmax()) if False else None
        lines.append(f"### {physics}")
        lines.append(f"- voltage error (pu), pooled n={ps['n']}: median={ps['median']:.4f} "
                     f"p95={ps['p95']:.4f} max={ps['max']:.4f}")
        lines.append(f"- J: mean={sub['J'].mean():.4e}  J/dof({DOF})={sub['J'].mean()/DOF:.2f}x  "
                     f"in chi2 95% CI [{chi2_lo:.1f},{chi2_hi:.1f}]: {n_in_ci}/{len(sub)}")
        lines.append(f"- converged: {int(sub['converged'].sum())}/{len(sub)}  "
                     f"mean iterations: {sub['iterations'].mean():.2f}")
        lnr_dom_counts = sub["dominant_channel"].value_counts()
        lines.append(f"- LNR: min={sub['lnr'].min():.4f} median={sub['lnr'].median():.4f} "
                     f"max={sub['lnr'].max():.4f}")
        lines.append(f"- dominant channel frequency: {lnr_dom_counts.to_dict()}")
        worst5 = sub.nlargest(5, "max_v_err_pu")[["i", "max_v_err_pu"]]
        lines.append(f"- 5 worst (point, max_v_err_pu): {list(worst5.itertuples(index=False, name=None))}\n")

    p0_ps = pooled_stats(pooled_err["P0"])
    ref_median, ref_p95, ref_max = 0.0239, 0.0408, 0.0626
    close = (abs(p0_ps["median"] - ref_median) < 0.01 and abs(p0_ps["p95"] - ref_p95) < 0.015
             and abs(p0_ps["max"] - ref_max) < 0.02)
    lines.append(f"**P0 vs scorecard reference** (median={ref_median}, p95={ref_p95}, max={ref_max}): "
                 f"this run gives median={p0_ps['median']:.4f}, p95={p0_ps['p95']:.4f}, max={p0_ps['max']:.4f}. "
                 f"Close to reference: {close}.\n")

    lines.append("## Step 5 -- mechanism checks\n")

    model = build_estimator_model("multiphase")
    unmetered_names = meta["unmetered_names"]
    per_hour_by_load = model.per_hour_by_load

    z_p0_list, z_p1_list = [], []
    S_list, J_p0_list, verr_p0_list = [], [], []
    for i in points:
        hour = truth[i]["hour"]
        z_p0_point, z_p1_point = [], []
        S = 0.0
        for name in unmetered_names:
            bucket = per_hour_by_load[name][hour]
            sigma_p = bucket["sigma_p"] if bucket["sigma_p"] else float("nan")
            nominal = meta["nominal_p"][name]
            p1_val = truth[i]["unmetered_p1_kw"][name]
            if sigma_p and sigma_p > 0:
                z_p0_point.append((bucket["mean_p"] - nominal) / sigma_p)
                z_p1_point.append((bucket["mean_p"] - p1_val) / sigma_p)
            S += abs(bucket["mean_p"] - nominal)
        z_p0_list.extend(z_p0_point)
        z_p1_list.extend(z_p1_point)
        S_list.append(S)
        row_j = out_df[(out_df["i"] == i) & (out_df["physics"] == "P0")]
        J_p0_list.append(float(row_j["J"].iloc[0]))
        verr_p0_list.append(float(row_j["max_v_err_pu"].iloc[0]))

    z_p0_arr, z_p1_arr = np.array(z_p0_list), np.array(z_p1_list)
    lines.append("### 5a -- z = (pseudo_mean - true_P) / sigma_pseudo, 42 unmetered loads x points")
    lines.append(f"- P0: RMS(z)={np.sqrt(np.mean(z_p0_arr**2)):.4f}  p95(|z|)={np.percentile(np.abs(z_p0_arr),95):.4f}")
    lines.append(f"- P1: RMS(z)={np.sqrt(np.mean(z_p1_arr**2)):.4f}  p95(|z|)={np.percentile(np.abs(z_p1_arr),95):.4f}\n")

    rho_J, p_J = sstats.spearmanr(S_list, J_p0_list)
    rho_V, p_V = sstats.spearmanr(S_list, verr_p0_list)
    lines.append("### 5b -- S = sum_42 |pseudo_mean - nominal_P| (kW), under P0, vs J and voltage error")
    lines.append(f"- Spearman(S, J_P0) = {rho_J:.4f} (p={p_J:.4g}, n={len(S_list)})")
    lines.append(f"- Spearman(S, max_v_err_pu_P0) = {rho_V:.4f} (p={p_V:.4g}, n={len(S_list)})\n")

    lines.append("### 5c -- mean SIGNED normalized residual of HEAD_P_A/B/C")
    for physics in ["P0", "P1"]:
        sub = out_df[out_df["physics"] == physics]
        for col, label in [("head_p_a_nr", "HEAD_P_A"), ("head_p_b_nr", "HEAD_P_B"), ("head_p_c_nr", "HEAD_P_C")]:
            vals = sub[col].dropna()
            lines.append(f"- {physics} {label}: mean={vals.mean():.4f}  n={len(vals)}")
    lines.append("")

    lines.append("### 5d -- accuracy and J by hour-of-day bucket")
    for physics in ["P0", "P1"]:
        sub = out_df[out_df["physics"] == physics]
        for lo, hi in HOUR_BUCKETS:
            bucket_errs = [arr for (i, hour, arr) in pooled_err_tagged[physics] if lo <= hour < hi]
            bsub = sub[(sub["hour"] >= lo) & (sub["hour"] < hi)]
            if len(bsub) == 0:
                continue
            bpooled = np.concatenate(bucket_errs)
            lines.append(f"- {physics} hour[{lo}-{hi}): n={len(bsub)}  J_mean={bsub['J'].mean():.4e}  "
                         f"pooled_median_v_err={np.median(bpooled):.4f}  "
                         f"pooled_p95_v_err={np.percentile(bpooled,95):.4f}  "
                         f"pooled_max_v_err={bpooled.max():.4f}")
    lines.append("")

    lines.append("## Runtime\n")
    lines.append(f"- truth-build (single process, {len(points)} points x 2 physics): {t_truth:.1f}s")
    lines.append(f"- estimator solves ({len(points)*2} calls, <= {MAX_WORKERS} workers): {t_est:.1f}s")
    lines.append(f"- total: {t_truth + t_est:.1f}s\n")

    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_REPORT}", flush=True)


if __name__ == "__main__":
    main()
