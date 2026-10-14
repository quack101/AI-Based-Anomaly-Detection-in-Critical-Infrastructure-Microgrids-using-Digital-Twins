"""
RES.md D10 Phase 8 -- does the estimator's own analytic measurement
model h(.) (observability/jacobian.py) agree with what OpenDSS itself
reports for the SAME real channels, evaluated at the TRUE (noise-
free) state? Diagnosis only -- does not change the estimator, solver,
Jacobian, bad-data code, registry files or the corpus, and does not
propose or apply a fix.

Same 120-point union as validation/exp_all91_physics.py (the
scorecard's 60 + the baseline's 90), under P0 physics (current corpus
physics: 49 metered loads driven, 42 unmetered held at DSS nominal).

For each point:
  x_true           = extract_solved_state(state) after a real OpenDSS
                      solve (observability/state_vector.py).
  z0                = the NOISE-FREE measurement for every real
                      (registry) channel, via
                      sensors.sensor_layer.build_measurement_dict(...,
                      add_noise=False) against that SAME live, solved
                      circuit -- i.e. OpenDSS's own CktElement/Bus
                      query path (sensors/measurement_functions.py).
  h(x_true)         = the estimator's own analytic forward-power-flow
                      Jacobian builder's h0 output
                      (observability/jacobian.py:build_jacobian),
                      evaluated at the TRUE complex voltage -- the
                      SAME function the solver and
                      estimation/bad_data_detection.py use, called
                      here with NO live OpenDSS query at all (pure
                      Y-bus/voltage-phasor algebra).
  r0 = z0 - h(x_true)   for real rows (pseudo/reg-pseudo rows use
                         their own static z_value as z0, per
                         estimation/state_estimator.py:364-367's own
                         "z_value if present else z_real[id]" rule).
Zero-injection rows: residual = h0_zi(x_true) - 0 (no z0 needed, exact
equality by construction).

Weights: whatever sigma estimation/state_estimator.py's own
_weighted_entries_for_hour would use right now (registry sigma for
real rows, the model's own estimated sigma for pseudo/reg-pseudo).

Part B (Step 3e) additionally re-runs StateEstimator.estimate() on the
ALREADY-CACHED noisy z (validation/exp_all91_physics_truth_cache.pkl,
written by exp_all91_physics.py's earlier run) to get the mean SIGNED
normalized residual at the ESTIMATE x_hat for the same channels --
up to 4 worker processes, per the task's allowance. All OpenDSS truth-
building (Part A) runs sequentially in ONE process.

Run: python -m validation.model_consistency_at_truth
"""

import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import opendssdirect as dss

from observability.estimator_model import build_estimator_model
from observability.state_vector import extract_solved_state, complex_v_from_state
from observability import jacobian as J
from observability.pseudo_measurements import build_pseudo_entries_for_hour
from sensors.sensor_layer import build_measurement_dict
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
DATASET_DIR = BASE_DIR / "dataset" / "node_loads_91"
REPRESENTATIVE_METERS = BASE_DIR / "simulation" / "representative_meters.json"
TRUTH_CACHE = BASE_DIR / "validation" / "exp_all91_physics_truth_cache.pkl"
EXP91_RESULTS = BASE_DIR / "validation" / "exp_all91_physics_results.csv"
OUT_CSV = BASE_DIR / "validation" / "model_consistency_at_truth.csv"
OUT_REPORT = BASE_DIR / "reports" / "model_consistency_at_truth.md"
THIS_SCRIPT = "validation/model_consistency_at_truth.py"

SEED = 777
N_PER_STRATUM = 10
N_EXTRA_CLEAN = 60
MAX_WORKERS = 4


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
    return sorted(scorecard60 | baseline90), sweep30


def build_p0_truth_and_residuals(points, df_full, z_cols):
    """Part A -- ONE OpenDSS process, sequential."""

    with open(REPRESENTATIVE_METERS) as f:
        representative_meters = json.load(f)
    metered_names = {m["load_name"] for m in representative_meters}
    meter_to_key = {m["load_name"]: m["load_name"].lower() for m in representative_meters}

    available = {p.stem[len("node_"):].lower(): p for p in DATASET_DIR.glob("node_*.csv")}
    unmetered_names = sorted(n for n in available if n not in {m.lower() for m in metered_names})
    datasets = {name: pd.read_csv(available[name]) for name in available}

    model = build_estimator_model("multiphase")
    state = model.state

    nominal_p, nominal_q = {}, {}
    for name in dss.Loads.AllNames():
        dss.Loads.Name(name)
        nominal_p[name.lower()] = dss.Loads.kW()
        nominal_q[name.lower()] = dss.Loads.kvar()

    zi_ids = [e["id"] for e in model.zi_entries]
    real_ids = [e["id"] for e in model.meas_entries]
    regzi_ids = [e["id"] for e in model.reg_pseudo_entries]

    rows = []
    t0 = time.time()
    for idx, i in enumerate(points):
        row = df_full.iloc[i]
        real_ts = str(row["real_timestamp"])
        t_global = int(row["timestamp"])
        hour = int(real_ts[11:13])

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
        x_v, x_delta = extract_solved_state(state)
        x_v, x_delta = np.asarray(x_v), np.asarray(x_delta)
        z0_dict = build_measurement_dict(model.meas_entries, add_noise=False)

        pseudo_entries = build_pseudo_entries_for_hour(model.bus_phase_by_load, model.per_hour_by_load, hour)
        all_entries = model.meas_entries + model.reg_pseudo_entries + pseudo_entries + model.zi_entries
        at_v = complex_v_from_state(state, x_v, x_delta)
        H_all, h0_all = J.build_jacobian(all_entries, model.Y, model.node_index, state, at_v=at_v)

        m_real = len(real_ids)
        m_regzi = len(regzi_ids)
        m_pseudo = len(pseudo_entries)
        h0_real = h0_all[:m_real]
        h0_regzi = h0_all[m_real:m_real + m_regzi]
        h0_pseudo = h0_all[m_real + m_regzi:m_real + m_regzi + m_pseudo]
        h0_zi = h0_all[m_real + m_regzi + m_pseudo:]

        z0_arr = np.array([z0_dict[eid] for eid in real_ids])
        r0_real = z0_arr - h0_real
        sigma_real = np.array([e["sigma"] for e in model.meas_entries])

        z_regzi = np.array([e["z_value"] for e in model.reg_pseudo_entries])
        r0_regzi = z_regzi - h0_regzi
        sigma_regzi = np.array([e["sigma"] for e in model.reg_pseudo_entries])

        z_pseudo = np.array([e["z_value"] for e in pseudo_entries])
        r0_pseudo = z_pseudo - h0_pseudo
        sigma_pseudo = np.array([e["sigma"] for e in pseudo_entries])

        for k, eid in enumerate(real_ids):
            quantity = model.meas_entries[k]["quantity"]
            rows.append({
                "i": i, "hour": hour, "group": "real", "sub": quantity, "id": eid,
                "z0": float(z0_arr[k]), "h_true": float(h0_real[k]), "r0": float(r0_real[k]),
                "sigma": float(sigma_real[k]), "r0_over_sigma": float(r0_real[k] / sigma_real[k]),
            })
        for k, e in enumerate(model.reg_pseudo_entries):
            rows.append({
                "i": i, "hour": hour, "group": "regzi", "sub": e["quantity"], "id": e["id"],
                "z0": float(z_regzi[k]), "h_true": float(h0_regzi[k]), "r0": float(r0_regzi[k]),
                "sigma": float(sigma_regzi[k]), "r0_over_sigma": float(r0_regzi[k] / sigma_regzi[k]),
            })
        for k, e in enumerate(pseudo_entries):
            rows.append({
                "i": i, "hour": hour, "group": "pseudo", "sub": e["quantity"], "id": e["id"],
                "z0": float(z_pseudo[k]), "h_true": float(h0_pseudo[k]), "r0": float(r0_pseudo[k]),
                "sigma": float(sigma_pseudo[k]), "r0_over_sigma": float(r0_pseudo[k] / sigma_pseudo[k]),
            })
        for k, eid in enumerate(zi_ids):
            rows.append({
                "i": i, "hour": hour, "group": "zi", "sub": "zero_injection", "id": eid,
                "z0": 0.0, "h_true": float(h0_zi[k]), "r0": float(h0_zi[k]),
                "sigma": None, "r0_over_sigma": None,
            })

        if (idx + 1) % 20 == 0 or idx == len(points) - 1:
            print(f"  truth [{idx+1}/{len(points)}] i={i} elapsed={time.time()-t0:.1f}s", flush=True)

    return pd.DataFrame(rows)


def estimate_case(args):
    i, z_dict, real_ts = args
    model = build_estimator_model("multiphase")
    est = StateEstimator(model)
    result = est.estimate(z_dict, real_ts)
    stats = compute_chi2_and_lnr(est, result, z_dict, real_ts)
    return {"i": i, "ids": stats["ids"], "nr": stats["normalized_residuals"].tolist(), "J": stats["J"]}


def what_if_head_p_sigma_x100(points30, df_full, z_cols, truth_cache):
    """Step 5 -- in-process only sigma override (no file write): a
    FRESH model's meas_entries list is mutated in memory (sigma for
    HEAD_P_A/B/C x100) before building the StateEstimator, then
    discarded at the end of this function. Nothing on disk changes."""

    rows = []
    for i in points30[:10]:
        row = df_full.iloc[i]
        real_ts = str(row["real_timestamp"])
        z_dict = truth_cache["truth"][i]["z_p0"]

        model_normal = build_estimator_model("multiphase")
        est_normal = StateEstimator(model_normal)
        result_n = est_normal.estimate(z_dict, real_ts)
        v_base = np.array([model_normal.state.v_base[n] for n in model_normal.state.all_node_names])
        x_v_true = np.asarray(truth_cache["truth"][i]["x_v_p0"])
        err_n = np.abs(result_n.x_v - x_v_true) / v_base

        model_mod = build_estimator_model("multiphase")
        for e in model_mod.meas_entries:
            if e["id"] in ("HEAD_P_A", "HEAD_P_B", "HEAD_P_C"):
                e["sigma"] = e["sigma"] * 100.0
        est_mod = StateEstimator(model_mod)
        result_m = est_mod.estimate(z_dict, real_ts)
        err_m = np.abs(result_m.x_v - x_v_true) / v_base

        rows.append({
            "i": i, "J_normal": result_n.J, "J_sigma100": result_m.J,
            "converged_normal": result_n.converged, "converged_sigma100": result_m.converged,
            "err_median_normal": float(np.median(err_n)), "err_p95_normal": float(np.percentile(err_n, 95)),
            "err_max_normal": float(err_n.max()),
            "err_median_sigma100": float(np.median(err_m)), "err_p95_sigma100": float(np.percentile(err_m, 95)),
            "err_max_sigma100": float(err_m.max()),
        })
    return pd.DataFrame(rows)


def main():
    t_script_start = time.time()
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df_full = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols)

    points, sweep30 = determine_points(df_full)
    print(f"union points: {len(points)}", flush=True)

    t_truth_start = time.time()
    resid_df = build_p0_truth_and_residuals(points, df_full, z_cols)
    t_truth = time.time() - t_truth_start
    resid_df.to_csv(OUT_CSV, index=False)
    print(f"truth+residual build: {t_truth:.1f}s, saved {len(resid_df)} rows to {OUT_CSV}", flush=True)

    with open(TRUTH_CACHE, "rb") as f:
        truth_cache = pickle.load(f)
    assert truth_cache["points"] == points, "point set mismatch vs cached exp_all91_physics run"

    jobs = [(i, truth_cache["truth"][i]["z_p0"], truth_cache["truth"][i]["real_ts"]) for i in points]
    t_est_start = time.time()
    est_rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(estimate_case, job): job for job in jobs}
        n_done = 0
        for fut in as_completed(futures):
            est_rows.append(fut.result())
            n_done += 1
            if n_done % 40 == 0 or n_done == len(jobs):
                print(f"  estimate [{n_done}/{len(jobs)}] elapsed={time.time()-t_est_start:.1f}s", flush=True)
    t_est = time.time() - t_est_start

    t_whatif_start = time.time()
    whatif_df = what_if_head_p_sigma_x100(sweep30, df_full, z_cols, truth_cache)
    t_whatif = time.time() - t_whatif_start

    write_report(resid_df, est_rows, whatif_df, points, t_truth, t_est, t_whatif,
                 time.time() - t_script_start)


def write_report(resid_df, est_rows, whatif_df, points, t_truth, t_est, t_whatif, t_total):
    lines = []
    lines.append("# Model consistency at the TRUE state\n")
    lines.append(f"Script: `{THIS_SCRIPT}`. Every number below comes from this one run unless noted.\n")
    lines.append(f"Points: {len(points)} (same union as validation/exp_all91_physics.py), P0 physics.\n")

    exp91 = pd.read_csv(EXP91_RESULTS)
    j_at_estimate = exp91[exp91["physics"] == "P0"].set_index("i")["J"]

    lines.append("## 3a -- J at truth vs J at the estimate\n")
    weighted = resid_df[resid_df["group"].isin(["real", "pseudo", "regzi"])]
    per_point_j_truth = weighted.groupby("i").apply(
        lambda g: float(np.sum(g["r0_over_sigma"] ** 2)), include_groups=False)
    for grp in ["real", "pseudo", "regzi"]:
        sub = weighted[weighted["group"] == grp]
        j_grp = sub.groupby("i").apply(lambda g: float(np.sum(g["r0_over_sigma"] ** 2)), include_groups=False)
        lines.append(f"- J_truth[{grp}]: mean={j_grp.mean():.4e}  median={j_grp.median():.4e}")
    lines.append(f"- J_truth[total] mean={per_point_j_truth.mean():.4e} median={per_point_j_truth.median():.4e}")
    j_est_aligned = j_at_estimate.reindex(per_point_j_truth.index)
    lines.append(f"- J_at_estimate (from exp_all91_physics_results.csv, P0) mean={j_est_aligned.mean():.4e} "
                 f"median={j_est_aligned.median():.4e}")
    lines.append(f"- J_truth / J_at_estimate ratio, mean of per-point ratios: "
                 f"{(per_point_j_truth / j_est_aligned).mean():.4f}\n")

    lines.append("## 3b -- share of J_truth by channel group\n")
    def group_label(row):
        if row["group"] == "real":
            if row["id"].startswith("HEAD_P"):
                return "HEAD_P"
            if row["id"].startswith("HEAD_Q"):
                return "HEAD_Q"
            if row["sub"] == "P_injection":
                return "meter_P"
            if row["sub"] == "Q_injection":
                return "meter_Q"
            if row["sub"] == "V_magnitude":
                return "V_magnitude"
            return "real_other"
        return row["group"]
    resid_df["label"] = resid_df.apply(group_label, axis=1)
    weighted2 = resid_df[resid_df["group"].isin(["real", "pseudo", "regzi"])].copy()
    weighted2["j_contrib"] = weighted2["r0_over_sigma"] ** 2
    total_j = weighted2["j_contrib"].sum()
    by_label = weighted2.groupby("label")["j_contrib"].sum().sort_values(ascending=False)
    for label, val in by_label.items():
        lines.append(f"- {label}: {val:.4e}  ({100*val/total_j:.2f}% of total J_truth)")
    zi = resid_df[resid_df["group"] == "zi"]
    lines.append(f"- zero-injection residual at truth: max|r0|={zi['r0'].abs().max():.6f}  "
                 f"mean|r0|={zi['r0'].abs().mean():.6f}\n")

    lines.append("## 3c -- top 15 REAL channels by |mean(r0)/sigma|\n")
    real = resid_df[resid_df["group"] == "real"]
    by_id = real.groupby("id").agg(mean_r0_over_sigma=("r0_over_sigma", "mean"),
                                    std_r0_over_sigma=("r0_over_sigma", "std"),
                                    mean_r0=("r0", "mean"))
    by_id["abs_mean"] = by_id["mean_r0_over_sigma"].abs()
    top15 = by_id.sort_values("abs_mean", ascending=False).head(15)
    for cid, row in top15.iterrows():
        sign = "+" if row["mean_r0_over_sigma"] > 0 else "-"
        lines.append(f"- {cid}: mean(r0/sigma)={row['mean_r0_over_sigma']:.4f}  "
                     f"std={row['std_r0_over_sigma']:.4f}  sign={sign}  mean(r0)={row['mean_r0']:.4f}")
    lines.append("")

    lines.append("## 3d -- HEAD_P_A/B/C: r0, percent of head P, linear fit vs P_head, hour dependence\n")
    for hid in ["HEAD_P_A", "HEAD_P_B", "HEAD_P_C"]:
        sub = real[real["id"] == hid].copy()
        h_true_std = float(sub["h_true"].std())
        if abs(sub["h_true"].mean()) > 1e-6:
            sub["pct"] = 100 * sub["r0"] / sub["h_true"]
            pct_mean = sub["pct"].mean()
        else:
            pct_mean = float("nan")  # h_true ~ 0 identically -- "percent of head P" is undefined
        if h_true_std > 1e-9:
            slope, intercept = np.polyfit(sub["h_true"], sub["r0"], 1)
            resid_fit = sub["r0"] - (intercept + slope * sub["h_true"])
            ss_res = float(np.sum(resid_fit ** 2))
            ss_tot = float(np.sum((sub["r0"] - sub["r0"].mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            fit_line = f"- fit r0 = a + b*P_head: a={intercept:.4f}  b={slope:.6f}  R2={r2:.4f}"
        else:
            fit_line = (f"- fit r0 = a + b*P_head: SKIPPED -- h_true has ~zero variance "
                        f"(std={h_true_std:.2e}, mean={sub['h_true'].mean():.2e}); "
                        f"r0 is essentially -z0 itself, not a function of h_true")
        by_hour = sub.groupby("hour")["r0"].mean()
        lines.append(f"### {hid}")
        lines.append(f"- h_true (kW): mean={sub['h_true'].mean():.2e}  std={h_true_std:.2e}")
        pct_str = f"{pct_mean:.2f}%" if pct_mean == pct_mean else "undefined (h_true~0)"
        lines.append(f"- r0 (kW): mean={sub['r0'].mean():.3f}  std={sub['r0'].std():.3f}  "
                     f"mean %head_P={pct_str}")
        lines.append(fit_line)
        lines.append(f"- by hour-of-day, mean r0 (kW): {by_hour.round(3).to_dict()}\n")

    lines.append("## 3e -- top 15 (same metric) at the ESTIMATE x_hat\n")
    all_ids = est_rows[0]["ids"]
    nr_matrix = np.array([r["nr"] for r in est_rows])
    mean_nr = np.nanmean(nr_matrix, axis=0)
    std_nr = np.nanstd(nr_matrix, axis=0)
    order = np.argsort(-np.abs(mean_nr))[:15]
    for idx in order:
        sign = "+" if mean_nr[idx] > 0 else "-"
        lines.append(f"- {all_ids[idx]}: mean(nr)={mean_nr[idx]:.4f}  std={std_nr[idx]:.4f}  sign={sign}")
    lines.append("")

    lines.append("## Step 5 -- diagnostic what-if: HEAD_P sigma x100, in-process only, 10 stratified points\n")
    lines.append("(mutates a fresh model's meas_entries dict in memory; nothing written to the registry file)\n")
    lines.append(whatif_df.to_string(index=False))
    lines.append("")
    lines.append(f"- mean J: normal={whatif_df['J_normal'].mean():.4e}  sigma100={whatif_df['J_sigma100'].mean():.4e}")
    lines.append(f"- median v-err: normal={whatif_df['err_median_normal'].median():.4f}  "
                 f"sigma100={whatif_df['err_median_sigma100'].median():.4f}")
    lines.append(f"- p95 v-err: normal={whatif_df['err_p95_normal'].median():.4f}  "
                 f"sigma100={whatif_df['err_p95_sigma100'].median():.4f}")
    lines.append(f"- max v-err: normal={whatif_df['err_max_normal'].median():.4f}  "
                 f"sigma100={whatif_df['err_max_sigma100'].median():.4f}\n")

    lines.append("## Runtime\n")
    lines.append(f"- truth+residual build (single process, {len(points)} points): {t_truth:.1f}s")
    lines.append(f"- estimator solves ({len(points)} calls, <= {MAX_WORKERS} workers): {t_est:.1f}s")
    lines.append(f"- what-if (10 points x 2 solves, sequential): {t_whatif:.1f}s")
    lines.append(f"- total script runtime: {t_total:.1f}s\n")

    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_REPORT}", flush=True)


if __name__ == "__main__":
    main()
