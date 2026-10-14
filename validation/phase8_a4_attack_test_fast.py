"""
RES.md D10 Phase 8 step 2 -- FAST/SMALL re-run of the A4 attack test.
Same mechanism as phase8_a4_attack_test.py, with three speed fixes:

  1. WARM START every attacked/control solve from the clean x_hat
     (not flat start) -- a flat-start solve on a perturbed z can hit
     the solver's own documented LM-ceiling non-convergence (up to
     ~80s/solve, measured directly -- see report), because the
     perturbed z is far from nominal. Starting from the clean
     solution (the operator's actual last-known-good state, the same
     "B_warm" condition used throughout the Phase 7 sweeps) avoids
     that for most cases.
  2. H and the null-space are computed ONCE per point and reused
     across all (mode, magnitude) combinations for that point.
  3. Points run in PARALLEL across processes (ProcessPoolExecutor);
     each result row is appended to the CSV as soon as its point's
     batch finishes.

Grid: first 8 of the 30 stratified points (seed=777), modes M1/M2/M3,
magnitudes 0.05/0.10 pu only -- 8*3*2 = 48 attack solves + 48 control
solves.

Safety check: times the first point's 12 solves SEQUENTIALLY first,
prints progress+ETA every 5 solves, and ABORTS before launching the
rest if the projected total wall-clock (accounting for running the
remaining points in parallel) exceeds 60 minutes.

Run: python -m validation.phase8_a4_attack_test_fast
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr, unattackable_jacobian
from attacks.a4_synthesis import synthesize_a4_attack, random_control

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase8_a4_attack_test_fast_results_frozen.csv"
THRESHOLDS_PATH = BASE_DIR / "config" / "bdd_thresholds_frozen.json"

SEED = 777
N_PER_STRATUM = 10
N_POINTS = 8
MODES = ["M1", "M2", "M3"]
MAGNITUDES = [0.05, 0.10]
N_SPARSE_METERS = 20
ZERO_REDUNDANCY_REAL_METER_IDS = {"M004_P", "M004_Q", "M040_P", "HEAD_Q_A", "HEAD_Q_B", "HEAD_Q_C"}

with open(THRESHOLDS_PATH) as _f:
    _thresholds = json.load(_f)
EMPIRICAL_J99 = float(_thresholds["J99"])
EMPIRICAL_LNR99 = float(_thresholds["LNR99"])
MAX_ETA_MINUTES = 60.0


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


def process_point(i, real_ts, z_real, seed_for_point, progress_cb=None):
    """Runs all (mode, magnitude) combos for ONE point. Rebuilds the
    model fresh (required -- OpenDSS COM state can't cross a process
    boundary). Returns a list of row dicts. progress_cb(n_done,
    elapsed), if given, is called after each solve pair for the
    sequential timing/ETA pass."""

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    v_base_full = np.array([state.v_base[n] for n in state.all_node_names])
    meas_ids = [e["id"] for e in model.meas_entries]
    fixed_idx = np.array([meas_ids.index(mid) for mid in ZERO_REDUNDANCY_REAL_METER_IDS])

    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)
    state.x0_v, state.x0_delta = flat_v, flat_delta
    result_clean = est.estimate(z_real, real_ts)
    stats_clean = compute_chi2_and_lnr(est, result_clean, z_real, real_ts)
    jac = unattackable_jacobian(est, result_clean, real_ts)
    Hs_attack, Hs_unattack, state_scale = jac["Hs_meas"], jac["Hs_unattackable"], jac["state_scale"]

    rng = np.random.default_rng(seed_for_point)
    rows = []
    n_done = 0
    t_start = time.time()

    for mode in MODES:
        for mag in MAGNITUDES:
            atk = synthesize_a4_attack(
                Hs_attack, Hs_unattack, state_scale, rng, mode, mag, state.n_phase_nodes,
                n_sparse_meters=N_SPARSE_METERS,
                fixed_meter_idx=fixed_idx if mode == "M3" else None,
            )
            a = atk["a_attackable"]
            max_a = np.max(np.abs(a))
            n_significant_meters = int(np.sum(np.abs(a) > 0.01 * max_a)) if max_a > 0 else 0

            z_bad = dict(z_real)
            for k, mid in enumerate(meas_ids):
                z_bad[mid] = z_real[mid] + a[k]

            # WARM START from the clean solution, not flat start
            state.x0_v = list(result_clean.x_v)
            state.x0_delta = list(result_clean.x_delta)
            result_bad = est.estimate(z_bad, real_ts)
            stats_bad = compute_chi2_and_lnr(est, result_bad, z_bad, real_ts)
            n_done += 1
            if progress_cb:
                progress_cb(n_done, time.time() - t_start)

            dx_v = result_bad.x_v - result_clean.x_v
            dx_delta = result_bad.x_delta - result_clean.x_delta
            actual_shift_norm = float(np.hypot(np.linalg.norm(dx_v), np.linalg.norm(dx_delta)))
            c_norm = float(np.linalg.norm(atk["c"]))
            norm_ratio = actual_shift_norm / c_norm if c_norm > 0 else float("nan")
            max_v_shift_pu = float(np.max(np.abs(dx_v) / v_base_full))

            dJ = stats_bad["J"] - stats_clean["J"]
            dLNR = stats_bad["lnr"] - stats_clean["lnr"]
            pass_chi2_emp = stats_bad["J"] <= EMPIRICAL_J99
            pass_lnr_emp = stats_bad["lnr"] <= EMPIRICAL_LNR99

            a_ctrl = random_control(a, rng)
            z_ctrl = dict(z_real)
            for k, mid in enumerate(meas_ids):
                z_ctrl[mid] = z_real[mid] + a_ctrl[k]

            state.x0_v = list(result_clean.x_v)
            state.x0_delta = list(result_clean.x_delta)
            result_ctrl = est.estimate(z_ctrl, real_ts)
            stats_ctrl = compute_chi2_and_lnr(est, result_ctrl, z_ctrl, real_ts)
            n_done += 1
            if progress_cb:
                progress_cb(n_done, time.time() - t_start)

            dJ_ctrl = stats_ctrl["J"] - stats_clean["J"]
            dLNR_ctrl = stats_ctrl["lnr"] - stats_clean["lnr"]
            pass_chi2_emp_ctrl = stats_ctrl["J"] <= EMPIRICAL_J99
            pass_lnr_emp_ctrl = stats_ctrl["lnr"] <= EMPIRICAL_LNR99
            dx_v_ctrl = result_ctrl.x_v - result_clean.x_v
            dx_delta_ctrl = result_ctrl.x_delta - result_clean.x_delta
            ctrl_shift_norm = float(np.hypot(np.linalg.norm(dx_v_ctrl), np.linalg.norm(dx_delta_ctrl)))

            rows.append({
                "i": i, "mode_requested": mode, "mode_used": atk["mode_used"],
                "feasible": atk["feasible"], "magnitude_target_pu": mag,
                "achieved_v_shift_pu_intended": atk["achieved_v_shift_pu"],
                "max_unattackable_residual": atk["max_unattackable_residual"],
                "n_significant_meters": n_significant_meters,
                "converged_clean": result_clean.converged, "converged_bad": result_bad.converged,
                "iters_bad": result_bad.iterations,
                "J_clean": stats_clean["J"], "J_bad": stats_bad["J"], "dJ": dJ,
                "lnr_clean": stats_clean["lnr"], "lnr_bad": stats_bad["lnr"], "dLNR": dLNR,
                "pass_chi2_empirical": pass_chi2_emp, "pass_lnr_empirical": pass_lnr_emp,
                "actual_shift_norm": actual_shift_norm, "c_norm": c_norm, "norm_ratio": norm_ratio,
                "max_v_shift_pu_actual": max_v_shift_pu,
                "converged_ctrl": result_ctrl.converged, "iters_ctrl": result_ctrl.iterations,
                "J_ctrl": stats_ctrl["J"], "dJ_ctrl": dJ_ctrl,
                "lnr_ctrl": stats_ctrl["lnr"], "dLNR_ctrl": dLNR_ctrl,
                "pass_chi2_empirical_ctrl": pass_chi2_emp_ctrl, "pass_lnr_empirical_ctrl": pass_lnr_emp_ctrl,
                "ctrl_shift_norm": ctrl_shift_norm,
            })
    return rows


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)

    p_cols = [c for c in df.columns if c.endswith("_P")]
    strata = stratified_picks(df, p_cols)
    all_sweep_idxs = sorted(set(strata["low"] + strata["mid"] + strata["high"]))
    sweep_idxs = all_sweep_idxs[:N_POINTS]
    print(f"using first {N_POINTS} of 30 stratified picks: {sweep_idxs}", flush=True)

    rows_point0 = []

    def cb(n_done, elapsed):
        if n_done % 5 == 0 or n_done == 12:
            per_solve = elapsed / n_done
            total_solves_all_points = N_POINTS * 12
            # ETA assuming remaining POINTS run in parallel (up to N_POINTS-1
            # workers), point 0 already counted
            remaining_points = N_POINTS - 1
            eta_parallel_s = per_solve * 12  # one more point's worth, run concurrently
            eta_total_min = (elapsed + eta_parallel_s) / 60.0
            print(f"  [point0 progress] {n_done}/12 solves, {elapsed:.1f}s elapsed, "
                  f"{per_solve:.2f}s/solve -> projected total ETA (parallel): {eta_total_min:.1f} min",
                  flush=True)

    row0 = df.iloc[sweep_idxs[0]]
    real_ts0 = str(row0["real_timestamp"])
    z_real0 = {c: float(row0[c]) for c in z_cols}
    t0 = time.time()
    rows_point0 = process_point(sweep_idxs[0], real_ts0, z_real0, SEED + 2, progress_cb=cb)
    point0_time = time.time() - t0
    per_solve = point0_time / 12
    projected_total_min = (point0_time + per_solve * 12) / 60.0  # point0 sequential + 1 more parallel batch
    print(f"point 0 done in {point0_time:.1f}s ({per_solve:.2f}s/solve). "
          f"Projected total wall-clock (parallel): {projected_total_min:.1f} min", flush=True)

    if projected_total_min > MAX_ETA_MINUTES:
        print(f"ABORTING: projected ETA {projected_total_min:.1f} min exceeds {MAX_ETA_MINUTES} min cap. "
              "Not running the remaining points.", flush=True)
        pd.DataFrame(rows_point0).to_csv(OUT_PATH, index=False)
        print("saved partial (point 0 only):", OUT_PATH)
        return

    all_rows = list(rows_point0)
    pd.DataFrame(all_rows).to_csv(OUT_PATH, index=False)

    remaining = sweep_idxs[1:]
    t_parallel_start = time.time()
    with ProcessPoolExecutor(max_workers=min(len(remaining), 8)) as ex:
        futures = {}
        for k, i in enumerate(remaining):
            row = df.iloc[i]
            real_ts = str(row["real_timestamp"])
            z_real = {c: float(row[c]) for c in z_cols}
            fut = ex.submit(process_point, i, real_ts, z_real, SEED + 2 + k + 1, None)
            futures[fut] = i

        n_points_done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            rows = fut.result()
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(OUT_PATH, index=False)
            n_points_done += 1
            elapsed = time.time() - t_parallel_start
            print(f"  [parallel] point i={i} done ({n_points_done}/{len(remaining)}), "
                  f"{elapsed:.1f}s elapsed since parallel batch start", flush=True)

    out = pd.DataFrame(all_rows)
    print()
    print(f"total rows: {len(out)}  (expected {N_POINTS * len(MODES) * len(MAGNITUDES)})")
    out["pass_emp_both"] = out["pass_chi2_empirical"] & out["pass_lnr_empirical"]
    out["pass_emp_both_ctrl"] = out["pass_chi2_empirical_ctrl"] & out["pass_lnr_empirical_ctrl"]

    print()
    print("=== Fraction under EMPIRICAL thresholds (both chi2 AND LNR), per mode x magnitude ===")
    print(out.groupby(["mode_requested", "magnitude_target_pu"])[["pass_emp_both", "pass_emp_both_ctrl"]].mean())
    print()
    print("=== Median paired dJ / dLNR: attack vs control, per mode x magnitude ===")
    print(out.groupby(["mode_requested", "magnitude_target_pu"])[["dJ", "dJ_ctrl", "dLNR", "dLNR_ctrl"]].median())
    print()
    print("=== Median ||x_bad - x_clean|| vs ||c||: norm_ratio, per mode x magnitude ===")
    print(out.groupby(["mode_requested", "magnitude_target_pu"])[["norm_ratio", "max_v_shift_pu_actual"]].median())
    print()
    print("=== Feasibility of requested mode (M2/M3 fallback rate) ===")
    print(out.groupby("mode_requested")["feasible"].mean())
    print()
    cases_ctrl_passes = out[out["pass_emp_both_ctrl"]]
    print(f"=== Cases where the CONTROL also passes both empirical thresholds: {len(cases_ctrl_passes)}/{len(out)} ===")
    if len(cases_ctrl_passes):
        print(cases_ctrl_passes[["i", "mode_requested", "magnitude_target_pu", "dJ_ctrl", "dLNR_ctrl"]].to_string(index=False))

    print()
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
