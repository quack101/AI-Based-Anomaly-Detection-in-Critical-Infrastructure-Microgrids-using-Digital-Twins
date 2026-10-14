"""
RES.md D10 Phase 8 step 2b -- A4/A5 pilot against the FROZEN estimator
(estimation/state_estimator.py's production StateEstimator.estimate(),
default truncated-SVD solve). This file previously ran attacked/control
solves through a hand-copied replica of the pre-freeze LM-damped loop
(warmstart_capped_solve) instead of the real estimator -- every one of
those solves came back SOLVER_FAIL (see validation/
phase8_a4_a5_pilot_frozen.csv). That function has been REMOVED; every
solve here now goes through est.estimate() directly, warm-started from
the clean solution (same convention as phase8_a4_attack_test_fast.py),
using parallel-across-points execution (ProcessPoolExecutor), same
pattern as that script.

Points: i=46, 94, 154 (3 of the stratified seed=777 picks used
throughout Phase 7/8).
A4 (linear, a=Hc): modes M1 (dense) and M3 (fixed to the 6 zero-
  redundancy real meters), magnitudes 0.05/0.10 pu.
A5 (exact nonlinear, a=h(x_bad)-h(x_clean)): mode M1 only, magnitudes
  0.01/0.05/0.10 pu, using the SAME dense null-space direction as
  A4-M1, Newton-corrected (up to 5 steps) to drive the unattackable-row
  residual back toward zero before the real estimator solve.

Two controls per attack case, both built in attacks/a4_synthesis.py:
  control 1 (random_control): same support, same L2 norm, random
    direction.
  control 2 (random_sign_control): same per-channel |a_i|, independent
    random +/-1 signs.

Decision thresholds (J99, LNR99) are loaded from
config/bdd_thresholds_frozen.json -- the single source of truth,
computed from validation/phase8_bdd_baseline_results.csv's n=90
clean-data run under this same frozen estimator. No EMPIRICAL_J99/
EMPIRICAL_LNR99 literal appears in this file.

Per solve, logs: converged, iterations, J before/after, lnr before/
after, the channel id with the largest normalized residual before and
after, the largest normalized residual EXCLUDING the before-channel
(before and after), and wall-clock seconds split into attack
synthesis, unattackable-Jacobian/null-space work, and est.estimate()
itself.

Run: python -m validation.phase8_a4_a5_boundary_scan
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import StateEstimator, DEFAULT_TOL_CONSTRAINT_KW
from estimation.bad_data_detection import (
    compute_chi2_and_lnr, unattackable_jacobian, nullspace_basis, RHO_CRITICAL_THRESHOLD,
)
from attacks.a4_synthesis import synthesize_a4_attack, random_control, random_sign_control

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase8_a4_a5_pilot2_frozen.csv"
THRESHOLDS_PATH = BASE_DIR / "config" / "bdd_thresholds_frozen.json"

with open(THRESHOLDS_PATH) as _f:
    _thresholds = json.load(_f)
EMPIRICAL_J99 = float(_thresholds["J99"])
EMPIRICAL_LNR99 = float(_thresholds["LNR99"])

TARGET_POINTS = [46, 94, 154]
A4_MODES = ["M1", "M3"]
A4_MAGNITUDES = [0.05, 0.10]
A5_MAGNITUDES = [0.01, 0.05, 0.10]
ZERO_REDUNDANCY_REAL_METER_IDS = {"M004_P", "M004_Q", "M040_P", "HEAD_Q_A", "HEAD_Q_B", "HEAD_Q_C"}
ETA_CAP_MIN = 30.0


def dominant_channel(stats, exclude_id=None):
    """(id, |normalized_residual|) of the largest-magnitude normalized
    residual among non-critical channels (redundancy >=
    RHO_CRITICAL_THRESHOLD, same exclusion compute_chi2_and_lnr's own
    'lnr' uses), optionally excluding one specific channel id. Returns
    (None, nan) if nothing is left to rank."""

    ids = stats["ids"]
    nr = np.asarray(stats["normalized_residuals"])
    redundancy = np.asarray(stats["redundancy"])
    kept = np.isfinite(nr) & (redundancy >= RHO_CRITICAL_THRESHOLD)
    if exclude_id is not None:
        kept = kept & np.array([i != exclude_id for i in ids])
    if not kept.any():
        return None, float("nan")
    abs_nr = np.where(kept, np.abs(nr), -np.inf)
    idx = int(np.argmax(abs_nr))
    return ids[idx], float(abs_nr[idx])


def process_point(i, real_ts, z_real, seed_for_point):
    """All A4/A5 x control1/control2 cases for ONE point. Rebuilds the
    model fresh (OpenDSS COM state can't cross a process boundary)."""

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

    t0 = time.time()
    jac = unattackable_jacobian(est, result_clean, real_ts)
    t_jac_point = time.time() - t0
    Hs_attack, Hs_unattack, state_scale = jac["Hs_meas"], jac["Hs_unattackable"], jac["state_scale"]
    unattack_entries = jac["unattackable_entries"]

    dom_id_clean, dom_val_clean = dominant_channel(stats_clean)
    lnr_excl_clean = dominant_channel(stats_clean, exclude_id=dom_id_clean)[1]

    rng = np.random.default_rng(seed_for_point)

    def solve_and_log(z_dict, kind, part, mode, mag, t_synth, t_jac_case,
                       c_norm, extra=None):
        state.x0_v = list(result_clean.x_v)
        state.x0_delta = list(result_clean.x_delta)
        t0 = time.time()
        result = est.estimate(z_dict, real_ts)
        t_estimate = time.time() - t0

        stats = compute_chi2_and_lnr(est, result, z_dict, real_ts)
        dom_id_after, dom_val_after = dominant_channel(stats)
        # value of the SAME before-dominant channel, re-evaluated after
        idx_before = stats["ids"].index(dom_id_clean)
        dom_val_after_same_channel = float(np.abs(stats["normalized_residuals"][idx_before]))
        lnr_excl_after = dominant_channel(stats, exclude_id=dom_id_clean)[1]

        dx_v = result.x_v - result_clean.x_v
        dx_delta = result.x_delta - result_clean.x_delta
        shift_norm = float(np.hypot(np.linalg.norm(dx_v), np.linalg.norm(dx_delta)))
        max_v_shift_pu = float(np.max(np.abs(dx_v) / v_base_full))
        norm_ratio = shift_norm / c_norm if c_norm > 0 else float("nan")

        row = {
            "i": i, "part": part, "mode": mode, "magnitude_pu": mag, "kind": kind,
            "converged": result.converged, "iterations": result.iterations,
            "J_before": stats_clean["J"], "J_after": stats["J"], "dJ": stats["J"] - stats_clean["J"],
            "lnr_before": stats_clean["lnr"], "lnr_after": stats["lnr"], "dLNR": stats["lnr"] - stats_clean["lnr"],
            "dominant_channel_before": dom_id_clean, "dominant_value_before": dom_val_clean,
            "dominant_channel_after": dom_id_after, "dominant_value_after": dom_val_after,
            "dominant_before_channel_value_after": dom_val_after_same_channel,
            "lnr_excl_dominant_before": lnr_excl_clean, "lnr_excl_dominant_after": lnr_excl_after,
            "max_v_shift_pu": max_v_shift_pu, "norm_ratio": norm_ratio,
            "detected_by_J99": bool(stats["J"] > EMPIRICAL_J99) if result.converged else None,
            "detected_by_LNR99": bool(stats["lnr"] > EMPIRICAL_LNR99) if result.converged else None,
            "t_attack_synth_s": t_synth, "t_jacobian_s": t_jac_case, "t_estimate_s": t_estimate,
        }
        if extra:
            row.update(extra)
        return row

    rows = []

    # ---------- A4: M1, M3 x 0.05, 0.10 ----------
    for mode in A4_MODES:
        for mag in A4_MAGNITUDES:
            t0 = time.time()
            atk = synthesize_a4_attack(
                Hs_attack, Hs_unattack, state_scale, rng, mode, mag, state.n_phase_nodes,
                fixed_meter_idx=fixed_idx if mode == "M3" else None,
            )
            t_synth = time.time() - t0
            a = atk["a_attackable"]
            c_norm = float(np.linalg.norm(atk["c"]))
            max_unattack = atk["max_unattackable_residual"]

            z_bad = dict(z_real)
            for k, mid in enumerate(meas_ids):
                z_bad[mid] = z_real[mid] + a[k]
            rows.append(solve_and_log(z_bad, "attack", "A4", mode, mag, t_synth, t_jac_point, c_norm,
                                       extra={"max_abs_attack_on_unattackable_row": max_unattack}))

            a_c1 = random_control(a, rng)
            z_c1 = dict(z_real)
            for k, mid in enumerate(meas_ids):
                z_c1[mid] = z_real[mid] + a_c1[k]
            rows.append(solve_and_log(z_c1, "control1_random_dir", "A4", mode, mag, 0.0, t_jac_point, c_norm))

            a_c2 = random_sign_control(a, rng)
            z_c2 = dict(z_real)
            for k, mid in enumerate(meas_ids):
                z_c2[mid] = z_real[mid] + a_c2[k]
            rows.append(solve_and_log(z_c2, "control2_random_sign", "A4", mode, mag, 0.0, t_jac_point, c_norm))

    # ---------- A5: M1 only x 0.01, 0.05, 0.10 ----------
    Z_un, nullity, _, _ = nullspace_basis(Hs_unattack, rank_tol=1e-8)
    y_dir = Z_un @ rng.standard_normal(nullity)
    raw_pu = float(np.max(np.abs(y_dir[:state.n_phase_nodes])))

    at_v_clean = complex_v_from_state(state, result_clean.x_v, result_clean.x_delta)
    _, h0_attack_clean = J.build_jacobian(model.meas_entries, model.Y, model.node_index, state, at_v=at_v_clean)
    _, h0_unattack_clean = J.build_jacobian(unattack_entries, model.Y, model.node_index, state, at_v=at_v_clean)

    for mag in A5_MAGNITUDES:
        t0 = time.time()
        scale = mag / raw_pu
        y = y_dir * scale
        c = state_scale * y
        x_bad_v = result_clean.x_v + c[:state.n_phase_nodes]
        x_bad_delta = result_clean.x_delta + c[state.n_phase_nodes:]
        c_norm = float(np.linalg.norm(c))

        at_v_bad = complex_v_from_state(state, x_bad_v, x_bad_delta)
        _, h0_unattack_bad = J.build_jacobian(unattack_entries, model.Y, model.node_index, state, at_v=at_v_bad)

        residual = h0_unattack_bad - h0_unattack_clean
        max_resid = float(np.max(np.abs(residual)))
        n_newton = 0
        while max_resid > DEFAULT_TOL_CONSTRAINT_KW and n_newton < 5:
            Hs_un_bad, _ = J.build_jacobian(unattack_entries, model.Y, model.node_index, state, at_v=at_v_bad)
            Hs_un_bad = Hs_un_bad * state_scale[None, :]
            dy = np.linalg.pinv(Hs_un_bad, rcond=1e-8) @ (-residual)
            dc = state_scale * dy
            c = c + dc
            x_bad_v = result_clean.x_v + c[:state.n_phase_nodes]
            x_bad_delta = result_clean.x_delta + c[state.n_phase_nodes:]
            at_v_bad = complex_v_from_state(state, x_bad_v, x_bad_delta)
            _, h0_unattack_bad = J.build_jacobian(unattack_entries, model.Y, model.node_index, state, at_v=at_v_bad)
            residual = h0_unattack_bad - h0_unattack_clean
            max_resid = float(np.max(np.abs(residual)))
            n_newton += 1
        _, h0_attack_bad = J.build_jacobian(model.meas_entries, model.Y, model.node_index, state, at_v=at_v_bad)
        t_synth = time.time() - t0

        a_att = h0_attack_bad - h0_attack_clean
        z_bad = dict(z_real)
        for k, mid in enumerate(meas_ids):
            z_bad[mid] = z_real[mid] + a_att[k]
        rows.append(solve_and_log(z_bad, "attack", "A5", "M1(nonlinear)", mag, t_synth, t_jac_point, c_norm,
                                   extra={"n_newton_corrections": n_newton, "unattack_residual_final": max_resid}))

        a_c1 = random_control(a_att, rng)
        z_c1 = dict(z_real)
        for k, mid in enumerate(meas_ids):
            z_c1[mid] = z_real[mid] + a_c1[k]
        rows.append(solve_and_log(z_c1, "control1_random_dir", "A5", "M1(nonlinear)", mag, 0.0, t_jac_point, c_norm))

        a_c2 = random_sign_control(a_att, rng)
        z_c2 = dict(z_real)
        for k, mid in enumerate(meas_ids):
            z_c2[mid] = z_real[mid] + a_c2[k]
        rows.append(solve_and_log(z_c2, "control2_random_sign", "A5", "M1(nonlinear)", mag, 0.0, t_jac_point, c_norm))

    return rows


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)

    print(f"points: {TARGET_POINTS}", flush=True)

    row0 = df.iloc[TARGET_POINTS[0]]
    real_ts0 = str(row0["real_timestamp"])
    z_real0 = {c: float(row0[c]) for c in z_cols}

    t0 = time.time()
    rows_point0 = process_point(TARGET_POINTS[0], real_ts0, z_real0, 777 + 200)
    point0_time = time.time() - t0
    n_per_point = len(rows_point0)
    per_solve = point0_time / n_per_point
    remaining_points = len(TARGET_POINTS) - 1
    projected_total_min = (point0_time + per_solve * n_per_point) / 60.0  # point0 + 1 more parallel batch
    print(f"point 0 (i={TARGET_POINTS[0]}) done in {point0_time:.1f}s ({n_per_point} rows, "
          f"{per_solve:.2f}s/row). Projected total wall-clock (parallel): {projected_total_min:.1f} min",
          flush=True)

    if projected_total_min > ETA_CAP_MIN:
        print(f"ABORTING: projected ETA {projected_total_min:.1f} min exceeds {ETA_CAP_MIN} min cap. "
              "Not running the remaining points.", flush=True)
        pd.DataFrame(rows_point0).to_csv(OUT_PATH, index=False)
        print("saved partial (point 0 only):", OUT_PATH)
        return

    all_rows = list(rows_point0)
    pd.DataFrame(all_rows).to_csv(OUT_PATH, index=False)

    remaining = TARGET_POINTS[1:]
    t_parallel_start = time.time()
    with ProcessPoolExecutor(max_workers=min(len(remaining), 8)) as ex:
        futures = {}
        for k, i in enumerate(remaining):
            row = df.iloc[i]
            real_ts = str(row["real_timestamp"])
            z_real = {c: float(row[c]) for c in z_cols}
            fut = ex.submit(process_point, i, real_ts, z_real, 777 + 200 + k + 1)
            futures[fut] = i

        for fut in as_completed(futures):
            i = futures[fut]
            rows = fut.result()
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(OUT_PATH, index=False)
            elapsed = time.time() - t_parallel_start
            print(f"  [parallel] point i={i} done, {elapsed:.1f}s elapsed since parallel batch start", flush=True)

    out = pd.DataFrame(all_rows)
    print()
    print(f"total rows: {len(out)}")
    print()
    print("=== SUMMARY: attack rows only ===")
    print(out[out["kind"] == "attack"][
        ["i", "part", "mode", "magnitude_pu", "converged", "iterations", "J_before", "J_after", "dJ",
         "lnr_before", "lnr_after", "dLNR", "dominant_channel_before", "dominant_channel_after",
         "detected_by_J99", "detected_by_LNR99"]
    ].to_string(index=False))
    print()
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
