"""
RES.md D10 Phase 8 -- size sweep for A4-M1 and A5-M1 against the
FROZEN estimator (estimation/state_estimator.py's production
StateEstimator.estimate(), default truncated-SVD solve). Every solve
goes through est.estimate() directly; no hand-copied solve loop.

20 of the 30 stratified (seed=777) points used throughout Phase 7/8
(the first 20 in sorted order, which includes i=46, 94, 154).
Magnitudes: 0.005/0.01/0.02/0.05/0.10/0.20 pu, for both A4 (linear,
mode M1 dense) and A5 (exact nonlinear, mode M1, Newton-corrected).
Each attack case also gets control 1 (random_control: same L2 norm,
random direction) and control 2 (random_sign_control: same per-
channel magnitude, random sign) -- both from attacks/a4_synthesis.py.

Decision thresholds (J99, LNR99) are loaded from
config/bdd_thresholds_frozen.json -- the single source of truth. No
literal J99/LNR99 number appears in this file.

"intended shift" = the target max voltage-magnitude shift (the
caller's `mag` argument). "achieved shift" = max |x_v_bad - x_v_clean|
/ v_base over all nodes (pu), i.e. what the re-estimated state
actually moved by.

At most 4 worker processes (ProcessPoolExecutor), one process per
point, same point-0-sequential-then-parallel-batch pattern as
validation/phase8_a4_a5_boundary_scan.py.

Run: python -m validation.phase8_a4_a5_scan_frozen
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
from estimation.bad_data_detection import compute_chi2_and_lnr, unattackable_jacobian, nullspace_basis
from attacks.a4_synthesis import synthesize_a4_attack, random_control, random_sign_control

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase8_a4_a5_scan_frozen.csv"
THRESHOLDS_PATH = BASE_DIR / "config" / "bdd_thresholds_frozen.json"

with open(THRESHOLDS_PATH) as _f:
    _thresholds = json.load(_f)
EMPIRICAL_J99 = float(_thresholds["J99"])
EMPIRICAL_LNR99 = float(_thresholds["LNR99"])

SEED = 777
N_PER_STRATUM = 10
N_POINTS = 20
MAGNITUDES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
MAX_WORKERS = 4
ETA_CAP_MIN = 60.0


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


def build_a5_attack(result_clean, model, state, Hs_unattack, state_scale, unattack_entries, mag, rng):
    Z_un, nullity, _, _ = nullspace_basis(Hs_unattack, rank_tol=1e-8)
    y_dir = Z_un @ rng.standard_normal(nullity)
    raw_pu = float(np.max(np.abs(y_dir[:state.n_phase_nodes])))
    scale = mag / raw_pu
    y = y_dir * scale
    c = state_scale * y
    x_bad_v = result_clean.x_v + c[:state.n_phase_nodes]
    x_bad_delta = result_clean.x_delta + c[state.n_phase_nodes:]

    at_v_clean = complex_v_from_state(state, result_clean.x_v, result_clean.x_delta)
    _, h0_attack_clean = J.build_jacobian(model.meas_entries, model.Y, model.node_index, state, at_v=at_v_clean)
    _, h0_unattack_clean = J.build_jacobian(unattack_entries, model.Y, model.node_index, state, at_v=at_v_clean)

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
    a_att = h0_attack_bad - h0_attack_clean
    return a_att, c, n_newton, max_resid


def process_point(i, real_ts, z_real, seed_for_point):
    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    v_base_full = np.array([state.v_base[n] for n in state.all_node_names])
    meas_ids = [e["id"] for e in model.meas_entries]

    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)
    state.x0_v, state.x0_delta = flat_v, flat_delta
    result_clean = est.estimate(z_real, real_ts)
    stats_clean = compute_chi2_and_lnr(est, result_clean, z_real, real_ts)

    jac = unattackable_jacobian(est, result_clean, real_ts)
    Hs_attack, Hs_unattack, state_scale = jac["Hs_meas"], jac["Hs_unattackable"], jac["state_scale"]
    unattack_entries = jac["unattackable_entries"]

    rng = np.random.default_rng(seed_for_point)
    rows = []

    def solve_and_log(a, kind, part, mag, intended_pu, unattack_resid, extra=None):
        z_dict = dict(z_real)
        for k, mid in enumerate(meas_ids):
            z_dict[mid] = z_real[mid] + a[k]
        state.x0_v = list(result_clean.x_v)
        state.x0_delta = list(result_clean.x_delta)
        t0 = time.time()
        result = est.estimate(z_dict, real_ts)
        solve_s = time.time() - t0
        stats = compute_chi2_and_lnr(est, result, z_dict, real_ts)

        dx_v = result.x_v - result_clean.x_v
        achieved_pu = float(np.max(np.abs(dx_v) / v_base_full))
        ratio = achieved_pu / intended_pu if intended_pu > 0 else float("nan")

        row = {
            "i": i, "part": part, "mode": "M1", "magnitude_pu": mag, "kind": kind,
            "converged": result.converged, "iterations": result.iterations, "solve_s": solve_s,
            "J_before": stats_clean["J"], "J_after": stats["J"], "dJ": stats["J"] - stats_clean["J"],
            "lnr_before": stats_clean["lnr"], "lnr_after": stats["lnr"], "dLNR": stats["lnr"] - stats_clean["lnr"],
            "intended_shift_pu": intended_pu, "achieved_shift_pu": achieved_pu, "achieved_over_intended": ratio,
            "unattack_residual_final": unattack_resid,
            "detected_by_J99": bool(stats["J"] > EMPIRICAL_J99) if result.converged else None,
            "detected_by_LNR99": bool(stats["lnr"] > EMPIRICAL_LNR99) if result.converged else None,
        }
        if extra:
            row.update(extra)
        return row

    for part in ["A4", "A5"]:
        for mag in MAGNITUDES:
            if part == "A4":
                atk = synthesize_a4_attack(
                    Hs_attack, Hs_unattack, state_scale, rng, "M1", mag, state.n_phase_nodes,
                )
                a = atk["a_attackable"]
                intended_pu = atk["achieved_v_shift_pu"]
                unattack_resid = atk["max_unattackable_residual"]
            else:
                a, c, n_newton, unattack_resid = build_a5_attack(
                    result_clean, model, state, Hs_unattack, state_scale, unattack_entries, mag, rng,
                )
                intended_pu = mag

            rows.append(solve_and_log(a, "attack", part, mag, intended_pu, unattack_resid))

            a_c1 = random_control(a, rng)
            rows.append(solve_and_log(a_c1, "control1_random_dir", part, mag, intended_pu, None))

            a_c2 = random_sign_control(a, rng)
            rows.append(solve_and_log(a_c2, "control2_random_sign", part, mag, intended_pu, None))

    return rows


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)

    p_cols = [c for c in df.columns if c.endswith("_P")]
    strata = stratified_picks(df, p_cols)
    all30 = sorted(set(strata["low"] + strata["mid"] + strata["high"]))
    points = all30[:N_POINTS]
    print(f"using first {N_POINTS} of 30 stratified picks: {points}", flush=True)
    assert all(x in points for x in (46, 94, 154)), "46/94/154 must be in the point set"

    row0 = df.iloc[points[0]]
    real_ts0 = str(row0["real_timestamp"])
    z_real0 = {c: float(row0[c]) for c in z_cols}

    t_total_start = time.time()
    t0 = time.time()
    rows_point0 = process_point(points[0], real_ts0, z_real0, SEED + 400)
    point0_time = time.time() - t0
    n_per_point = len(rows_point0)
    per_row = point0_time / n_per_point
    remaining = points[1:]
    n_batches = -(-len(remaining) // MAX_WORKERS)  # ceil
    projected_total_min = (point0_time + n_batches * per_row * n_per_point) / 60.0
    print(f"point 0 (i={points[0]}) done in {point0_time:.1f}s ({n_per_point} rows, {per_row:.2f}s/row). "
          f"Projected total wall-clock (<= {MAX_WORKERS} workers): {projected_total_min:.1f} min", flush=True)

    if projected_total_min > ETA_CAP_MIN:
        print(f"ABORTING: projected ETA {projected_total_min:.1f} min exceeds {ETA_CAP_MIN} min cap.", flush=True)
        pd.DataFrame(rows_point0).to_csv(OUT_PATH, index=False)
        print("saved partial (point 0 only):", OUT_PATH)
        return

    all_rows = list(rows_point0)
    pd.DataFrame(all_rows).to_csv(OUT_PATH, index=False)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for k, i in enumerate(remaining):
            row = df.iloc[i]
            real_ts = str(row["real_timestamp"])
            z_real = {c: float(row[c]) for c in z_cols}
            fut = ex.submit(process_point, i, real_ts, z_real, SEED + 400 + k + 1)
            futures[fut] = i

        n_done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            rows = fut.result()
            all_rows.extend(rows)
            n_done += 1
            pd.DataFrame(all_rows).to_csv(OUT_PATH, index=False)
            elapsed = time.time() - t_total_start
            print(f"  [parallel] point i={i} done ({n_done}/{len(remaining)}), {elapsed:.1f}s total elapsed",
                  flush=True)

    out = pd.DataFrame(all_rows)
    total_elapsed = time.time() - t_total_start
    print()
    print(f"total rows: {len(out)}  total elapsed: {total_elapsed:.1f}s")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
