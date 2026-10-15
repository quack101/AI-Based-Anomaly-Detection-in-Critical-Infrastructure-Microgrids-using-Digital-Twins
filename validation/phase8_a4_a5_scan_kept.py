"""
Branch est-pf-start -- A4/A5 size sweep with the attack built INSIDE
the directions the estimator keeps (attacks/a4_kept.py), against the
estimator's default start (init="pseudo_pf", adopted on this branch:
every solve -- clean and attacked -- starts from the power flow of the
telemetry it is given, so an attack also moves the start point).

20 of the 30 stratified (seed=777) points (first 20 sorted, includes
46, 94, 154); per-point rng seed SEED+400+k as before (the draw ORDER
differs from the old scan, so individual draws are not comparable).
Sizes 0.005/0.01/0.02/0.05/0.10/0.20 pu, for:
  A4 -- linear: a = H_attackable c, c from attacks/a4_kept.py.
  A5 -- exact nonlinear: x_bad = x_hat + c (same kept-direction c),
        Newton-corrected so h_unattackable(x_bad) = h_unattackable(x_hat)
        (REGZI_*, PSEUDO_*, ZI rows), a = h_att(x_bad) - h_att(x_hat).
        "Works" = residual below DEFAULT_TOL_CONSTRAINT_KW within 5
        Newton steps; logged per row (a5_newton_ok).
Both with a = 0 on the 3 P_head rows, the angle part of c bounded by
the size in rad, and two controls:
  control1_weighted -- random direction, same ||W^0.5 a|| as the attack.
  control2_random_sign -- same per-channel |a_i|, random signs.

rel_mismatch = ||dx - c|| / ||c|| in the estimator's scaled coordinates
(V in pu, angle in rad), dx = x_after - x_hat. For control rows it is
computed against the attack's c (the control is not built from c).

One process only (every solve calls the OpenDSS power flow start).
Thresholds: validation/bdd_thresholds.py (active source).

Run: python -m validation.phase8_a4_a5_scan_kept            (full sweep)
     python -m validation.phase8_a4_a5_scan_kept --point 46  (one point,
       plus a diagnostic re-solve of each attack warm-started at x_hat
       with init="flat"; writes nothing)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator, DEFAULT_TOL_CONSTRAINT_KW
from estimation.bad_data_detection import compute_chi2_and_lnr, unattackable_jacobian
from attacks.a4_synthesis import random_sign_control
from attacks.a4_kept import kept_subspace_basis, synthesize_kept_attack, random_weighted_control, a5_from_c
from validation.bdd_thresholds import load_active_thresholds

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase8_a4_a5_scan_kept.csv"
OUT_SUMMARY = BASE_DIR / "validation" / "phase8_a4_a5_scan_kept_summary.csv"
OUT_POINTS = BASE_DIR / "validation" / "phase8_a4_a5_scan_kept_points.csv"

J99, LNR99 = load_active_thresholds()
SEED = 777
N_PER_STRATUM = 10
N_POINTS = 20
MAGNITUDES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


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


def process_point(i, real_ts, z_real, seed_for_point, model, est, warm_diag=False):
    state = model.state
    n = state.n_phase_nodes
    meas_ids = [e["id"] for e in model.meas_entries]

    t_pt = time.time()
    result_clean = est.estimate(z_real, real_ts)
    stats_clean = compute_chi2_and_lnr(est, result_clean, z_real, real_ts)
    basis = kept_subspace_basis(est, result_clean, real_ts)
    unattack_entries = unattackable_jacobian(est, result_clean, real_ts)["unattackable_entries"]
    s = basis["s"]
    x_clean = np.concatenate([result_clean.x_v, result_clean.x_delta])
    point_info = {"i": i, "clean_converged": result_clean.converged, "clean_iterations": result_clean.iterations,
                  "clean_J": stats_clean["J"], "clean_lnr": stats_clean["lnr"],
                  "clean_flag_J99": bool(stats_clean["J"] > J99), "clean_flag_LNR99": bool(stats_clean["lnr"] > LNR99),
                  "dim_N": basis["dim_N"], "dim_K": basis["dim_K"], "n_intersection": basis["n_intersection"],
                  "cos_max": float(basis["cosines"].max()),
                  "cos_2nd": float(np.sort(basis["cosines"])[-2]) if basis["cosines"].size > 1 else float("nan"),
                  "fallback_ratio": basis["fallback_ratio"]}

    rng = np.random.default_rng(seed_for_point)
    rows = []

    def solve_and_log(a, kind, part, mag, atk, extra):
        z_dict = dict(z_real)
        for k, mid in enumerate(meas_ids):
            z_dict[mid] = z_real[mid] + a[k]
        t0 = time.time()
        result = est.estimate(z_dict, real_ts)
        solve_s = time.time() - t0
        stats = compute_chi2_and_lnr(est, result, z_dict, real_ts)
        y_c = atk["c"] / s
        dy = (np.concatenate([result.x_v, result.x_delta]) - x_clean) / s
        row = {
            "i": i, "part": part, "magnitude_pu": mag, "kind": kind, "branch": atk["branch"],
            "converged": result.converged, "iterations": result.iterations, "solve_s": solve_s,
            "J_before": stats_clean["J"], "J_after": stats["J"], "dJ": stats["J"] - stats_clean["J"],
            "lnr_before": stats_clean["lnr"], "lnr_after": stats["lnr"], "dLNR": stats["lnr"] - stats_clean["lnr"],
            "rel_mismatch": float(np.linalg.norm(dy - y_c) / np.linalg.norm(y_c)),
            "c_max_v_pu": float(np.max(np.abs(y_c[:n]))), "c_max_ang_rad": float(np.max(np.abs(y_c[n:]))),
            "achieved_v_pu": float(np.max(np.abs(dy[:n]))),
            "achieved_over_intended": float(np.max(np.abs(dy[:n]))) / mag,
            "weighted_norm_a": float(np.linalg.norm(a / basis["sigma_meas"])),
            "angle_bound_binds": atk["angle_bound_binds"], "predicted_dJ": atk["predicted_dJ"],
            "frac_c_in_K": atk["frac_in_K"],
            "detected_by_J99": bool(stats["J"] > J99) if result.converged else None,
            "detected_by_LNR99": bool(stats["lnr"] > LNR99) if result.converged else None,
        }
        row.update(extra)
        if warm_diag and kind == "attack":
            state.x0_v, state.x0_delta = list(result_clean.x_v), list(result_clean.x_delta)
            rw = est.estimate(z_dict, real_ts, init="flat")
            sw = compute_chi2_and_lnr(est, rw, z_dict, real_ts)
            dyw = (np.concatenate([rw.x_v, rw.x_delta]) - x_clean) / s
            row.update({"diag_warm_rel_mismatch": float(np.linalg.norm(dyw - y_c) / np.linalg.norm(y_c)),
                        "diag_warm_dJ": sw["J"] - stats_clean["J"], "diag_warm_dLNR": sw["lnr"] - stats_clean["lnr"],
                        "diag_warm_converged": rw.converged, "diag_warm_iterations": rw.iterations})
        return row

    for part in ["A4", "A5"]:
        for mag in MAGNITUDES:
            atk = synthesize_kept_attack(basis, rng, mag)
            extra = {}
            if part == "A4":
                a = atk["a"]
            else:
                a, c5, n_newton, resid, frac5 = a5_from_c(model, state, result_clean, unattack_entries, basis, atk["c"])
                atk = dict(atk, c=c5, frac_in_K=frac5)
                extra = {"a5_newton_steps": n_newton, "a5_unattack_resid": resid,
                         "a5_newton_ok": bool(resid <= DEFAULT_TOL_CONSTRAINT_KW),
                         "a5_c_max_ang_rad": float(np.max(np.abs(c5[state.n_phase_nodes:])))}
            rows.append(solve_and_log(a, "attack", part, mag, atk, extra))
            rows.append(solve_and_log(random_weighted_control(a, basis["sigma_meas"], rng),
                                      "control1_weighted", part, mag, atk, extra))
            rows.append(solve_and_log(random_sign_control(a, rng), "control2_random_sign", part, mag, atk, extra))

    point_info["runtime_s"] = time.time() - t_pt
    return rows, point_info


def summarize(df, points):
    out = []
    for (part, mag, kind), g in df.groupby(["part", "magnitude_pu", "kind"], sort=True):
        conv = g[g["converged"].astype(bool)]
        dj = conv["detected_by_J99"].astype(bool)
        dl = conv["detected_by_LNR99"].astype(bool)
        out.append({
            "part": part, "magnitude_pu": mag, "kind": kind, "n": len(g), "n_converged": len(conv),
            "convergence_rate": len(conv) / len(g),
            "det_rate_J99": dj.mean() if len(conv) else np.nan,
            "det_rate_LNR99": dl.mean() if len(conv) else np.nan,
            "det_rate_either": (dj | dl).mean() if len(conv) else np.nan,
            "median_dJ": g["dJ"].median(), "p95_dJ": g["dJ"].quantile(0.95),
            "median_dLNR": g["dLNR"].median(), "p95_dLNR": g["dLNR"].quantile(0.95),
            "median_rel_mismatch": g["rel_mismatch"].median(),
            "median_achieved_over_intended": g["achieved_over_intended"].median(),
            "angle_bound_binds_rate": g["angle_bound_binds"].mean(),
            "median_predicted_dJ": g["predicted_dJ"].median(),
            "a5_newton_ok_rate": g["a5_newton_ok"].mean() if "a5_newton_ok" in g and g["a5_newton_ok"].notna().any() else np.nan,
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", type=int, default=None)
    args = ap.parse_args()

    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    strata = stratified_picks(df, [c for c in df.columns if c.endswith("_P")])
    points = sorted(set(strata["low"] + strata["mid"] + strata["high"]))[:N_POINTS]
    assert all(x in points for x in (46, 94, 154))

    model = build_estimator_model("multiphase")
    est = StateEstimator(model)
    print(f"J99={J99:.4f} LNR99={LNR99:.4f}  points={points}", flush=True)

    todo = [args.point] if args.point is not None else points
    all_rows, infos = [], []
    t0 = time.time()
    for i in todo:
        k = points.index(i)
        row = df.iloc[i]
        rows, info = process_point(i, str(row["real_timestamp"]), {c: float(row[c]) for c in z_cols},
                                   SEED + 400 + k, model, est, warm_diag=args.point is not None)
        all_rows += rows
        infos.append(info)
        print(f"  point i={i} done in {info['runtime_s']:.1f}s  (N={info['dim_N']} K={info['dim_K']} "
              f"n_inter={info['n_intersection']} cos_max={info['cos_max']:.4f})", flush=True)
        if args.point is None:
            pd.DataFrame(all_rows).to_csv(OUT_PATH, index=False)

    out = pd.DataFrame(all_rows)
    pts = pd.DataFrame(infos)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    if args.point is not None:
        sel = out[(out["magnitude_pu"].isin([0.005, 0.05]))]
        cols = ["part", "magnitude_pu", "kind", "branch", "converged", "iterations", "rel_mismatch", "dJ", "dLNR",
                "achieved_v_pu", "angle_bound_binds", "predicted_dJ", "frac_c_in_K", "weighted_norm_a"]
        cols += [c for c in out.columns if c.startswith("diag_warm") or c.startswith("a5_")]
        print(sel[cols].to_string(index=False))
        print(pts.to_string(index=False))
        return

    out.to_csv(OUT_PATH, index=False)
    pts.to_csv(OUT_POINTS, index=False)
    summ = summarize(out, pts)
    summ.to_csv(OUT_SUMMARY, index=False)
    print(summ.to_string(index=False))
    print()
    print(pts.to_string(index=False))
    fj, fl = pts["clean_flag_J99"], pts["clean_flag_LNR99"]
    print(f"clean points flagged: J99 {int(fj.sum())}/{len(pts)}  LNR99 {int(fl.sum())}/{len(pts)}  "
          f"either {int((fj | fl).sum())}/{len(pts)}")
    print(f"total elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
