"""
Branch exp-tsvd -- EXPERIMENT ONLY. Does the TSVD cutoff in the
reduced step (estimation/state_estimator.py, TSVD path; default
rcond TSVD_RCOND=1e-8) cause both the clean voltage error and the
A4 attack mismatch (validation/phase8_a4_linear_check.py: 99.2% of c
lies in the directions TSVD drops)?

Variants (est.estimate kwargs; V0 == main-line default):
  V0 rcond 1e-8, V1 1e-10, V2 1e-12, V3 1e-14,
  V4 column-equilibrated A_reduced (unit column norms) + rcond 1e-8,
  V5 equilibrated + rcond 1e-12.

Per variant:
  scorecard -- the 60-solve Phase 7 scorecard protocol of
    validation/phase7_scorecard_headpexcl.py (30 stratified seed-777
    points from flat start + their i+1 neighbours warm-started).
  start     -- the first 10 of the sorted 30 stratified points, solved
    from (a) flat start and (b) x_true as the start.
  attack    -- point 46, A4-M1, sizes 0.005 and 0.05, synthesised at
    THIS variant's own clean x_hat with the scan's rng (seed SEED+400,
    same draw order incl. both controls), then est.estimate() on
    z_clean + a warm-started at x_hat.

At most 4 worker processes, one variant per process.

Run: python -m validation.exp_tsvd_cutoff
"""

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr, unattackable_jacobian
from attacks.a4_synthesis import synthesize_a4_attack, random_control, random_sign_control

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_DIR = BASE_DIR / "validation"
SEED = 777
N_PER_STRATUM = 10
N_START_POINTS = 10
ATTACK_POINT = 46
ATTACK_SIZES = [0.005, 0.05]
SCAN_MAGNITUDES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]  # scan draw order, for the rng sequence
SCAN_SEED_OFFSET = 400
MAX_WORKERS = 4

VARIANTS = {
    "V0": {"tsvd_rcond": 1e-8, "tsvd_equilibrate": False},
    "V1": {"tsvd_rcond": 1e-10, "tsvd_equilibrate": False},
    "V2": {"tsvd_rcond": 1e-12, "tsvd_equilibrate": False},
    "V3": {"tsvd_rcond": 1e-14, "tsvd_equilibrate": False},
    "V4": {"tsvd_rcond": 1e-8, "tsvd_equilibrate": True},
    "V5": {"tsvd_rcond": 1e-12, "tsvd_equilibrate": True},
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


def load_inputs():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    return df, z_cols, npz["x_v"][:1440], npz["x_delta"][:1440]


def _record(variant, kind, i, start, res, err_v):
    kept = res.tsvd_kept or [np.nan]
    return {"variant": variant, "kind": kind, "i": i, "start": start, "converged": res.converged,
            "iterations": res.iterations, "J": res.J, "constraint_residual_inf": res.constraint_residual_inf,
            "max_v_err_pu": float(err_v.max()), "median_v_err_pu": float(np.median(err_v)),
            "kept_final": kept[-1], "kept_min": min(kept), "kept_max": max(kept)}


def run_variant(variant):
    kw = VARIANTS[variant]
    t0 = time.time()
    df, z_cols, x_v_all, x_d_all = load_inputs()
    p_cols = [c for c in df.columns if c.endswith("_P")]
    strata = stratified_picks(df, p_cols)
    sorted30 = sorted(set(strata["low"] + strata["mid"] + strata["high"]))

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    n = state.n_phase_nodes
    v_base = np.array([state.v_base[name] for name in state.all_node_names])
    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)

    def z_at(i):
        row = df.iloc[i]
        return {c: float(row[c]) for c in z_cols}, str(row["real_timestamp"])

    def solve(z, ts, x0_v, x0_d):
        state.x0_v, state.x0_delta = list(x0_v), list(x0_d)
        return est.estimate(z, ts, **kw)

    # --- scorecard (60 solves), same protocol as phase7_scorecard_headpexcl.py
    score, node_errs = [], []
    for idxs in strata.values():
        for i in idxs:
            z, ts = z_at(i)
            ra = solve(z, ts, flat_v, flat_delta)
            ea = np.abs(ra.x_v - x_v_all[i]) / v_base
            score.append(_record(variant, "scorecard", i, "A_flat", ra, ea))
            node_errs.append(ea)
            j = i + 1
            if j >= len(df):
                continue
            z, ts = z_at(j)
            rb = solve(z, ts, ra.x_v, ra.x_delta)
            eb = np.abs(rb.x_v - x_v_all[j]) / v_base
            score.append(_record(variant, "scorecard", j, "B_warm", rb, eb))
            node_errs.append(eb)
    pooled = np.concatenate(node_errs)
    sc_err = {"sc_v_median": float(np.median(pooled)), "sc_v_p95": float(np.percentile(pooled, 95)),
              "sc_v_max": float(pooled.max())}
    node_max = np.max(np.vstack(node_errs), axis=0)
    worst3 = [(state.all_node_names[k], float(node_max[k])) for k in np.argsort(-node_max)[:3]]

    # --- start-point test (10 points, flat vs x_true)
    start_rows, start_errs, start_diff = [], {"flat": [], "x_true": []}, []
    for i in sorted30[:N_START_POINTS]:
        z, ts = z_at(i)
        rf = solve(z, ts, flat_v, flat_delta)
        rt = solve(z, ts, x_v_all[i], x_d_all[i])
        ef = np.abs(rf.x_v - x_v_all[i]) / v_base
        et = np.abs(rt.x_v - x_v_all[i]) / v_base
        start_rows.append(_record(variant, "start", i, "flat", rf, ef))
        start_rows.append(_record(variant, "start", i, "x_true", rt, et))
        start_errs["flat"].append(ef)
        start_errs["x_true"].append(et)
        dv = float(np.max(np.abs(rf.x_v - rt.x_v) / v_base))
        dd = float(np.max(np.abs(rf.x_delta - rt.x_delta)))
        start_diff.append({"variant": variant, "i": i, "max_abs_dV_flat_vs_xtrue_pu": dv,
                           "max_abs_ddelta_flat_vs_xtrue_rad": dd})

    # --- attack check (point 46, A4-M1)
    k_point = sorted30[:20].index(ATTACK_POINT)
    z, ts = z_at(ATTACK_POINT)
    rc = solve(z, ts, flat_v, flat_delta)
    sc = compute_chi2_and_lnr(est, rc, z, ts)
    jac = unattackable_jacobian(est, rc, ts)
    meas_ids = jac["meas_ids"]
    s = jac["state_scale"]
    rng = np.random.default_rng(SEED + SCAN_SEED_OFFSET + k_point)
    atk_rows = []
    for mag in SCAN_MAGNITUDES:
        atk = synthesize_a4_attack(jac["Hs_meas"], jac["Hs_unattackable"], s, rng, "M1", mag, n)
        random_control(atk["a_attackable"], rng)
        random_sign_control(atk["a_attackable"], rng)
        if mag not in ATTACK_SIZES:
            continue
        zb = dict(z)
        for k, mid in enumerate(meas_ids):
            zb[mid] = z[mid] + atk["a_attackable"][k]
        rb = solve(zb, ts, rc.x_v, rc.x_delta)
        sb = compute_chi2_and_lnr(est, rb, zb, ts)
        dy = np.concatenate([rb.x_v - rc.x_v, rb.x_delta - rc.x_delta]) / s
        y_c = atk["c"] / s
        atk_rows.append({
            "variant": variant, "i": ATTACK_POINT, "size_pu": mag,
            "clean_converged": rc.converged, "clean_J": sc["J"], "clean_lnr": sc["lnr"],
            "converged": rb.converged, "iterations": rb.iterations,
            "rel_mismatch": float(np.linalg.norm(dy - y_c) / np.linalg.norm(y_c)),
            "achieved_v_pu": float(np.max(np.abs(dy[:n]))),
            "dJ": sb["J"] - sc["J"], "dLNR": sb["lnr"] - sc["lnr"],
            "kept_final": (rb.tsvd_kept or [np.nan])[-1],
        })

    return {"variant": variant, "score": score, "sc_err": sc_err, "worst3": worst3, "start": start_rows,
            "start_errs": {k: np.concatenate(v) for k, v in start_errs.items()},
            "start_diff": start_diff, "attack": atk_rows, "dof": est.dof, "runtime_s": time.time() - t0}


def main():
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(run_variant, list(VARIANTS)))

    score = pd.DataFrame([r for res in results for r in res["score"]])
    start = pd.DataFrame([r for res in results for r in res["start"]])
    start_diff = pd.DataFrame([r for res in results for r in res["start_diff"]])
    attack = pd.DataFrame([r for res in results for r in res["attack"]])

    summary = []
    for res in results:
        v = res["variant"]
        sc = score[score["variant"] == v]
        row = {"variant": v, **VARIANTS[v], "runtime_s": res["runtime_s"],
               "sc_n": len(sc), "sc_converged": int(sc["converged"].sum()),
               "sc_mean_iterations": sc["iterations"].mean(), **res["sc_err"],
               "sc_J_mean": sc["J"].mean(), "sc_J_over_dof": sc["J"].mean() / res["dof"],
               "sc_feasibility_max": sc["constraint_residual_inf"].max(),
               "sc_kept_min": sc["kept_final"].min(), "sc_kept_median": sc["kept_final"].median(),
               "sc_kept_max": sc["kept_final"].max(),
               "sc_worst3_nodes": "; ".join(f"{nm}={e:.4f}" for nm, e in res["worst3"])}
        for st in ["flat", "x_true"]:
            ss = start[(start["variant"] == v) & (start["start"] == st)]
            e = res["start_errs"][st]
            row.update({f"st_{st}_v_median": float(np.median(e)), f"st_{st}_v_p95": float(np.percentile(e, 95)),
                        f"st_{st}_v_max": float(e.max()), f"st_{st}_J_mean": ss["J"].mean(),
                        f"st_{st}_J_over_dof": ss["J"].mean() / res["dof"],
                        f"st_{st}_converged": int(ss["converged"].sum()),
                        f"st_{st}_mean_iterations": ss["iterations"].mean(),
                        f"st_{st}_kept_final_median": ss["kept_final"].median(),
                        f"st_{st}_kept_min": ss["kept_min"].min(), f"st_{st}_kept_max": ss["kept_max"].max()})
        sd = start_diff[start_diff["variant"] == v]
        row["st_max_abs_dV_flat_vs_xtrue_pu"] = sd["max_abs_dV_flat_vs_xtrue_pu"].max()
        row["st_max_abs_ddelta_flat_vs_xtrue_rad"] = sd["max_abs_ddelta_flat_vs_xtrue_rad"].max()
        for _, a in attack[attack["variant"] == v].iterrows():
            tag = f"atk_{a['size_pu']}"
            row.update({f"{tag}_rel_mismatch": a["rel_mismatch"], f"{tag}_dJ": a["dJ"], f"{tag}_dLNR": a["dLNR"],
                        f"{tag}_converged": a["converged"], f"{tag}_iterations": a["iterations"]})
        summary.append(row)
    summary = pd.DataFrame(summary)

    score.to_csv(OUT_DIR / "exp_tsvd_scorecard.csv", index=False)
    start.to_csv(OUT_DIR / "exp_tsvd_start.csv", index=False)
    start_diff.to_csv(OUT_DIR / "exp_tsvd_start_diff.csv", index=False)
    attack.to_csv(OUT_DIR / "exp_tsvd_attack.csv", index=False)
    summary.to_csv(OUT_DIR / "exp_tsvd_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 200)
    print(summary.T.to_string())
    print(f"total elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
