"""
One-timestep detection check of attacks/library.py A1-A3 against the
adopted estimator (default init="pseudo_pf") and the active thresholds
(validation/bdd_thresholds.py).

20 of the 30 stratified (seed=777) points (first 20 sorted, includes 46,
94, 154). Each case is applied as a step on a 1-row window at that
point; seed = 1000 * point_index + case_index.
  A1: k = 5, 20, 165  x  m = 1, 3, 5, 10
  A2: one random meter (P, Q, V), m = 3, 5, 10, 20
  A3: dP = 10, 20, 50 % of the smaller load, access full / partial

impact = max_nodes |V_att - V_true| / v_base  -  max_nodes |V_clean - V_true| / v_base
shift  = max_nodes |V_att - V_clean| / v_base
x_true (test_x_true.npz) is read ONLY for these two scores.

One process (A3 and the power-flow start call OpenDSS).

Run: python -m validation.phase8_attack_library_check
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from estimation.bad_data_detection import compute_chi2_and_lnr
from attacks.library import attack, AttackSpec, AttackContext
from validation.bdd_thresholds import load_active_thresholds

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase8_attack_library_check.csv"
OUT_SUMMARY = BASE_DIR / "validation" / "phase8_attack_library_check_summary.csv"
SEED = 777
N_PER_STRATUM = 10
N_POINTS = 20

CASES = (
    [("A1", dict(k=k, magnitude=m), f"A1_k{k}_m{m}") for k in (5, 20, 165) for m in (1, 3, 5, 10)]
    + [("A2", dict(magnitude=m), f"A2_m{m}") for m in (3, 5, 10, 20)]
    + [("A3", dict(magnitude=f, access=acc), f"A3_{int(f * 100)}pct_{acc}")
       for f in (0.10, 0.20, 0.50) for acc in ("full", "partial")]
)


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
    j99, lnr99 = load_active_thresholds()
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    x_v_true = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)["x_v"][:1440]  # scoring only
    points = sorted(set(sum(stratified_picks(df, [c for c in z_cols if c.endswith("_P")]).values(), [])))[:N_POINTS]
    assert all(x in points for x in (46, 94, 154))

    ctx = AttackContext()
    est, state = ctx.est, ctx.model.state
    v_base = np.array([state.v_base[n] for n in state.all_node_names])
    _ = ctx.caps
    print(f"J99={j99:.4f} LNR99={lnr99:.4f} points={points}", flush=True)

    rows = []
    t_all = time.time()
    for pi, i in enumerate(points):
        t_pt = time.time()
        zrow = df.iloc[[i]][z_cols]
        ts = [str(df.iloc[i]["real_timestamp"])]
        z0 = zrow.iloc[0].to_dict()
        clean = est.estimate(z0, ts[0])
        sc = compute_chi2_and_lnr(est, clean, z0, ts[0])
        err_clean = np.abs(clean.x_v - x_v_true[i]) / v_base
        rows.append({"i": i, "case": "clean", "type": "clean", "converged": clean.converged, "J": sc["J"],
                     "lnr": sc["lnr"], "flag_J99": bool(sc["J"] > j99), "flag_LNR99": bool(sc["lnr"] > lnr99)})
        for ci, (typ, kw, name) in enumerate(CASES):
            spec = AttackSpec(typ, 0, 1, "step", seed=1000 * pi + ci, **kw)
            za, lb = attack(zrow, ts, spec, ctx)
            z1 = za.iloc[0].to_dict()
            res = est.estimate(z1, ts[0])
            st = compute_chi2_and_lnr(est, res, z1, ts[0])
            err = np.abs(res.x_v - x_v_true[i]) / v_base
            lab = lb.iloc[0]
            a = (za.iloc[0] - zrow.iloc[0])
            row = {"i": i, "case": name, "type": typ, "seed": spec.seed, **kw,
                   "converged": res.converged, "iterations": res.iterations,
                   "J": st["J"], "lnr": st["lnr"], "dJ": st["J"] - sc["J"], "dLNR": st["lnr"] - sc["lnr"],
                   "flag_J99": bool(st["J"] > j99) if res.converged else None,
                   "flag_LNR99": bool(st["lnr"] > lnr99) if res.converged else None,
                   "impact_max_err_change_pu": float(err.max() - err_clean.max()),
                   "shift_max_dV_pu": float(np.max(np.abs(res.x_v - clean.x_v) / v_base)),
                   "n_channels": int((a != 0).sum()), "targets": lab["target_ids"],
                   "max_abs_a_over_abs_z": float(np.max(np.abs(a[a != 0]) / np.maximum(np.abs(zrow.iloc[0][a != 0]), 1e-12)))
                   if (a != 0).any() else 0.0}
            for c in ("n_capped", "n_targets", "a3_load_up", "a3_load_down", "a3_dP_kw", "a3_dQ_total_kvar",
                      "a3_dP_consumed_kw", "a3_dLoss_kw", "a3_pf_converged"):
                if c in lb.columns:
                    row[c] = lab[c]
            rows.append(row)
        print(f"  point i={i} done in {time.time() - t_pt:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    att = out[out["case"] != "clean"]
    summ = []
    for name in [c[2] for c in CASES]:
        g = att[att["case"] == name]
        conv = g[g["converged"].astype(bool)]
        fj, fl = conv["flag_J99"].astype(bool), conv["flag_LNR99"].astype(bool)
        d = {"case": name, "n": len(g), "converged_rate": len(conv) / len(g),
             "det_rate_J99": fj.mean(), "det_rate_LNR99": fl.mean(), "det_rate_either": (fj | fl).mean(),
             "median_dJ": g["dJ"].median(), "median_dLNR": g["dLNR"].median(),
             "median_impact_max_err_change_pu": g["impact_max_err_change_pu"].median(),
             "median_shift_max_dV_pu": g["shift_max_dV_pu"].median(),
             "median_max_abs_a_over_abs_z": g["max_abs_a_over_abs_z"].median()}
        if "n_capped" in g and g["n_capped"].notna().any():
            d["cap_binds_channels"] = int(g["n_capped"].sum())
            d["cap_total_channels"] = int(g["n_targets"].sum())
        summ.append(d)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT_SUMMARY, index=False)
    clean = out[out["case"] == "clean"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(summ.to_string(index=False))
    print(f"clean points flagged: J99 {int(clean['flag_J99'].sum())}/{len(clean)}  "
          f"LNR99 {int(clean['flag_LNR99'].sum())}/{len(clean)}  "
          f"either {int((clean['flag_J99'] | clean['flag_LNR99']).sum())}/{len(clean)}")
    print(f"total elapsed {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
