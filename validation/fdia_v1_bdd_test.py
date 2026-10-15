"""
Part C -- residual-based bad-data detector baseline on dataset/fdia_v1
test split, with the FROZEN estimator (estimator-v5-hfix defaults:
network-Y measurement model, S1 + structural sigmas, pseudo_pf start)
and config/bdd_thresholds_v5.json (J99, LNR99).

Per timestep: J (chi-square) and max normalized residual (LNR,
estimation/bad_data_detection.py), flagged if J > J99 or LNR > LNR99.
max |V_est - V_true| is recorded for scoring only (test_x_true.npz;
never used to detect). The z for each row comes from test_z.csv; the
hour for the pseudo bucket comes from the corpus test split's own
real_timestamp (the clock, available at measurement time). The row
position i aligns test_z, test_labels, the corpus telemetry and
test_x_true.

Timing: 50 solves are timed and the full run projected; if <= 5 h every
test timestep is run, else every attacked timestep is kept and clean
timesteps are strided (stride reported). Progress is checkpointed every
500 rows.

Window summary: a window is detected if any of its attacked timesteps
is flagged. Detection rate by type/class/level/profile/duration;
timestep false-alarm rate on clean timesteps; median detection delay.

Sensitivity: the window summary is repeated with the OTHER sigma
setting U1 = network_y + S0 (registry sigmas, no overrides), using
U1's own J99/LNR99 from a 90-point clean baseline, on a 2,000-timestep
sample (every attacked window's timesteps, sampled evenly, plus clean
timesteps to fill 2,000).

Run: python -m validation.fdia_v1_bdd_test
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr
from validation.bdd_thresholds import load_active_thresholds

BASE_DIR = Path(__file__).resolve().parent.parent
FDIA = BASE_DIR / "dataset" / "fdia_v1"
CORPUS = BASE_DIR / "simulation" / "phase6_corpus"
OUT = BASE_DIR / "validation"
TIME_BUDGET_H = 5.0
CHECKPOINT = 500
U1_SAMPLE = 2000
SEED = 4242


def load_inputs():
    z = pd.read_csv(FDIA / "test_z.csv")
    labels = pd.read_csv(FDIA / "test_labels.csv")
    corpus = pd.read_csv(CORPUS / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"])
    x_v_true = np.load(CORPUS / "test_x_true.npz", allow_pickle=True)["x_v"]
    assert (z["timestamp"].values == corpus["timestamp"].values).all()
    assert len(z) == len(labels) == len(x_v_true)
    z_cols = [c for c in z.columns if c != "timestamp"]
    return z, labels, corpus["real_timestamp"].tolist(), x_v_true, z_cols


def solve_rows(est, z, real_ts, x_v_true, z_cols, rows, j99, lnr99, v_base, out_path=None, label=""):
    recs = []
    t0 = time.time()
    for n, i in enumerate(rows):
        zr = {c: float(z.iat[i, z.columns.get_loc(c)]) for c in z_cols}
        ts = str(real_ts[i])
        res = est.estimate(zr, ts)
        st = compute_chi2_and_lnr(est, res, zr, ts)
        err = float(np.max(np.abs(res.x_v - x_v_true[i]) / v_base))
        recs.append({"row": i, "timestamp": int(z.iat[i, 0]), "converged": res.converged,
                     "iterations": res.iterations, "J": st["J"], "max_lnr": st["lnr"],
                     "top_lnr_channel": st["ids"][int(np.nanargmax(np.where(
                         np.isfinite(st["normalized_residuals"]) &
                         ~(np.asarray(st["redundancy"]) < 1e-3),
                         np.abs(st["normalized_residuals"]), -1)))],
                     "flag_J99": bool(st["J"] > j99), "flag_LNR99": bool(st["lnr"] > lnr99),
                     "max_v_err_pu": err})
        if out_path is not None and (n + 1) % CHECKPOINT == 0:
            pd.DataFrame(recs).to_csv(out_path, index=False)
            el = time.time() - t0
            print(f"  [{label}] {n+1}/{len(rows)}  {el:.0f}s  ({el/(n+1):.3f}s/row)", flush=True)
    df = pd.DataFrame(recs)
    if out_path is not None:
        df.to_csv(out_path, index=False)
    return df


def window_summary(labels, flags_by_row):
    """flags_by_row: {row -> flagged bool} over the rows actually solved.
    Returns (window rows, clean false-alarm rate, detection-delay list)."""
    aw = labels[labels["attacked"] == 1]
    rows = []
    delays = []
    for wid, g in aw.groupby("window_id"):
        ridx = [r for r in g.index if r in flags_by_row]
        if not ridx:
            continue
        flagged = [r for r in ridx if flags_by_row[r]]
        detected = len(flagged) > 0
        first = min(flagged) if detected else None
        delay = (first - min(ridx)) if detected else None
        if detected:
            delays.append(delay)
        r0 = g.iloc[0]
        rows.append({"window_id": wid, "type": r0["type"], "class": r0["class"], "level": r0["level"],
                     "profile": r0["profile"], "duration": int(r0["duration"]),
                     "n_timesteps_solved": len(ridx), "n_flagged": len(flagged),
                     "detected": detected, "detection_delay": delay})
    clean_rows = [r for r in labels.index[labels["attacked"] == 0] if r in flags_by_row]
    fa = np.mean([flags_by_row[r] for r in clean_rows]) if clean_rows else float("nan")
    return pd.DataFrame(rows), fa, delays


def detection_rates(win):
    out = []
    for dim in ["type", "class", "level", "profile", "duration"]:
        g = win.groupby(dim)["detected"].agg(n="size", detected="sum").reset_index()
        g["detection_rate"] = g["detected"] / g["n"]
        g.insert(0, "dimension", dim)
        out.append(g.rename(columns={dim: "value"}))
    overall = pd.DataFrame([{"dimension": "ALL", "value": "all", "n": len(win),
                             "detected": int(win["detected"].sum()),
                             "detection_rate": win["detected"].mean()}])
    return pd.concat([overall] + out, ignore_index=True)


def baseline_thresholds(est, v_base):
    """90-point clean baseline (same protocol as phase8_bdd_baseline_pfstart)
    for an arbitrary estimator; returns (J99, LNR99)."""
    telem = pd.read_csv(CORPUS / "test_telemetry.csv")
    zc = [c for c in telem.columns if c not in ("timestamp", "real_timestamp")]
    df = telem.iloc[:1440]
    tot = df[[c for c in zc if c.endswith("_P")]].sum(axis=1)
    q1, q2 = tot.quantile([1 / 3, 2 / 3])
    rng = np.random.default_rng(777)
    pick = lambda idx, k: sorted(rng.choice(idx, size=k, replace=False).tolist())
    strat = pick(df.index[tot <= q1], 10) + pick(df.index[(tot > q1) & (tot <= q2)], 10) + pick(df.index[tot > q2], 10)
    rng2 = np.random.default_rng(778)
    extra = sorted(rng2.choice([i for i in range(len(telem)) if i not in strat], size=60, replace=False).tolist())
    Js, Ls = [], []
    for i in sorted(set(strat)) + extra:
        row = telem.iloc[i]
        zr = {c: float(row[c]) for c in zc}
        ts = str(row["real_timestamp"])
        st = compute_chi2_and_lnr(est, est.estimate(zr, ts), zr, ts)
        Js.append(st["J"]); Ls.append(st["lnr"])
    return float(np.quantile(Js, 0.99)), float(np.quantile(Ls, 0.99))


def main():
    j99, lnr99 = load_active_thresholds()
    z, labels, real_ts, x_v_true, z_cols = load_inputs()
    model = build_estimator_model("multiphase")          # v5 defaults
    est = StateEstimator(model)
    v_base = np.array([model.state.v_base[n] for n in model.state.all_node_names])
    print(f"estimator defaults: h_model={model.h_model}  m_weighted={est.m_weighted}  dof={est.dof}", flush=True)
    print(f"thresholds v5: J99={j99:.4f} LNR99={lnr99:.4f}  test rows={len(z)}", flush=True)

    # --- timing + projection
    t0 = time.time()
    _ = solve_rows(est, z, real_ts, x_v_true, z_cols, list(range(50)), j99, lnr99, v_base)
    per = (time.time() - t0) / 50
    proj_h = per * len(z) / 3600
    print(f"50-solve timing: {per:.3f}s/row  -> full {len(z)} rows ~ {proj_h:.2f} h", flush=True)
    att_rows = labels.index[labels["attacked"] == 1].tolist()
    if proj_h <= TIME_BUDGET_H:
        rows = list(range(len(z)))
        stride = 1
    else:
        stride = int(np.ceil(per * (len(z) - len(att_rows)) / ((TIME_BUDGET_H * 3600) - per * len(att_rows))))
        stride = max(stride, 1)
        clean = labels.index[labels["attacked"] == 0].tolist()[::stride]
        rows = sorted(set(att_rows) | set(clean))
        print(f"projection over budget: keeping all {len(att_rows)} attacked + clean stride {stride} "
              f"-> {len(rows)} rows", flush=True)

    # --- main run (v5)
    per_ts = solve_rows(est, z, real_ts, x_v_true, z_cols, rows, j99, lnr99, v_base,
                        out_path=OUT / "fdia_v1_bdd_test_v5_pertimestep.csv", label="v5")
    flags = {int(r["row"]): bool(r["flag_J99"] or r["flag_LNR99"]) for _, r in per_ts.iterrows()}
    win, fa, delays = window_summary(labels, flags)
    win.to_csv(OUT / "fdia_v1_bdd_test_v5_windows.csv", index=False)
    rates = detection_rates(win)
    rates.to_csv(OUT / "fdia_v1_bdd_test_v5_rates.csv", index=False)
    conv = per_ts["converged"].mean()
    print(f"\n[v5] solved {len(per_ts)} rows, converged {conv*100:.1f}%, stride {stride}")
    print(f"[v5] windows {len(win)}, detected {int(win['detected'].sum())}, "
          f"clean-timestep false-alarm rate {fa*100:.2f}%, median detection delay "
          f"{np.median(delays) if delays else float('nan')} timesteps")
    pd.set_option("display.width", 200)
    print(rates.to_string(index=False))

    # --- sensitivity: U1 = network_y + S0, own thresholds, 2000-sample
    est_u1 = StateEstimator(build_estimator_model("multiphase", sigma_overrides={}))
    j99_u1, lnr99_u1 = baseline_thresholds(est_u1, v_base)
    print(f"\n[U1 network_y+S0] own baseline J99={j99_u1:.4f} LNR99={lnr99_u1:.4f}", flush=True)
    rng = np.random.default_rng(SEED)
    sample = set()
    for wid, g in labels[labels["attacked"] == 1].groupby("window_id"):
        idx = g.index.tolist()
        take = idx if len(idx) <= 20 else [idx[int(round(x))] for x in np.linspace(0, len(idx) - 1, 20)]
        sample |= set(take)
    clean_idx = labels.index[labels["attacked"] == 0].tolist()
    n_clean = max(0, U1_SAMPLE - len(sample))
    sample |= set(rng.choice(clean_idx, size=min(n_clean, len(clean_idx)), replace=False).tolist())
    sample = sorted(sample)
    print(f"[U1] 2000-sample: every attacked window evenly to <=20 timesteps ({sum(labels.loc[list(sample),'attacked'])} "
          f"attacked) + {len(sample)-int(labels.loc[list(sample),'attacked'].sum())} random clean = {len(sample)} rows",
          flush=True)
    per_u1 = solve_rows(est_u1, z, real_ts, x_v_true, z_cols, sample, j99_u1, lnr99_u1, v_base,
                        out_path=OUT / "fdia_v1_bdd_test_u1_pertimestep.csv", label="U1")
    flags_u1 = {int(r["row"]): bool(r["flag_J99"] or r["flag_LNR99"]) for _, r in per_u1.iterrows()}
    win_u1, fa_u1, delays_u1 = window_summary(labels, flags_u1)
    win_u1.to_csv(OUT / "fdia_v1_bdd_test_u1_windows.csv", index=False)
    rates_u1 = detection_rates(win_u1)
    rates_u1.to_csv(OUT / "fdia_v1_bdd_test_u1_rates.csv", index=False)
    json.dump({"v5": {"J99": j99, "LNR99": lnr99, "stride": stride, "n_rows": len(per_ts),
                      "converged_frac": float(conv), "clean_fa_rate": float(fa),
                      "median_delay": float(np.median(delays)) if delays else None,
                      "windows_detected": int(win["detected"].sum()), "windows": len(win)},
               "u1": {"J99": j99_u1, "LNR99": lnr99_u1, "n_sample": len(sample),
                      "clean_fa_rate": float(fa_u1),
                      "median_delay": float(np.median(delays_u1)) if delays_u1 else None,
                      "windows_detected": int(win_u1["detected"].sum()), "windows": len(win_u1)}},
              open(OUT / "fdia_v1_bdd_test_summary.json", "w"), indent=2)
    print(f"\n[U1] windows {len(win_u1)}, detected {int(win_u1['detected'].sum())}, "
          f"clean false-alarm {fa_u1*100:.2f}%, median delay "
          f"{np.median(delays_u1) if delays_u1 else float('nan')}")
    print(rates_u1.to_string(index=False))


if __name__ == "__main__":
    main()
