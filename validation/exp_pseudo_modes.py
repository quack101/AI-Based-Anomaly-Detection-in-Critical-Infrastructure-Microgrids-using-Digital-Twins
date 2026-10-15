"""
Branch exp-pseudo, A3 -- pseudo values as today vs the P/Q each
unmetered load actually draws in the pseudo_pf start power flow.

  T0 = S0 + pseudo_mode "static"   (today)
  T1 = S0 + pseudo_mode "pf_drawn"
  T2 = S1 + pseudo_mode "pf_drawn" (S1 = config/sigma_overrides_s1.json)

Corpus A (P0), the 60 scorecard and 90 baseline points (same indices as
validation/exp_p1_corpus.py). Estimator otherwise unchanged (init
"pseudo_pf", TSVD, HEAD_P handling). pf_drawn reads only telemetry z,
the static pseudo tables and the compiled model (observability/
pf_start.py); x_true is used only to score voltage error and
e = (z_pseudo - h(x_true)) / sigma, where z_pseudo is the value the
setting actually used and h is the estimator's own measurement function.

One process.

Run: python -m validation.exp_pseudo_modes
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import StateEstimator, TSVD_MAX_ITER
from estimation.bad_data_detection import compute_chi2_and_lnr
from attacks.a4_kept import kept_subspace_basis
from validation.exp_p1_corpus import load_corpus, point_sets, CORPORA, S1_PATH

OUT = Path(__file__).resolve().parent


def run(name, ov, mode, df, z_cols, x_v, x_d, score60, base90):
    model = build_estimator_model("multiphase", sigma_overrides=ov)
    est = StateEstimator(model, pseudo_mode=mode)
    state, dof = model.state, est.dof
    v_base = np.array([state.v_base[n] for n in state.all_node_names])
    lo95, hi95 = sstats.chi2.ppf(0.025, dof), sstats.chi2.ppf(0.975, dof)
    rows, summ, pseudo = [], [], []
    for set_name, idxs in [("scorecard60", score60), ("baseline90", base90)]:
        t0 = time.time()
        errs, dom, sub = [], Counter(), []
        for i in idxs:
            row = df.iloc[i]
            ts = str(row["real_timestamp"])
            z = {c: float(row[c]) for c in z_cols}
            res = est.estimate(z, ts)
            st = compute_chi2_and_lnr(est, res, z, ts)
            e = np.abs(res.x_v - x_v[i]) / v_base
            errs.append(e)
            nr = np.abs(np.asarray(st["normalized_residuals"], dtype=float))
            keep = np.isfinite(nr) & ~(np.asarray(st["redundancy"]) < 1e-3)
            dom[st["ids"][int(np.argmax(np.where(keep, nr, -1)))]] += 1
            if set_name == "baseline90":
                ps = [w for w in est._weighted_entries_for_hour(int(ts[11:13]), z) if w["id"].startswith("PSEUDO_")]
                _, h0 = J.build_jacobian(ps, model.Y, model.node_index, state,
                                         at_v=complex_v_from_state(state, x_v[i], x_d[i]))
                pseudo += [{"setting": name, "i": i, "id": w["id"], "e": (w["z_value"] - h) / w["sigma"]}
                           for w, h in zip(ps, h0)]
            sub.append({"setting": name, "set": set_name, "i": i, "converged": res.converged,
                        "iterations": res.iterations, "J": st["J"], "lnr": st["lnr"],
                        "constraint_residual_inf": res.constraint_residual_inf,
                        "kept": kept_subspace_basis(est, res, ts)["dim_K"],
                        "median_v_err_pu": float(np.median(e)), "max_v_err_pu": float(e.max())})
        sr = pd.DataFrame(sub)
        rows += sub
        pooled = np.concatenate(errs)
        summ.append({
            "setting": name, "set": set_name, "n": len(sr), "dof": dof,
            "converged": int(sr["converged"].sum()), "mean_iterations": sr["iterations"].mean(),
            "n_at_iter_cap": int((sr["iterations"] >= TSVD_MAX_ITER).sum()),
            "v_err_median": float(np.median(pooled)), "v_err_p95": float(np.percentile(pooled, 95)),
            "v_err_max": float(pooled.max()), "J_mean": sr["J"].mean(), "J_over_dof": sr["J"].mean() / dof,
            "n_in_chi2_95": int(((sr["J"] >= lo95) & (sr["J"] <= hi95)).sum()),
            "feasibility_max": sr["constraint_residual_inf"].max(),
            "kept_min": sr["kept"].min(), "kept_median": sr["kept"].median(), "kept_max": sr["kept"].max(),
            "J99": sr["J"].quantile(0.99), "LNR99": sr["lnr"].quantile(0.99),
            "top5_lnr_channels": "; ".join(f"{c}={n}" for c, n in dom.most_common(5)),
            "runtime_s": time.time() - t0})
        print({k: summ[-1][k] for k in ("setting", "set", "converged", "mean_iterations", "v_err_median",
                                        "v_err_max", "J_over_dof", "runtime_s")}, flush=True)
    return rows, summ, pseudo


def main():
    with open(S1_PATH) as f:
        s1 = json.load(f)["sigma"]
    df, z_cols, x_v, x_d = load_corpus(CORPORA["P0"])
    score60, base90 = point_sets(df, z_cols)
    all_rows, all_summ, all_pseudo = [], [], []
    for name, ov, mode in [("T0", None, "static"), ("T1", None, "pf_drawn"), ("T2", s1, "pf_drawn")]:
        r, s, p = run(name, ov, mode, df, z_cols, x_v, x_d, score60, base90)
        all_rows += r
        all_summ += s
        all_pseudo += p
    pd.DataFrame(all_rows).to_csv(OUT / "exp_pseudo_modes_rows.csv", index=False)
    ps = pd.DataFrame(all_pseudo)
    per_row = ps.groupby(["setting", "id"])["e"].agg(
        mean="mean", rms=lambda x: float(np.sqrt(np.mean(np.square(x))))).reset_index()
    per_row.to_csv(OUT / "exp_pseudo_modes_pseudo_rows.csv", index=False)
    summary = pd.DataFrame(all_summ)
    for name in ("T0", "T1", "T2"):
        pr = per_row[per_row["setting"] == name]
        summary.loc[summary["setting"] == name, "pseudo_median_abs_mean_e"] = float(pr["mean"].abs().median())
        summary.loc[summary["setting"] == name, "pseudo_median_mean_e"] = float(pr["mean"].median())
        summary.loc[summary["setting"] == name, "pseudo_median_rms_e"] = float(pr["rms"].median())
    summary.to_csv(OUT / "exp_pseudo_modes_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 110)
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
