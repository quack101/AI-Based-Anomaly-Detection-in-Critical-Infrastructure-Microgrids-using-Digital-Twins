"""
Branch fix-h-yload, A4 -- settings on corpus A's 60 scorecard and 90
baseline points (indices as validation/exp_p1_corpus.py):

  U0  = legacy h, S0 sigmas, pseudo_pf start (today)
  U1  = network_y, S0, pseudo_pf
  U2  = network_y, S1 + structural overrides, pseudo_pf
  U3  = network_y, S1 + structural overrides, flat start
  PF0 = the pseudo_pf start state alone (no WLS iteration); J/LNR of that
        state are evaluated with U2's model and sigmas. Scoring only.

S1 = config/sigma_overrides_s1.json; structural = config/
sigma_overrides_structural.json (validation/fix_h_check.py). x_true is
used only to score voltage error and the pseudo-row e. One process.

Run: python -m validation.fix_h_settings
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
from estimation.state_estimator import StateEstimator, TSVD_MAX_ITER, EstimationResult
from estimation.bad_data_detection import compute_chi2_and_lnr
from attacks.a4_kept import kept_subspace_basis
from validation.exp_p1_corpus import load_corpus, point_sets, CORPORA, S1_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "validation"
STRUCT_PATH = BASE_DIR / "config" / "sigma_overrides_structural.json"


def settings():
    with open(S1_PATH) as f:
        s1 = json.load(f)["sigma"]
    with open(STRUCT_PATH) as f:
        st = json.load(f)["sigma"]
    s1s = dict(s1, **st)
    return [("U0", "legacy", None, "pseudo_pf"), ("U1", "network_y", None, "pseudo_pf"),
            ("U2", "network_y", s1s, "pseudo_pf"), ("U3", "network_y", s1s, "flat"),
            ("PF0", "network_y", s1s, "pf_only")]


def run(name, h_model, ov, init, df, z_cols, x_v, x_d, score60, base90):
    model = build_estimator_model("multiphase", sigma_overrides=ov, h_model=h_model)
    est = StateEstimator(model)
    state, dof = model.state, est.dof
    v_base = np.array([state.v_base[n] for n in state.all_node_names])
    flat_v, flat_d = list(state.x0_v), list(state.x0_delta)
    lo95, hi95 = sstats.chi2.ppf(0.025, dof), sstats.chi2.ppf(0.975, dof)
    rows, summ, pseudo = [], [], []
    for set_name, idxs in [("scorecard60", score60), ("baseline90", base90)]:
        t0 = time.time()
        errs, dom, sub = [], Counter(), []
        for i in idxs:
            row = df.iloc[i]
            ts = str(row["real_timestamp"])
            z = {c: float(row[c]) for c in z_cols}
            if init == "pf_only":
                xv0, xd0, conv = model.pf_start(z, int(ts[11:13]))
                res = EstimationResult(x_v=xv0, x_delta=xd0, converged=conv, iterations=0, max_dx_v_pu=0.0,
                                       max_dx_delta_rad=0.0, J=float("nan"), dof=dof, m_weighted=est.m_weighted,
                                       p_constraints=est.p_constraints, constraint_residual_inf=float("nan"))
            elif init == "flat":
                state.x0_v, state.x0_delta = list(flat_v), list(flat_d)
                res = est.estimate(z, ts, init="flat")
            else:
                res = est.estimate(z, ts, init="pseudo_pf")
            stt = compute_chi2_and_lnr(est, res, z, ts)
            e = np.abs(res.x_v - x_v[i]) / v_base
            errs.append(e)
            nr = np.abs(np.asarray(stt["normalized_residuals"], dtype=float))
            keep = np.isfinite(nr) & ~(np.asarray(stt["redundancy"]) < 1e-3)
            dom[stt["ids"][int(np.argmax(np.where(keep, nr, -1)))]] += 1
            if set_name == "baseline90" and init != "pf_only":
                ps = [w for w in est._weighted_entries_for_hour(int(ts[11:13])) if w["id"].startswith("PSEUDO_")]
                _, h0 = J.build_jacobian(ps, model.Y, model.node_index, state,
                                         at_v=complex_v_from_state(state, x_v[i], x_d[i]))
                pseudo += [{"id": w["id"], "e": (w["z_value"] - h) / w["sigma"]} for w, h in zip(ps, h0)]
            sub.append({"setting": name, "set": set_name, "i": i, "converged": res.converged,
                        "iterations": res.iterations, "J": stt["J"], "lnr": stt["lnr"],
                        "constraint_residual_inf": res.constraint_residual_inf,
                        "kept": kept_subspace_basis(est, res, ts)["dim_K"] if init != "pf_only" else np.nan,
                        "median_v_err_pu": float(np.median(e)), "max_v_err_pu": float(e.max())})
        sr = pd.DataFrame(sub)
        rows += sub
        pooled = np.concatenate(errs)
        node_max = np.max(np.vstack(errs), axis=0)
        summ.append({
            "setting": name, "set": set_name, "n": len(sr), "dof": dof,
            "converged": int(sr["converged"].sum()), "mean_iterations": sr["iterations"].mean(),
            "n_at_iter_cap": int((sr["iterations"] >= TSVD_MAX_ITER).sum()),
            "v_err_median": float(np.median(pooled)), "v_err_p95": float(np.percentile(pooled, 95)),
            "v_err_max": float(pooled.max()), "J_mean": sr["J"].mean(), "J_over_dof": sr["J"].mean() / dof,
            "n_in_chi2_95": int(((sr["J"] >= lo95) & (sr["J"] <= hi95)).sum()),
            "feasibility_max": sr["constraint_residual_inf"].max(),
            "kept_min": sr["kept"].min(), "kept_median": sr["kept"].median(), "kept_max": sr["kept"].max(),
            "worst3_nodes": "; ".join(f"{state.all_node_names[k]}={node_max[k]:.4f}"
                                      for k in np.argsort(-node_max)[:3]),
            "J99": sr["J"].quantile(0.99), "LNR99": sr["lnr"].quantile(0.99),
            "top5_lnr_channels": "; ".join(f"{c}={n}" for c, n in dom.most_common(5)),
            "runtime_s": time.time() - t0})
        print({k: summ[-1][k] for k in ("setting", "set", "converged", "mean_iterations", "v_err_median",
                                        "v_err_max", "J_over_dof", "n_in_chi2_95", "runtime_s")}, flush=True)
    if pseudo:
        ps = pd.DataFrame(pseudo).groupby("id")["e"].mean()
        for s_ in summ:
            s_["pseudo_median_abs_mean_e"] = float(ps.abs().median())
    return rows, summ


def main():
    df, z_cols, x_v, x_d = load_corpus(CORPORA["P0"])
    score60, base90 = point_sets(df, z_cols)
    all_rows, all_summ = [], []
    for name, hm, ov, init in settings():
        r, s = run(name, hm, ov, init, df, z_cols, x_v, x_d, score60, base90)
        all_rows += r
        all_summ += s
    pd.DataFrame(all_rows).to_csv(OUT / "fix_h_settings_rows.csv", index=False)
    summary = pd.DataFrame(all_summ)
    summary.to_csv(OUT / "fix_h_settings_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 110)
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
