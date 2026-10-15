"""
Branch exp-sigma -- EXPERIMENT: revert the 81ff01a sigma recalibration
at model build time (observability/estimator_model.py sigma_overrides;
the registry file is not edited here).

  S0 -- current registry.
  S1 -- every sigma that differs between ae0fcfb and HEAD in
        config/sensor_registry_multiphase.json set back to its ae0fcfb
        value (the corpus noise sigmas; reports/corpus_A_repro_manifest.md).
        M042_Q's ae0fcfb value is 0.0999960592236547, the corpus value.

Per variant, with the adopted estimator (init="pseudo_pf" default):
  scorecard -- the 60 solves of validation/phase7_scorecard_pfstart.py
               (30 stratified seed-777 points and their i+1 neighbours,
               each from the power-flow start).
  baseline  -- the 90 clean points of validation/phase8_bdd_baseline_pfstart.py
               (30 stratified + 60 extra, seed 778); J99/LNR99 from these.
x_true is read only to score voltage error. Kept TSVD directions are
counted at the converged estimate (attacks/a4_kept.py kept_subspace_basis,
same A_reduced and TSVD_RCOND as the solver), not per iteration.

One process (the power-flow start calls OpenDSS).

Run: python -m validation.exp_sigma_revert
"""

import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator, TSVD_MAX_ITER
from estimation.bad_data_detection import compute_chi2_and_lnr
from attacks.a4_kept import kept_subspace_basis

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_ROWS = BASE_DIR / "validation" / "exp_sigma_revert_rows.csv"
OUT_SUMMARY = BASE_DIR / "validation" / "exp_sigma_revert_summary.csv"
OUT_OVERRIDES = BASE_DIR / "validation" / "exp_sigma_revert_S1_overrides.csv"
REGISTRY = "config/sensor_registry_multiphase.json"
CORPUS_SIGMA_REV = "ae0fcfb"
SEED = 777
N_PER_STRATUM = 10
N_EXTRA_CLEAN = 60


def registry_at(rev, path=REGISTRY):
    txt = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True, check=True,
                         cwd=BASE_DIR).stdout
    d = json.loads(txt)
    ents = d if isinstance(d, list) else d.get("entries", d.get("sensors", d))
    return {e["id"]: e for e in ents}


def s1_overrides():
    old, new = registry_at(CORPUS_SIGMA_REV), registry_at("HEAD")
    return {i: old[i]["sigma"] for i in new if new[i]["sigma"] != old[i]["sigma"]}, old, new


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
    overrides, old, new = s1_overrides()
    pd.DataFrame([{"id": i, "quantity": new[i].get("quantity"), "sigma_ae0fcfb": overrides[i],
                   "sigma_head": new[i]["sigma"], "ratio_head_over_ae0fcfb": new[i]["sigma"] / overrides[i]}
                  for i in overrides]).to_csv(OUT_OVERRIDES, index=False)
    print(f"S1 overrides: {len(overrides)} ids; M042_Q -> {overrides.get('M042_Q')}", flush=True)

    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols)
    x_v_all = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)["x_v"]  # scoring only
    p_cols = [c for c in z_cols if c.endswith("_P")]
    strata = stratified_picks(df.iloc[:1440], p_cols)
    sweep30 = sorted(set(strata["low"] + strata["mid"] + strata["high"]))
    score_idx = [j for idxs in strata.values() for i in idxs for j in (i, i + 1)]
    rng2 = np.random.default_rng(SEED + 1)
    remaining = [i for i in range(len(df)) if i not in sweep30]
    extra = sorted(rng2.choice(remaining, size=N_EXTRA_CLEAN, replace=False).tolist())
    base_idx = sweep30 + extra

    rows, summary = [], []
    for variant, ov in [("S0", None), ("S1", overrides)]:
        model = build_estimator_model("multiphase", sigma_overrides=ov)
        est = StateEstimator(model)
        state = model.state
        dof = est.dof
        v_base = np.array([state.v_base[n] for n in state.all_node_names])
        lo95, hi95 = sstats.chi2.ppf(0.025, dof), sstats.chi2.ppf(0.975, dof)
        for set_name, idxs in [("scorecard60", score_idx), ("baseline90", base_idx)]:
            t0 = time.time()
            errs, dominant = [], Counter()
            sub_rows = []
            for i in idxs:
                row = df.iloc[i]
                ts = str(row["real_timestamp"])
                z = {c: float(row[c]) for c in z_cols}
                res = est.estimate(z, ts)
                st = compute_chi2_and_lnr(est, res, z, ts)
                e = np.abs(res.x_v - x_v_all[i]) / v_base
                errs.append(e)
                nr = np.abs(np.asarray(st["normalized_residuals"], dtype=float))
                kept = np.isfinite(nr) & ~(np.asarray(st["redundancy"]) < 1e-3)
                dom = st["ids"][int(np.argmax(np.where(kept, nr, -1)))]
                dominant[dom] += 1
                sub_rows.append({"variant": variant, "set": set_name, "i": i, "converged": res.converged,
                                 "iterations": res.iterations, "J": st["J"], "lnr": st["lnr"],
                                 "constraint_residual_inf": res.constraint_residual_inf,
                                 "kept_final": kept_subspace_basis(est, res, ts)["dim_K"],
                                 "median_v_err_pu": float(np.median(e)), "max_v_err_pu": float(e.max()),
                                 "lnr_channel": dom})
            sr = pd.DataFrame(sub_rows)
            rows += sub_rows
            pooled = np.concatenate(errs)
            node_max = np.max(np.vstack(errs), axis=0)
            summary.append({
                "variant": variant, "set": set_name, "n": len(sr), "dof": dof,
                "converged": int(sr["converged"].sum()), "mean_iterations": sr["iterations"].mean(),
                "n_at_iter_cap": int((sr["iterations"] >= TSVD_MAX_ITER).sum()),
                "v_err_median": float(np.median(pooled)), "v_err_p95": float(np.percentile(pooled, 95)),
                "v_err_max": float(pooled.max()), "J_mean": sr["J"].mean(), "J_over_dof": sr["J"].mean() / dof,
                "n_in_chi2_95": int(((sr["J"] >= lo95) & (sr["J"] <= hi95)).sum()),
                "feasibility_max": sr["constraint_residual_inf"].max(),
                "kept_min": sr["kept_final"].min(), "kept_median": sr["kept_final"].median(),
                "kept_max": sr["kept_final"].max(),
                "worst3_nodes": "; ".join(f"{state.all_node_names[k]}={node_max[k]:.4f}"
                                          for k in np.argsort(-node_max)[:3]),
                "J99": sr["J"].quantile(0.99), "LNR99": sr["lnr"].quantile(0.99),
                "top5_lnr_channels": "; ".join(f"{c}={n}" for c, n in dominant.most_common(5)),
                "runtime_s": time.time() - t0,
            })
            print({k: summary[-1][k] for k in ("variant", "set", "converged", "mean_iterations", "v_err_median",
                                               "v_err_max", "J_over_dof", "n_in_chi2_95", "runtime_s")}, flush=True)

    pd.DataFrame(rows).to_csv(OUT_ROWS, index=False)
    out = pd.DataFrame(summary)
    out.to_csv(OUT_SUMMARY, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(out.T.to_string())


if __name__ == "__main__":
    main()
