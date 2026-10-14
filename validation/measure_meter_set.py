"""
Measure one meter set (registry + corpus) with the frozen estimator,
using the SAME protocols as the set-A reference scripts, so A and B are
compared like for like. Nothing here changes the estimator, solver,
Jacobian or bad-data detection code; it only calls them.

Protocols reproduced exactly (same seeds, same picks, same formulas):
  1. validation/phase7_scorecard_tsvd_default.py -- 60-point stratified
     sweep of the first 1440 test rows (seed 777, 10 per stratum, each
     point A_flat then B_warm on the next row): per-node |V| error pooled
     over all phase-nodes (median / p95 / max, pu), J mean and J/dof,
     convergence, mean iterations.
  2. validation/phase8_bdd_baseline.py -- n=90 clean points (the 30
     stratified rows + 60 extra test rows, seed 778), A_flat:
     99th-percentile J and largest normalised residual (LNR); null-space
     dimension of the unattackable rows at the first / middle / last
     stratified row.
Added:
  3. Weakest-observed nodes: per-phase-node median |V| error (pu) over
     the 60 scorecard points (and p95), largest first. EMPIRICAL on
     purpose: the WLS covariance pinv(H_reduced^T W H_reduced) that
     bad_data_detection uses for Omega truncates the ill-conditioned
     directions to ZERO variance, so it ranks the worst-observed nodes
     as the most precise (measured: ~1e-7 pu "std" against ~0.02 pu
     real error on set A). It cannot rank nodes by precision.

The reference scripts write fixed set-A result CSVs; this one writes
only to reports/measurements/<label>.json (+ a per-point CSV next to it).

Run: python -m validation.measure_meter_set --label A_exact \
        --registry config/sensor_registry_multiphase.json \
        --corpus simulation/phase6_corpus_Arebuilt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr, unattackable_nullspace_dimension

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "reports" / "measurements"
SEED = 777
N_PER_STRATUM = 10
N_EXTRA_CLEAN = 60
SCORECARD_ROWS = 1440
REF_DOF = 57  # the reference scripts' fixed DOF, reported alongside the measured one


def stratified_picks(df, p_cols):
    """Identical to both reference scripts."""
    total_p = df[p_cols].sum(axis=1)
    q1, q2 = total_p.quantile([1 / 3, 2 / 3])
    low = df.index[total_p <= q1]
    mid = df.index[(total_p > q1) & (total_p <= q2)]
    high = df.index[total_p > q2]
    rng = np.random.default_rng(SEED)

    def pick(idx, k):
        return sorted(rng.choice(idx, size=k, replace=False).tolist())

    return {"low": pick(low, N_PER_STRATUM), "mid": pick(mid, N_PER_STRATUM), "high": pick(high, N_PER_STRATUM)}


def main(label, registry, corpus, meter_set=None):
    corpus = Path(corpus)
    t_start = time.time()
    telem_cols = pd.read_csv(corpus / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(corpus / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols)
    npz = np.load(corpus / "test_x_true.npz", allow_pickle=True)
    x_v_all = npz["x_v"]

    model = build_estimator_model("multiphase", registry_path=registry, corpus_dir=corpus)
    state = model.state
    est = StateEstimator(model)
    reg_ids = [e["id"] for e in model.meas_entries]
    if set(reg_ids) != set(z_cols):
        raise ValueError(f"registry enabled ids and corpus columns differ: "
                         f"only in registry {sorted(set(reg_ids)-set(z_cols))[:5]}, "
                         f"only in corpus {sorted(set(z_cols)-set(reg_ids))[:5]}")
    v_base = np.array([state.v_base[n] for n in state.all_node_names])
    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)

    def reset():
        state.x0_v, state.x0_delta = list(flat_v), list(flat_delta)

    def z_of(i):
        row = df.iloc[i]
        return {c: float(row[c]) for c in z_cols}, str(row["real_timestamp"])

    # ---- 1. scorecard (phase7_scorecard_tsvd_default.py protocol)
    sc_df = df.iloc[:SCORECARD_ROWS]
    p_cols = [c for c in sc_df.columns if c.endswith("_P")]
    strata = stratified_picks(sc_df, p_cols)
    records, errs, dofs = [], [], []
    for stratum, idxs in strata.items():
        for i in idxs:
            reset()
            z_a, ts_a = z_of(i)
            t0 = time.time()
            r_a = est.estimate(z_a, ts_a)
            dt = time.time() - t0
            e_a = np.abs(r_a.x_v - x_v_all[i]) / v_base
            errs.append(e_a)
            records.append({"condition": "A_flat", "stratum": stratum, "i": i, "converged": r_a.converged,
                            "iterations": r_a.iterations, "time_s": dt, "max_v_err_pu": e_a.max(), "J": r_a.J})
            if r_a.converged:
                st = compute_chi2_and_lnr(est, r_a, z_a, ts_a)
                dofs.append(st["m_weighted"] - st["n_w"])
            j = i + 1
            if j >= len(sc_df):
                continue
            z_b, ts_b = z_of(j)
            state.x0_v, state.x0_delta = list(r_a.x_v), list(r_a.x_delta)
            t0 = time.time()
            r_b = est.estimate(z_b, ts_b)
            dt = time.time() - t0
            e_b = np.abs(r_b.x_v - x_v_all[j]) / v_base
            errs.append(e_b)
            records.append({"condition": "B_warm", "stratum": stratum, "i": j, "converged": r_b.converged,
                            "iterations": r_b.iterations, "time_s": dt, "max_v_err_pu": e_b.max(), "J": r_b.J})
    sc = pd.DataFrame(records)
    pooled = np.concatenate(errs)

    # ---- 3. weakest-observed nodes (empirical, see module docstring)
    node_errs = np.array(errs)  # (points, phase-nodes)
    med = np.median(node_errs, axis=0)
    p95 = np.percentile(node_errs, 95, axis=0)
    order = np.argsort(-med)
    weakest = [{"node": state.all_node_names[k], "median_err_pu": float(med[k]), "p95_err_pu": float(p95[k])}
               for k in order[:10]]

    # ---- 2. BDD thresholds + null space (phase8_bdd_baseline.py protocol)
    sweep_idxs = sorted(set(strata["low"] + strata["mid"] + strata["high"]))
    rng2 = np.random.default_rng(SEED + 1)
    remaining = [i for i in range(len(df)) if i not in sweep_idxs]
    extra_idxs = sorted(rng2.choice(remaining, size=min(N_EXTRA_CLEAN, len(remaining)), replace=False).tolist())
    bdd = []
    for i in sweep_idxs + extra_idxs:
        reset()
        z, ts = z_of(i)
        r = est.estimate(z, ts)
        st = compute_chi2_and_lnr(est, r, z, ts)
        bdd.append({"i": i, "converged": r.converged, "J": st["J"], "lnr": st["lnr"]})
    bdd = pd.DataFrame(bdd)
    nulls = []
    for i in [sweep_idxs[0], sweep_idxs[len(sweep_idxs) // 2], sweep_idxs[-1]]:
        reset()
        z, ts = z_of(i)
        r = est.estimate(z, ts)
        ns = unattackable_nullspace_dimension(est, r, ts)
        nulls.append({"i": i, **{k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                                  for k, v in ns.items() if isinstance(v, (int, float, np.integer, np.floating))}})

    summary = {
        "label": label, "registry": str(registry), "corpus": str(corpus),
        "n_meter_channels": len(reg_ids),
        "scorecard": {
            "source": "validation/measure_meter_set.py (protocol of validation/phase7_scorecard_tsvd_default.py)",
            "n_points": int(len(sc)),
            "v_err_pu_median": float(np.median(pooled)), "v_err_pu_p95": float(np.percentile(pooled, 95)),
            "v_err_pu_max": float(pooled.max()), "v_err_pu_mean": float(pooled.mean()),
            "converged": int(sc["converged"].sum()), "mean_iterations": float(sc["iterations"].mean()),
            "J_mean": float(sc["J"].mean()), "J_over_ref_dof57": float(sc["J"].mean() / REF_DOF),
            "dof_measured_median": float(np.median(dofs)) if dofs else None,
            "mean_time_s": float(sc["time_s"].mean()),
        },
        "weakest_nodes": {
            "source": "validation/measure_meter_set.py (per-node median |V| error over the 60 scorecard points)",
            "worst_node_median_err_pu": float(med.max()),
            "median_over_nodes_of_median_err_pu": float(np.median(med)),
            "top10": weakest,
        },
        "bdd_clean": {
            "source": "validation/measure_meter_set.py (protocol of validation/phase8_bdd_baseline.py)",
            "n_points": int(len(bdd)), "converged": int(bdd["converged"].sum()),
            "J99": float(bdd["J"].quantile(0.99)), "LNR99": float(bdd["lnr"].quantile(0.99)),
        },
        "nullspace": {"source": "estimation.bad_data_detection.unattackable_nullspace_dimension", "points": nulls},
        "runtime_s": time.time() - t_start,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{label}.json").write_text(json.dumps(summary, indent=2))
    sc.to_csv(OUT_DIR / f"{label}_scorecard_points.csv", index=False)
    print(json.dumps({k: summary[k] for k in ("label", "scorecard", "bdd_clean", "nullspace")}, indent=2))
    print("weakest:", weakest[:5])
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--registry", required=True)
    p.add_argument("--corpus", required=True)
    a = p.parse_args()
    # resolve now: compiling the feeder changes the working directory
    main(a.label, Path(a.registry).resolve(), Path(a.corpus).resolve())
