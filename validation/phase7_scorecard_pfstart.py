"""
Branch est-pf-start -- Phase 7 scorecard (same 60 solves, seed 777,
as validation/phase7_scorecard_headpexcl.py) under two start policies:
  flat      -- today's protocol: 30 stratified points from the flat
               start, each followed by its i+1 neighbour warm-started
               from the previous estimate.
  pseudo_pf -- every one of the same 60 solves started from the
               power-flow start (est.estimate(init="pseudo_pf"),
               observability/pf_start.py). The start is built from
               telemetry + pseudo tables + model only; x_true below is
               used ONLY to score the result.
One process (OpenDSS power flows stay in this process).

Run: python -m validation.phase7_scorecard_pfstart
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase7_scorecard_pfstart_results.csv"
OUT_SUMMARY = BASE_DIR / "validation" / "phase7_scorecard_pfstart_summary.csv"
SEED = 777
N_PER_STRATUM = 10


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
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    x_v_all = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)["x_v"][:1440]  # scoring only

    strata = stratified_picks(df, [c for c in df.columns if c.endswith("_P")])
    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    dof = est.dof
    v_base = np.array([state.v_base[n] for n in state.all_node_names])
    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)

    def z_at(i):
        row = df.iloc[i]
        return {c: float(row[c]) for c in z_cols}, str(row["real_timestamp"])

    records, errs, summary = [], {}, []
    for policy in ["flat", "pseudo_pf"]:
        t0 = time.time()
        errs[policy] = []
        for idxs in strata.values():
            for i in idxs:
                for j, cond in [(i, "A"), (i + 1, "B")]:
                    z, ts = z_at(j)
                    if policy == "pseudo_pf":
                        ts0 = time.time()
                        pf_v, _, pf_conv = model.pf_start(z, int(ts[11:13]))
                        pf_s = time.time() - ts0
                        start_err = float(np.max(np.abs(pf_v - x_v_all[j]) / v_base))
                        res = est.estimate(z, ts, init="pseudo_pf")
                    else:
                        pf_conv, pf_s, start_err = None, None, None
                        if cond == "A":
                            state.x0_v, state.x0_delta = list(flat_v), list(flat_delta)
                        else:
                            state.x0_v, state.x0_delta = list(prev.x_v), list(prev.x_delta)
                        res = est.estimate(z, ts, init="flat")
                    prev = res
                    e = np.abs(res.x_v - x_v_all[j]) / v_base
                    errs[policy].append(e)
                    records.append({"policy": policy, "condition": cond, "i": j, "converged": res.converged,
                                    "iterations": res.iterations, "J": res.J,
                                    "constraint_residual_inf": res.constraint_residual_inf,
                                    "median_v_err_pu": float(np.median(e)), "max_v_err_pu": float(e.max()),
                                    "pf_converged": pf_conv, "pf_solve_s": pf_s,
                                    "start_max_v_err_pu": start_err})
        runtime = time.time() - t0
        sub = pd.DataFrame([r for r in records if r["policy"] == policy])
        pooled = np.concatenate(errs[policy])
        node_max = np.max(np.vstack(errs[policy]), axis=0)
        worst3 = "; ".join(f"{state.all_node_names[k]}={node_max[k]:.4f}" for k in np.argsort(-node_max)[:3])
        summary.append({
            "policy": policy, "n": len(sub), "converged": int(sub["converged"].sum()),
            "mean_iterations": sub["iterations"].mean(),
            "v_err_median": float(np.median(pooled)), "v_err_p95": float(np.percentile(pooled, 95)),
            "v_err_max": float(pooled.max()), "J_mean": sub["J"].mean(), "J_over_dof": sub["J"].mean() / dof,
            "feasibility_max": sub["constraint_residual_inf"].max(), "worst3_nodes": worst3,
            "runtime_s": runtime,
            "pf_converged": int(sub["pf_converged"].sum()) if policy == "pseudo_pf" else None,
        })
        print(summary[-1], flush=True)

    pd.DataFrame(records).to_csv(OUT_PATH, index=False)
    pd.DataFrame(summary).to_csv(OUT_SUMMARY, index=False)
    print(f"dof={dof}")
    print("saved:", OUT_PATH, OUT_SUMMARY)


if __name__ == "__main__":
    main()
