"""
RES.md D10 Phase 8 -- re-run of the Phase 7 scorecard (same 60 points,
same seed, same TSVD default, same sigmas) AFTER the HEAD_P
measurement-model fix (observability/jacobian.py build_head_p_row,
branch fix-head-p-model). Does not overwrite the pre-fix results
(validation/phase7_scorecard_tsvd_default_results.csv) -- writes to
its own _headpfix-suffixed output. Identical logic to
phase7_scorecard_tsvd_default.py otherwise.

Run: python -m validation.phase7_scorecard_headpfix
"""

import numpy as np
import pandas as pd
from scipy import stats as sstats

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator

BASE_DIR_CORPUS = __import__("pathlib").Path(__file__).resolve().parent.parent / "simulation" / "phase6_corpus"
OUT_PATH = __import__("pathlib").Path(__file__).resolve().parent.parent / "validation" / "phase7_scorecard_headpfix_results.csv"
SEED = 777
N_PER_STRATUM = 10
DOF = 57


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
    telem_cols = pd.read_csv(BASE_DIR_CORPUS / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(BASE_DIR_CORPUS / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    npz = np.load(BASE_DIR_CORPUS / "test_x_true.npz", allow_pickle=True)
    x_v_all, x_delta_all = npz["x_v"][:1440], npz["x_delta"][:1440]

    p_cols = [c for c in df.columns if c.endswith("_P")]
    strata = stratified_picks(df, p_cols)

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    v_base = np.array([state.v_base[n] for n in state.all_node_names])
    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)

    records = []
    all_node_errs = []

    for stratum_name, idxs in strata.items():
        for i in idxs:
            state.x0_v = list(flat_v)
            state.x0_delta = list(flat_delta)
            row_a = df.iloc[i]
            z_a = {c: float(row_a[c]) for c in z_cols}
            result_a = est.estimate(z_a, row_a["real_timestamp"])
            err_v_a = np.abs(result_a.x_v - x_v_all[i]) / v_base
            all_node_errs.append(err_v_a)
            records.append({"condition": "A_flat", "i": i, "converged": result_a.converged,
                             "iterations": result_a.iterations,
                             "max_v_err_pu": err_v_a.max(), "J": result_a.J,
                             "constraint_residual_inf": result_a.constraint_residual_inf})

            j = i + 1
            if j >= len(df):
                continue
            row_b = df.iloc[j]
            z_b = {c: float(row_b[c]) for c in z_cols}
            state.x0_v = list(result_a.x_v)
            state.x0_delta = list(result_a.x_delta)
            result_b = est.estimate(z_b, row_b["real_timestamp"])
            err_v_b = np.abs(result_b.x_v - x_v_all[j]) / v_base
            all_node_errs.append(err_v_b)
            records.append({"condition": "B_warm", "i": j, "converged": result_b.converged,
                             "iterations": result_b.iterations,
                             "max_v_err_pu": err_v_b.max(), "J": result_b.J,
                             "constraint_residual_inf": result_b.constraint_residual_inf})

    out = pd.DataFrame(records)
    out.to_csv(OUT_PATH, index=False)
    pooled = np.concatenate(all_node_errs)

    lo95, hi95 = sstats.chi2.ppf(0.025, DOF), sstats.chi2.ppf(0.975, DOF)
    n_in_ci = ((out["J"] >= lo95) & (out["J"] <= hi95)).sum()

    print(f"n_points={len(out)}")
    print(f"feasibility (constraint_residual_inf): mean={out['constraint_residual_inf'].mean():.4f} "
          f"max={out['constraint_residual_inf'].max():.4f}")
    print(f"accuracy (per-node error, pooled n={len(pooled)}): mean={pooled.mean():.5f} "
          f"median={np.median(pooled):.5f} p95={np.percentile(pooled,95):.5f} max={pooled.max():.5f}")
    print(f"J: mean={out['J'].mean():.4e}  J/dof={out['J'].mean()/DOF:.2f}x")
    print(f"in chi2({DOF}) 95% CI [{lo95:.1f},{hi95:.1f}]: {n_in_ci}/{len(out)}")
    print(f"converged fraction: {out['converged'].mean():.3f} ({int(out['converged'].sum())}/{len(out)})")
    print(f"mean iterations: {out['iterations'].mean():.2f}")

    all_node_names = state.all_node_names
    worst5 = []
    for arr, idx in zip(all_node_errs, [r["i"] for r in records]):
        order = np.argsort(-arr)[:1]
        for o in order:
            worst5.append((idx, all_node_names[o], arr[o]))
    worst5.sort(key=lambda t: -t[2])
    print("worst 5 (point, node, err_pu):", worst5[:5])
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
