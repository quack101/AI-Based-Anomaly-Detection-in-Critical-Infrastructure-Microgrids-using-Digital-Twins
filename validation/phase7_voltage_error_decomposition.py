"""
RES.md D10 Phase 7 VOLTAGE-ERROR DECOMPOSITION -- per-node error,
same 60-point stratified sweep (seed=777) used for the J decomposition.

For each of the 60 points, records |x_v_hat - x_v_true| / v_base at
EVERY one of the 278 phase-nodes (not just the max), then reports:
  - top 15 nodes by average error across all 60 points, and % of
    total (summed) error they account for
  - concentrated vs diffuse
  - category breakdown: metered-load nodes (registry P/Q entries),
    unmetered-load nodes (model.bus_phase_by_load), zero-injection
    nodes (model.zi_entries), and whatever is left over (uncategorized)

Uses the real StateEstimator, unmodified. Read-only diagnostic.

Run: python -m validation.phase7_voltage_error_decomposition
"""

from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase7_voltage_error_decomposition_results.csv"

SEED = 777
N_PER_STRATUM = 10
PHASE_NUM = {"A": 1, "B": 2, "C": 3}


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
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_all = npz["x_v"][:1440]

    p_cols = [c for c in df.columns if c.endswith("_P")]
    strata = stratified_picks(df, p_cols)
    print("stratum picks:", strata, flush=True)

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    v_base = np.array([state.v_base[n] for n in state.all_node_names])
    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)

    # node categories
    registry = est.model.meas_entries
    metered_nodes = set()
    for e in registry:
        if e["quantity"] in ("P_injection", "Q_injection"):
            metered_nodes.add(f"{e['node']}.{PHASE_NUM[e['phase']]}")

    unmetered_nodes = set()
    for load_name, (bus, phase) in model.bus_phase_by_load.items():
        unmetered_nodes.add(f"{bus}.{PHASE_NUM[phase]}")

    zi_nodes = set()
    for e in model.zi_entries:
        zi_nodes.add(f"{e['node']}.{PHASE_NUM[e['phase']]}")

    all_nodes = set(state.all_node_names)
    categorized = metered_nodes | unmetered_nodes | zi_nodes
    uncategorized_nodes = all_nodes - categorized
    overlap_metered_unmetered = metered_nodes & unmetered_nodes
    overlap_metered_zi = metered_nodes & zi_nodes
    overlap_unmetered_zi = unmetered_nodes & zi_nodes

    print(f"node categories: metered={len(metered_nodes)} unmetered={len(unmetered_nodes)} "
          f"zero_injection={len(zi_nodes)} uncategorized={len(uncategorized_nodes)} "
          f"(total nodes={len(all_nodes)})", flush=True)
    print(f"overlaps: metered&unmetered={len(overlap_metered_unmetered)} "
          f"metered&zi={len(overlap_metered_zi)} unmetered&zi={len(overlap_unmetered_zi)}", flush=True)

    all_rows = []
    for stratum_name, idxs in strata.items():
        for i in idxs:
            state.x0_v = list(flat_v)
            state.x0_delta = list(flat_delta)
            row_a = df.iloc[i]
            real_ts_a = row_a["real_timestamp"]
            z_a = {c: float(row_a[c]) for c in z_cols}
            result_a = est.estimate(z_a, real_ts_a)
            err_a = np.abs(result_a.x_v - x_v_all[i]) / v_base
            for node, e in zip(state.all_node_names, err_a):
                all_rows.append({"point": f"A_flat_{i}", "node": node, "err_pu": e})
            print(f"[A_flat] i={i} max_err={err_a.max():.5f}", flush=True)

            j = i + 1
            if j >= len(df):
                continue
            row_b = df.iloc[j]
            real_ts_b = row_b["real_timestamp"]
            z_b = {c: float(row_b[c]) for c in z_cols}
            state.x0_v = list(result_a.x_v)
            state.x0_delta = list(result_a.x_delta)
            result_b = est.estimate(z_b, real_ts_b)
            err_b = np.abs(result_b.x_v - x_v_all[j]) / v_base
            for node, e in zip(state.all_node_names, err_b):
                all_rows.append({"point": f"B_warm_{j}", "node": node, "err_pu": e})
            print(f"[B_warm] i={j} max_err={err_b.max():.5f}", flush=True)

    long_df = pd.DataFrame(all_rows)
    long_df.to_csv(OUT_PATH, index=False)

    n_points = long_df["point"].nunique()
    n_nodes = long_df["node"].nunique()
    per_node_mean = long_df.groupby("node")["err_pu"].mean().sort_values(ascending=False)
    total_err_mean = per_node_mean.sum()

    print()
    print(f"n_points={n_points} n_nodes={n_nodes} sum_of_mean_node_err={total_err_mean:.4f}")
    print()
    print("=== TOP 15 NODES BY AVERAGE VOLTAGE ERROR (pu) ===")
    top15 = per_node_mean.head(15)
    for node, val in top15.items():
        print(f"  {node:<10s} mean_err_pu={val:.5f}  pct_of_total_summed_err={100*val/total_err_mean:.3f}%")
    top15_sum = top15.sum()
    print()
    print(f"TOP 15 NODES ACCOUNT FOR {100*top15_sum/total_err_mean:.3f}% OF TOTAL SUMMED ERROR")
    uniform_share = 100.0 / n_nodes
    print(f"if diffuse/uniform across {n_nodes} nodes, each would be ~{uniform_share:.3f}% on average "
          f"(top 15 uniform would be ~{15*uniform_share:.3f}%)")

    def cat_of(node):
        in_m, in_u, in_z = node in metered_nodes, node in unmetered_nodes, node in zi_nodes
        if in_m and not in_u and not in_z:
            return "metered"
        if in_u and not in_m and not in_z:
            return "unmetered"
        if in_z and not in_m and not in_u:
            return "zero_injection"
        if not in_m and not in_u and not in_z:
            return "uncategorized"
        return "overlap"

    long_df["category"] = long_df["node"].map(cat_of)
    print()
    print("=== ERROR BY NODE CATEGORY (mean err_pu over all point x node rows in that category) ===")
    cat_stats = long_df.groupby("category")["err_pu"].agg(["mean", "median", "max", "count"])
    print(cat_stats)
    print()
    n_nodes_per_cat = long_df.groupby("category")["node"].nunique()
    print("n_unique_nodes_per_category:")
    print(n_nodes_per_cat)

    print()
    print("saved per-(point,node) long table to:", OUT_PATH)


if __name__ == "__main__":
    main()
