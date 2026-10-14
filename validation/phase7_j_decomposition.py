"""
RES.md D10 Phase 7 J-DECOMPOSITION -- per-channel contribution to J.

Read-only diagnostic: reuses the EXACT same stratified sample
(seed=777, same 30 points x {A_flat, B_warm} = 60 calls) as
validation/phase7_handrolled_stratified_AB.py, via the real
estimation.state_estimator.StateEstimator (unmodified). For each of
the 60 points, after the real solve converges, recomputes the final
weighted residual W_k * (z_k - h_k(x_hat))^2 for every individual
measurement channel k (m_weighted=364 channels), not just the J total.
Does not modify estimation/state_estimator.py.

Run: python -m validation.phase7_j_decomposition
"""

from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import StateEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase7_j_decomposition_results.csv"

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


def per_channel_contributions(est, model, state, x_v, x_delta, hour, z):
    weighted_entries = est._weighted_entries_for_hour(hour)
    sigma = np.array([e["sigma"] for e in weighted_entries])
    W = 1.0 / (sigma ** 2)
    at_v = complex_v_from_state(state, x_v, x_delta)
    H, h0 = J.build_jacobian(weighted_entries, model.Y, model.node_index, state, at_v=at_v)
    dz = z - h0
    contrib = W * dz ** 2
    ids = [e["id"] for e in weighted_entries]
    return ids, contrib


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)

    p_cols = [c for c in df.columns if c.endswith("_P")]
    strata = stratified_picks(df, p_cols)
    print("stratum picks:", strata, flush=True)

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    print("m_weighted:", est.m_weighted, "p_constraints:", est.p_constraints, "dof:", est.dof, flush=True)
    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)

    all_rows = []  # one row per (point, channel)
    point_totals = []

    for stratum_name, idxs in strata.items():
        for i in idxs:
            state.x0_v = list(flat_v)
            state.x0_delta = list(flat_delta)

            row_a = df.iloc[i]
            real_ts_a = row_a["real_timestamp"]
            z_a_dict = {c: float(row_a[c]) for c in z_cols}
            result_a = est.estimate(z_a_dict, real_ts_a)

            hour_a = int(real_ts_a[11:13])
            weighted_entries_a = est._weighted_entries_for_hour(hour_a)
            z_a = np.array([e["z_value"] if "z_value" in e else z_a_dict[e["id"]] for e in weighted_entries_a])
            ids_a, contrib_a = per_channel_contributions(est, model, state, result_a.x_v, result_a.x_delta, hour_a, z_a)
            for cid, c in zip(ids_a, contrib_a):
                all_rows.append({"point": f"A_flat_{i}", "channel": cid, "contribution": c})
            point_totals.append({"point": f"A_flat_{i}", "J_reported": result_a.J, "J_recomputed": contrib_a.sum()})
            print(f"[A_flat] i={i} J_reported={result_a.J:.4e} J_recomputed_from_channels={contrib_a.sum():.4e}", flush=True)

            j = i + 1
            if j >= len(df):
                continue
            row_b = df.iloc[j]
            real_ts_b = row_b["real_timestamp"]
            z_b_dict = {c: float(row_b[c]) for c in z_cols}

            state.x0_v = list(result_a.x_v)
            state.x0_delta = list(result_a.x_delta)
            result_b = est.estimate(z_b_dict, real_ts_b)

            hour_b = int(real_ts_b[11:13])
            weighted_entries_b = est._weighted_entries_for_hour(hour_b)
            z_b = np.array([e["z_value"] if "z_value" in e else z_b_dict[e["id"]] for e in weighted_entries_b])
            ids_b, contrib_b = per_channel_contributions(est, model, state, result_b.x_v, result_b.x_delta, hour_b, z_b)
            for cid, c in zip(ids_b, contrib_b):
                all_rows.append({"point": f"B_warm_{j}", "channel": cid, "contribution": c})
            point_totals.append({"point": f"B_warm_{j}", "J_reported": result_b.J, "J_recomputed": contrib_b.sum()})
            print(f"[B_warm] i={j} J_reported={result_b.J:.4e} J_recomputed_from_channels={contrib_b.sum():.4e}", flush=True)

    long_df = pd.DataFrame(all_rows)
    long_df.to_csv(OUT_PATH, index=False)

    totals_df = pd.DataFrame(point_totals)
    print()
    print("=== sanity check: J_reported vs J_recomputed from summed channel contributions ===")
    print(totals_df.describe())
    max_mismatch = (totals_df["J_reported"] - totals_df["J_recomputed"]).abs().max()
    print(f"max |J_reported - J_recomputed| across all 60 points: {max_mismatch:.6e}")
    print()

    n_points = long_df["point"].nunique()
    per_channel_mean = long_df.groupby("channel")["contribution"].mean().sort_values(ascending=False)
    total_J_mean = totals_df["J_recomputed"].mean()

    print(f"n_points={n_points}  n_unique_channels={long_df['channel'].nunique()}  mean_total_J={total_J_mean:.4e}")
    print()
    print("=== TOP 10 CHANNELS BY AVERAGE CONTRIBUTION TO J (across all 60 points) ===")
    top10 = per_channel_mean.head(10)
    for cid, val in top10.items():
        print(f"  {cid:<12s} mean_contribution={val:.4e}  pct_of_mean_total_J={100*val/total_J_mean:.3f}%")

    top10_sum = top10.sum()
    print()
    print(f"sum of top 10 channels' mean contribution: {top10_sum:.4e}")
    print(f"mean total J: {total_J_mean:.4e}")
    print(f"TOP 10 CHANNELS ACCOUNT FOR {100*top10_sum/total_J_mean:.3f}% OF MEAN TOTAL J")
    print()

    n_channels = long_df["channel"].nunique()
    uniform_share_per_channel = 100.0 / n_channels
    print(f"if diffuse/uniform across all {n_channels} channels, each would contribute ~{uniform_share_per_channel:.3f}% on average")
    print(f"top 10 channels' actual average single-channel shares (%): "
          + ", ".join(f"{100*v/total_J_mean:.3f}" for v in top10.values))

    print()
    print("saved per-(point,channel) long table to:", OUT_PATH)


if __name__ == "__main__":
    main()
