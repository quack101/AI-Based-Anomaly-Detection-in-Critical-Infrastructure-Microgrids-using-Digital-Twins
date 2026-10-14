"""
RES.md D10 Phase 7 REMEDIATION -- Step 3: direct circularity diagnostic.

Sets x_0 = x_true + delta at a handful of magnitudes (delta as a fraction
of each node's own V_base for the voltage block, and as a fraction of a
characteristic 3-phase angle separation (2*pi/3 rad) for the delta block)
at several representative timestamps, and reports iteration count vs
delta. If iteration count stays ~flat across all delta magnitudes
(including delta=0, i.e. starting EXACTLY at x_true), the shipped "mean
1.00 iterations" reflects a loose tolerance, not circularity. If
iteration count is only ~1 near delta=0 and rises sharply with delta,
that is direct evidence the "mean 1.00" depended on already starting at
the answer.

Run: python validation/phase7_circularity_diagnostic.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"

# Trimmed to 2 timestamps (off-peak + peak) per Step 2's own stratified
# sample (seed 777, low/high tertile of total feeder P) for a free
# cross-check against the Step 2 flat-start result at the SAME indices,
# instead of 5 independently-chosen timestamps -- all 5 delta
# fractions are kept (that axis is the actual diagnostic; Step 3's
# verdict reads off the SHAPE of iterations vs delta, which a thinner
# delta sweep would blunt).
PICKS = [46, 368]  # i=46: low-load tertile; i=368: high-load tertile
DELTA_FRACS = [0.0, 0.001, 0.01, 0.05, 0.10]
SEED = 12345


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df_full = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_all, x_delta_all = npz["x_v"], npz["x_delta"]

    picks = PICKS

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    v_base = np.array([state.v_base[name] for name in state.all_node_names])
    delta_ref = 2.0 * np.pi / 3.0  # characteristic 3-phase angle separation

    rng = np.random.default_rng(SEED)

    records = []
    for i in picks:
        row = df_full.iloc[i]
        real_ts = row["real_timestamp"]
        z_real = {c: float(row[c]) for c in z_cols}
        xt_v, xt_delta = x_v_all[i], x_delta_all[i]

        for frac in DELTA_FRACS:
            # same perturbation direction reused across fracs for a given
            # timestamp (scaled by frac), so the delta=0 row is the exact
            # x_true start and larger fracs are consistent larger nudges
            # along the SAME random direction, not independently redrawn
            # noise each time.
            unit_v = rng.uniform(-1.0, 1.0, size=v_base.shape)
            unit_delta = rng.uniform(-1.0, 1.0, size=xt_delta.shape)

            x0_v = xt_v + frac * unit_v * v_base
            x0_delta = xt_delta + frac * unit_delta * delta_ref

            state.x0_v = list(x0_v)
            state.x0_delta = list(x0_delta)

            t0 = time.time()
            result = est.estimate(z_real, real_ts)
            dt = time.time() - t0

            err_v_pu = np.abs(result.x_v - xt_v) / v_base
            err_delta = np.abs(result.x_delta - xt_delta)

            records.append({
                "i": i, "real_timestamp": real_ts, "delta_frac": frac,
                "converged": result.converged, "iterations": result.iterations,
                "time_s": dt, "max_v_err_pu": err_v_pu.max(),
                "max_delta_err_rad": err_delta.max(),
                "constraint_residual_inf": result.constraint_residual_inf,
            })
            print(f"i={i} real_ts={real_ts} frac={frac:<6} converged={result.converged} "
                  f"iters={result.iterations:<3d} time_s={dt:.2f} "
                  f"maxVerr={err_v_pu.max():.5f} cres={result.constraint_residual_inf:.3f}", flush=True)

    out = pd.DataFrame(records)
    out_path = BASE_DIR / "validation" / "phase7_circularity_results.csv"
    out.to_csv(out_path, index=False)

    print()
    print("=== ITERATIONS vs DELTA_FRAC (pivot, mean over timestamps) ===")
    pivot = out.groupby("delta_frac")["iterations"].agg(["mean", "median", "max"])
    print(pivot)
    print()
    print("saved:", out_path)


if __name__ == "__main__":
    main()
