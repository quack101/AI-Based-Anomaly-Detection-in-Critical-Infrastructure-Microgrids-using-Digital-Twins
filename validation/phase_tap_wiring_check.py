"""
RES.md D10 Phase 7 fix-jacobian-bugs branch -- Part A DIAGNOSTIC.
Confirms the true taps actually reach the estimator (not just the
consistency check), then gives per-node TSVD voltage-error stats
(nominal vs true taps) at i=46/94/95, split by hop-distance from any
regulator bus. Does not change the default solver.

Run: python -m validation.phase_tap_wiring_check
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import opendssdirect as dss

from simulation.opendss_utils import compile_feeder
from placement.electrical_distance import get_system_y_index
from observability.state_vector import build_state_vector, extract_solved_state, complex_v_from_state
from observability.estimator_model import build_estimator_model
from observability import jacobian as J
from estimation.state_estimator import StateEstimator
from estimation.experimental_tsvd_solver import estimate_tsvd

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
DATASET_DIR = BASE_DIR / "dataset" / "node_loads_91"
REPRESENTATIVE_METERS = BASE_DIR / "simulation" / "representative_meters.json"
TARGETS = [46, 94, 95]
REPLAY_UPTO = max(TARGETS)
REG_BUSES = {"9", "9r", "25", "25r", "150", "150r", "160", "160r"}


def main():
    with open(REPRESENTATIVE_METERS) as f:
        representative_meters = json.load(f)
    available = {f.stem[len("node_"):].lower(): f for f in DATASET_DIR.glob("node_*.csv")}
    datasets = [pd.read_csv(available[m["load_name"].lower()]) for m in representative_meters]

    compile_feeder()
    state = build_state_vector()
    reg_names = [n for n in dss.Transformers.AllNames() if n.lower().startswith("reg")]
    model_nominal = build_estimator_model("multiphase")
    est_nominal = StateEstimator(model_nominal)
    Y_nominal, idx_nominal = model_nominal.Y, model_nominal.node_index

    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_true_all = npz["x_v"]
    v_base_full = np.array([state.v_base[n] for n in state.all_node_names])

    # sequential replay 0..95, capturing true-tap Y at each target
    checkpoint = {}
    for t in range(REPLAY_UPTO + 1):
        for meter, ds in zip(representative_meters, datasets):
            row = ds.iloc[t]
            dss.Loads.Name(meter["load_name"])
            dss.Loads.kW(float(row["P_kW"]))
            dss.Loads.kvar(float(row["Q_kvar"]))
        dss.Solution.Solve()
        if t in TARGETS:
            taps = {}
            for n in reg_names:
                dss.Transformers.Name(n)
                taps[n] = dss.Transformers.Tap()
            Y_true, idx_true = get_system_y_index()
            x_v_live, x_delta_live = extract_solved_state(state)
            checkpoint[t] = {"taps": taps, "Y": Y_true, "idx": idx_true,
                             "x_v": np.array(x_v_live), "x_delta": np.array(x_delta_live)}

    # ---------------- A1: prove taps reach the estimator ----------------
    print("=== A1: do the true taps actually change H/h(x) and x_hat? (i=46) ===")
    cd46 = checkpoint[46]
    at_v_46 = complex_v_from_state(state, cd46["x_v"], cd46["x_delta"])
    H_nom, h0_nom = J.build_jacobian(model_nominal.meas_entries, Y_nominal, idx_nominal, state, at_v=at_v_46)
    H_true, h0_true = J.build_jacobian(model_nominal.meas_entries, cd46["Y"], cd46["idx"], state, at_v=at_v_46)
    print(f"max|H_nominal - H_true| = {np.max(np.abs(H_nom - H_true)):.6e}")
    print(f"max|h0_nominal - h0_true| = {np.max(np.abs(h0_nom - h0_true)):.6e} (kW/kvar/pu, mixed units)")

    row46 = df.iloc[46]
    real_ts46 = str(row46["real_timestamp"])
    z46 = {c: float(row46[c]) for c in z_cols}

    res_nom = estimate_tsvd(est_nominal, z46, real_ts46, state.x0_v, state.x0_delta, rcond=1e-8, max_iter=40)

    model_nominal.Y, model_nominal.node_index = cd46["Y"], cd46["idx"]
    res_true = estimate_tsvd(est_nominal, z46, real_ts46, state.x0_v, state.x0_delta, rcond=1e-8, max_iter=40)
    model_nominal.Y, model_nominal.node_index = Y_nominal, idx_nominal  # restore

    max_dx_v = float(np.max(np.abs(res_nom["x_v"] - res_true["x_v"])))
    max_dx_delta = float(np.max(np.abs(res_nom["x_delta"] - res_true["x_delta"])))
    print(f"max|x_hat_v(nominal) - x_hat_v(true taps)| = {max_dx_v:.6f} V")
    print(f"max|x_hat_delta(nominal) - x_hat_delta(true taps)| = {max_dx_delta:.6e} rad")
    if max_dx_v < 1e-6 and max_dx_delta < 1e-9:
        print("IDENTICAL -- taps were NOT wired into the solve. "
              f"The estimator reads its Y-bus from model.Y, set once in "
              f"observability/estimator_model.py's build_estimator_model() at compile time.")
    else:
        print("DIFFERENT -- confirmed: the true taps DO reach the solve (via model.Y).")
    print()

    # ---------------- A2 + A3: per-node error, nominal vs true taps ----------------
    # adjacency for hop-distance-from-regulator and "measurements near node"
    adj = {}
    for L in dss.Lines.AllNames():
        dss.Lines.Name(L)
        b1 = dss.Lines.Bus1().split(".")[0]
        b2 = dss.Lines.Bus2().split(".")[0]
        adj.setdefault(b1, set()).add(b2)
        adj.setdefault(b2, set()).add(b1)
    for x in dss.Transformers.AllNames():
        dss.Transformers.Name(x)
        buses = [b.split(".")[0] for b in dss.CktElement.BusNames()]
        for a_ in buses:
            for b_ in buses:
                if a_ != b_:
                    adj.setdefault(a_, set()).add(b_)

    hop_from_reg = {}
    dq = deque()
    for rb in REG_BUSES:
        if rb not in hop_from_reg:
            hop_from_reg[rb] = 0
            dq.append(rb)
    while dq:
        u = dq.popleft()
        for v in adj.get(u, []):
            if v not in hop_from_reg:
                hop_from_reg[v] = hop_from_reg[u] + 1
                dq.append(v)

    meas_count_by_bus = {}
    for e in model_nominal.meas_entries:
        meas_count_by_bus[e["node"]] = meas_count_by_bus.get(e["node"], 0) + 1

    def meas_near(bus):
        total = meas_count_by_bus.get(bus, 0)
        for nb in adj.get(bus, []):
            total += meas_count_by_bus.get(nb, 0)
        return total

    print("=== A2 + A3: per-node voltage error, TSVD rcond=1e-8, nominal vs true taps ===")
    for t in TARGETS:
        row = df.iloc[t]
        real_ts = str(row["real_timestamp"])
        z_real = {c: float(row[c]) for c in z_cols}
        cd = checkpoint[t]

        res_nom = estimate_tsvd(est_nominal, z_real, real_ts, state.x0_v, state.x0_delta, rcond=1e-8, max_iter=40)
        model_nominal.Y, model_nominal.node_index = cd["Y"], cd["idx"]
        res_true = estimate_tsvd(est_nominal, z_real, real_ts, state.x0_v, state.x0_delta, rcond=1e-8, max_iter=40)
        model_nominal.Y, model_nominal.node_index = Y_nominal, idx_nominal

        for label, res in [("nominal taps", res_nom), ("true taps", res_true)]:
            err = np.abs(res["x_v"] - x_v_true_all[t]) / v_base_full
            print(f"  i={t} [{label}]: median={np.median(err):.5f} mean={np.mean(err):.5f} "
                  f"p95={np.percentile(err, 95):.5f} max={np.max(err):.5f}")
            order = np.argsort(-err)[:5]
            for idx in order:
                node = state.all_node_names[idx]
                bus = node.split(".")[0]
                hop = hop_from_reg.get(bus, "unreachable")
                print(f"      worst: {node:<10s} err={err[idx]:.5f} pu  hop_from_reg={hop}  "
                      f"n_measurements_at_or_adjacent={meas_near(bus)}")

            v_idx_610 = [i2 for i2, n2 in enumerate(state.all_node_names) if n2.startswith("610.")]
            if v_idx_610:
                print(f"      bus 610 errors: {[f'{err[i2]:.5f}' for i2 in v_idx_610]}")

            bus_names = [n2.split(".")[0] for n2 in state.all_node_names]
            near_reg_mask = np.array([hop_from_reg.get(b, 999) <= 2 for b in bus_names])
            for grp, mask in [("within 2 hops of regulator", near_reg_mask), ("rest", ~near_reg_mask)]:
                sub = err[mask]
                print(f"      [{grp}, n={mask.sum()}]: median={np.median(sub):.5f} mean={np.mean(sub):.5f} "
                      f"p95={np.percentile(sub,95):.5f} max={np.max(sub):.5f}")
        print()


if __name__ == "__main__":
    main()
