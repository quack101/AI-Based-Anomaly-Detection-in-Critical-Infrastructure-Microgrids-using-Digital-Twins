"""
RES.md D10 Phase 7 fix-jacobian-bugs branch -- DIAGNOSTIC ONLY, Part A.
Model consistency check: does our analytic h(x) (observability/
jacobian.py) reproduce OpenDSS's OWN native measurement at the
data-generating true state x_true, for 5 stratified points?

Independent ground truth: re-solves OpenDSS at the exact historical
loads for each target timestamp (same procedure
validation/generate_phase6_corpus.py uses), then reads
sensors.measurement_functions.measure() directly -- NOT through our
own Jacobian module -- for every metered (meas_entries) and unmetered
-but-real-load (PSEUDO_*) channel. True zero-injection (ZI_*) and
regulator-pseudo (REGZI_*) rows have no single OpenDSS element to
query this way (they are a net-injection-at-a-bus concept, not one
element) -- for those, the check is h_zi(x_true) vs the EXACT target
(0 for ZI_*, that row's own empirical mean for REGZI_*), which tests
whether our Y-bus-derived network equations reproduce zero/the
expected value at the SAME state OpenDSS itself solved for.

Run: python -m validation.phase_model_consistency_check
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

import opendssdirect as dss

from simulation.opendss_utils import compile_feeder
from observability.state_vector import build_state_vector, extract_solved_state, complex_v_from_state
from observability.estimator_model import build_estimator_model
from observability.pseudo_measurements import build_pseudo_entries_for_hour
from observability import jacobian as J
from sensors.measurement_functions import measure
from estimation.state_estimator import StateEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
DATASET_DIR = BASE_DIR / "dataset" / "node_loads_91"
REPRESENTATIVE_METERS = BASE_DIR / "simulation" / "representative_meters.json"
TARGET_IDXS = [46, 94, 824, 368, 1104]


def qgroup(q):
    if q == "P_injection":
        return "P_inj"
    if q == "Q_injection":
        return "Q_inj"
    if q == "V_magnitude":
        return "V_mag"
    if q in ("P_head", "Q_head"):
        return "head"
    if q in ("P_flow", "Q_flow"):
        return "flow"
    return q


def main():
    with open(REPRESENTATIVE_METERS) as f:
        representative_meters = json.load(f)
    available = {f.stem[len("node_"):].lower(): f for f in DATASET_DIR.glob("node_*.csv")}
    datasets = [pd.read_csv(available[m["load_name"].lower()]) for m in representative_meters]

    compile_feeder()
    state = build_state_vector()
    model = build_estimator_model("multiphase")
    est = StateEstimator(model)

    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_true_all, x_delta_true_all = npz["x_v"], npz["x_delta"]

    all_rows = []
    zi_summary = []
    j_x_true_rows = []

    for i in TARGET_IDXS:
        row = df.iloc[i]
        real_ts = str(row["real_timestamp"])
        z_real = {c: float(row[c]) for c in z_cols}

        # replicate generate_phase6_corpus.py's load-setting procedure
        t = int(datasets[0].index[datasets[0]["timestamp"] == real_ts][0])
        for meter, ds in zip(representative_meters, datasets):
            drow = ds.iloc[t]
            dss.Loads.Name(meter["load_name"])
            dss.Loads.kW(float(drow["P_kW"]))
            dss.Loads.kvar(float(drow["Q_kvar"]))
        dss.Solution.Solve()

        x_v_live, x_delta_live = extract_solved_state(state)
        x_v_stored, x_delta_stored = x_v_true_all[i], x_delta_true_all[i]
        v_mismatch = float(np.max(np.abs(np.array(x_v_live) - x_v_stored)))
        print(f"[i={i}] live-resolve vs stored x_true: max|dV|={v_mismatch:.6f} V "
              f"(sanity check -- should be ~0)")

        hour = int(real_ts[11:13])
        pseudo_entries = build_pseudo_entries_for_hour(model.bus_phase_by_load, model.per_hour_by_load, hour)
        measurable_entries = model.meas_entries + pseudo_entries  # have load_name/element_name -> real OpenDSS element
        zi_like_entries = model.reg_pseudo_entries + model.zi_entries  # net-injection-at-bus, no single element

        at_v = complex_v_from_state(state, x_v_live, x_delta_live)

        # --- OpenDSS-native ground truth for measurable entries ---
        for e in measurable_entries:
            z_exact = measure(e)
            H_single, h0_single = J.build_jacobian([e], model.Y, model.node_index, state, at_v=at_v)
            h_ours = h0_single[0]
            sigma = e["sigma"]
            err_kw = h_ours - z_exact
            err_sigma = err_kw / sigma
            all_rows.append({
                "i": i, "id": e["id"], "quantity": e["quantity"], "qgroup": qgroup(e["quantity"]),
                "node": e.get("node", "?"), "sigma": sigma, "h_ours": h_ours, "z_exact": z_exact,
                "err_kw": err_kw, "err_sigma": err_sigma,
                "is_pseudo": e["id"].startswith("PSEUDO_"),
            })

        # --- zero-injection / reg-pseudo network-equation residual ---
        H_zi, h0_zi = J.build_jacobian(zi_like_entries, model.Y, model.node_index, state, at_v=at_v)
        for e, h0v in zip(zi_like_entries, h0_zi):
            target = e.get("z_value", 0.0)
            resid = h0v - target
            zi_summary.append({"i": i, "id": e["id"], "quantity": e["quantity"],
                                "node": e.get("node", "?"), "target": target, "h_ours": h0v, "resid_kw": resid,
                                "is_regzi": e["id"].startswith("REGZI_")})

        # --- J(x_true) using the real NOISY delivered z ---
        weighted_entries = est._weighted_entries_for_hour(hour)
        z_w = np.array([e["z_value"] if "z_value" in e else z_real[e["id"]] for e in weighted_entries])
        sigma_w = np.array([e["sigma"] for e in weighted_entries])
        W_w = 1.0 / sigma_w ** 2
        _, h0_w = J.build_jacobian(weighted_entries, model.Y, model.node_index, state, at_v=at_v)
        J_x_true = float(np.sum(W_w * (z_w - h0_w) ** 2))
        j_x_true_rows.append({"i": i, "J_x_true": J_x_true, "m_weighted": len(weighted_entries)})

    out = pd.DataFrame(all_rows)
    zi_out = pd.DataFrame(zi_summary)

    print()
    print(f"=== OVERALL (measurable entries, n={len(out)}) ===")
    print(f"median |err_sigma| = {out['err_sigma'].abs().median():.4f}")
    print(f"max |err_sigma|    = {out['err_sigma'].abs().max():.4f}")
    print()
    print("=== PER CHANNEL TYPE ===")
    for grp, sub in out.groupby("qgroup"):
        print(f"  {grp:<10s} n={len(sub):<5d} median|err_sigma|={sub['err_sigma'].abs().median():.4f}  "
              f"max|err_sigma|={sub['err_sigma'].abs().max():.4f}  median|err_kw|={sub['err_kw'].abs().median():.5f}  "
              f"max|err_kw|={sub['err_kw'].abs().max():.5f}")
    # pseudo as its own explicit group
    pseudo_sub = out[out["is_pseudo"]]
    if len(pseudo_sub):
        print(f"  {'pseudo':<10s} n={len(pseudo_sub):<5d} median|err_sigma|={pseudo_sub['err_sigma'].abs().median():.4f}  "
              f"max|err_sigma|={pseudo_sub['err_sigma'].abs().max():.4f}")
    print()

    print("=== TOP 15 WORST ROWS (by |err_sigma|) ===")
    worst = out.reindex(out["err_sigma"].abs().sort_values(ascending=False).index).head(15)
    for _, r in worst.iterrows():
        print(f"  i={r['i']:<5d} {r['id']:<16s} node={str(r['node']):<6s} err_kw={r['err_kw']:+.5f}  "
              f"err_sigma={r['err_sigma']:+.3f}  sigma={r['sigma']:.4f}")
    print()

    print("=== ZERO-INJECTION / REG-PSEUDO network-equation residual (h_ours - target, kW/kvar) ===")
    zi_true = zi_out[~zi_out["is_regzi"]]
    zi_reg = zi_out[zi_out["is_regzi"]]
    print(f"TRUE zero-injection (ZI_*, target=0): n={len(zi_true)}  "
          f"median|resid|={zi_true['resid_kw'].abs().median():.6f}  max|resid|={zi_true['resid_kw'].abs().max():.6f}")
    worst_zi = zi_true.reindex(zi_true["resid_kw"].abs().sort_values(ascending=False).index).head(10)
    for _, r in worst_zi.iterrows():
        print(f"    i={r['i']:<5d} {r['id']:<16s} node={str(r['node']):<6s} resid={r['resid_kw']:+.5f} kW/kvar")
    print(f"REGZI (demoted pseudo, target=own empirical mean): n={len(zi_reg)}  "
          f"median|resid|={zi_reg['resid_kw'].abs().median():.4f}  max|resid|={zi_reg['resid_kw'].abs().max():.4f}")
    print()

    print("=== J(x_true) using the real noisy delivered z ===")
    for r in j_x_true_rows:
        print(f"  i={r['i']:<5d} J(x_true)={r['J_x_true']:.4e}  m_weighted={r['m_weighted']}")


if __name__ == "__main__":
    main()
