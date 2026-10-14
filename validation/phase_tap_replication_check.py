"""
RES.md D10 Phase 7 fix-jacobian-bugs branch -- DIAGNOSTIC ONLY.
Replicates the EXACT sequential OpenDSS solve history (regulator
auto-control ON, same load-setting order as
validation/generate_phase6_corpus.py) from row 0 through row 1104,
recording every regulator's tap position at every row. At each of 6
checkpoint rows (46, 94, 95, 368, 824, 1104), rebuilds the Y-bus FRESH
from the live (now historically-correct) circuit state and redoes
the model-consistency check (h(x_true) vs OpenDSS-native measure()).
If it passes, re-runs the experimental truncated-SVD solver with this
corrected Y-bus on i=46, 94, 95.

Does not change the default solver or estimator_model.py's own
compile-time Y-bus construction -- this is a standalone, parallel
investigation using its own live OpenDSS session.

Run: python -m validation.phase_tap_replication_check
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import opendssdirect as dss

from simulation.opendss_utils import compile_feeder
from placement.electrical_distance import get_system_y_index
from observability.state_vector import build_state_vector, extract_solved_state, complex_v_from_state
from observability.estimator_model import build_estimator_model
from observability.pseudo_measurements import build_pseudo_entries_for_hour
from observability import jacobian as J
from sensors.measurement_functions import measure
from estimation.state_estimator import StateEstimator
from estimation.experimental_tsvd_solver import estimate_tsvd

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
DATASET_DIR = BASE_DIR / "dataset" / "node_loads_91"
REPRESENTATIVE_METERS = BASE_DIR / "simulation" / "representative_meters.json"
CHECKPOINTS = [46, 94, 95, 368, 824, 1104]
REPLAY_UPTO = max(CHECKPOINTS)


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
    reg_names = [n for n in dss.Transformers.AllNames() if n.lower().startswith("reg")]

    # nominal (compile-time) taps -- what our static estimator model actually assumes
    nominal_taps = {}
    for n in reg_names:
        dss.Transformers.Name(n)
        nominal_taps[n] = dss.Transformers.Tap()
    step_size_pu = 0.2 / 32  # (max-min)/numtaps, same for all 7 units
    print(f"=== STEP 1: regulator inventory ===")
    print(f"step size: {step_size_pu} pu/tap ({step_size_pu * 2401.77:.2f} V/tap at 2401.77V base)")
    print(f"nominal (compile-time, what the static estimator model assumes): {nominal_taps}")
    print()

    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_true_all, x_delta_true_all = npz["x_v"], npz["x_delta"]

    model_nominal = build_estimator_model("multiphase")  # for Part-4 StateEstimator reuse at nominal (unused for Y)

    tap_history = {n: [] for n in reg_names}
    checkpoint_data = {}

    t0 = time.time()
    for t in range(REPLAY_UPTO + 1):
        for meter, ds in zip(representative_meters, datasets):
            row = ds.iloc[t]
            dss.Loads.Name(meter["load_name"])
            dss.Loads.kW(float(row["P_kW"]))
            dss.Loads.kvar(float(row["Q_kvar"]))
        dss.Solution.Solve()

        for n in reg_names:
            dss.Transformers.Name(n)
            tap_history[n].append(dss.Transformers.Tap())

        if t in CHECKPOINTS:
            x_v_live, x_delta_live = extract_solved_state(state)
            taps_now = {n: tap_history[n][-1] for n in reg_names}
            # capture Y RIGHT HERE, in-sequence -- do not re-solve this row
            # later out of order (taps are path-dependent; a later
            # out-of-sequence re-solve does not reliably reproduce this
            # exact tap state, confirmed as a bug in an earlier version
            # of this script).
            Y_here, node_index_here = get_system_y_index()
            checkpoint_data[t] = {"x_v": np.array(x_v_live), "x_delta": np.array(x_delta_live), "taps": taps_now,
                                   "Y": Y_here, "node_index": node_index_here}
    replay_time = time.time() - t0
    print(f"sequential replay of {REPLAY_UPTO + 1} rows took {replay_time:.2f}s")
    print()

    print("=== STEP 1 (cont.): tap range over the replayed 0..{} span ===".format(REPLAY_UPTO))
    for n in reg_names:
        arr = np.array(tap_history[n])
        frac_diff = float(np.mean(np.abs(arr - nominal_taps[n]) > 1e-9))
        print(f"  {n:<8s} min={arr.min():.4f} max={arr.max():.4f}  nominal={nominal_taps[n]:.4f}  "
              f"frac_timestamps_differing_from_nominal={frac_diff:.3f}")
    print()

    print("=== STEP 2: EVIDENCE -- mismatch vs tap-difference x step size, at the 5 original Part-A points ===")
    for t in [46, 94, 824, 368, 1104]:
        cd = checkpoint_data[t]
        taps_true = cd["taps"]
        print(f"  i={t}:")
        for n in reg_names:
            diff_taps = round((taps_true[n] - nominal_taps[n]) / step_size_pu)
            predicted_v = abs(diff_taps) * step_size_pu * 2401.77
            if abs(diff_taps) > 0:
                print(f"    {n:<8s} true_tap={taps_true[n]:.4f} nominal={nominal_taps[n]:.4f} "
                      f"tap_diff={diff_taps:+d}  predicted_V_shift={predicted_v:.2f} V")

    print()
    print("=== STEP 3: consistency re-check with HISTORICALLY-CORRECT taps (live Y-bus at each checkpoint) ===")

    all_rows = []
    for t in CHECKPOINTS:
        cd = checkpoint_data[t]
        # restore the circuit to exactly this checkpoint's solved state (taps + loads) before rebuilding Y
        for meter, ds in zip(representative_meters, datasets):
            row = ds.iloc[t]
            dss.Loads.Name(meter["load_name"])
            dss.Loads.kW(float(row["P_kW"]))
            dss.Loads.kvar(float(row["Q_kvar"]))
        # FREEZE taps at the historically-correct values before solving --
        # re-solving with auto-control ON lets taps drift away from the
        # correct, path-dependent value captured during the one true
        # sequential pass (confirmed as a bug in an earlier version of
        # this script: taps are NOT simply a function of (loads, bus) at
        # a point in isolation, they depend on the full prior history).
        for rc in dss.RegControls.AllNames():
            dss.RegControls.Name(rc)
            dss.RegControls.MaxTapChange(0)
        for n in reg_names:
            dss.Transformers.Name(n)
            dss.Transformers.Tap(cd["taps"][n])
        dss.Solution.Solve()

        Y_correct, node_index_correct = get_system_y_index()

        real_ts = str(df.iloc[t]["real_timestamp"]) if t < 1440 else None
        if real_ts is None:
            continue
        hour = int(real_ts[11:13])

        # reuse the nominal model's registry/entries (structure doesn't depend on Y), swap in the correct Y
        pseudo_entries = build_pseudo_entries_for_hour(model_nominal.bus_phase_by_load, model_nominal.per_hour_by_load, hour)
        measurable_entries = model_nominal.meas_entries + pseudo_entries
        zi_like_entries = model_nominal.reg_pseudo_entries + model_nominal.zi_entries

        x_v_fresh, x_delta_fresh = extract_solved_state(state)
        at_v = complex_v_from_state(state, x_v_fresh, x_delta_fresh)

        for e in measurable_entries:
            z_exact = measure(e)
            _, h0_single = J.build_jacobian([e], Y_correct, node_index_correct, state, at_v=at_v)
            h_ours = h0_single[0]
            sigma = e["sigma"]
            err_kw = h_ours - z_exact
            all_rows.append({"i": t, "id": e["id"], "quantity": e["quantity"], "qgroup": qgroup(e["quantity"]),
                              "node": e.get("node", "?"), "sigma": sigma, "err_kw": err_kw, "err_sigma": err_kw / sigma})

    out = pd.DataFrame(all_rows)
    print(f"overall (n={len(out)}): median|err_sigma|={out['err_sigma'].abs().median():.4f}  "
          f"max|err_sigma|={out['err_sigma'].abs().max():.4f}")
    for grp, sub in out.groupby("qgroup"):
        print(f"  {grp:<10s} n={len(sub):<5d} median|err_sigma|={sub['err_sigma'].abs().median():.4f}  "
              f"max|err_sigma|={sub['err_sigma'].abs().max():.4f}")
    print()
    print("top 15 worst rows:")
    worst = out.reindex(out["err_sigma"].abs().sort_values(ascending=False).index).head(15)
    for _, r in worst.iterrows():
        print(f"  i={r['i']:<5d} {r['id']:<16s} node={str(r['node']):<6s} err_kw={r['err_kw']:+.5f}  err_sigma={r['err_sigma']:+.3f}")

    median_ok = out["err_sigma"].abs().median() < 1.0
    max_ok = out["err_sigma"].abs().max() < 10.0  # a genuinely strict "near noise level" bar
    print()
    print(f"median|err_sigma|={out['err_sigma'].abs().median():.4f}  ({'OK' if median_ok else 'FAILS'}, <1.0)")
    print(f"max|err_sigma|={out['err_sigma'].abs().max():.4f}  ({'OK' if max_ok else 'FAILS'}, <10.0)")
    sanity_pass = median_ok and max_ok
    print(f"SANITY CHECK {'PASSES' if sanity_pass else 'STILL FAILS'} overall -- "
          f"{'median looks fine but the MAX does not, do not round this up to a pass' if median_ok and not max_ok else ''}")

    print()
    print("=== STEP 4: TSVD re-run with historically-correct Y-bus (frozen, from the sequential pass), rcond=1e-8 ===")
    print("(running regardless of the sanity-check outcome above, since the fixed-tap Y is still the best-available model)")
    est_nominal = StateEstimator(model_nominal)
    for t, label in [(46, "i=46"), (94, "i=94"), (95, "i=95")]:
        row = df.iloc[t]
        real_ts = str(row["real_timestamp"])
        z_real = {c: float(row[c]) for c in z_cols}
        cd = checkpoint_data[t]

        for meter, ds in zip(representative_meters, datasets):
            drow = ds.iloc[t]
            dss.Loads.Name(meter["load_name"])
            dss.Loads.kW(float(drow["P_kW"]))
            dss.Loads.kvar(float(drow["Q_kvar"]))
        for rc in dss.RegControls.AllNames():
            dss.RegControls.Name(rc)
            dss.RegControls.MaxTapChange(0)
        for n in reg_names:
            dss.Transformers.Name(n)
            dss.Transformers.Tap(cd["taps"][n])
        dss.Solution.Solve()
        Y_correct, node_index_correct = get_system_y_index()

        # monkey-patch the model's Y/node_index for this run only
        orig_Y, orig_idx = model_nominal.Y, model_nominal.node_index
        model_nominal.Y, model_nominal.node_index = Y_correct, node_index_correct

        res = estimate_tsvd(est_nominal, z_real, real_ts, state.x0_v, state.x0_delta, rcond=1e-8, max_iter=40)
        model_nominal.Y, model_nominal.node_index = orig_Y, orig_idx

        v_base_full = np.array([state.v_base[n] for n in state.all_node_names])
        v_err = float(np.max(np.abs(res["x_v"] - x_v_true_all[t]) / v_base_full))
        print(f"  [{label}] iters={res['iterations']} converged={res['converged']} J={res['J']:.4e} "
              f"v_err_pu={v_err:.5f}  cres={res['constraint_residual_inf']:.4f}")


if __name__ == "__main__":
    main()
