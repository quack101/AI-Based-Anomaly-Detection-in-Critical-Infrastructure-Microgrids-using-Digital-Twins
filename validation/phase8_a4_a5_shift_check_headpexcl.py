"""
RES.md D10 Phase 8 -- does the re-estimated state follow x_before + c?

For points i=46, 94, 154 and A4-M1 / A5-M1 at 0.05, 0.10, 0.20 pu,
re-creates the exact attack vectors of validation/phase8_a4_a5_scan_
frozen.py (same per-point seed SEED+400+k, same rng call order incl.
the two controls, which are drawn but not solved here) and records the
full state vectors that the scan CSV does not keep:
  c                  -- the intended state shift (volts, radians)
  x_before, x_after  -- est.estimate() clean / attacked (warm start)
  mismatch = (x_after - x_before) - c
V-block reported in pu (/ v_base), angle block in rad. That is also
the estimator's y-space (state_scale = [v_base, 1]), so states are
ranked by |mismatch| in that space.

Reproduction check: J_after and achieved_shift_pu are compared with
the matching rows of validation/phase8_a4_a5_scan_headpexcl.csv.

Run: python -m validation.phase8_a4_a5_shift_check_headpexcl
"""

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr, unattackable_jacobian
from attacks.a4_synthesis import synthesize_a4_attack, random_control, random_sign_control
from validation import phase8_a4_a5_scan_frozen as scan

BASE_DIR = scan.BASE_DIR
OUT_SUMMARY = BASE_DIR / "validation" / "phase8_a4_a5_shift_check_headpexcl.csv"
OUT_TOP5 = BASE_DIR / "validation" / "phase8_a4_a5_shift_check_top5_headpexcl.csv"
POINTS = [46, 94, 154]
CHECK_MAGS = [0.05, 0.10, 0.20]
TOP_K = 5


def shift_check_point(i, real_ts, z_real, seed_for_point):
    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    n = state.n_phase_nodes
    v_base = np.array([state.v_base[name] for name in state.all_node_names])
    delta_names = [None] * (state.dim - n)
    for name, idx in state.delta_index.items():
        delta_names[idx - n] = name
    state_names = [f"V|{name}" for name in state.all_node_names] + [f"ang|{name}" for name in delta_names]
    meas_ids = [e["id"] for e in model.meas_entries]

    state.x0_v, state.x0_delta = list(state.x0_v), list(state.x0_delta)
    result_clean = est.estimate(z_real, real_ts)
    x_before = np.concatenate([result_clean.x_v, result_clean.x_delta])

    jac = unattackable_jacobian(est, result_clean, real_ts)
    Hs_attack, Hs_unattack, state_scale = jac["Hs_meas"], jac["Hs_unattackable"], jac["state_scale"]
    unattack_entries = jac["unattackable_entries"]
    rng = np.random.default_rng(seed_for_point)

    summary, top = [], []
    for part in ["A4", "A5"]:
        for mag in scan.MAGNITUDES:
            if part == "A4":
                atk = synthesize_a4_attack(Hs_attack, Hs_unattack, state_scale, rng, "M1", mag, n)
                a, c = atk["a_attackable"], atk["c"]
            else:
                a, c, _, _ = scan.build_a5_attack(
                    result_clean, model, state, Hs_unattack, state_scale, unattack_entries, mag, rng,
                )
            random_control(a, rng)       # consumed only to keep the scan's rng sequence
            random_sign_control(a, rng)
            if mag not in CHECK_MAGS:
                continue

            z_dict = dict(z_real)
            for k, mid in enumerate(meas_ids):
                z_dict[mid] = z_real[mid] + a[k]
            state.x0_v, state.x0_delta = list(result_clean.x_v), list(result_clean.x_delta)
            result = est.estimate(z_dict, real_ts)
            stats = compute_chi2_and_lnr(est, result, z_dict, real_ts)
            x_after = np.concatenate([result.x_v, result.x_delta])

            dx = x_after - x_before
            mismatch = dx - c
            c_y, dx_y, mm_y = c / state_scale, dx / state_scale, mismatch / state_scale
            summary.append({
                "i": i, "part": part, "magnitude_pu": mag, "converged": result.converged,
                "iterations": result.iterations, "J_after": stats["J"],
                "max_abs_c_v_pu": float(np.max(np.abs(c_y[:n]))),
                "max_abs_c_ang_rad": float(np.max(np.abs(c_y[n:]))),
                "max_abs_dx_v_pu": float(np.max(np.abs(dx_y[:n]))),
                "max_abs_dx_ang_rad": float(np.max(np.abs(dx_y[n:]))),
                "inf_norm_mismatch_v_pu": float(np.max(np.abs(mm_y[:n]))),
                "inf_norm_mismatch_ang_rad": float(np.max(np.abs(mm_y[n:]))),
                "rel_mismatch_v": float(np.linalg.norm(mm_y[:n]) / np.linalg.norm(c_y[:n])),
                "rel_mismatch_ang": float(np.linalg.norm(mm_y[n:]) / np.linalg.norm(c_y[n:])),
            })
            for rank, k in enumerate(np.argsort(-np.abs(mm_y))[:TOP_K], start=1):
                top.append({
                    "i": i, "part": part, "magnitude_pu": mag, "rank": rank, "state": state_names[k],
                    "unit": "pu" if k < n else "rad", "c": float(c_y[k]), "dx": float(dx_y[k]),
                    "mismatch": float(mm_y[k]),
                })
    return summary, top


def main():
    telem_cols = pd.read_csv(scan.CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(scan.CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols,
                     nrows=1440)
    p_cols = [c for c in df.columns if c.endswith("_P")]
    strata = scan.stratified_picks(df, p_cols)
    points = sorted(set(strata["low"] + strata["mid"] + strata["high"]))[:scan.N_POINTS]

    summary, top = [], []
    for i in POINTS:
        k = points.index(i)
        row = df.iloc[i]
        s, t = shift_check_point(i, str(row["real_timestamp"]), {c: float(row[c]) for c in z_cols},
                                 scan.SEED + 400 + k)
        summary += s
        top += t
        print(f"point i={i} (scan index {k}) done", flush=True)

    out = pd.DataFrame(summary)
    scan_df = pd.read_csv(scan.OUT_PATH)
    ref = scan_df[scan_df["kind"] == "attack"][["i", "part", "magnitude_pu", "J_after", "achieved_shift_pu"]]
    chk = out.merge(ref, on=["i", "part", "magnitude_pu"], suffixes=("", "_scan"))
    print(f"reproduction vs scan CSV: {len(chk)} rows, "
          f"max |dJ_after| = {np.max(np.abs(chk['J_after'] - chk['J_after_scan'])):.3e}, "
          f"max |d achieved_v_pu| = {np.max(np.abs(chk['max_abs_dx_v_pu'] - chk['achieved_shift_pu'])):.3e}")

    out.to_csv(OUT_SUMMARY, index=False)
    pd.DataFrame(top).to_csv(OUT_TOP5, index=False)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    print()
    print(pd.DataFrame(top).to_string(index=False))
    print("saved:", OUT_SUMMARY)
    print("saved:", OUT_TOP5)


if __name__ == "__main__":
    main()
