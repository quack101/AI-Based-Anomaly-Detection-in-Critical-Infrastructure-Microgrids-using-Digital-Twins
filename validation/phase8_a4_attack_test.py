"""
RES.md D10 Phase 8 step 2 -- TEST the A4 attack (a = Hc) against the
REAL, nonlinear, equality-constrained AC estimator. Read-only on the
estimator; does NOT touch estimation/state_estimator.py and is NOT
wired into streaming/injector.py.

For each of the 30 stratified points (seed=777) x 3 modes (M1 dense,
M2 sparse ~20 meters, M3 the 6 zero-redundancy real meters) x 3
magnitudes (0.01/0.05/0.10 pu max voltage shift), synthesizes an A4
attack via attacks/a4_synthesis.py, re-runs the real StateEstimator on
z_delivered = z + a, and compares against a same-size random-direction
control on the same meters.

Run: python -m validation.phase8_a4_attack_test
"""

from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr, unattackable_jacobian
from attacks.a4_synthesis import synthesize_a4_attack, random_control

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase8_a4_attack_test_results.csv"

SEED = 777
N_PER_STRATUM = 10
MODES = ["M1", "M2", "M3"]
MAGNITUDES = [0.01, 0.05, 0.10]
N_SPARSE_METERS = 20
ZERO_REDUNDANCY_REAL_METER_IDS = {"M004_P", "M004_Q", "M040_P", "HEAD_Q_A", "HEAD_Q_B", "HEAD_Q_C"}
CHI2_THRESH = 75.6237  # chi2.ppf(0.95, 57) -- textbook
LNR_THRESH = 3.0       # textbook
EMPIRICAL_J99 = 207272.6213
EMPIRICAL_LNR99 = 158.7156


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

    p_cols = [c for c in df.columns if c.endswith("_P")]
    strata = stratified_picks(df, p_cols)
    sweep_idxs = sorted(set(strata["low"] + strata["mid"] + strata["high"]))
    print("stratified picks:", sweep_idxs, flush=True)

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)
    v_base_full = np.array([state.v_base[n] for n in state.all_node_names])

    meas_ids = [e["id"] for e in model.meas_entries]
    fixed_idx = np.array([meas_ids.index(mid) for mid in ZERO_REDUNDANCY_REAL_METER_IDS])
    print(f"M3 fixed meter indices (of {len(meas_ids)} attackable channels): "
          f"{list(zip(ZERO_REDUNDANCY_REAL_METER_IDS, fixed_idx))}", flush=True)

    rng = np.random.default_rng(SEED + 2)
    records = []

    for idx_num, i in enumerate(sweep_idxs):
        state.x0_v = list(flat_v)
        state.x0_delta = list(flat_delta)
        row = df.iloc[i]
        real_ts = str(row["real_timestamp"])
        z_real = {c: float(row[c]) for c in z_cols}

        result_clean = est.estimate(z_real, real_ts)
        stats_clean = compute_chi2_and_lnr(est, result_clean, z_real, real_ts)
        jac = unattackable_jacobian(est, result_clean, real_ts)
        Hs_attack, Hs_unattack, state_scale = jac["Hs_meas"], jac["Hs_unattackable"], jac["state_scale"]

        print(f"[{idx_num+1}/{len(sweep_idxs)}] i={i} J_clean={stats_clean['J']:.3e} "
              f"lnr_clean={stats_clean['lnr']:.3f}", flush=True)

        for mode in MODES:
            for mag in MAGNITUDES:
                atk = synthesize_a4_attack(
                    Hs_attack, Hs_unattack, state_scale, rng, mode, mag, state.n_phase_nodes,
                    n_sparse_meters=N_SPARSE_METERS,
                    fixed_meter_idx=fixed_idx if mode == "M3" else None,
                )
                a = atk["a_attackable"]
                max_a = np.max(np.abs(a))
                n_significant_meters = int(np.sum(np.abs(a) > 0.01 * max_a)) if max_a > 0 else 0
                z_bad = dict(z_real)
                for k, mid in enumerate(meas_ids):
                    z_bad[mid] = z_real[mid] + a[k]

                state.x0_v = list(flat_v)
                state.x0_delta = list(flat_delta)
                result_bad = est.estimate(z_bad, real_ts)
                stats_bad = compute_chi2_and_lnr(est, result_bad, z_bad, real_ts)

                dx_v = result_bad.x_v - result_clean.x_v
                dx_delta = result_bad.x_delta - result_clean.x_delta
                actual_shift_norm = float(np.hypot(np.linalg.norm(dx_v), np.linalg.norm(dx_delta)))
                c_norm = float(np.linalg.norm(atk["c"]))
                norm_ratio = actual_shift_norm / c_norm if c_norm > 0 else float("nan")
                max_v_shift_pu = float(np.max(np.abs(dx_v) / v_base_full))

                dJ = stats_bad["J"] - stats_clean["J"]
                dLNR = stats_bad["lnr"] - stats_clean["lnr"]
                pass_chi2_textbook = stats_bad["J"] <= CHI2_THRESH
                pass_lnr_textbook = stats_bad["lnr"] <= LNR_THRESH
                pass_chi2_emp = stats_bad["J"] <= EMPIRICAL_J99
                pass_lnr_emp = stats_bad["lnr"] <= EMPIRICAL_LNR99

                # control: same norm, same meters, random direction
                a_ctrl = random_control(a, rng)
                z_ctrl = dict(z_real)
                for k, mid in enumerate(meas_ids):
                    z_ctrl[mid] = z_real[mid] + a_ctrl[k]

                state.x0_v = list(flat_v)
                state.x0_delta = list(flat_delta)
                result_ctrl = est.estimate(z_ctrl, real_ts)
                stats_ctrl = compute_chi2_and_lnr(est, result_ctrl, z_ctrl, real_ts)
                dJ_ctrl = stats_ctrl["J"] - stats_clean["J"]
                dLNR_ctrl = stats_ctrl["lnr"] - stats_clean["lnr"]
                pass_chi2_emp_ctrl = stats_ctrl["J"] <= EMPIRICAL_J99
                pass_lnr_emp_ctrl = stats_ctrl["lnr"] <= EMPIRICAL_LNR99

                dx_v_ctrl = result_ctrl.x_v - result_clean.x_v
                dx_delta_ctrl = result_ctrl.x_delta - result_clean.x_delta
                ctrl_shift_norm = float(np.hypot(np.linalg.norm(dx_v_ctrl), np.linalg.norm(dx_delta_ctrl)))

                records.append({
                    "i": i, "mode_requested": mode, "mode_used": atk["mode_used"],
                    "feasible": atk["feasible"], "magnitude_target_pu": mag,
                    "achieved_v_shift_pu_intended": atk["achieved_v_shift_pu"],
                    "max_unattackable_residual": atk["max_unattackable_residual"],
                    "n_significant_meters": n_significant_meters,
                    "converged_clean": result_clean.converged, "converged_bad": result_bad.converged,
                    "J_clean": stats_clean["J"], "J_bad": stats_bad["J"], "dJ": dJ,
                    "lnr_clean": stats_clean["lnr"], "lnr_bad": stats_bad["lnr"], "dLNR": dLNR,
                    "pass_chi2_textbook": pass_chi2_textbook, "pass_lnr_textbook": pass_lnr_textbook,
                    "pass_chi2_empirical": pass_chi2_emp, "pass_lnr_empirical": pass_lnr_emp,
                    "actual_shift_norm": actual_shift_norm, "c_norm": c_norm, "norm_ratio": norm_ratio,
                    "max_v_shift_pu_actual": max_v_shift_pu,
                    "converged_ctrl": result_ctrl.converged,
                    "J_ctrl": stats_ctrl["J"], "dJ_ctrl": dJ_ctrl,
                    "lnr_ctrl": stats_ctrl["lnr"], "dLNR_ctrl": dLNR_ctrl,
                    "pass_chi2_empirical_ctrl": pass_chi2_emp_ctrl, "pass_lnr_empirical_ctrl": pass_lnr_emp_ctrl,
                    "ctrl_shift_norm": ctrl_shift_norm,
                })
                pd.DataFrame(records).to_csv(OUT_PATH, index=False)
                print(f"    [{mode}/{mag}] feasible={atk['feasible']} mode_used={atk['mode_used']} "
                      f"dJ={dJ:+.3e} dJ_ctrl={dJ_ctrl:+.3e} dLNR={dLNR:+.2f} dLNR_ctrl={dLNR_ctrl:+.2f} "
                      f"pass_emp(attack)={pass_chi2_emp and pass_lnr_emp} "
                      f"pass_emp(ctrl)={pass_chi2_emp_ctrl and pass_lnr_emp_ctrl}", flush=True)

    out = pd.DataFrame(records)
    print()
    print("=== Fraction under EMPIRICAL thresholds (both chi2 AND LNR pass), per mode x magnitude ===")
    out["pass_emp_both"] = out["pass_chi2_empirical"] & out["pass_lnr_empirical"]
    out["pass_emp_both_ctrl"] = out["pass_chi2_empirical_ctrl"] & out["pass_lnr_empirical_ctrl"]
    print(out.groupby(["mode_requested", "magnitude_target_pu"])[["pass_emp_both", "pass_emp_both_ctrl"]].mean())
    print()
    print("=== Median paired dJ / dLNR: attack vs control, per mode x magnitude ===")
    print(out.groupby(["mode_requested", "magnitude_target_pu"])[["dJ", "dJ_ctrl", "dLNR", "dLNR_ctrl"]].median())
    print()
    print("=== Median ||x_bad - x_clean|| vs ||c||: norm_ratio, per mode x magnitude ===")
    print(out.groupby(["mode_requested", "magnitude_target_pu"])[["norm_ratio", "max_v_shift_pu_actual"]].median())
    print()
    print("=== Feasibility of requested mode (M2/M3 fallback rate) ===")
    print(out.groupby("mode_requested")["feasible"].mean())
    print()
    cases_ctrl_passes = out[out["pass_emp_both_ctrl"]]
    print(f"=== Cases where the CONTROL also passes both empirical thresholds: {len(cases_ctrl_passes)}/{len(out)} ===")
    if len(cases_ctrl_passes):
        print(cases_ctrl_passes[["i", "mode_requested", "magnitude_target_pu", "dJ_ctrl", "dLNR_ctrl"]].to_string(index=False))

    print()
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
