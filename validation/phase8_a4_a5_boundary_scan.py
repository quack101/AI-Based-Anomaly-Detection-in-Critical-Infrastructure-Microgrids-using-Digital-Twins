"""
RES.md D10 Phase 8 step 2b -- sequential, single-point (i=46) A4
boundary scan (small sizes) + A5 nonlinear comparison. No
multiprocessing (the earlier cross-process attempt hung -- see
report; opendssdirect's COM interface does not appear to survive a
Python multiprocessing worker boundary on this machine). Every solve
is WARM-STARTED from the clean x_hat and capped at 20s wall-clock via
an explicit per-iteration deadline check inside a verbatim replica of
the real solve loop (same pattern as validation/phase7_bus83_*.py --
does not modify estimation/state_estimator.py).

Part A -- A4 (linear, a=Hc) boundary scan: modes M1 (dense null-space
direction) and M2 (sparse, ~20 meters), magnitudes
0.001/0.003/0.01/0.02 pu, each with a matching same-norm/same-meters
random control.

Part B -- A5 (exact nonlinear, a=h(x_bad)-h(x_clean)) at magnitudes
0.01/0.05/0.10 pu, using the SAME dense null-space direction as A4-M1.
Checks the residual nonlinearity leaves on the unattackable rows, and
Newton-corrects c (up to 5 steps) to drive it back toward zero before
running the real estimator.

Labels: STEALTH (converged, under both empirical thresholds, real
effect), DETECTED (converged, fails a threshold), SOLVER_FAIL (did
not converge or hit the 20s cap -- NEVER counted as detected),
NO_EFFECT (converged, under thresholds, but norm_ratio ~ 0 -- not
really an attack).

Run: python -m validation.phase8_a4_a5_boundary_scan
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import (
    StateEstimator, DEFAULT_TOL_V_PU, DEFAULT_TOL_DELTA_RAD,
    DEFAULT_TOL_CONSTRAINT_KW, LM_LAMBDA_INIT, LM_MAX_TRIES, LM_DECREASE_FACTOR,
    LM_INCREASE_FACTOR, LM_LAMBDA_MAX, LM_LAMBDA_MIN, STEP_FRACTIONS,
    MAX_STEP_V_PU, MAX_STEP_DELTA_RAD, NULLSPACE_RANK_TOL, LINESEARCH_CONSTRAINT_PENALTY,
)
from estimation.bad_data_detection import compute_chi2_and_lnr, unattackable_jacobian, nullspace_basis
from attacks.a4_synthesis import synthesize_a4_attack, random_control

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
OUT_PATH = BASE_DIR / "validation" / "phase8_a4_a5_boundary_scan_results.csv"

TARGET_I = 46
SOLVE_TIMEOUT_S = 20.0
ETA_CAP_MIN = 30.0
A4_MAGNITUDES = [0.001, 0.003, 0.01, 0.02]
A5_MAGNITUDES = [0.01, 0.05, 0.10]
N_SPARSE_METERS = 20
NEAR_ZERO_NORM_RATIO = 0.05   # below this -> NO_EFFECT
NEAR_ONE_LO, NEAR_ONE_HI = 0.5, 1.5  # "norm_ratio near 1" band for the final summary
EMPIRICAL_J99 = 207272.6213
EMPIRICAL_LNR99 = 158.7156
ZERO_REDUNDANCY_REAL_METER_IDS = {"M004_P", "M004_Q", "M040_P", "HEAD_Q_A", "HEAD_Q_B", "HEAD_Q_C"}


def warmstart_capped_solve(est, model, state, z_real, real_ts, x0_v, x0_delta, timeout_s):
    """Verbatim replica of StateEstimator.estimate()'s solve loop,
    warm-started from (x0_v, x0_delta) instead of flat start, with an
    explicit wall-clock deadline check at the top of every iteration.
    Returns (converged, iterations, x_v, x_delta, J, timed_out,
    elapsed_s)."""

    hour = int(real_ts[11:13])
    weighted_entries = est._weighted_entries_for_hour(hour)
    z = np.array([e["z_value"] if "z_value" in e else z_real[e["id"]] for e in weighted_entries])
    sigma = np.array([e["sigma"] for e in weighted_entries])
    W = 1.0 / (sigma ** 2)
    alpha = 1.0 / W.max()
    sqrt_Ws = np.sqrt(alpha * W)

    all_entries = weighted_entries + model.zi_entries
    m = len(weighted_entries)
    p = est.p_constraints
    s = est._state_scale
    v_base_scale = s[:state.n_phase_nodes]

    x_v = np.array(x0_v, dtype=float)
    x_delta = np.array(x0_delta, dtype=float)

    def _merit(xv, xd):
        at_v = complex_v_from_state(state, xv, xd)
        H_c, h0_c = J.build_jacobian(all_entries, model.Y, model.node_index, state, at_v=at_v)
        dz_c = z - h0_c[:m]
        dc_c = -h0_c[m:]
        merit_val = float(np.sum(W * dz_c ** 2) + LINESEARCH_CONSTRAINT_PENALTY * np.sum(dc_c ** 2))
        return merit_val, H_c, h0_c

    t_start = time.time()
    converged = False
    timed_out = False
    lm_lambda = LM_LAMBDA_INIT
    current_merit, H_all, h0_all = _merit(x_v, x_delta)

    for iterations in range(1, 31):
        if time.time() - t_start > timeout_s:
            timed_out = True
            break

        H_meas, h0_meas = H_all[:m], h0_all[:m]
        C, h0_zi = H_all[m:], h0_all[m:]
        dz = z - h0_meas
        dc = -h0_zi
        Hs_meas = H_meas * s[None, :]
        Cs = C * s[None, :]

        y_particular, *_ = np.linalg.lstsq(Cs, dc, rcond=None)
        _, Sigma, Vt = np.linalg.svd(Cs, full_matrices=True)
        rank_c = int(np.sum(Sigma > NULLSPACE_RANK_TOL * Sigma[0])) if Sigma.size else 0
        Z = Vt[rank_c:].T

        A_reduced = (Hs_meas * sqrt_Ws[:, None]) @ Z
        b_reduced = sqrt_Ws * (dz - Hs_meas @ y_particular)
        n_w = A_reduced.shape[1]

        improved = False
        for _ in range(LM_MAX_TRIES):
            A_aug = np.vstack([A_reduced, np.sqrt(lm_lambda) * np.eye(n_w)])
            b_aug = np.concatenate([b_reduced, np.zeros(n_w)])
            w, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
            y_full = y_particular + Z @ w

            for frac in STEP_FRACTIONS:
                y = frac * y_full
                dx = s * y
                dx_v0, dx_delta0 = dx[:state.n_phase_nodes], dx[state.n_phase_nodes:]
                step_ratio = max(
                    np.max(np.abs(dx_v0) / v_base_scale) / MAX_STEP_V_PU,
                    np.max(np.abs(dx_delta0)) / MAX_STEP_DELTA_RAD,
                    1.0,
                )
                cand_dx_v, cand_dx_delta = dx_v0 / step_ratio, dx_delta0 / step_ratio
                cand_v = x_v + cand_dx_v
                cand_delta = x_delta + cand_dx_delta
                cand_merit, cand_H, cand_h0 = _merit(cand_v, cand_delta)

                if cand_merit < current_merit:
                    best = (cand_merit, cand_v, cand_delta, cand_dx_v, cand_dx_delta, cand_H, cand_h0)
                    lm_lambda = max(lm_lambda / LM_DECREASE_FACTOR, LM_LAMBDA_MIN)
                    improved = True
                    break
            if improved:
                break
            lm_lambda = min(lm_lambda * LM_INCREASE_FACTOR, LM_LAMBDA_MAX)

        if not improved:
            dx_v = np.zeros(state.n_phase_nodes)
            dx_delta = np.zeros(state.dim - state.n_phase_nodes)
        else:
            current_merit, x_v, x_delta, dx_v, dx_delta, H_all, h0_all = best

        max_dx_v_pu = float(np.max(np.abs(dx_v) / v_base_scale))
        max_dx_delta_rad = float(np.max(np.abs(dx_delta)))
        constraint_residual_iter = float(np.max(np.abs(h0_all[m:]))) if p else 0.0

        if (max_dx_v_pu < DEFAULT_TOL_V_PU and max_dx_delta_rad < DEFAULT_TOL_DELTA_RAD
                and constraint_residual_iter < DEFAULT_TOL_CONSTRAINT_KW):
            converged = True
            break

    Jval = float(np.sum(W * (z - h0_all[:m]) ** 2))
    return converged, iterations, x_v, x_delta, Jval, timed_out, time.time() - t_start


def label_row(converged, timed_out, norm_ratio, pass_chi2, pass_lnr):
    if timed_out or not converged:
        return "SOLVER_FAIL"
    if abs(norm_ratio) < NEAR_ZERO_NORM_RATIO:
        return "NO_EFFECT"
    if pass_chi2 and pass_lnr:
        return "STEALTH"
    return "DETECTED"


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    v_base_full_np = None  # filled after model build

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    v_base_full = np.array([state.v_base[n] for n in state.all_node_names])
    meas_ids = [e["id"] for e in model.meas_entries]
    fixed_idx = np.array([meas_ids.index(mid) for mid in ZERO_REDUNDANCY_REAL_METER_IDS])

    row = df.iloc[TARGET_I]
    real_ts = str(row["real_timestamp"])
    z_real = {c: float(row[c]) for c in z_cols}

    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)
    state.x0_v, state.x0_delta = flat_v, flat_delta
    result_clean = est.estimate(z_real, real_ts)
    stats_clean = compute_chi2_and_lnr(est, result_clean, z_real, real_ts)
    jac = unattackable_jacobian(est, result_clean, real_ts)
    Hs_attack, Hs_unattack, state_scale = jac["Hs_meas"], jac["Hs_unattackable"], jac["state_scale"]
    print(f"clean: J={stats_clean['J']:.4e} lnr={stats_clean['lnr']:.4f} converged={result_clean.converged}",
          flush=True)

    rng = np.random.default_rng(777 + 100)
    records = []
    solve_times = []
    n_done = 0
    t_run_start = time.time()

    def report_progress(total_planned):
        elapsed = time.time() - t_run_start
        if n_done == 0:
            return
        rate = elapsed / n_done
        eta_min = rate * (total_planned - n_done) / 60.0
        print(f"  [progress] {n_done} solves done, {elapsed:.1f}s elapsed, {rate:.2f}s/solve, "
              f"ETA for remaining {total_planned-n_done} solves: {eta_min:.1f} min", flush=True)
        return eta_min

    total_planned = len(["M1", "M2"]) * len(A4_MAGNITUDES) * 2 + len(A5_MAGNITUDES) * 2
    print(f"total planned solves: {total_planned}", flush=True)

    def do_solve(z_bad):
        t0 = time.time()
        converged, iters, xv, xd, Jval, timed_out, dt = warmstart_capped_solve(
            est, model, state, z_bad, real_ts, result_clean.x_v, result_clean.x_delta, SOLVE_TIMEOUT_S,
        )
        solve_times.append(dt)
        # LNR needs a fabricated "result" shape; build a tiny shim
        class _R: pass
        r = _R(); r.x_v, r.x_delta, r.converged, r.iterations = xv, xd, converged, iters
        stats = compute_chi2_and_lnr(est, r, z_bad, real_ts)
        dx_v = xv - result_clean.x_v
        dx_delta = xd - result_clean.x_delta
        shift_norm = float(np.hypot(np.linalg.norm(dx_v), np.linalg.norm(dx_delta)))
        return converged, timed_out, Jval, stats["lnr"], shift_norm, dt

    # ---------- Part A: A4 boundary scan ----------
    for mode in ["M1", "M2"]:
        for mag in A4_MAGNITUDES:
            atk = synthesize_a4_attack(
                Hs_attack, Hs_unattack, state_scale, rng, mode, mag, state.n_phase_nodes,
                n_sparse_meters=N_SPARSE_METERS,
            )
            a = atk["a_attackable"]
            c_norm = float(np.linalg.norm(atk["c"]))
            z_bad = dict(z_real)
            for k, mid in enumerate(meas_ids):
                z_bad[mid] = z_real[mid] + a[k]

            converged, timed_out, Jbad, lnr_bad, shift_norm, dt = do_solve(z_bad)
            n_done += 1
            dJ, dLNR = Jbad - stats_clean["J"], lnr_bad - stats_clean["lnr"]
            norm_ratio = shift_norm / c_norm if c_norm > 0 else float("nan")
            label = label_row(converged, timed_out, norm_ratio, Jbad <= EMPIRICAL_J99, lnr_bad <= EMPIRICAL_LNR99)

            a_ctrl = random_control(a, rng)
            z_ctrl = dict(z_real)
            for k, mid in enumerate(meas_ids):
                z_ctrl[mid] = z_real[mid] + a_ctrl[k]
            converged_c, timed_out_c, Jctrl, lnr_ctrl, shift_norm_c, dt_c = do_solve(z_ctrl)
            n_done += 1
            dJ_c, dLNR_c = Jctrl - stats_clean["J"], lnr_ctrl - stats_clean["lnr"]
            norm_ratio_c = shift_norm_c / c_norm if c_norm > 0 else float("nan")
            label_c = label_row(converged_c, timed_out_c, norm_ratio_c, Jctrl <= EMPIRICAL_J99, lnr_ctrl <= EMPIRICAL_LNR99)

            records.append({
                "part": "A4", "mode": mode, "magnitude_pu": mag, "kind": "attack",
                "converged": converged, "timed_out": timed_out, "solve_s": dt,
                "J": Jbad, "dJ": dJ, "lnr": lnr_bad, "dLNR": dLNR,
                "norm_ratio": norm_ratio, "label": label,
            })
            records.append({
                "part": "A4", "mode": mode, "magnitude_pu": mag, "kind": "control",
                "converged": converged_c, "timed_out": timed_out_c, "solve_s": dt_c,
                "J": Jctrl, "dJ": dJ_c, "lnr": lnr_ctrl, "dLNR": dLNR_c,
                "norm_ratio": norm_ratio_c, "label": label_c,
            })
            pd.DataFrame(records).to_csv(OUT_PATH, index=False)
            print(f"[A4/{mode}/{mag}] attack: label={label} dJ={dJ:+.3e} dLNR={dLNR:+.2f} "
                  f"norm_ratio={norm_ratio:.4f} ({dt:.2f}s)  |  control: label={label_c} "
                  f"dJ={dJ_c:+.3e} ({dt_c:.2f}s)", flush=True)
            eta = report_progress(total_planned)
            if eta is not None and eta > ETA_CAP_MIN:
                print(f"ABORTING: ETA {eta:.1f} min exceeds {ETA_CAP_MIN} min cap.", flush=True)
                pd.DataFrame(records).to_csv(OUT_PATH, index=False)
                return

    # ---------- Part B: A5 nonlinear comparison ----------
    # same dense (M1-style) null-space DIRECTION, reused at each magnitude
    Z_un, nullity, _, _ = nullspace_basis(Hs_unattack, rank_tol=1e-8)
    y_dir = Z_un @ rng.standard_normal(nullity)
    raw_pu = float(np.max(np.abs(y_dir[:state.n_phase_nodes])))

    at_v_clean = complex_v_from_state(state, result_clean.x_v, result_clean.x_delta)
    _, h0_attack_clean = J.build_jacobian(model.meas_entries, model.Y, model.node_index, state, at_v=at_v_clean)
    unattack_entries = jac["unattackable_entries"]
    _, h0_unattack_clean = J.build_jacobian(unattack_entries, model.Y, model.node_index, state, at_v=at_v_clean)

    for mag in A5_MAGNITUDES:
        scale = mag / raw_pu
        y = y_dir * scale
        c = state_scale * y
        x_bad_v = result_clean.x_v + c[:state.n_phase_nodes]
        x_bad_delta = result_clean.x_delta + c[state.n_phase_nodes:]
        c_norm = float(np.linalg.norm(c))

        at_v_bad = complex_v_from_state(state, x_bad_v, x_bad_delta)
        _, h0_attack_bad = J.build_jacobian(model.meas_entries, model.Y, model.node_index, state, at_v=at_v_bad)
        _, h0_unattack_bad = J.build_jacobian(unattack_entries, model.Y, model.node_index, state, at_v=at_v_bad)

        residual = h0_unattack_bad - h0_unattack_clean
        max_resid = float(np.max(np.abs(residual)))
        n_newton = 0
        while max_resid > DEFAULT_TOL_CONSTRAINT_KW and n_newton < 5:
            Hs_un_bad, _ = J.build_jacobian(unattack_entries, model.Y, model.node_index, state, at_v=at_v_bad)
            Hs_un_bad = Hs_un_bad * state_scale[None, :]
            dy = np.linalg.pinv(Hs_un_bad, rcond=1e-8) @ (-residual)
            dc = state_scale * dy
            c = c + dc
            x_bad_v = result_clean.x_v + c[:state.n_phase_nodes]
            x_bad_delta = result_clean.x_delta + c[state.n_phase_nodes:]
            at_v_bad = complex_v_from_state(state, x_bad_v, x_bad_delta)
            _, h0_unattack_bad = J.build_jacobian(unattack_entries, model.Y, model.node_index, state, at_v=at_v_bad)
            residual = h0_unattack_bad - h0_unattack_clean
            max_resid = float(np.max(np.abs(residual)))
            n_newton += 1
        _, h0_attack_bad = J.build_jacobian(model.meas_entries, model.Y, model.node_index, state, at_v=at_v_bad)

        a_att = h0_attack_bad - h0_attack_clean
        z_bad = dict(z_real)
        for k, mid in enumerate(meas_ids):
            z_bad[mid] = z_real[mid] + a_att[k]

        converged, timed_out, Jbad, lnr_bad, shift_norm, dt = do_solve(z_bad)
        n_done += 1
        dJ, dLNR = Jbad - stats_clean["J"], lnr_bad - stats_clean["lnr"]
        norm_ratio = shift_norm / c_norm if c_norm > 0 else float("nan")
        label = label_row(converged, timed_out, norm_ratio, Jbad <= EMPIRICAL_J99, lnr_bad <= EMPIRICAL_LNR99)

        a_ctrl = random_control(a_att, rng)
        z_ctrl = dict(z_real)
        for k, mid in enumerate(meas_ids):
            z_ctrl[mid] = z_real[mid] + a_ctrl[k]
        converged_c, timed_out_c, Jctrl, lnr_ctrl, shift_norm_c, dt_c = do_solve(z_ctrl)
        n_done += 1
        dJ_c, dLNR_c = Jctrl - stats_clean["J"], lnr_ctrl - stats_clean["lnr"]
        norm_ratio_c = shift_norm_c / c_norm if c_norm > 0 else float("nan")
        label_c = label_row(converged_c, timed_out_c, norm_ratio_c, Jctrl <= EMPIRICAL_J99, lnr_ctrl <= EMPIRICAL_LNR99)

        records.append({
            "part": "A5", "mode": "M1(nonlinear)", "magnitude_pu": mag, "kind": "attack",
            "converged": converged, "timed_out": timed_out, "solve_s": dt,
            "J": Jbad, "dJ": dJ, "lnr": lnr_bad, "dLNR": dLNR,
            "norm_ratio": norm_ratio, "label": label,
            "unattack_residual_before_newton": max_resid if n_newton == 0 else None,
            "n_newton_corrections": n_newton, "unattack_residual_final": max_resid,
        })
        records.append({
            "part": "A5", "mode": "M1(nonlinear)", "magnitude_pu": mag, "kind": "control",
            "converged": converged_c, "timed_out": timed_out_c, "solve_s": dt_c,
            "J": Jctrl, "dJ": dJ_c, "lnr": lnr_ctrl, "dLNR": dLNR_c,
            "norm_ratio": norm_ratio_c, "label": label_c,
        })
        pd.DataFrame(records).to_csv(OUT_PATH, index=False)
        print(f"[A5/{mag}] n_newton={n_newton} resid_final={max_resid:.4f}  attack: label={label} "
              f"dJ={dJ:+.3e} dLNR={dLNR:+.2f} norm_ratio={norm_ratio:.4f} ({dt:.2f}s)  |  "
              f"control: label={label_c} dJ={dJ_c:+.3e} ({dt_c:.2f}s)", flush=True)
        eta = report_progress(total_planned)
        if eta is not None and eta > ETA_CAP_MIN:
            print(f"ABORTING: ETA {eta:.1f} min exceeds {ETA_CAP_MIN} min cap.", flush=True)
            pd.DataFrame(records).to_csv(OUT_PATH, index=False)
            return

    out = pd.DataFrame(records)
    pd.DataFrame(records).to_csv(OUT_PATH, index=False)
    print()
    print("=== SUMMARY TABLE: size vs label vs dJ vs norm_ratio ===")
    print(out[out["kind"] == "attack"][["part", "mode", "magnitude_pu", "label", "dJ", "dLNR", "norm_ratio"]]
          .to_string(index=False))
    print()
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
