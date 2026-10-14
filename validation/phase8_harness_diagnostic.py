"""
RES.md D10 Phase 8 step 2c -- DIAGNOSTIC ONLY, point i=46. Checks
whether the A4 test harness reproduces Liu-Ning-Reiter Theorem 3.1
(a=Hc => x_bad-x_hat=c, residual unchanged) at tiny sizes. Does not
modify estimation/state_estimator.py.

NOTE: the task asked to use "the 229-dim nullspace" from an earlier
report. That number was measured with an UNSCALED Jacobian and was
wrong (flagged and corrected in the immediately-preceding turn) --
the correct, column-scaled nullity at this point is 113. This script
uses 113, not 229; see printed output for the direct re-verification.

Run: python -m validation.phase8_harness_diagnostic
"""

import numpy as np
import pandas as pd
from pathlib import Path

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr, unattackable_jacobian, nullspace_basis
from attacks.a4_synthesis import synthesize_a4_attack

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
TARGET_I = 46
SIZES = [1e-6, 1e-5, 1e-4, 1e-3]


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=50)

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    meas_ids = [e["id"] for e in model.meas_entries]

    row = df.iloc[TARGET_I]
    real_ts = str(row["real_timestamp"])
    z_real = {c: float(row[c]) for c in z_cols}

    result_clean = est.estimate(z_real, real_ts)
    stats_clean = compute_chi2_and_lnr(est, result_clean, z_real, real_ts)
    x_hat_v, x_hat_delta = result_clean.x_v, result_clean.x_delta
    print(f"=== clean baseline ===")
    print(f"converged={result_clean.converged} iters={result_clean.iterations} "
          f"J_clean={stats_clean['J']:.6e} lnr_clean={stats_clean['lnr']:.4f}")
    print()

    # ---------------- STEP 1: H consistency ----------------
    print("=== STEP 1: H consistency (analytic vs finite-difference) ===")
    at_v = complex_v_from_state(state, x_hat_v, x_hat_delta)
    H_attack, h0_attack = J.build_jacobian(model.meas_entries, model.Y, model.node_index, state, at_v=at_v)

    eps = 1e-3  # volts / rad
    n = H_attack.shape[1]
    H_fd = np.zeros_like(H_attack)
    for col in range(n):
        xv_p, xd_p = x_hat_v.copy(), x_hat_delta.copy()
        xv_m, xd_m = x_hat_v.copy(), x_hat_delta.copy()
        if col < state.n_phase_nodes:
            xv_p[col] += eps; xv_m[col] -= eps
        else:
            d = col - state.n_phase_nodes
            xd_p[d] += eps; xd_m[d] -= eps
        _, h0_p = J.build_jacobian(model.meas_entries, model.Y, model.node_index, state,
                                    at_v=complex_v_from_state(state, xv_p, xd_p))
        _, h0_m = J.build_jacobian(model.meas_entries, model.Y, model.node_index, state,
                                    at_v=complex_v_from_state(state, xv_m, xd_m))
        H_fd[:, col] = (h0_p - h0_m) / (2 * eps)

    denom = np.where(np.abs(H_attack) > 1e-9, np.abs(H_attack), 1.0)
    rel_err = np.abs(H_fd - H_attack) / denom
    max_rel_err = float(np.max(rel_err))
    worst = np.unravel_index(np.argmax(rel_err), rel_err.shape)
    print(f"max relative error (analytic vs FD), all {H_attack.shape[0]}x{n} entries: {max_rel_err:.3e}")
    print(f"  worst entry: row={worst[0]} ({meas_ids[worst[0]]}), col={worst[1]}, "
          f"analytic={H_attack[worst]:.6f}, fd={H_fd[worst]:.6f}")
    print(f"row order: weighted_entries order, meas_entries FIRST ({len(meas_ids)} rows) -- "
          f"matches est._weighted_entries_for_hour's concatenation order, which the real solver uses verbatim.")
    print(f"state ordering: columns 0..{state.n_phase_nodes-1} = V-block (state.v_index order), "
          f"{state.n_phase_nodes}..{n-1} = delta-block (state.delta_index order) -- "
          f"same convention build_jacobian and the solver's main loop both use.")
    print(f"row scaling: H_attack here is RAW (unscaled). The solver scales COLUMNS by state_scale "
          f"(s = v_base for V-block, 1.0 for delta-block) before any null-space/LM work -- "
          f"a separate ROW scaling (sqrt(alpha*W), alpha=Sha25 eq.6) is applied only to the WEIGHTED "
          f"measurement block for the WLS fit itself, never to the constraint rows.")
    print(f"zero-injection constraints (p={est.p_constraints}): HARD Lagrangian equality constraints, "
          f"solved via the NULL-SPACE method (estimation/state_estimator.py module docstring) -- "
          f"NOT heavy weights, NOT simple variable elimination. Each iteration: (1) a minimum-norm "
          f"PARTICULAR solution y_particular to C@y=dc via lstsq, (2) an orthonormal NULL-SPACE basis Z "
          f"for C via SVD, (3) the weighted LS fit is solved ONLY over free parameter w with x=y_particular+Z@w -- "
          f"mathematically equivalent to the textbook KKT/Lagrange-multiplier block system, chosen instead "
          f"because forming H^T W H for the KKT system squared this feeder's condition number past float64 precision.")
    print()

    # ---------------- unattackable jacobian / nullspace (corrected) ----------------
    jac = unattackable_jacobian(est, result_clean, real_ts)
    Hs_attack, Hs_unattack, state_scale = jac["Hs_meas"], jac["Hs_unattackable"], jac["state_scale"]
    Z_un, nullity, sv_max, sv_min = nullspace_basis(Hs_unattack, rank_tol=1e-8)
    print(f"*** CORRECTED nullity at this point: {nullity} (NOT 229 -- that number used an "
          f"unscaled Jacobian, a bug found and fixed two turns ago). Using {nullity} here. ***")
    print()

    # ---------------- STEP 2: synthetic noiseless case ----------------
    print("=== STEP 2: synthetic noiseless case, z0 = h(x_hat) on attackable rows ===")
    z0 = dict(z_real)
    for k, mid in enumerate(meas_ids):
        z0[mid] = h0_attack[k]
    state.x0_v, state.x0_delta = list(x_hat_v), list(x_hat_delta)  # warm start AT x_hat
    result_z0 = est.estimate(z0, real_ts)
    stats_z0 = compute_chi2_and_lnr(est, result_z0, z0, real_ts)
    print(f"converged={result_z0.converged} iters={result_z0.iterations} J={stats_z0['J']:.6e} "
          f"lnr={stats_z0['lnr']:.4f}")
    print("NOTE: z0=h(x_hat) only on the 168 ATTACKABLE rows -- the 196 pseudo/reg rows keep their "
          "own historical z_value (independent of x_hat), so J is NOT exactly 0 even here; "
          f"compare to J_clean={stats_clean['J']:.4e} -- this isolates how much of J_clean was real "
          "AMI measurement noise vs structural pseudo-measurement mismatch.")
    print()

    # ---------------- STEP 3 + 4: tiny attacks, both scenarios ----------------
    print("=== STEP 3 + 4: tiny attacks, s in", SIZES, "===")
    rng = np.random.default_rng(999)
    y_dir = Z_un @ rng.standard_normal(nullity)
    v_block_dir = y_dir[:state.n_phase_nodes]   # already pu (state_scale's V-block IS v_base)
    d_block_dir = y_dir[state.n_phase_nodes:]   # already rad (state_scale's delta-block is 1.0)
    max_v_dir = float(np.max(np.abs(v_block_dir)))
    max_d_dir = float(np.max(np.abs(d_block_dir)))
    print(f"single fixed direction drawn once; raw max|V|={max_v_dir:.4f} pu, raw max|delta|={max_d_dir:.4f} rad")
    print("scaling note: cannot hit BOTH max|dV|=s pu AND max|d_angle|=s rad exactly with one scalar "
          "unless the direction's own V/delta maxima are equal (they are not, generally) -- scaled by "
          "the MORE RESTRICTIVE of the two so NEITHER exceeds s; the dominant block hits s exactly, "
          "the other is reported alongside, not silently hidden.")
    print()

    step3_rows = []
    step4_rows = []

    for s in SIZES:
        scale = s / max(max_v_dir, max_d_dir)
        y = y_dir * scale
        c = state_scale * y
        achieved_v = float(np.max(np.abs(y[:state.n_phase_nodes])))
        achieved_d = float(np.max(np.abs(y[state.n_phase_nodes:])))
        c_norm = float(np.linalg.norm(c))

        a_linear = Hs_attack @ y  # = H_attack @ c

        # exact nonlinear a and unattackable residual (step 4, z-independent)
        x_pert_v = x_hat_v + c[:state.n_phase_nodes]
        x_pert_delta = x_hat_delta + c[state.n_phase_nodes:]
        at_v_pert = complex_v_from_state(state, x_pert_v, x_pert_delta)
        _, h0_attack_pert = J.build_jacobian(model.meas_entries, model.Y, model.node_index, state, at_v=at_v_pert)
        a_exact = h0_attack_pert - h0_attack
        rel_nonlin = float(np.linalg.norm(a_exact - a_linear) / np.linalg.norm(a_linear))

        unattack_entries = jac["unattackable_entries"]
        _, h0_un_clean = J.build_jacobian(unattack_entries, model.Y, model.node_index, state, at_v=at_v)
        _, h0_un_pert = J.build_jacobian(unattack_entries, model.Y, model.node_index, state, at_v=at_v_pert)
        unattack_resid = float(np.max(np.abs(h0_un_pert - h0_un_clean)))
        step4_rows.append({"s": s, "rel_nonlinearity_attackable": rel_nonlin, "max_unattack_residual": unattack_resid})

        for scenario, z_base, J_base in [("synthetic_z0", z0, stats_z0["J"]), ("real_z", z_real, stats_clean["J"])]:
            z_bad = dict(z_base)
            for k, mid in enumerate(meas_ids):
                z_bad[mid] = z_base[mid] + a_linear[k]
            state.x0_v, state.x0_delta = list(x_hat_v), list(x_hat_delta)
            result_bad = est.estimate(z_bad, real_ts)
            stats_bad = compute_chi2_and_lnr(est, result_bad, z_bad, real_ts)

            dx_v = result_bad.x_v - x_hat_v
            dx_delta = result_bad.x_delta - x_hat_delta
            shift_norm = float(np.hypot(np.linalg.norm(dx_v), np.linalg.norm(dx_delta)))
            ratio = shift_norm / c_norm if c_norm > 0 else float("nan")
            dJ = stats_bad["J"] - J_base
            dJ_rel = dJ / J_base if abs(J_base) > 1e-9 else float("nan")

            scenario_base_lnr = stats_z0["lnr"] if scenario == "synthetic_z0" else stats_clean["lnr"]
            dLNR = stats_bad["lnr"] - scenario_base_lnr

            step3_rows.append({
                "s": s, "scenario": scenario, "ratio_shift_over_c": ratio, "dJ_over_Jclean": dJ_rel,
                "dJ": dJ, "dLNR": dLNR, "converged": result_bad.converged, "iters": result_bad.iterations,
                "J_bad": stats_bad["J"], "lnr_bad": stats_bad["lnr"],
            })

    step3 = pd.DataFrame(step3_rows)
    step4 = pd.DataFrame(step4_rows)
    print(step3.to_string(index=False))
    print()
    print("--- step 4: nonlinearity vs s (z-independent) ---")
    print(step4.to_string(index=False))
    print()

    # ---------------- STEP 5: where does the J jump come from (M1/0.001) ----------------
    print("=== STEP 5: J-jump decomposition, M1/0.001 case from the PREVIOUS run ===")
    rng5 = np.random.default_rng(777 + 100)  # same seed as the boundary-scan script, first draw = M1/0.001
    state.x0_v, state.x0_delta = list(x_hat_v), list(x_hat_delta)
    atk = synthesize_a4_attack(Hs_attack, Hs_unattack, state_scale, rng5, "M1", 0.001, state.n_phase_nodes)
    a5 = atk["a_attackable"]
    z_bad5 = dict(z_real)
    for k, mid in enumerate(meas_ids):
        z_bad5[mid] = z_real[mid] + a5[k]
    state.x0_v, state.x0_delta = list(x_hat_v), list(x_hat_delta)
    result_bad5 = est.estimate(z_bad5, real_ts)
    stats_bad5 = compute_chi2_and_lnr(est, result_bad5, z_bad5, real_ts)
    print(f"reproduced: converged={result_bad5.converged} J_bad={stats_bad5['J']:.6e} "
          f"dJ={stats_bad5['J']-stats_clean['J']:.6e}  lnr_bad={stats_bad5['lnr']:.4f} "
          f"dLNR={stats_bad5['lnr']-stats_clean['lnr']:.6f}  (compare to last run's dJ=3.284e+06, lnr unchanged)")

    weighted_entries = est._weighted_entries_for_hour(int(real_ts[11:13]))
    sigma = np.array([e["sigma"] for e in weighted_entries])
    W = 1.0 / sigma ** 2
    ids = [e["id"] for e in weighted_entries]
    m = len(weighted_entries)

    at_v_clean_full = complex_v_from_state(state, x_hat_v, x_hat_delta)
    at_v_bad5_full = complex_v_from_state(state, result_bad5.x_v, result_bad5.x_delta)
    _, h0_clean_full = J.build_jacobian(weighted_entries, model.Y, model.node_index, state, at_v=at_v_clean_full)
    _, h0_bad5_full = J.build_jacobian(weighted_entries, model.Y, model.node_index, state, at_v=at_v_bad5_full)

    z_full_clean = np.array([e["z_value"] if "z_value" in e else z_real[e["id"]] for e in weighted_entries])
    z_full_bad = np.array([e["z_value"] if "z_value" in e else z_bad5[e["id"]] for e in weighted_entries])

    contrib_clean = W * (z_full_clean - h0_clean_full) ** 2
    contrib_bad = W * (z_full_bad - h0_bad5_full) ** 2
    delta_contrib = contrib_bad - contrib_clean

    order = np.argsort(-delta_contrib)
    print("top 10 rows by INCREASE in w*r^2:")
    from estimation.bad_data_detection import RHO_CRITICAL_THRESHOLD
    stats_full = compute_chi2_and_lnr(est, result_clean, z_real, real_ts, rho_critical_threshold=RHO_CRITICAL_THRESHOLD)
    critical_ids = set(stats_full["excluded_ids"])
    for k in order[:10]:
        tag = "ZERO-REDUNDANCY" if ids[k] in critical_ids else ""
        print(f"  {ids[k]:<16s} delta_contrib={delta_contrib[k]:14.4e}  W={W[k]:.6e}  sigma={sigma[k]:.4f}  {tag}")
    print()

    # ---------------- STEP 6: dLNR flat at +0.0 despite huge dJ ----------------
    print("=== STEP 6: why dLNR stays +0.0 for M1/0.001 despite dJ in the millions ===")
    stats_clean_full = compute_chi2_and_lnr(est, result_clean, z_real, real_ts)
    stats_bad5_full = compute_chi2_and_lnr(est, result_bad5, z_bad5, real_ts)
    nr_clean = stats_clean_full["normalized_residuals"]
    nr_bad = stats_bad5_full["normalized_residuals"]
    ids_full = stats_clean_full["ids"]
    excl_clean = set(stats_clean_full["excluded_ids"])

    def top_lnr(nr, excl):
        masked = np.array([abs(v) if ids_full[i] not in excl and np.isfinite(v) else -1 for i, v in enumerate(nr)])
        idx = int(np.argmax(masked))
        return ids_full[idx], masked[idx]

    id_before, val_before = top_lnr(nr_clean, excl_clean)
    id_after, val_after = top_lnr(nr_bad, excl_clean)
    print(f"max-LNR row BEFORE attack: {id_before} = {val_before:.4f}")
    print(f"max-LNR row AFTER attack:  {id_after} = {val_after:.4f}")
    idx_before_in_after = ids_full.index(id_before)
    print(f"that same row ({id_before}) AFTER the attack: {abs(nr_bad[idx_before_in_after]):.4f}  "
          f"(barely moved? that's why the reported max/LNR is unchanged)")


if __name__ == "__main__":
    main()
