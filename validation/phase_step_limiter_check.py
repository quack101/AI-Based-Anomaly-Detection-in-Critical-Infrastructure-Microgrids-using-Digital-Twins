"""
RES.md D10 Phase 7 fix-jacobian-bugs branch -- DIAGNOSTIC ONLY,
steps 2-5. Point i=46 (A_flat), 300-iteration forced run. Does not
modify estimation/state_estimator.py.

Run: python -m validation.phase_step_limiter_check
"""

import numpy as np
import pandas as pd
from pathlib import Path

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import (
    StateEstimator, DEFAULT_TOL_V_PU, DEFAULT_TOL_DELTA_RAD,
    DEFAULT_TOL_CONSTRAINT_KW, LM_LAMBDA_INIT, LM_MAX_TRIES, LM_DECREASE_FACTOR,
    LM_INCREASE_FACTOR, LM_LAMBDA_MAX, LM_LAMBDA_MIN, STEP_FRACTIONS,
    MAX_STEP_V_PU, MAX_STEP_DELTA_RAD, NULLSPACE_RANK_TOL, LINESEARCH_CONSTRAINT_PENALTY,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
TARGET_I = 46
FORCE_MAX_ITER = 300
CHECKPOINTS_STEP2 = {1, 5, 10, 20, 30, 60}
CHECKPOINTS_STEP5 = {30, 60, 100, 150, 200, 300}
COND_ITER = 30


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_true = npz["x_v"][TARGET_I]

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    v_base_full = np.array([state.v_base[n] for n in state.all_node_names])
    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)

    row = df.iloc[TARGET_I]
    real_ts = str(row["real_timestamp"])
    z_real = {c: float(row[c]) for c in z_cols}

    hour = int(real_ts[11:13])
    weighted_entries = est._weighted_entries_for_hour(hour)
    z = np.array([e["z_value"] if "z_value" in e else z_real[e["id"]] for e in weighted_entries])
    sigma = np.array([e["sigma"] for e in weighted_entries])
    W = 1.0 / sigma ** 2
    alpha = 1.0 / W.max()
    sqrt_Ws = np.sqrt(alpha * W)

    all_entries = weighted_entries + model.zi_entries
    m = len(weighted_entries)
    p = est.p_constraints
    s = est._state_scale
    v_base_scale = s[:state.n_phase_nodes]
    node_names_v = state.all_node_names
    delta_name_by_idx = {idx: name for name, idx in state.delta_index.items()}

    x_v = np.array(flat_v, dtype=float)
    x_delta = np.array(flat_delta, dtype=float)

    def _merit(xv, xd):
        at_v = complex_v_from_state(state, xv, xd)
        H_c, h0_c = J.build_jacobian(all_entries, model.Y, model.node_index, state, at_v=at_v)
        dz_c = z - h0_c[:m]
        dc_c = -h0_c[m:]
        merit_val = float(np.sum(W * dz_c ** 2) + LINESEARCH_CONSTRAINT_PENALTY * np.sum(dc_c ** 2))
        return merit_val, H_c, h0_c

    lm_lambda = LM_LAMBDA_INIT
    current_merit, H_all, h0_all = _merit(x_v, x_delta)

    print("=== STEP 2: applied step vs full (lambda->0, unclipped) GN step norm ===")
    print("=== STEP 3: constraint_residual_iter (kW) at same checkpoints ===")
    print(f"{'iter':<6s}{'cost':<14s}{'cres_kW':<12s}{'applied_step':<16s}{'full_GN_step':<16s}{'ratio':<10s}{'v_err_pu':<10s}")

    trace_all = []
    cond_report = None

    for iterations in range(1, FORCE_MAX_ITER + 1):
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

        full_gn_step_norm = None
        if iterations in CHECKPOINTS_STEP2:
            w_full, *_ = np.linalg.lstsq(A_reduced, b_reduced, rcond=None)  # lambda=0, no clip
            y_full_un = y_particular + Z @ w_full
            dx_full_un = s * y_full_un
            full_gn_step_norm = float(np.hypot(
                np.linalg.norm(dx_full_un[:state.n_phase_nodes] / v_base_scale),
                np.linalg.norm(dx_full_un[state.n_phase_nodes:])))

        if iterations == COND_ITER:
            sv = np.linalg.svd(A_reduced, compute_uv=False)
            _, _, VtA = np.linalg.svd(A_reduced, full_matrices=True)
            cond_num = sv[0] / sv[-1] if sv[-1] > 0 else float("inf")
            n_below = int(np.sum(sv < 1e-6 * sv[0]))
            v_weak = VtA[-1]  # weakest right-singular vector, in w-space
            full_weak = Z @ v_weak  # map to y-space
            idx_sorted = np.argsort(-np.abs(full_weak))[:5]
            weak_states = []
            for idx in idx_sorted:
                if idx < state.n_phase_nodes:
                    weak_states.append((node_names_v[idx], "V", float(full_weak[idx])))
                else:
                    nm = delta_name_by_idx.get(idx, f"col{idx}")
                    weak_states.append((nm, "angle", float(full_weak[idx])))
            cond_report = {
                "cond_number": float(cond_num), "n_below_1e-6": n_below,
                "smallest_10": sv[-10:].tolist(), "largest_10": sv[:10].tolist(),
                "weak_states": weak_states,
            }

        improved = False
        n_lm_tries = 0
        for _ in range(LM_MAX_TRIES):
            n_lm_tries += 1
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

        applied_step_norm = float(np.hypot(np.linalg.norm(dx_v / v_base_scale), np.linalg.norm(dx_delta)))
        constraint_residual_iter = float(np.max(np.abs(h0_all[m:]))) if p else 0.0
        v_err_pu = float(np.max(np.abs(x_v - x_v_true) / v_base_full))
        Jval = float(np.sum(W * (z - h0_all[:m]) ** 2))

        trace_all.append({"iter": iterations, "cost": current_merit, "cres": constraint_residual_iter,
                           "v_err_pu": v_err_pu, "J": Jval, "applied_step_norm": applied_step_norm})

        if iterations in CHECKPOINTS_STEP2:
            ratio = applied_step_norm / full_gn_step_norm if full_gn_step_norm else float("nan")
            print(f"{iterations:<6d}{current_merit:<14.4e}{constraint_residual_iter:<12.4f}"
                  f"{applied_step_norm:<16.6f}{full_gn_step_norm:<16.6f}{ratio:<10.4f}{v_err_pu:<10.5f}")

    print()
    print("=== STEP 4: conditioning of reduced scaled Jacobian A_reduced = sqrt(W)*H*Z at iter=30 ===")
    print(f"condition number: {cond_report['cond_number']:.4e}")
    print(f"n singular values below 1e-6 of largest: {cond_report['n_below_1e-6']}")
    print(f"10 largest singular values: {[f'{v:.4e}' for v in cond_report['largest_10']]}")
    print(f"10 smallest singular values: {[f'{v:.4e}' for v in cond_report['smallest_10']]}")
    print("top 5 states in weakest singular direction (node, block, component):")
    for nm, block, val in cond_report["weak_states"]:
        print(f"  {nm:<10s} {block:<6s} {val:+.4f}")
    print()

    print("=== STEP 5: long-run checkpoints (cost, J, v_err_pu) ===")
    trace_df = pd.DataFrame(trace_all)
    for it in sorted(CHECKPOINTS_STEP5):
        r = trace_df[trace_df["iter"] == it].iloc[0]
        print(f"  iter={it:<5d} cost={r['cost']:.4e}  J={r['J']:.4e}  v_err_pu={r['v_err_pu']:.5f}  cres={r['cres']:.4f}")

    below = trace_df[trace_df["v_err_pu"] < 0.015]
    if len(below):
        print(f"\nv_err_pu first falls below 0.015 pu at iteration {int(below.iloc[0]['iter'])}")
    else:
        print(f"\nv_err_pu NEVER falls below 0.015 pu within {FORCE_MAX_ITER} iterations "
              f"(final value at iter={FORCE_MAX_ITER}: {trace_df.iloc[-1]['v_err_pu']:.5f})")

    # plateau check: compare last 50 iterations' cost spread
    tail = trace_df.tail(50)
    cost_range = tail["cost"].max() - tail["cost"].min()
    cost_rel_range = cost_range / tail["cost"].mean()
    print(f"\nplateau check (last 50 iterations): cost range={cost_range:.4e} "
          f"({100*cost_rel_range:.2f}% of mean) -- {'PLATEAUED' if cost_rel_range < 0.01 else 'STILL CHANGING'}")


if __name__ == "__main__":
    main()
