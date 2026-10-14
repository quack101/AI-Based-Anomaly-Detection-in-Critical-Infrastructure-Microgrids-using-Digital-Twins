"""
RES.md D10 Phase 7 fix-jacobian-bugs branch -- DIAGNOSTIC ONLY, step 1.
Gradient check: compares the analytic gradient (as the solver's own
null-space-reduced Gauss-Newton loop uses it) against a finite-
difference directional derivative of cost(w) = 0.5*sum(W*r(w)^2),
r = z - h(x(w)), x(w) = x_particular + s*(Z@w) -- the SAME null-space
parameterization estimation/state_estimator.py's main loop uses.
Does not modify the solver.

Run: python -m validation.phase_gradient_check
"""

import numpy as np
import pandas as pd
from pathlib import Path

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import StateEstimator, NULLSPACE_RANK_TOL

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
TARGET_I = 46
N_DIRS = 20
EPS = 1e-6


def setup(est, model, state, z_real, real_ts, x_v, x_delta):
    hour = int(real_ts[11:13])
    weighted_entries = est._weighted_entries_for_hour(hour)
    z = np.array([e["z_value"] if "z_value" in e else z_real[e["id"]] for e in weighted_entries])
    sigma = np.array([e["sigma"] for e in weighted_entries])
    W = 1.0 / sigma ** 2
    m = len(weighted_entries)
    s = est._state_scale

    all_entries = weighted_entries + model.zi_entries
    at_v = complex_v_from_state(state, x_v, x_delta)
    H_all, h0_all = J.build_jacobian(all_entries, model.Y, model.node_index, state, at_v=at_v)
    H_meas, h0_meas = H_all[:m], h0_all[:m]
    C = H_all[m:]
    Cs = C * s[None, :]
    Hs_meas = H_meas * s[None, :]

    _, Sigma, Vt = np.linalg.svd(Cs, full_matrices=True)
    rank_c = int(np.sum(Sigma > NULLSPACE_RANK_TOL * Sigma[0])) if Sigma.size else 0
    Z = Vt[rank_c:].T
    H_reduced = Hs_meas @ Z  # m x n_w

    r = z - h0_meas
    grad_w = -H_reduced.T @ (W * r)  # d(0.5*sum(W r^2))/dw ; r(w) chain rule: dr/dw=-H_reduced

    def cost_at(w):
        y = Z @ w
        xv = x_v + s[:state.n_phase_nodes] * y[:state.n_phase_nodes]
        xd = x_delta + s[state.n_phase_nodes:] * y[state.n_phase_nodes:]
        at_v2 = complex_v_from_state(state, xv, xd)
        _, h0_2 = J.build_jacobian(weighted_entries, model.Y, model.node_index, state, at_v=at_v2)
        r2 = z - h0_2
        return 0.5 * float(np.sum(W * r2 ** 2))

    return grad_w, cost_at, Z, s, state.n_phase_nodes


def run_check(label, est, model, state, z_real, real_ts, x_v, x_delta):
    grad_w, cost_at, Z, s, nV = setup(est, model, state, z_real, real_ts, x_v, x_delta)
    n_w = Z.shape[1]
    rng = np.random.default_rng(2024)

    print(f"--- {label}: n_w={n_w}, ||grad_w||={np.linalg.norm(grad_w):.4e} ---")

    def test_directions(name, direction_sampler):
        rows = []
        for _ in range(N_DIRS):
            d = direction_sampler()
            d = d / np.linalg.norm(d)
            analytic_deriv = float(grad_w @ d)
            c_p = cost_at(EPS * d)
            c_m = cost_at(-EPS * d)
            c0 = cost_at(np.zeros(n_w))
            fd_deriv = (c_p - c_m) / (2 * EPS)
            denom = max(abs(analytic_deriv), abs(fd_deriv), 1e-12)
            relerr = abs(analytic_deriv - fd_deriv) / denom
            rows.append((analytic_deriv, fd_deriv, relerr))
        rows_arr = np.array(rows)
        worst = rows_arr[np.argmax(rows_arr[:, 2])]
        print(f"  [{name}] median relerr={np.median(rows_arr[:,2]):.3e}  max relerr={np.max(rows_arr[:,2]):.3e}  "
              f"worst: analytic={worst[0]:.4e} fd={worst[1]:.4e}")

    test_directions("all (generic w-space)", lambda: rng.standard_normal(n_w))

    def v_only_dir():
        full = np.zeros(Z.shape[0])
        full[:nV] = rng.standard_normal(nV)
        w = Z.T @ full  # project the V-only full-space vector into w-coordinates
        return w

    def angle_only_dir():
        full = np.zeros(Z.shape[0])
        full[nV:] = rng.standard_normal(Z.shape[0] - nV)
        w = Z.T @ full
        return w

    test_directions("V-states-only (projected)", v_only_dir)
    test_directions("angle-states-only (projected)", angle_only_dir)
    print()


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=50)

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    flat_v, flat_delta = list(state.x0_v), list(state.x0_delta)

    row = df.iloc[TARGET_I]
    real_ts = str(row["real_timestamp"])
    z_real = {c: float(row[c]) for c in z_cols}

    # flat start
    run_check("FLAT START (x0)", est, model, state, z_real, real_ts, np.array(flat_v), np.array(flat_delta))

    # final iterate
    state.x0_v, state.x0_delta = flat_v, flat_delta
    result = est.estimate(z_real, real_ts)
    run_check(f"FINAL ITERATE (converged={result.converged}, iters={result.iterations})",
              est, model, state, z_real, real_ts, result.x_v, result.x_delta)


if __name__ == "__main__":
    main()
