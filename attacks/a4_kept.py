"""
Branch est-pf-start -- A4/A5 attack directions built INSIDE the
directions the state estimator actually resolves.

validation/phase8_a4_linear_check.py (branch fix-a4-synthesis) showed
that c drawn from null(H_unattackable) alone lies ~99% in directions
the estimator's TSVD step drops (estimation/state_estimator.py, TSVD
path, rcond TSVD_RCOND), so the estimate never followed x_hat + c.
Here everything is built in the estimator's OWN reduced, scaled
coordinates at the clean estimate x_hat:

  y = x / s            (s = estimator._state_scale: v_base for V, 1 for angles)
  y = Z w              (Z: null space of the scaled zero-injection rows Cs,
                        same SVD + NULLSPACE_RANK_TOL as the solver)
  A = sqrt(alpha W) Hs_weighted Z     (the solver's A_reduced)
  K = span of A's right singular vectors with sv > TSVD_RCOND * sv_max
  N = null(Hs_unattackable_weighted Z)  (REGZI_* + PSEUDO_* rows; ZI is
                                         already enforced by Z)

The intersection is taken as the principal vectors of N whose cosine
with K exceeds cos_threshold (default 0.95). If there are none, the
fallback is the direction u in K minimising
||W_un^0.5 H_un c|| / ||W_att^0.5 H_att c||.

Attack vector: a = H_attackable c (model.meas_entries order), with a = 0
on the P_head rows, which the estimator does not weight
(observability/estimator_model.py, exclude_dead_head_p). The angle part
of c is bounded by the size in radians as well as the V part by the
size in pu: c is scaled by min(size / max|y_V|, size / max|y_angle|).
"""

import numpy as np

from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import TSVD_RCOND, NULLSPACE_RANK_TOL, DEFAULT_TOL_CONSTRAINT_KW

MAX_NEWTON = 5


def kept_subspace_basis(est, result, real_timestamp, cos_threshold=0.95, rank_tol=1e-8):
    """Everything the kept-direction attacks need, linearised at
    result (the clean estimate)."""

    model = est.model
    state = model.state
    n = state.n_phase_nodes
    s = est._state_scale
    weighted = est._weighted_entries_for_hour(int(real_timestamp[11:13]))
    m = len(weighted)
    at_v = complex_v_from_state(state, result.x_v, result.x_delta)

    H_all, h0_all = J.build_jacobian(weighted + model.zi_entries, model.Y, model.node_index, state, at_v=at_v)
    sigma = np.array([e["sigma"] for e in weighted])
    W = 1.0 / sigma ** 2
    alpha = 1.0 / W.max()

    Cs = H_all[m:] * s[None, :]
    _, sig_c, Vt_c = np.linalg.svd(Cs, full_matrices=True)
    rank_c = int(np.sum(sig_c > NULLSPACE_RANK_TOL * sig_c[0]))
    Z = Vt_c[rank_c:].T

    Hs_w = H_all[:m] * s[None, :]
    A = (Hs_w * np.sqrt(alpha * W)[:, None]) @ Z
    _, sv, Vt_a = np.linalg.svd(A, full_matrices=False)
    keep = sv > TSVD_RCOND * sv[0]
    Q_K = Vt_a[keep].T

    attackable_ids = {e["id"] for e in model.meas_entries}
    un_mask = np.array([e["id"] not in attackable_ids for e in weighted])
    att_mask = ~un_mask
    Hs_un_red = Hs_w[un_mask] @ Z
    _, sig_u, Vt_u = np.linalg.svd(Hs_un_red, full_matrices=True)
    rank_u = int(np.sum(sig_u > rank_tol * sig_u[0]))
    Q_N = Vt_u[rank_u:].T

    U_m, cosines, _ = np.linalg.svd(Q_N.T @ Q_K, full_matrices=False)
    inter = cosines > cos_threshold
    B = Q_N @ U_m[:, inter]  # w-space, orthonormal; empty if no direction passes

    H_meas, h0_meas = J.build_jacobian(model.meas_entries, model.Y, model.node_index, state, at_v=at_v)
    meas_ids = [e["id"] for e in model.meas_entries]
    zero_mask = np.array([e["quantity"] == "P_head" for e in model.meas_entries])

    # fallback direction: min ||W_un^.5 H_un c|| / ||W_att^.5 H_att c|| over c in K
    P = (Hs_w[un_mask] / sigma[un_mask][:, None]) @ Z @ Q_K
    Qm = (Hs_w[att_mask] / sigma[att_mask][:, None]) @ Z @ Q_K
    _, sq, Vq = np.linalg.svd(Qm, full_matrices=False)
    ok = sq > rank_tol * sq[0]
    T = Vq[ok].T / sq[ok][None, :]  # u = T v gives ||Qm u|| = ||v||
    _, sp, Vp = np.linalg.svd(P @ T, full_matrices=False)
    u_fb = T @ Vp[-1]
    fallback_ratio = float(sp[-1])
    w_fallback = Q_K @ u_fb

    return {
        "n": n, "s": s, "Z": Z, "Q_N": Q_N, "Q_K": Q_K, "cosines": cosines, "B": B,
        "dim_N": Q_N.shape[1], "dim_K": Q_K.shape[1], "n_intersection": int(inter.sum()),
        "H_meas": H_meas, "meas_ids": meas_ids, "zero_mask": zero_mask,
        "sigma_meas": np.array([e["sigma"] for e in model.meas_entries]),
        "w_fallback": w_fallback, "fallback_ratio": fallback_ratio,
        "Hs_un_weighted": Hs_w[un_mask], "sigma_un": sigma[un_mask],
    }


def _scale_to_size(y, n, size):
    max_v = float(np.max(np.abs(y[:n])))
    max_a = float(np.max(np.abs(y[n:]))) if y.size > n else 0.0
    scale_v = size / max_v if max_v > 0 else np.inf
    scale_a = size / max_a if max_a > 0 else np.inf
    scale = min(scale_v, scale_a)
    return y * scale, bool(scale_a < scale_v)


def synthesize_kept_attack(basis, rng, size):
    """A4 (linear) attack inside the kept directions. Returns dict with
    c (actual units), a (meas_entries order, 0 on P_head), y, angle
    bound binding flag, which branch was used, and the predicted dJ
    (= ||W_un^0.5 H_un c||^2, the weighted change the estimator cannot
    absorb on the unattackable rows; ~0 for the intersection branch)."""

    n, s, Z = basis["n"], basis["s"], basis["Z"]
    if basis["n_intersection"] > 0:
        g = rng.standard_normal(basis["B"].shape[1])
        w = basis["B"] @ g
        branch = "intersection"
    else:
        w = basis["w_fallback"] * (1.0 if rng.standard_normal() >= 0 else -1.0)
        branch = "fallback_min_ratio"
    y, binds = _scale_to_size(Z @ w, n, size)
    c = s * y
    a = basis["H_meas"] @ c
    a[basis["zero_mask"]] = 0.0
    un_resid = (basis["Hs_un_weighted"] @ y) / basis["sigma_un"]
    return {"c": c, "y": y, "a": a, "angle_bound_binds": binds, "branch": branch,
            "predicted_dJ": float(un_resid @ un_resid),
            "frac_in_K": float(np.linalg.norm(basis["Q_K"].T @ (Z.T @ y)) / np.linalg.norm(y))}


def random_weighted_control(a, sigma, rng):
    """Control 1: random direction on a's support with the SAME weighted
    norm ||W^0.5 a|| (W = 1/sigma^2) as the attack, not the same L2 norm."""

    support = a != 0
    g = np.zeros_like(a)
    g[support] = rng.standard_normal(int(support.sum()))
    target = np.linalg.norm(a[support] / sigma[support])
    gw = np.linalg.norm(g[support])
    return g * sigma * (target / gw) if gw > 0 else g


def a5_from_c(model, state, result_clean, unattack_entries, basis, c):
    """Exact nonlinear attack from the kept-direction c, Newton-corrected
    on the unattackable rows (same procedure as the old scan's
    build_a5_attack, applied to this c)."""
    n, s = state.n_phase_nodes, basis["s"]

    def h(entries, c_):
        at_v = complex_v_from_state(state, result_clean.x_v + c_[:n], result_clean.x_delta + c_[n:])
        return J.build_jacobian(entries, model.Y, model.node_index, state, at_v=at_v)

    _, h_un_clean = h(unattack_entries, np.zeros_like(c))
    _, h_att_clean = h(model.meas_entries, np.zeros_like(c))
    H_un_bad, h_un_bad = h(unattack_entries, c)
    resid = h_un_bad - h_un_clean
    max_resid = float(np.max(np.abs(resid)))
    n_newton = 0
    while max_resid > DEFAULT_TOL_CONSTRAINT_KW and n_newton < MAX_NEWTON:
        dy = np.linalg.pinv(H_un_bad * s[None, :], rcond=1e-8) @ (-resid)
        c = c + s * dy
        H_un_bad, h_un_bad = h(unattack_entries, c)
        resid = h_un_bad - h_un_clean
        max_resid = float(np.max(np.abs(resid)))
        n_newton += 1
    _, h_att_bad = h(model.meas_entries, c)
    a = h_att_bad - h_att_clean
    a[basis["zero_mask"]] = 0.0
    y = c / s
    frac_in_K = float(np.linalg.norm(basis["Q_K"].T @ (basis["Z"].T @ y)) / np.linalg.norm(y))
    return a, c, n_newton, max_resid, frac_in_K
