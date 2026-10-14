"""
RES.md D10 Phase 7 classical BDD / Phase 8 pre-work -- chi-square (J)
and Largest Normalized Residual (LNR) bad-data detection statistics
for the real, equality-constrained null-space StateEstimator.

Read-only: computes statistics FROM a converged EstimationResult, does
not modify estimation/state_estimator.py. Reuses the estimator's own
private helpers (_weighted_entries_for_hour, _state_scale) rather than
re-deriving the weighted-entry composition, so this module can never
drift out of sync with what the real solver actually weights.

## Why LNR needs more than result.J

J (chi-square statistic) is already returned by StateEstimator.estimate().
LNR requires the per-measurement NORMALIZED residual, r_N,i = r_i /
sqrt(Omega_ii), where Omega is the residual covariance -- NOT simply
sigma_i (that ignores the fact that the residual is partly "used up"
fitting the state, the textbook distinction this module exists to
get right). For an UNCONSTRAINED WLS estimator this is the standard
Omega = R - H*Cov(x_hat)*H^T. Our estimator is EQUALITY-constrained
(hard zero-injection via null-space, module docstring of
estimation/state_estimator.py) -- the analogous quantity in the
null-space parameterization w (x = x_particular(z) + s*(Z@w), Z the
null-space basis at the converged iterate) is:

  H_reduced = (H_meas * s[None,:]) @ Z      -- m x n_w, dz/dw holding
                                                the linearised
                                                constraint exactly
  Cov(w_hat) = (H_reduced^T W H_reduced)^-1  -- standard WLS covariance
                                                 of the FREE (unconstrained
                                                 given the constraint)
                                                 parameter w
  Omega = diag(1/W) - H_reduced @ Cov(w_hat) @ H_reduced^T
  r_N,i = r_i / sqrt(Omega_ii)

This is the same null-space machinery the solver itself uses at each
iteration (see estimation/state_estimator.py's main loop) evaluated
ONCE at the converged solution, not iterated.

Run as a module for a quick self-test: python -m estimation.bad_data_detection
"""

import numpy as np

from observability.state_vector import complex_v_from_state
from observability import jacobian as J


# REDUNDANCY, rho_i = Omega_ii / sigma_i^2, is the textbook measure of
# how much of measurement i's variance budget is left after the state
# fit uses up what it can (classical SE theory: rho_i in [0,1]; rho_i=0
# is a "critical measurement" for which LNR is undefined BY DEFINITION
# -- any residual there, however small, divides by an arbitrarily small
# denominator). Measured directly on this feeder (i=0, hour 0): 364
# channels split cleanly into 16 with rho_i < 1e-4 (all regulator/
# switch-adjacent REGZI_* entries plus the 3 HEAD_Q_* and M004_P/Q/
# M040_P channels already flagged pathological in the sigma-
# recalibration work) and the remaining 348 with rho_i >= 0.5 -- a
# clean gap, not a judgment call. RHO_CRITICAL_THRESHOLD sits in that
# gap; everything below it is excluded from the reported LNR as
# mathematically untestable, not hidden.
RHO_CRITICAL_THRESHOLD = 1e-3


def compute_chi2_and_lnr(estimator, result, z_real, real_timestamp, rho_critical_threshold=RHO_CRITICAL_THRESHOLD):
    """Returns a dict: J, lnr (max |normalized residual| excluding
    critical/near-critical measurements, rho_i < rho_critical_threshold),
    lnr_all (max over every channel including the critical ones --
    reported separately so nothing is hidden), normalized_residuals
    (array, aligned to weighted_entries order), ids (list, same order),
    omega_diag, redundancy (rho_i per channel), n_negative_omega
    (diagnostic -- should be 0 after the rcond truncation; >0 would
    mean numerical trouble in the covariance calc, not a real finding)."""

    model = estimator.model
    state = model.state

    hour = int(real_timestamp[11:13])
    weighted_entries = estimator._weighted_entries_for_hour(hour)
    z = np.array([e["z_value"] if "z_value" in e else z_real[e["id"]] for e in weighted_entries])
    sigma = np.array([e["sigma"] for e in weighted_entries])
    W = 1.0 / (sigma ** 2)
    m = len(weighted_entries)
    s = estimator._state_scale

    all_entries = weighted_entries + model.zi_entries
    at_v = complex_v_from_state(state, result.x_v, result.x_delta)
    H_all, h0_all = J.build_jacobian(all_entries, model.Y, model.node_index, state, at_v=at_v)

    H_meas, h0_meas = H_all[:m], h0_all[:m]
    C = H_all[m:]

    r = z - h0_meas  # final residual, actual units
    Jval = float(np.sum(W * r ** 2))

    Cs = C * s[None, :]
    _, Sigma_sv, Vt = np.linalg.svd(Cs, full_matrices=True)
    rank_c = int(np.sum(Sigma_sv > 1e-8 * Sigma_sv[0])) if Sigma_sv.size else 0
    Z = Vt[rank_c:].T  # n x n_w null-space basis

    Hs_meas = H_meas * s[None, :]
    H_reduced = Hs_meas @ Z  # m x n_w

    G = H_reduced.T @ (W[:, None] * H_reduced)  # n_w x n_w, = H_reduced^T W H_reduced
    cond_G = float(np.linalg.cond(G))
    # G is severely ill-conditioned (same root cause the solver's own LM
    # damping exists for -- module docstring of estimation/state_estimator.py).
    # A naive pinv (numpy default rcond~1e-15) "inverts" numerically
    # meaningless near-zero singular directions, producing Omega_ii < 0
    # (impossible for a real variance) -- confirmed empirically: 35/364
    # rows went negative before this truncation. Use the SAME rank
    # tolerance (1e-8 relative) the solver itself uses for NULLSPACE_RANK_TOL,
    # for consistency.
    Cov_w = np.linalg.pinv(G, rcond=1e-8)

    M = H_reduced @ Cov_w  # m x n_w
    omega_diag = (1.0 / W) - np.einsum("ij,ij->i", M, H_reduced)

    n_negative_omega = int(np.sum(omega_diag <= 0))
    omega_diag_safe = np.where(omega_diag > 0, omega_diag, np.nan)
    normalized_residuals = r / np.sqrt(omega_diag_safe)
    redundancy = omega_diag / (1.0 / W)  # omega_ii / sigma_i^2

    ids = [e["id"] for e in weighted_entries]
    finite_mask = np.isfinite(normalized_residuals)
    lnr_all = float(np.max(np.abs(normalized_residuals[finite_mask]))) if finite_mask.any() else float("nan")

    excl_mask = redundancy < rho_critical_threshold
    kept_mask = finite_mask & ~excl_mask
    lnr = float(np.max(np.abs(normalized_residuals[kept_mask]))) if kept_mask.any() else float("nan")

    return {
        "J": Jval,
        "lnr": lnr,
        "lnr_all": lnr_all,
        "n_excluded": int(excl_mask.sum()),
        "excluded_ids": [i for i, e in zip(ids, excl_mask) if e],
        "normalized_residuals": normalized_residuals,
        "redundancy": redundancy,
        "ids": ids,
        "omega_diag": omega_diag,
        "n_negative_omega": n_negative_omega,
        "cond_G": cond_G,
        "m_weighted": m,
        "n_w": Z.shape[1],
    }


def unattackable_jacobian(estimator, result, real_timestamp):
    """Returns H_unattackable (the stacked Jacobian of every row the
    attacker CANNOT touch on the wire: reg_pseudo_entries (REGZI_*),
    the per-unmetered-load PSEUDO_* pseudo-entries, and the hard
    zero-injection constraint rows) at the given state, plus the
    attackable Jacobian H_meas (model.meas_entries -- the registry:
    real AMI P/Q/V meters + substation head, the ONLY block
    streaming/injector.py's apply_attack() actually sees on
    telemetry.raw) and the meas_entries list itself (for ids)."""

    model = estimator.model
    state = model.state
    hour = int(real_timestamp[11:13])

    from observability.pseudo_measurements import build_pseudo_entries_for_hour
    pseudo_entries = build_pseudo_entries_for_hour(
        model.bus_phase_by_load, model.per_hour_by_load, hour,
    )
    unattackable_entries = model.reg_pseudo_entries + pseudo_entries + model.zi_entries

    at_v = complex_v_from_state(state, result.x_v, result.x_delta)
    H_un, _ = J.build_jacobian(unattackable_entries, model.Y, model.node_index, state, at_v=at_v)
    H_meas, _ = J.build_jacobian(model.meas_entries, model.Y, model.node_index, state, at_v=at_v)

    # COLUMN-SCALE by the solver's own state scale (s = v_base for the
    # V-block, 1 for the delta-block) before any SVD/null-space work --
    # the SAME convention the real solver always uses (Cs = C*s[None,:]
    # in estimation/state_estimator.py's main loop). Without this, a
    # raw-unit SVD conflates volts (~2400) and radians (~0.01-1) and
    # lets a relative rank_tol keep directions that are NOT actually
    # small once multiplied back through H -- confirmed empirically:
    # a "null space" vector drawn without this scaling left a residual
    # of hundreds to thousands on the supposedly-zeroed unattackable
    # rows, not ~0.
    s = estimator._state_scale
    Hs_un = H_un * s[None, :]
    Hs_meas = H_meas * s[None, :]

    return {
        "H_unattackable": H_un,
        "H_meas": H_meas,
        "Hs_unattackable": Hs_un,
        "Hs_meas": Hs_meas,
        "state_scale": s,
        "meas_ids": [e["id"] for e in model.meas_entries],
        "unattackable_entries": unattackable_entries,
    }


def nullspace_basis(H, rank_tol=1e-8):
    """Orthonormal basis (n x nullity) for {c : H @ c = 0}, via SVD."""

    n = H.shape[1]
    if H.shape[0] == 0:
        return np.eye(n), n, 0.0, 0.0
    _, sv, Vt = np.linalg.svd(H, full_matrices=True)
    rank = int(np.sum(sv > rank_tol * sv[0])) if sv.size else 0
    Z = Vt[rank:].T
    return Z, Z.shape[1], (float(sv[0]) if sv.size else 0.0), (float(sv[-1]) if sv.size else 0.0)


def unattackable_nullspace_dimension(estimator, result, real_timestamp, rank_tol=1e-8):
    """Dimension of {c : H_unattackable @ c = 0}. See unattackable_jacobian
    for exactly what rows are stacked. Evaluated at the real, converged
    state (result.x_v/x_delta), via the same Jacobian builder the
    solver itself uses. Uses the COLUMN-SCALED Jacobian (Hs_unattackable),
    same convention as the solver -- see unattackable_jacobian's note."""

    jac = unattackable_jacobian(estimator, result, real_timestamp)
    H_un = jac["Hs_unattackable"]
    Z, nullity, sv_max, sv_min = nullspace_basis(H_un, rank_tol=rank_tol)

    return {
        "n_unattackable_rows": H_un.shape[0],
        "n_state_vars": H_un.shape[1],
        "rank_H_unattackable": H_un.shape[1] - nullity,
        "nullity": nullity,
        "max_singular_value": sv_max,
        "min_singular_value": sv_min,
    }


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    from observability.estimator_model import build_estimator_model
    from estimation.state_estimator import StateEstimator

    BASE_DIR = Path(__file__).resolve().parent.parent
    CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=50)

    model = build_estimator_model("multiphase")
    est = StateEstimator(model)

    row = df.iloc[0]
    real_ts = str(row["real_timestamp"])
    z_dict = {c: float(row[c]) for c in z_cols}
    result = est.estimate(z_dict, real_ts)

    stats = compute_chi2_and_lnr(est, result, z_dict, real_ts)
    print(f"self-test i=0: J(module)={stats['J']:.4e} J(solver)={result.J:.4e} "
          f"match={abs(stats['J']-result.J)<1e-6}")
    print(f"LNR(excl. {stats['n_excluded']} critical/near-critical)={stats['lnr']:.4f}  "
          f"LNR(all channels)={stats['lnr_all']:.4f}  cond(G)={stats['cond_G']:.3e}  "
          f"n_negative_omega={stats['n_negative_omega']}")

    ns = unattackable_nullspace_dimension(est, result, real_ts)
    print(ns)
