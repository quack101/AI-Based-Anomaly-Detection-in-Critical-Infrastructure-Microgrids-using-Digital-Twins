"""
RES.md D10 Phase 8 step 2 -- A4 attack synthesis (a = Hc), TEST ONLY.

Pure functions, read-only on the estimator's own Jacobian. Does NOT
touch estimation/state_estimator.py and is NOT wired into
streaming/injector.py -- this is a test harness for whether the A4
construction (RES.md D6 A4, Liu-Ning-Reiter Theorem 3.1) survives the
real, nonlinear, equality-constrained AC estimator on this feeder, not
a deployed attack path.

## The construction

c is drawn from the null space of H_unattackable (see
estimation/bad_data_detection.py's unattackable_jacobian /
nullspace_basis -- this module reuses those, doesn't re-derive them),
so that a = H @ c is (to first order) exactly zero on every row the
attacker cannot actually touch on the wire (pseudo-measurements,
zero-injection constraints). Three modes differ in which ADDITIONAL
rows (beyond the unattackable ones) are also forced to zero:

  M1  dense: no additional constraint. c is a generic vector in
      null(H_unattackable); a is generically nonzero on every
      attackable (real meter) row.
  M2  sparse: additionally force a to be ~0 on all but ~n_sparse_meters
      attackable rows, by stacking those extra rows onto
      H_unattackable before taking the null space. This shrinks the
      null space further (every row removed can cost one dimension);
      tried against several random meter subsets, falls back to M1
      (and says so) if no subset yields a nonempty null space.
  M3  fixed: same idea as M2, but the kept meter set is FIXED (the
      caller passes which attackable row indices to keep), not random.

Magnitude: callers want a specific max voltage shift in pu. Since the
state scale's V-block IS v_base (estimator._state_scale), the raw pu
voltage shift of a UNIT draw is just max(|c_raw[:n_phase_nodes]|)
directly (no separate division needed) -- scale c so that quantity
hits the target.
"""

import numpy as np

from estimation.bad_data_detection import nullspace_basis


def synthesize_a4_attack(Hs_attackable, Hs_unattackable, state_scale, rng, mode,
                          target_v_shift_pu, n_phase_nodes,
                          n_sparse_meters=20, fixed_meter_idx=None,
                          max_sparse_attempts=10, rank_tol=1e-8):
    """Hs_attackable, Hs_unattackable: COLUMN-SCALED Jacobians (H * s,
    s = estimator._state_scale -- see estimation/bad_data_detection.py's
    unattackable_jacobian) for the attackable (model.meas_entries) and
    unattackable (pseudo + zero-injection) rows respectively. The null
    space is taken in this SCALED space -- same convention the real
    solver uses for its own null-space step -- then mapped back to
    actual units (c = s * y) before being returned. Taking the null
    space of the RAW (unscaled) Jacobian instead is wrong: confirmed
    empirically, it leaves a residual of hundreds-to-thousands on the
    "zeroed" unattackable rows instead of ~0, because a relative SVD
    rank tolerance on raw units conflates volts (~2400) with radians
    (~0.01-1).
    state_scale: (n,) same array (s). Its V-block IS v_base, so a
    y-space V-component is already in pu directly -- no separate
    v_base argument needed.
    mode: 'M1', 'M2', or 'M3' (M3 requires fixed_meter_idx).

    Returns dict: c (n,), a_attackable (m_attack,), feasible (bool),
    mode_used (str -- may show a fallback), max_unattackable_residual_pu
    (diagnostic -- should be ~0), selected_meter_idx (M2/M3 only),
    achieved_v_shift_pu."""

    m_attack, n = Hs_attackable.shape

    def _draw_from(Hs_extra_rows):
        Hs_stack = Hs_unattackable if Hs_extra_rows is None else np.vstack([Hs_unattackable, Hs_extra_rows])
        Z, nullity, _, _ = nullspace_basis(Hs_stack, rank_tol=rank_tol)
        if nullity == 0:
            return None
        coeffs = rng.standard_normal(nullity)
        return Z @ coeffs  # y-space (scaled) direction

    feasible = True
    selected_meter_idx = None
    mode_used = mode

    if mode == "M1":
        y_dir = _draw_from(None)

    elif mode == "M2":
        y_dir = None
        for _ in range(max_sparse_attempts):
            keep = rng.choice(m_attack, size=min(n_sparse_meters, m_attack), replace=False)
            keep_mask = np.zeros(m_attack, dtype=bool)
            keep_mask[keep] = True
            extra_rows = Hs_attackable[~keep_mask]
            cand = _draw_from(extra_rows)
            if cand is not None:
                y_dir = cand
                selected_meter_idx = np.sort(keep)
                break
        if y_dir is None:
            feasible = False
            mode_used = "M1(fallback_from_M2)"
            y_dir = _draw_from(None)

    elif mode == "M3":
        if fixed_meter_idx is None:
            raise ValueError("M3 requires fixed_meter_idx")
        keep_mask = np.zeros(m_attack, dtype=bool)
        keep_mask[fixed_meter_idx] = True
        extra_rows = Hs_attackable[~keep_mask]
        y_dir = _draw_from(extra_rows)
        if y_dir is None:
            feasible = False
            mode_used = "M1(fallback_from_M3)"
            y_dir = _draw_from(None)
        else:
            selected_meter_idx = np.sort(np.asarray(fixed_meter_idx))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # y-space's V-block IS pu directly (state_scale's V-block is v_base,
    # so x_v = s_v * y_v = v_base * y_v, and x_v/v_base = y_v) -- no
    # separate v_base division needed here, unlike the raw-unit version
    # this replaced.
    raw_pu = float(np.max(np.abs(y_dir[:n_phase_nodes])))
    scale = 0.0 if raw_pu == 0.0 else target_v_shift_pu / raw_pu
    y = y_dir * scale
    c = state_scale * y  # actual units (volts, radians)

    a_attackable = Hs_attackable @ y
    unattack_residual = Hs_unattackable @ y
    max_unattack_residual = float(np.max(np.abs(unattack_residual))) if unattack_residual.size else 0.0

    achieved_v_shift_pu = float(np.max(np.abs(y[:n_phase_nodes])))

    return {
        "c": c,
        "a_attackable": a_attackable,
        "feasible": feasible,
        "mode_used": mode_used,
        "selected_meter_idx": selected_meter_idx,
        "max_unattackable_residual": max_unattack_residual,
        "achieved_v_shift_pu": achieved_v_shift_pu,
    }


def random_control(a_attackable, rng):
    """Same support (same nonzero indices), same L2 norm, RANDOM
    direction -- not derived from H or any nullspace at all. This is
    the negative control: does an arbitrary perturbation of the same
    size, on the same meters, evade detection just as well?"""

    nonzero_mask = a_attackable != 0.0
    norm = float(np.linalg.norm(a_attackable))
    direction = rng.standard_normal(a_attackable.shape[0]) * nonzero_mask
    dir_norm = np.linalg.norm(direction)
    if dir_norm == 0.0 or norm == 0.0:
        return np.zeros_like(a_attackable)
    return direction * (norm / dir_norm)


def random_sign_control(a_attackable, rng):
    """Control 2: the SAME per-channel magnitudes as the real attack
    (|a_i| unchanged on every channel, including the zero ones), each
    with an independently random +/-1 sign. Unlike random_control
    (same L2 norm, new random direction spread arbitrarily across the
    support), this keeps each channel's own perturbation size fixed --
    it only asks whether the per-channel SIGN pattern null(H_unattackable)
    picks is doing any work, holding the per-channel magnitude profile
    exactly as the real attack left it."""

    signs = rng.choice([-1.0, 1.0], size=a_attackable.shape[0])
    return np.abs(a_attackable) * signs
