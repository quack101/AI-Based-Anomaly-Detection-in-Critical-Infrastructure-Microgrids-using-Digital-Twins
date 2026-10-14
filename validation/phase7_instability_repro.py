"""
RES.md D10 Phase 7 SOLVER-INSTABILITY FIX -- Step 1 repro/diagnostic.

Minimal, single-timestep, single-iteration repro of the catastrophic
constraint-residual blowup found in reports/
phase7_convergence_remediation_report.txt Step 3 (a 0.1% perturbation
from x_true at i=46 causes constraint_residual_inf to jump from 361.5
(delta=0) to 560,763.1 (delta=0.001) within 30 iterations).

Reproduces ONLY the first Gauss-Newton iteration's linear algebra
(estimation/state_estimator.py:364-382) OUTSIDE the full solve loop, to
inspect exactly where the blowup enters: the particular solution
y_particular = lstsq(Cs, dc, rcond=None) (line 375), or the null-space
reduced system (line 380-382 onward, already LM-damped).

Hypothesis under test: y_particular's lstsq call uses numpy's own
machine-precision-based rcond (rcond=None -> cutoff ~ max(M,N)*eps,
~1e-13 relative for this system's shapes), while the EXPLICIT SVD used
to build the null space Z truncates at NULLSPACE_RANK_TOL=1e-8 relative
-- a much more aggressive cutoff. Any singular value of Cs between
these two thresholds is RETAINED (and divided by) when solving for
y_particular, but treated as "null" (contributes nothing, is not
corrected for) when building Z -- i.e. y_particular can be amplified
by inverting against a near-zero singular value that the rest of the
solver has already decided is numerically meaningless.

Run: python -m validation.phase7_instability_repro
"""

import numpy as np
import pandas as pd
from pathlib import Path

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import NULLSPACE_RANK_TOL

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"

SEED = 12345
PICK_I = 46
DELTA_FRAC = 0.001


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_all, x_delta_all = npz["x_v"], npz["x_delta"]

    model = build_estimator_model("multiphase")
    state = model.state
    v_base = np.array([state.v_base[name] for name in state.all_node_names])
    delta_ref = 2.0 * np.pi / 3.0

    row = df.iloc[PICK_I]
    real_ts = row["real_timestamp"]
    z_real = {c: float(row[c]) for c in z_cols}
    xt_v, xt_delta = x_v_all[PICK_I], x_delta_all[PICK_I]

    rng = np.random.default_rng(SEED)
    unit_v = rng.uniform(-1.0, 1.0, size=v_base.shape)
    unit_delta = rng.uniform(-1.0, 1.0, size=xt_delta.shape)
    x0_v = xt_v + DELTA_FRAC * unit_v * v_base
    x0_delta = xt_delta + DELTA_FRAC * unit_delta * delta_ref

    from observability.pseudo_measurements import build_pseudo_entries_for_hour
    hour = int(real_ts[11:13])
    weighted_entries = model.meas_entries + model.reg_pseudo_entries + build_pseudo_entries_for_hour(
        model.bus_phase_by_load, model.per_hour_by_load, hour)
    all_entries = weighted_entries + model.zi_entries
    m = len(weighted_entries)
    p = len(model.zi_entries)

    s = np.concatenate([v_base, np.ones(state.dim - state.n_phase_nodes)])

    at_v = complex_v_from_state(state, x0_v, x0_delta)
    H_all, h0_all = J.build_jacobian(all_entries, model.Y, model.node_index, state, at_v=at_v)

    C, h0_zi = H_all[m:], h0_all[m:]
    dc = -h0_zi
    Cs = C * s[None, :]

    print(f"=== repro: i={PICK_I}, real_ts={real_ts}, delta_frac={DELTA_FRAC} ===")
    print("dc (constraint residual at x0): max|dc| =", np.max(np.abs(dc)), " ||dc||_2 =", np.linalg.norm(dc))
    print()

    # --- current code path: y_particular via plain lstsq, rcond=None ---
    y_particular_current, residuals, rank_lstsq, sv_lstsq = np.linalg.lstsq(Cs, dc, rcond=None)
    print("--- CURRENT CODE: y_particular = np.linalg.lstsq(Cs, dc, rcond=None) ---")
    print("lstsq-reported effective rank:", rank_lstsq, " (Cs shape:", Cs.shape, ")")
    print("||y_particular||_2 (scaled-y space):", np.linalg.norm(y_particular_current))
    print("||s * y_particular||_inf (actual dx units):", np.max(np.abs(s * y_particular_current)))
    print()

    # --- explicit SVD, same matrix, to inspect the spectrum and the
    # rank cutoff mismatch directly ---
    U, Sigma, Vt = np.linalg.svd(Cs, full_matrices=True)
    rank_c_nullspace_tol = int(np.sum(Sigma > NULLSPACE_RANK_TOL * Sigma[0]))
    # numpy's own default rcond cutoff for lstsq(rcond=None): max(M,N)*eps
    lstsq_default_rcond = max(Cs.shape) * np.finfo(Cs.dtype if Cs.dtype.kind == 'f' else float).eps
    rank_c_lstsq_tol = int(np.sum(Sigma > lstsq_default_rcond * Sigma[0]))

    print("--- SVD SPECTRUM OF Cs ---")
    print("Sigma[0] (largest):", Sigma[0])
    print("Sigma[-1] (smallest):", Sigma[-1])
    print(f"NULLSPACE_RANK_TOL={NULLSPACE_RANK_TOL:.1e} relative cutoff -> rank {rank_c_nullspace_tol} (used to build Z)")
    print(f"lstsq default rcond={lstsq_default_rcond:.3e} relative cutoff -> rank {rank_c_lstsq_tol} (used for y_particular)")
    print("MISMATCH (extra singular values retained by y_particular's lstsq "
          "but treated as null by Z):", rank_c_lstsq_tol - rank_c_nullspace_tol)
    print()
    print("Singular values in the mismatch band (between the two cutoffs):")
    lo = NULLSPACE_RANK_TOL * Sigma[0]
    hi = lstsq_default_rcond * Sigma[0]
    band = Sigma[(Sigma > lo * 0) & (Sigma <= lo) & (Sigma > 0)]
    # print the smallest ~10 singular values regardless, for context
    print(" smallest 10 singular values:", Sigma[-10:])
    print(" NULLSPACE_RANK_TOL absolute threshold:", lo)
    print(" lstsq default absolute threshold:", hi)
    mismatch_band = Sigma[(Sigma > hi) & (Sigma <= lo)]
    print(" singular values retained by lstsq but nulled by Z (hi < sigma <= lo):", mismatch_band)
    print()

    # --- proposed fix: truncated-SVD particular solution, SAME rank
    # cutoff as Z (NULLSPACE_RANK_TOL), reusing the SAME SVD ---
    rank_c = rank_c_nullspace_tol
    if rank_c > 0:
        y_particular_fixed = Vt[:rank_c].T @ ((U[:, :rank_c].T @ dc) / Sigma[:rank_c])
    else:
        y_particular_fixed = np.zeros(Cs.shape[1])

    print("--- PROPOSED FIX: truncated-SVD y_particular, same cutoff as Z ---")
    print("||y_particular_fixed||_2 (scaled-y space):", np.linalg.norm(y_particular_fixed))
    print("||s * y_particular_fixed||_inf (actual dx units):", np.max(np.abs(s * y_particular_fixed)))
    print()
    print("RATIO ||y_particular_current|| / ||y_particular_fixed||:",
          np.linalg.norm(y_particular_current) / max(np.linalg.norm(y_particular_fixed), 1e-300))


if __name__ == "__main__":
    main()
