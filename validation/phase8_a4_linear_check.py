"""
RES.md D10 Phase 8 -- linear checks on the A4-M1 attack a = H c, using
the estimator's OWN Jacobian (weighted rows + zero-injection rows, in
the estimator's own row order, estimation/state_estimator.py), to find
why est.estimate() does not move the state by c.

Point i=46, A4-M1, sizes 0.005 and 0.05; the attack vectors are the
exact ones the scan draws (validation/phase8_a4_a5_scan_frozen.py,
seed SEED+400, same rng call order incl. both controls).

Checks per (linearisation point x_hat | x_true, size):
  a) ||H_unatt c|| / ||H_att c||  (sigma-weighted; ZI rows reported raw)
  b) per attackable row |a_i - (H c)_i| / |(H c)_i|, 10 worst
  c) row alignment: synthesis index -> id vs estimator row -> id
  d) one linearised constrained WLS step with z_clean + a from the
     linearisation point; ||dx_lin - c|| / ||c|| (scaled y-space),
     exact (full-rank pinv) and with the estimator's TSVD rcond
  e) fraction of c (in the estimator's scaled reduced coordinates w)
     lying in directions TSVD drops (sv < TSVD_RCOND * sv_max), plus
     the fraction in the null space of A_reduced itself

Run: python -m validation.phase8_a4_linear_check
"""

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import StateEstimator, TSVD_RCOND, NULLSPACE_RANK_TOL
from estimation.bad_data_detection import unattackable_jacobian
from attacks.a4_synthesis import synthesize_a4_attack, random_control, random_sign_control
from validation import phase8_a4_a5_scan_frozen as scan

POINT = 46
SIZES = [0.005, 0.05]


def estimator_blocks(est, at_v, real_ts):
    """Estimator's own stacked Jacobian at at_v: weighted rows (order of
    _weighted_entries_for_hour) then ZI constraint rows."""
    model = est.model
    weighted = est._weighted_entries_for_hour(int(real_ts[11:13]))
    H_all, h0_all = J.build_jacobian(weighted + model.zi_entries, model.Y, model.node_index, model.state, at_v=at_v)
    return weighted, H_all, h0_all


def linear_checks(est, weighted, H_all, h0_all, z_clean, a_by_id, c, tag):
    model = est.model
    s = est._state_scale
    m = len(weighted)
    ids = [e["id"] for e in weighted]
    sigma = np.array([e["sigma"] for e in weighted])
    W = 1.0 / sigma ** 2
    attackable_ids = {e["id"] for e in model.meas_entries}
    att = np.array([i in attackable_ids for i in ids])
    H_w, C = H_all[:m], H_all[m:]
    Hc_w = H_w @ c
    Cc = C @ c
    out = {"tag": tag}

    # a) unattackable vs attackable response, sigma-weighted
    out["a_norm_W_Hc_att"] = float(np.linalg.norm((Hc_w / sigma)[att]))
    out["a_norm_W_Hc_unatt_weighted"] = float(np.linalg.norm((Hc_w / sigma)[~att]))
    out["a_ratio_unatt_over_att"] = out["a_norm_W_Hc_unatt_weighted"] / out["a_norm_W_Hc_att"]
    out["a_max_abs_Cc_kw"] = float(np.max(np.abs(Cc)))
    out["a_max_abs_Hc_att_raw"] = float(np.max(np.abs(Hc_w[att])))

    # b) a vs Hc per attackable row
    a_w = np.array([a_by_id.get(i, 0.0) for i in ids])
    err = np.abs(a_w - Hc_w)
    rel = err / np.maximum(np.abs(Hc_w), 1e-12)
    out["b_max_rel_err"] = float(np.max(rel[att]))
    out["b_median_rel_err"] = float(np.median(rel[att]))
    worst = np.argsort(-np.where(att, rel, -1))[:10]
    out["b_worst"] = [(ids[k], float(a_w[k]), float(Hc_w[k]), float(rel[k])) for k in worst]
    out["b_a_on_unattackable_rows"] = float(np.max(np.abs(a_w[~att]))) if (~att).any() else 0.0

    # d) one linearised constrained WLS step, same algebra as the solver
    dz_clean = np.array([e["z_value"] if "z_value" in e else z_clean[e["id"]] for e in weighted]) - h0_all[:m]
    dc = -h0_all[m:]
    alpha = 1.0 / W.max()
    sqrt_Ws = np.sqrt(alpha * W)
    Hs, Cs = H_w * s[None, :], C * s[None, :]
    y_part, *_ = np.linalg.lstsq(Cs, dc, rcond=None)
    _, Sig, Vt = np.linalg.svd(Cs, full_matrices=True)
    rank_c = int(np.sum(Sig > NULLSPACE_RANK_TOL * Sig[0]))
    Z = Vt[rank_c:].T
    A = (Hs * sqrt_Ws[:, None]) @ Z
    U, sv, Vt2 = np.linalg.svd(A, full_matrices=False)
    keep = sv > TSVD_RCOND * sv[0]
    y_c = c / s

    def step(b, mask):
        inv = np.zeros_like(sv)
        inv[mask] = 1.0 / sv[mask]
        return (Vt2.T * inv) @ (U.T @ b)

    for name, dz in [("clean", dz_clean), ("attacked", dz_clean + a_w)]:
        b = sqrt_Ws * (dz - Hs @ y_part)
        for mode, mask in [("exact", sv > 1e-14 * sv[0]), ("tsvd", keep)]:
            y = y_part + Z @ step(b, mask)
            out[f"d_y_{name}_{mode}"] = y
    for mode in ["exact", "tsvd"]:
        dy = out.pop(f"d_y_attacked_{mode}") - out.pop(f"d_y_clean_{mode}")
        out[f"d_rel_mismatch_{mode}"] = float(np.linalg.norm(dy - y_c) / np.linalg.norm(y_c))
        out[f"d_max_abs_dy_v_pu_{mode}"] = float(np.max(np.abs(dy[:model.state.n_phase_nodes])))
    # a-only step (no clean residual), TSVD -- isolates the attack's own response
    y_a = Z @ step(sqrt_Ws * a_w, keep)
    out["d_rel_mismatch_a_only_tsvd"] = float(np.linalg.norm(y_a - y_c) / np.linalg.norm(y_c))

    # e) TSVD-dropped fraction of c in the reduced coordinates
    w_c = Z.T @ y_c
    out["e_frac_c_outside_constraint_nullspace"] = float(np.linalg.norm(y_c - Z @ w_c) / np.linalg.norm(y_c))
    coef = Vt2 @ w_c
    out["e_n_w"] = Z.shape[1]
    out["e_rank_A_kept"] = int(keep.sum())
    out["e_sv_max"], out["e_sv_min"] = float(sv[0]), float(sv[-1])
    out["e_frac_c_in_dropped_sv"] = float(np.linalg.norm(coef[~keep]) / np.linalg.norm(w_c))
    w_in_rows = Vt2.T @ coef
    out["e_frac_c_in_null_of_A"] = float(np.linalg.norm(w_c - w_in_rows) / np.linalg.norm(w_c))
    out["e_frac_c_kept"] = float(np.linalg.norm(coef[keep]) / np.linalg.norm(w_c))
    return out


def main():
    telem_cols = pd.read_csv(scan.CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(scan.CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols,
                     nrows=1440)
    p_cols = [c for c in df.columns if c.endswith("_P")]
    strata = scan.stratified_picks(df, p_cols)
    points = sorted(set(strata["low"] + strata["mid"] + strata["high"]))[:scan.N_POINTS]
    k_point = points.index(POINT)
    row = df.iloc[POINT]
    real_ts = str(row["real_timestamp"])
    z_clean = {c: float(row[c]) for c in z_cols}

    xt = np.load(scan.CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)
    n = state.n_phase_nodes
    assert [str(r[0]) for r in xt["v_index"]] == list(state.all_node_names) or True
    v_names = [str(r[0]) for r in xt["v_index"]]
    d_names = [str(r[0]) for r in xt["delta_index"]]
    delta_names = [None] * (state.dim - n)
    for name, idx in state.delta_index.items():
        delta_names[idx - n] = name
    print("x_true V order matches state:", v_names == list(state.all_node_names),
          " delta order matches:", d_names == delta_names, flush=True)
    vpos = {nm: k for k, nm in enumerate(v_names)}
    dpos = {nm: k for k, nm in enumerate(d_names)}
    x_true_v = np.array([xt["x_v"][POINT][vpos[nm]] for nm in state.all_node_names])
    x_true_d = np.array([xt["x_delta"][POINT][dpos[nm]] for nm in delta_names])

    result_clean = est.estimate(z_clean, real_ts)
    print(f"clean: converged={result_clean.converged} it={result_clean.iterations} J={result_clean.J:.3f}")
    jac = unattackable_jacobian(est, result_clean, real_ts)
    Hs_attack, Hs_unattack, state_scale = jac["Hs_meas"], jac["Hs_unattackable"], jac["state_scale"]
    meas_ids = jac["meas_ids"]
    rng = np.random.default_rng(scan.SEED + 400 + k_point)

    attacks = {}
    for mag in scan.MAGNITUDES:
        atk = synthesize_a4_attack(Hs_attack, Hs_unattack, state_scale, rng, "M1", mag, n)
        random_control(atk["a_attackable"], rng)
        random_sign_control(atk["a_attackable"], rng)
        attacks[mag] = atk
        # A5 draws also consume rng, but they come after all A4 sizes -- not needed here.

    weighted_hat, H_hat, h0_hat = estimator_blocks(
        est, complex_v_from_state(state, result_clean.x_v, result_clean.x_delta), real_ts)
    weighted_true, H_true, h0_true = estimator_blocks(est, complex_v_from_state(state, x_true_v, x_true_d), real_ts)
    w_ids = [e["id"] for e in weighted_hat]

    # c) row alignment
    a0 = attacks[SIZES[0]]["a_attackable"]
    nz = np.flatnonzero(np.abs(a0) > 0)
    idx_mismatch = [(k, meas_ids[k], w_ids[k]) for k in nz if k < len(w_ids) and meas_ids[k] != w_ids[k]]
    print(f"\n[c] a nonzero on {len(nz)}/{len(a0)} synthesis rows; synthesis id != estimator id AT SAME INDEX: "
          f"{len(idx_mismatch)}; first 5: {idx_mismatch[:5]}")
    by_id_missing = [meas_ids[k] for k in nz if meas_ids[k] not in set(w_ids)]
    print(f"[c] by-ID mapping (what the scan uses, z_dict[id] += a[k]): ids not among estimator weighted rows: "
          f"{by_id_missing}  |a| on them: {[float(abs(a0[meas_ids.index(i)])) for i in by_id_missing]}")
    print(f"[c] meas_entries order == weighted meter-row order after dropping P_head: "
          f"{[i for i in meas_ids if not i.startswith('HEAD_P')] == [i for i in w_ids if i in set(meas_ids)]}")

    rows = []
    for mag in SIZES:
        atk = attacks[mag]
        a_by_id = dict(zip(meas_ids, atk["a_attackable"]))
        c = atk["c"]
        print(f"\n===== size {mag} pu: max|c_V|/vb={np.max(np.abs(c[:n] / state_scale[:n])):.4f} "
              f"max|c_ang|={np.max(np.abs(c[n:])):.4f} rad, synthesis max_unattack_resid={atk['max_unattackable_residual']:.3e}")
        for tag, wl, H_all, h0_all in [("x_hat", weighted_hat, H_hat, h0_hat), ("x_true", weighted_true, H_true, h0_true)]:
            o = linear_checks(est, wl, H_all, h0_all, z_clean, a_by_id, c, tag)
            worst = o.pop("b_worst")
            print(f"--- linearised at {tag}")
            for kk, v in o.items():
                if kk != "tag":
                    print(f"   {kk}: {v:.6g}" if isinstance(v, float) else f"   {kk}: {v}")
            print("   b) 10 worst rows (id, a, Hc, rel_err):")
            for w in worst:
                print(f"      {w[0]:>14s}  a={w[1]: .5e}  Hc={w[2]: .5e}  rel={w[3]:.3e}")
            o["size"] = mag
            rows.append(o)

        # actual nonlinear solve from x_hat, for reference
        z_bad = dict(z_clean)
        for k, mid in enumerate(meas_ids):
            z_bad[mid] = z_clean[mid] + atk["a_attackable"][k]
        state.x0_v, state.x0_delta = list(result_clean.x_v), list(result_clean.x_delta)
        res = est.estimate(z_bad, real_ts)
        dy = np.concatenate([res.x_v - result_clean.x_v, res.x_delta - result_clean.x_delta]) / state_scale
        y_c = c / state_scale
        print(f"--- est.estimate from x_hat: it={res.iterations} conv={res.converged} J={res.J:.3f} "
              f"(dJ={res.J - result_clean.J:.3f})  ||dx-c||/||c||={np.linalg.norm(dy - y_c) / np.linalg.norm(y_c):.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(scan.BASE_DIR / "validation" / "phase8_a4_linear_check.csv", index=False)


if __name__ == "__main__":
    main()
