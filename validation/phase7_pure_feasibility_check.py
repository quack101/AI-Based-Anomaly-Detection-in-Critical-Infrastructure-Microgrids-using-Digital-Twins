"""
RES.md D10 Phase 7 CONSTRAINT-FEASIBILITY INVESTIGATION -- Step 1.

Pure feasibility check: minimize ||c(x)||^2 subject to NOTHING else --
no measurement objective at all. Uses ONLY the existing zero-injection
constraint function/Jacobian (observability.jacobian.build_jacobian
over model.zi_entries), from flat start. If this ALSO lands near
86.15 kvar (the identical residual 6 different real datasets produced
via two independent solvers, reports/
phase7_scipy_trustconstr_test_report.txt), it confirms the floor is
pure constraint geometry, independent of ANY data or objective.

Run: python -m validation.phase7_pure_feasibility_check
"""

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    model = build_estimator_model("multiphase")
    state = model.state
    entries = model.zi_entries
    p = len(entries)

    v_base = np.array([state.v_base[name] for name in state.all_node_names])
    s = np.concatenate([v_base, np.ones(state.dim - state.n_phase_nodes)])

    x0 = np.concatenate([np.array(state.x0_v, dtype=float), np.array(state.x0_delta, dtype=float)])
    y0 = x0 / s

    cache = {"y": None, "C": None, "c0": None}

    def _ensure(y):
        if cache["y"] is not None and np.array_equal(cache["y"], y):
            return
        x = s * y
        x_v, x_delta = x[: state.n_phase_nodes], x[state.n_phase_nodes:]
        at_v = complex_v_from_state(state, x_v, x_delta)
        C, c0 = J.build_jacobian(entries, model.Y, model.node_index, state, at_v=at_v)
        cache["y"] = y.copy()
        cache["C"] = C
        cache["c0"] = c0

    def resid(y):
        _ensure(y)
        return cache["c0"]  # target 0

    def jac(y):
        _ensure(y)
        return cache["C"] * s[None, :]

    print("=== STEP 1: pure feasibility check, minimize ||c(x)||^2, flat start ===")
    print("p_constraints:", p, " dim(x):", state.dim)
    print("max|c(x0)| (flat start):", np.max(np.abs(resid(y0))))
    print()

    result = least_squares(resid, y0, jac=jac, method="trf", max_nfev=2000,
                            xtol=1e-14, ftol=1e-14, gtol=1e-14)

    c_final = resid(result.x)
    max_c = float(np.max(np.abs(c_final)))
    print("scipy least_squares status:", result.status, " message:", result.message)
    print("nfev:", result.nfev)
    print("max|c(x_final)| (constraint_residual_inf, pure feasibility):", max_c)
    print("||c(x_final)||_2:", np.linalg.norm(c_final))
    print()

    # save the final point + per-row residuals for Step 2's decomposition
    out_path = BASE_DIR / "validation" / "phase7_pure_feasibility_result.npz"
    np.savez(out_path, y_final=result.x, c_final=c_final,
             entry_ids=np.array([e["id"] for e in entries]))
    print("saved:", out_path)

    print()
    print("=== STEP 2: top 15 |c_i(x_final)| rows, descending ===")
    order = np.argsort(-np.abs(c_final))
    for k in order[:15]:
        print(f"  {entries[k]['id']:20s}  c={c_final[k]: .6f}")


if __name__ == "__main__":
    main()
