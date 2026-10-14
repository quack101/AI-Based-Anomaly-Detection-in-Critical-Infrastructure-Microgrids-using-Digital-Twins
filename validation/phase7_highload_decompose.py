"""
RES.md D10 Phase 7 HIGH-LOAD INVESTIGATION -- Step 2.

Same residual-decomposition method as reports/
phase7_constraint_feasibility_report.txt's Step 2, but at the ACTUAL
point where the scipy trust-constr solve (WITH the bus-83 capacitor
fix already in the model) lands for a high-load timestamp (i=368) --
not flat start. Prints every zero-injection row's |c_i(x_final)|,
sorted descending, top 15.

Run: python -m validation.phase7_highload_decompose
"""

from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator_scipy import ScipyTrustConstrEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"

TARGET_I = 368


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)
    x_v_all, x_delta_all = npz["x_v"][:1440], npz["x_delta"][:1440]

    model = build_estimator_model("multiphase")
    state = model.state
    est = ScipyTrustConstrEstimator(model)
    v_base = np.array([state.v_base[n] for n in state.all_node_names])

    row = df.iloc[TARGET_I]
    real_ts = row["real_timestamp"]
    z_real = {c: float(row[c]) for c in z_cols}

    print(f"=== i={TARGET_I}, real_ts={real_ts} (high-load, with capacitor fix already in model) ===")
    result = est.estimate(z_real, real_ts, max_iter=500)
    err_v = np.abs(result.x_v - x_v_all[TARGET_I]) / v_base
    print("converged:", result.converged, "iters:", result.iterations, "nfev:", result.nfev)
    print("constraint_residual_inf:", result.constraint_residual_inf)
    print("max_v_err_pu:", err_v.max())
    print("status:", result.status, "msg:", result.message)
    print()

    # recompute c(x_final) for ALL zero-injection rows directly
    from observability.state_vector import complex_v_from_state
    from observability import jacobian as J

    at_v = complex_v_from_state(state, result.x_v, result.x_delta)
    C, c_final = J.build_jacobian(model.zi_entries, model.Y, model.node_index, state, at_v=at_v)
    ids = [e["id"] for e in model.zi_entries]

    print("=== top 15 |c_i(x_final)| rows, descending, AT THE ACTUAL SOLVE LANDING POINT ===")
    order = np.argsort(-np.abs(c_final))
    for k in order[:15]:
        print(f"  {ids[k]:20s}  c={c_final[k]: .6f}")

    np.savez(BASE_DIR / "validation" / "phase7_highload_decompose_result.npz",
             x_v=result.x_v, x_delta=result.x_delta, c_final=c_final,
             entry_ids=np.array(ids))


if __name__ == "__main__":
    main()
