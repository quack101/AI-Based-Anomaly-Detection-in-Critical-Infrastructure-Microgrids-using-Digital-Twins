"""
RES.md D10 Phase 7 GLOBAL SIGMA RECALIBRATION -- TEST, not committed.

Generalizes the M042_Q-style amplified-sigma fix to every enabled
P_injection/Q_injection/P_head/Q_head channel in BOTH registry files
(config/sensor_registry.json, config/sensor_registry_multiphase.json).

For each such channel, computes its sensitivity from the Jacobian at
a fixed reference state (i=46's converged solution, same point used
throughout this investigation):
  - Q-type (Q_injection, Q_head): dQ/dV only, per the M042 formula.
  - P-type (P_injection, P_head): dP/dV AND dP/ddelta, combined in
    quadrature. dP/dV is scaled by a typical ~4% voltage error
    (0.04*v_base, same as M042). dP/ddelta is scaled by a typical
    angle error of 0.03 rad -- EMPIRICALLY MEASURED at i=46 (mean
    |x_delta_hat - x_delta_true| = 0.0311 rad, median 0.0328 rad),
    not an assumed number, chosen to parallel the empirically-
    grounded ~4% voltage figure.

sigma_new = sqrt(sigma_old^2 + amplified_term^2), amplified_term =
  Q-type: |dQ/dV| * 0.04 * v_base
  P-type: sqrt((dP/dV * 0.04 * v_base)^2 + (dP/ddelta * 0.03)^2)

V_magnitude channels are left untouched (direct linear measurement
of the quantity being estimated -- h(x)=V, dV/dV=1, no amplification
mechanism applies). REGZI_* pseudo-entries are NOT in the registry
JSON files (they are built at runtime in
observability/estimator_model.py) and are explicitly out of scope
for this script, per the task's literal "both registry files" scope.

Writes the recalibrated sigma + a note back into BOTH JSON files
in place. Does NOT git commit.

Run: python -m validation.phase7_global_sigma_recalibration
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from observability.state_vector import complex_v_from_state
from observability import jacobian as J
from estimation.state_estimator import StateEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"
REGISTRY_FILES = [
    BASE_DIR / "config" / "sensor_registry.json",
    BASE_DIR / "config" / "sensor_registry_multiphase.json",
]
TARGET_I = 46
TYPICAL_V_ERR_FRAC = 0.04
TYPICAL_DELTA_ERR_RAD = 0.03  # empirically measured at i=46, see module docstring

PHASE_NUM = {"A": 1, "B": 2, "C": 3}
P_TYPES = {"P_injection", "P_head"}
Q_TYPES = {"Q_injection", "Q_head"}


def main():
    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)

    model = build_estimator_model("multiphase")
    state = model.state
    est = StateEstimator(model)

    row = df.iloc[TARGET_I]
    real_timestamp = str(row["real_timestamp"])
    z_real = {c: float(row[c]) for c in z_cols}
    result = est.estimate(z_real, real_timestamp)
    x_v, x_delta = result.x_v.copy(), result.x_delta.copy()
    at_v = complex_v_from_state(state, x_v, x_delta)

    # load both registries, collect every enabled P/Q-type entry (id -> entry dict, per file)
    registries = {}
    for path in REGISTRY_FILES:
        with open(path) as f:
            registries[path] = json.load(f)

    computed = {}  # id -> (sigma_old, sigma_new, sensitivity_info)
    skipped_no_delta = []

    for path, entries in registries.items():
        for e in entries:
            if not e.get("enabled"):
                continue
            q = e["quantity"]
            if q not in P_TYPES and q not in Q_TYPES:
                continue
            cid = e["id"]
            if cid in computed:
                continue  # already computed from the other file

            node_key = f"{e['node']}.{PHASE_NUM[e['phase']]}"
            v_idx = state.v_index[node_key]
            v_base = state.v_base[node_key]
            delta_idx = state.delta_index.get(node_key)

            H, h0 = J.build_jacobian([e], model.Y, model.node_index, state, at_v=at_v)
            dQdV = H[0, v_idx]
            dPdV = H[0, v_idx]
            dPddelta = H[0, delta_idx] if delta_idx is not None else 0.0
            if delta_idx is None:
                skipped_no_delta.append(cid)

            if q in Q_TYPES:
                amplified = abs(dQdV) * TYPICAL_V_ERR_FRAC * v_base
                sens_repr = f"dQ/dV={dQdV:.4f}"
            else:
                term_v = dPdV * TYPICAL_V_ERR_FRAC * v_base
                term_delta = dPddelta * TYPICAL_DELTA_ERR_RAD
                amplified = float(np.hypot(term_v, term_delta))
                sens_repr = f"dP/dV={dPdV:.4f} dP/ddelta={dPddelta:.4f}"

            sigma_old = e["sigma"]
            sigma_new = float(np.hypot(sigma_old, amplified))
            computed[cid] = (sigma_old, sigma_new, sens_repr, amplified)

    print(f"computed recalibrated sigma for {len(computed)} channels "
          f"({len(skipped_no_delta)} had no delta state -- reference bus, dP/ddelta term skipped for those)")
    print()

    # apply back to both files
    for path, entries in registries.items():
        changed = 0
        for e in entries:
            if e["id"] in computed:
                sigma_old, sigma_new, sens_repr, amplified = computed[e["id"]]
                e["sigma"] = sigma_new
                e["note"] = (f"sigma recalibrated (global pass): old={sigma_old:.6f}, "
                             f"new={sigma_new:.6f}, {sens_repr}, amplified_term={amplified:.4f} "
                             f"(sensitivity x typical error, combined in quadrature with old sigma). "
                             f"TEST ONLY, not yet committed.")
                changed += 1
        with open(path, "w") as f:
            json.dump(entries, f, indent=2)
            f.write("\n")
        print(f"{path}: updated {changed} entries")

    # print a sorted summary (biggest sigma inflation first)
    print()
    print("=== all recalibrated channels, sorted by inflation ratio ===")
    rows = []
    for cid, (sigma_old, sigma_new, sens_repr, amplified) in computed.items():
        rows.append((cid, sigma_old, sigma_new, sigma_new / sigma_old, sens_repr))
    rows.sort(key=lambda r: -r[3])
    for cid, sold, snew, ratio, sens in rows[:20]:
        print(f"  {cid:<14s} old={sold:>12.6f} new={snew:>14.4f} ratio={ratio:>12.1f}x  {sens}")
    print(f"  ... ({len(rows)} total channels recalibrated)")
    print(f"  min ratio={min(r[3] for r in rows):.2f}x  max ratio={max(r[3] for r in rows):.1f}x  "
          f"median ratio={sorted(r[3] for r in rows)[len(rows)//2]:.1f}x")


if __name__ == "__main__":
    main()
