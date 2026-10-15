"""
Branch est-pf-start -- "pseudo_pf" start point for the state estimator.

For one timestamp: every METERED load is set to its own telemetry P/Q
(sum over that load's per-phase meters -- the meters read the load
element's own terminal power, sensors/measurement_functions.py's
_load_terminal_power), every UNMETERED load to its static pseudo value
for that hour (the same train-bucket mean_p/mean_q the estimator's
PSEUDO_* rows use, observability/pseudo_measurements.py), and the
compiled feeder model is solved once. The solved node voltages, in the
estimator's own (x_v, x_delta) layout, are the start state.

Inputs are ONLY: the telemetry dict z (Layer 2), the pseudo tables
(per_hour_by_load -- offline, train split of dataset/node_loads_91),
and the compiled network model. No ground-truth file or column is read:
all 91 loads are overwritten on every call, and the regulator taps are
reset to their post-compile positions before every solve, so nothing
left on the circuit by an earlier solve carries into this one.

Lives in observability/ (allowed to touch the compiled circuit for
network-model purposes, see estimator_model.py's docstring);
estimation/ only calls the returned function through the model bundle
and never imports opendssdirect itself (tests/test_import_boundaries.py).
"""

import numpy as np
import opendssdirect as dss

from observability.state_vector import extract_solved_state


def build_pf_start_fn(state, meas_entries, per_hour_by_load):
    """Call right after compile_feeder() (build_estimator_model does).
    Returns pf_start(z, hour) -> (x_v, x_delta, converged)."""

    metered = {}
    for e in meas_entries:
        if e["quantity"] in ("P_injection", "Q_injection") and "load_name" in e:
            ids = metered.setdefault(e["load_name"], {"P": [], "Q": []})
            ids["P" if e["quantity"] == "P_injection" else "Q"].append(e["id"])

    all_loads = set(dss.Loads.AllNames())
    unmetered = sorted(per_hour_by_load)
    if set(metered) | set(unmetered) != all_loads or set(metered) & set(unmetered):
        raise ValueError("metered + unmetered loads must partition the feeder's loads exactly")

    initial_taps = {}
    for name in dss.RegControls.AllNames():
        dss.RegControls.Name(name)
        initial_taps[name] = dss.RegControls.TapNumber()

    def pf_start(z, hour):
        for name, tap in initial_taps.items():
            dss.RegControls.Name(name)
            dss.RegControls.TapNumber(tap)
        for load, ids in metered.items():
            dss.Loads.Name(load)
            dss.Loads.kW(float(sum(z[i] for i in ids["P"])))
            dss.Loads.kvar(float(sum(z[i] for i in ids["Q"])))
        for load in unmetered:
            bucket = per_hour_by_load[load][hour]
            dss.Loads.Name(load)
            dss.Loads.kW(float(bucket["mean_p"]))
            dss.Loads.kvar(float(bucket["mean_q"]))
        dss.Solution.Solve()
        converged = bool(dss.Solution.Converged())
        x_v, x_delta = extract_solved_state(state)
        return np.asarray(x_v, dtype=float), np.asarray(x_delta, dtype=float), converged

    return pf_start
