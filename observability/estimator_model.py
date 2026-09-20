"""
RES.md D10 Phase 7 -- builds the FIXED model a StateEstimator needs
(Y-bus, state-vector layout, the 112 measurement entries, the zero-
injection constraint entries, and per-load hourly pseudo-measurement
tables) entirely OUTSIDE estimation/. This module -- like every other
observability/ module -- is permitted to touch a compiled circuit for
NETWORK-TOPOLOGY purposes (tests/test_import_boundaries.py's
FORBIDDEN_MODULES list does not restrict observability/*, only
sensors.measurement_functions/sensor_layer/generate_registry,
core.ieee123_model, simulation.opendss_utils and opendssdirect itself,
and only inside estimation/detection/evaluation). estimation/
state_estimator.py must NOT import opendssdirect or
simulation.opendss_utils directly -- it is Layer 3 (inference from
already-published z), not Layer 1/2 -- so this module does the one-time
circuit compile and hands estimation/ a plain data bundle instead.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import opendssdirect as dss

from placement.electrical_distance import get_system_y_index
from observability.state_vector import build_state_vector, complex_v_from_state, StateVector
from observability import jacobian as J
from observability.zero_injection import (
    build_zero_injection_entries,
    enumerate_regulator_adjacent_nodes,
    build_regulator_pseudo_entries,
)
from observability.pseudo_measurements import estimate_sigma_pseudo
from sensors.registry import load_registry, enabled_entries
from simulation.opendss_utils import compile_feeder

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE6_TRAIN_X_TRUE = BASE_DIR / "simulation" / "phase6_corpus" / "train_x_true.npz"

# ~300 samples spread across the 42-day (60,480-timestep) train split --
# enough to see the regulators cycle through their real tap range without
# paying to evaluate every timestep for a handful of nodes.
REG_SIGMA_SAMPLE_STRIDE = 200
REG_SIGMA_FLOOR = 1e-3  # kW/kvar -- avoid a degenerate sigma=0 if ever 0 samples land nonzero


@dataclass
class EstimatorModel:
    state: StateVector
    Y: object
    node_index: dict
    meas_entries: list        # 112: 106 meter P,Q + 6 head P,Q (multiphase variant)
    zi_entries: list          # HARD zero-injection constraint entries (exact, Lagrangian)
    reg_pseudo_entries: list  # WEIGHTED regulator-adjacent pseudo-entries (target 0, empirical sigma)
    bus_phase_by_load: dict   # unmetered load_name -> (bus, phase)
    per_hour_by_load: dict    # unmetered load_name -> {hour: {mean_p,mean_q,sigma_p,sigma_q,...}}


def _unmetered_load_names(registry):
    metered = {e["load_name"] for e in registry if "load_name" in e}
    all_loads = set(dss.Loads.AllNames())
    return sorted(all_loads - metered, key=str.lower)


def _resolve_bus_and_phase(load_name):
    dss.Circuit.SetActiveElement(f"Load.{load_name}")
    bus = dss.CktElement.BusNames()[0].split(".")[0]
    node_order = dss.CktElement.NodeOrder()
    phase = {1: "A", 2: "B", 3: "C"}[node_order[0]]
    return bus, phase


def _estimate_regulator_mean_sigma(nodes, Y, node_index, state):
    """PER-NODE empirical (mean_p, mean_q, sigma_p, sigma_q) for
    build_regulator_pseudo_entries -- ESTIMATED, not hand-picked, same
    principle Phase 5 applied to load pseudo-measurement sigma (RES.md:
    "a fixed hand-picked sigma is the first thing a reviewer familiar
    with this line will attack").

    Evaluates h_P_injection/h_Q_injection (the module's own analytic
    forward-power-flow formula -- observability.jacobian, the SAME code
    Gauss-Newton itself calls -- NOT a second OpenDSS solve) at each
    regulator-adjacent node, AT the Phase 6 corpus's own persisted
    x_true, across a stride-sampled subset of the TRAIN split (held out
    from the val/test splits this phase validates against -- the same
    train/held-out discipline Phase 5 used for load pseudo-measurements).

    CORRECTED FINDING (found empirically, not assumed): an earlier
    version of this function pooled all regulator-adjacent nodes
    together and targeted a HARDCODED 0 (treating them as "should read
    zero, just uncertain", the same story as _regulator_buses's tap-
    drift account). Direct inspection of the per-node distribution
    falsified that story for SOME of these nodes -- bus 25's P/Q
    injection reads a CONSISTENT ~120-130 (kW or kvar) across nearly
    every sampled ground-truth timestamp, not an occasional tap-drift
    spike around 0. Regulator-adjacent buses on this feeder are not all
    the same: some (e.g. the *r buses immediately across a near-zero-
    impedance regulator branch) DO read near-0 most of the time with
    rare large tap-change-driven spikes (consistent with tap drift);
    others read a small but PERSISTENTLY nonzero value (more consistent
    with how the regulator's own admittance is represented in the
    exported Y-bus than with a discrete control-state change). Rather
    than adjudicate that distinction node-by-node under this phase's
    time budget, this function estimates BOTH the mean AND the spread,
    per node, exactly as Phase 5 does for unmetered loads -- the
    pseudo-measurement's target is whatever the data says it typically
    is, not an assumption of exactly 0, and sigma is the spread AROUND
    that empirical mean (not around 0), which is the statistically
    correct quantity regardless of which of the two stories above is
    the real explanation for any given node."""

    if not nodes:
        return {}

    entries = []
    for node_name in nodes:
        bus_root, phase_str = node_name.split(".")
        phase = {1: "A", 2: "B", 3: "C"}[int(phase_str)]
        entries.append({"id": f"{node_name}_P", "node": bus_root, "phase": phase, "quantity": "P_injection"})
        entries.append({"id": f"{node_name}_Q", "node": bus_root, "phase": phase, "quantity": "Q_injection"})

    npz = np.load(PHASE6_TRAIN_X_TRUE, allow_pickle=True)
    x_v_all = npz["x_v"][::REG_SIGMA_SAMPLE_STRIDE]
    x_delta_all = npz["x_delta"][::REG_SIGMA_SAMPLE_STRIDE]

    samples = []
    for x_v, x_delta in zip(x_v_all, x_delta_all):
        at_v = complex_v_from_state(state, x_v, x_delta)
        _, h0 = J.build_jacobian(entries, Y, node_index, state, at_v=at_v)
        samples.append(h0)
    samples = np.array(samples)  # (num_samples, 2*len(nodes)) -- P,Q interleaved per node

    stats_by_node = {}
    for i, node_name in enumerate(nodes):
        p_col, q_col = samples[:, 2 * i], samples[:, 2 * i + 1]
        stats_by_node[node_name] = {
            "mean_p": float(np.mean(p_col)), "mean_q": float(np.mean(q_col)),
            "sigma_p": max(float(np.std(p_col)), REG_SIGMA_FLOOR),
            "sigma_q": max(float(np.std(q_col)), REG_SIGMA_FLOOR),
        }
    return stats_by_node


def build_estimator_model(registry_variant="multiphase"):
    """Compiles the feeder ONCE and builds everything a StateEstimator
    needs -- none of which depends on the timestamp being estimated."""

    compile_feeder()
    state = build_state_vector()
    Y, node_index = get_system_y_index()

    registry = load_registry(variant=registry_variant)
    meas_entries = enabled_entries(registry)

    zi_entries = build_zero_injection_entries(state.all_node_names)

    reg_nodes = enumerate_regulator_adjacent_nodes(state.all_node_names)
    reg_stats_by_node = _estimate_regulator_mean_sigma(reg_nodes, Y, node_index, state)
    reg_pseudo_entries = build_regulator_pseudo_entries(state.all_node_names, reg_stats_by_node)

    unmetered = _unmetered_load_names(registry)
    bus_phase_by_load = {name: _resolve_bus_and_phase(name) for name in unmetered}
    per_hour_by_load = {name: estimate_sigma_pseudo(name) for name in unmetered}

    return EstimatorModel(
        state=state, Y=Y, node_index=node_index,
        meas_entries=meas_entries, zi_entries=zi_entries,
        reg_pseudo_entries=reg_pseudo_entries,
        bus_phase_by_load=bus_phase_by_load, per_hour_by_load=per_hour_by_load,
    )
