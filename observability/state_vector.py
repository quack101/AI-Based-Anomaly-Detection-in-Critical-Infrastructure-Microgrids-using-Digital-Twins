"""
RES.md D10 Phase 4 / D4 Sec 4.2(a) -- the state vector x and its
FLAT-START evaluation point for the Jacobian.

x = (V_i^mu for every phase-node) + (delta_i^mu for every phase-node
EXCEPT the 3 source-bus phase angles, fixed as references per RES.md's
resolution: "Model verification confirms the OpenDSS Vsource is
balanced to within 0.002 deg of 0/-120/+120 at both bus 150 and the
regulated 150r, so all three source angles are known exactly by
construction and are fixed as references rather than estimated."
Source MAGNITUDES remain states (the substation-head sensor measures
|V| per phase) -- only the 3 angles at bus "150" are excluded.

dim(x) = 2 * N_phase_nodes - 3.

Units: V is carried in ACTUAL volts (line-to-neutral), not per-unit --
the registry mixes unit systems (P/Q/|I| in actual kW/kvar/A, |V| in
per-unit), so the state vector itself is kept in one consistent
physical unit (actual volts) and each measurement function's Jacobian
row applies its OWN unit conversion (see jacobian.py), rather than
forcing a per-unit state that would need a different base at every
node anyway once regulators are involved.
"""

import math
from dataclasses import dataclass, field

import opendssdirect as dss

PHASE_NUM_TO_ANGLE_DEG = {1: 0.0, 2: -120.0, 3: 120.0}


@dataclass
class StateVector:
    all_node_names: list           # length N_phase_nodes, canonical order
    reference_nodes: list          # the 3 node names excluded from delta
    v_index: dict                  # node_name -> column index in V-block
    delta_index: dict              # node_name -> column index in delta-block (absent for reference_nodes)
    v_base: dict                   # node_name -> nominal L-N volts
    dim: int
    n_phase_nodes: int
    x0_v: list = field(default_factory=list)      # flat-start V, actual volts, per V-block index
    x0_delta: list = field(default_factory=list)  # flat-start delta, radians, per delta-block index


def _source_bus():
    dss.Circuit.SetActiveElement("Vsource.source")
    bus = dss.CktElement.BusNames()[0].split(".")[0]
    return bus


def build_state_vector():
    """Requires a compiled circuit (simulation.opendss_utils.compile_feeder()
    or core.ieee123_model.compile_and_solve()) already active."""

    all_node_names = dss.Circuit.AllNodeNames()
    n_phase_nodes = len(all_node_names)

    source_bus = _source_bus()
    reference_nodes = sorted(
        n for n in all_node_names if n.split(".")[0].lower() == source_bus.lower()
    )
    if len(reference_nodes) != 3:
        raise RuntimeError(
            f"expected exactly 3 source-bus phase-nodes at bus {source_bus!r} "
            f"to fix as angle references, found {len(reference_nodes)}: "
            f"{reference_nodes}"
        )

    v_index = {name: i for i, name in enumerate(all_node_names)}

    # delta-block columns are OFFSET past the V-block (columns
    # n_phase_nodes .. dim-1) so the two blocks occupy disjoint columns
    # of the same length-dim row vector -- without this offset, V and
    # delta indices collide (both 0-based) and Jacobian contributions
    # silently overwrite each other in the shared row array.
    delta_index = {}
    j = n_phase_nodes
    for name in all_node_names:
        if name in reference_nodes:
            continue
        delta_index[name] = j
        j += 1

    dim = n_phase_nodes + (n_phase_nodes - 3)
    if dim != 2 * n_phase_nodes - 3:
        raise RuntimeError("dim(x) bookkeeping mismatch")

    v_base = {}
    for name in all_node_names:
        bus_root = name.split(".")[0]
        dss.Circuit.SetActiveBus(bus_root)
        kv_base = dss.Bus.kVBase()  # line-to-neutral kV, per OpenDSS convention
        v_base[name] = kv_base * 1000.0

    x0_v = [v_base[name] for name in all_node_names]  # flat start: 1.0 pu everywhere
    # x0_delta is COMPACT (length len(delta_index), 0-based) -- the
    # SAME shape extract_solved_state() below already returns for
    # x_true (and what Phase 6's persisted corpus files store), so flat-
    # start x0 and any actual solved/estimated state are directly
    # comparable without a reindexing step. This is deliberately NOT
    # delta_index's own offset-into-a-553-length-Jacobian-row
    # convention (columns n_phase_nodes..dim-1 of a JACOBIAN ROW) --
    # that offset exists so V and delta columns don't collide inside a
    # single Jacobian row (jacobian.py); a plain STATE VALUE array has
    # no such collision risk and gains nothing from padding out 278
    # unused leading slots.
    x0_delta = [0.0] * len(delta_index)
    for name, idx in delta_index.items():
        phase_num = int(name.split(".")[-1])
        x0_delta[idx - n_phase_nodes] = math.radians(PHASE_NUM_TO_ANGLE_DEG[phase_num])

    return StateVector(
        all_node_names=all_node_names,
        reference_nodes=reference_nodes,
        v_index=v_index,
        delta_index=delta_index,
        v_base=v_base,
        dim=dim,
        n_phase_nodes=n_phase_nodes,
        x0_v=x0_v,
        x0_delta=x0_delta,
    )


def extract_solved_state(state):
    """RES.md D10 Phase 6: the ACTUAL solved x_true, in the SAME
    (v_index, delta_index) layout build_state_vector() defines for the
    flat-start x0 -- so Phase 7's estimator can compare x_hat against
    x_true using one consistent state definition across phases. Reads
    dss.Circuit.AllBusVolts() (actual volts, real/imag interleaved, in
    AllNodeNames() order -- confirmed empirically, not assumed) on
    whatever circuit state is CURRENTLY SOLVED; caller must have already
    called dss.Solution.Solve() first. Returns (x_v: length
    n_phase_nodes, x_delta: length n_phase_nodes-3), matching x0_v/
    x0_delta's shapes exactly."""

    volts = dss.Circuit.AllBusVolts()

    x_v = [0.0] * state.n_phase_nodes
    x_delta = [0.0] * len(state.delta_index)

    for i, name in enumerate(state.all_node_names):
        re, im = volts[2 * i], volts[2 * i + 1]
        v_c = complex(re, im)
        x_v[state.v_index[name]] = abs(v_c)

        if name not in state.reference_nodes:
            x_delta[state.delta_index[name] - state.n_phase_nodes] = math.atan2(im, re)

    return x_v, x_delta


def complex_v_from_state(state, x_v, x_delta):
    """RES.md D10 Phase 7 -- {node_name: complex V in actual volts} at an
    ARBITRARY state (x_v, x_delta), not just flat start. Reference-node
    angles are always fixed at their known nominal value (0/-120/120
    deg) regardless of x_delta's contents -- they are never state
    variables, per build_state_vector()'s own dim(x) accounting, so
    there is nothing to read for them from x_delta. Gauss-Newton
    iterates (observability.jacobian.build_jacobian's `at_v` parameter)
    call this once per iterate to re-evaluate h(x_k)/H_k away from flat
    start; flat_start_complex_v() below is the x0 special case."""

    result = {}
    for name in state.all_node_names:
        vmag = x_v[state.v_index[name]]
        if name in state.reference_nodes:
            phase_num = int(name.split(".")[-1])
            delta = math.radians(PHASE_NUM_TO_ANGLE_DEG[phase_num])
        else:
            delta = x_delta[state.delta_index[name] - state.n_phase_nodes]
        result[name] = vmag * complex(math.cos(delta), math.sin(delta))
    return result


def flat_start_complex_v(state):
    """{node_name: complex V in actual volts} at flat start -- convenience
    for the Jacobian's forward power-flow evaluation."""

    return complex_v_from_state(state, state.x0_v, state.x0_delta)
