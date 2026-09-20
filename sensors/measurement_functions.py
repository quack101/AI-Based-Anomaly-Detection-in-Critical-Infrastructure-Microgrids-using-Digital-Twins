"""
h_abc(.) -- the nonlinear map from a solved circuit's state to what each
sensor reads, per RES.md D10 Phase 2. This module (with sensor_layer.py,
which orchestrates it) is the ONLY code permitted to read a solved
OpenDSS circuit and turn it into a Layer-2 measurement (RES.md D2: "only
the sensors module may read a solved circuit and emit measurements" --
tests/test_import_boundaries.py enforces the far side of this rule for
whichever estimation/detection/evaluation modules come later).

Implements, phase-resolved:
  - P_i^mu, Q_i^mu load-bus injection         -- Sayghe eq. (14)-(15)
  - P_ij^mu, Q_ij^mu branch flow               -- Anwar eq. (17.5)-(17.6)
  - |I_ij^mu| branch current magnitude         -- Anwar eq. (17.7)
  - |V_i^mu| bus voltage magnitude (p.u.)      -- direct
  - substation-head P, Q, |I| (same forms, applied to the Vsource
    element, with the sign convention corrected to report power
    DELIVERED into the feeder as positive -- see _source_terminal)

Each function takes ONE registry entry (dict) and reads whatever is
CURRENTLY SOLVED in the active OpenDSS circuit -- it does not solve or
compile anything itself. Units match what the registry declares: kW,
kvar, per-unit V, amperes.

JACOBIAN (H = dh/dx), STRUCTURE ONLY -- not built in this module.
RES.md D10 explicitly assigns building H to Phase 4 ("build H_abc(x) =
dh/dx at flat start"). The equations above are standard AC power-flow
functions of the state x = (V, delta) at every phase-node and the
network's Y-bus admittance structure -- e.g. for injections,
  P_i^mu = V_i^mu * sum_{j,eta} V_j^eta * (G^{mu,eta}_ij cos(delta_i^mu
           - delta_j^eta) + B^{mu,eta}_ij sin(delta_i^mu - delta_j^eta))
(Sayghe eq. (2)-(3), generalised from single-phase to phase pairs per
eq. (14)-(15)), with an analogous form for Q and for branch flows using
the pi-section branch admittance instead of the full Y-bus row. Phase 4
should differentiate THESE algebraic forms analytically -- closed-form
per measurement type, reusing the same Y-bus already exposed by
core.ieee123_model / placement.electrical_distance.get_system_y_index()
-- rather than finite-differencing through repeated OpenDSS solves,
which would be slower and a worse-conditioned Jacobian estimate for a
stiff AC power flow. The (node, phase) indexing this registry uses is
exactly the (i, mu) indexing those equations use, so Phase 4's H rows
line up with this module's z entries with no re-indexing.
"""

import opendssdirect as dss

PHASE_LETTER_TO_NUM = {"A": 1, "B": 2, "C": 3}


# --------------------------------------------------
# Load-bus injection (Sayghe eq. 14-15)
# --------------------------------------------------

def _load_terminal_power(entry):
    dss.Circuit.SetActiveElement(f"Load.{entry['load_name']}")
    node_order = dss.CktElement.NodeOrder()
    powers = dss.CktElement.Powers()

    phase_num = PHASE_LETTER_TO_NUM[entry["phase"]]
    idx = node_order.index(phase_num)

    return powers[2 * idx], powers[2 * idx + 1]


def measure_load_p_injection(entry):
    p, _ = _load_terminal_power(entry)
    return p


def measure_load_q_injection(entry):
    _, q = _load_terminal_power(entry)
    return q


# --------------------------------------------------
# Voltage magnitude, per-unit -- reused for meter |V| AND head |V|,
# since both are just "the voltage at this bus/phase"
# --------------------------------------------------

def measure_voltage_magnitude_pu(entry):
    dss.Circuit.SetActiveBus(entry["node"])
    nodes = dss.Bus.Nodes()
    pu = dss.Bus.puVmagAngle()

    phase_num = PHASE_LETTER_TO_NUM[entry["phase"]]
    idx = nodes.index(phase_num)

    return pu[2 * idx]


# --------------------------------------------------
# Branch flow and current magnitude (Anwar eq. 17.5-17.7)
# --------------------------------------------------

def _branch_terminal_index(entry):
    """Index, within the FIRST terminal's phase block, of this entry's
    phase -- Line/Transformer CktElement.Powers()/CurrentsMagAng() list
    both terminals; restricting the search to [:num_phases] pins this to
    the terminal-1 ("from" bus) side."""

    element_kind = entry.get("element_kind", "Line")
    dss.Circuit.SetActiveElement(f"{element_kind}.{entry['element_name']}")

    num_phases = dss.CktElement.NumPhases()
    node_order = dss.CktElement.NodeOrder()[:num_phases]

    phase_num = PHASE_LETTER_TO_NUM[entry["phase"]]
    return node_order.index(phase_num)


def measure_branch_p_flow(entry):
    idx = _branch_terminal_index(entry)
    return dss.CktElement.Powers()[2 * idx]


def measure_branch_q_flow(entry):
    idx = _branch_terminal_index(entry)
    return dss.CktElement.Powers()[2 * idx + 1]


def measure_branch_current_magnitude(entry):
    idx = _branch_terminal_index(entry)
    return dss.CktElement.CurrentsMagAng()[2 * idx]


# --------------------------------------------------
# Substation head (RES.md D4 4.3 row 2). Same Powers()/CurrentsMagAng()
# mechanics as a branch, applied to the Vsource element -- but OpenDSS
# reports source power with generator sign convention (negative =
# delivered), so P/Q are negated to report power DELIVERED into the
# feeder as positive, matching feeder_head_p_kw's convention already
# established in core.ieee123_model.get_solution_metrics().
# --------------------------------------------------

def _source_terminal_index(entry):
    element_name = entry.get("element_name", "source")
    dss.Circuit.SetActiveElement(f"Vsource.{element_name}")

    num_phases = dss.CktElement.NumPhases()
    node_order = dss.CktElement.NodeOrder()[:num_phases]

    phase_num = PHASE_LETTER_TO_NUM[entry["phase"]]
    return node_order.index(phase_num)


def measure_head_p(entry):
    idx = _source_terminal_index(entry)
    return -dss.CktElement.Powers()[2 * idx]


def measure_head_q(entry):
    idx = _source_terminal_index(entry)
    return -dss.CktElement.Powers()[2 * idx + 1]


def measure_head_current_magnitude(entry):
    idx = _source_terminal_index(entry)
    return dss.CktElement.CurrentsMagAng()[2 * idx]


# --------------------------------------------------
# Registry quantity name -> h() function. This dict IS the formal
# definition of "every quantity type this twin knows how to sense" --
# a registry entry naming any quantity not in this dict is a
# configuration error, not a silently-ignored field.
# --------------------------------------------------

MEASUREMENT_FUNCTIONS = {
    "P_injection": measure_load_p_injection,
    "Q_injection": measure_load_q_injection,
    "V_magnitude": measure_voltage_magnitude_pu,
    "P_flow": measure_branch_p_flow,
    "Q_flow": measure_branch_q_flow,
    "I_magnitude_branch": measure_branch_current_magnitude,
    "P_head": measure_head_p,
    "Q_head": measure_head_q,
    "I_magnitude_head": measure_head_current_magnitude,
}


def measure(entry):
    """h_k(x_true) for one registry entry -- noise-free."""

    quantity = entry["quantity"]

    if quantity not in MEASUREMENT_FUNCTIONS:
        raise KeyError(
            f"Registry entry {entry.get('id')!r} names unknown quantity "
            f"{quantity!r}. Known quantities: {sorted(MEASUREMENT_FUNCTIONS)}"
        )

    return MEASUREMENT_FUNCTIONS[quantity](entry)
