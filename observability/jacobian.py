"""
RES.md D10 Phase 4 / D4 Sec 4.3 acceptance test -- H_abc(x) = dh/dx at
flat start, built ANALYTICALLY from the compiled circuit's Y-bus (per
RES.md Phase 2's documented decision: differentiate the algebraic form,
do NOT finite-difference through OpenDSS solves).

Standard AC power-flow injection Jacobian (Sayghe eq. 2-3, generalised
per-phase, e.g. Grainger & Stevenson Ch.9), applied at TWO scales:

  - Bus injection (P_injection, Q_injection, zero-injection, pseudo-
    measurement rows): uses the FULL system Y-bus row for the target
    phase-node (placement.electrical_distance.get_system_y_index()).
  - Branch/head flow (P_flow, Q_flow, I_magnitude_branch, P_head,
    Q_head, I_magnitude_head): uses THAT ELEMENT'S OWN primitive Y
    (CktElement.YPrim()), not the aggregate system Y -- a two-terminal
    branch's flow depends only on its own two endpoints (all phases,
    capturing genuine 3-phase mutual coupling), not on the whole
    network row. For the substation head, terminal 2 of Vsource.source
    is the ideal internal EMF (fixed magnitude/angle, not a state) --
    handled as a "one-sided branch" against a constant, exactly as
    Sha25's fixed-reference resolution treats the 3 source angles.

SCOPE NOTE on Y-bus contamination: OpenDSS's system Y-bus folds in
constant-impedance/constant-current load admittances (Model != 1) at
their CURRENTLY-SET value -- confirmed empirically (bus 6.3's diagonal,
where load s6c (Model=2) attaches, shifts measurably when s6c's kW is
scaled). A textbook state-estimation H should be built from network
topology alone (lines/transformers/regulators/shunt caps), independent
of any particular load operating point. Measured magnitude of this
effect at a representative well-connected node: ~0.1-1% of the nodal
admittance (s6c's own contribution is small relative to the branch
admittance already present at a well-meshed node). Disabling all loads
and re-solving to obtain a load-free Y was tested and works, but
introduces its own uncertainty (regulator tap positions may respond to
a near-zero-load operating point rather than the nominal one). Given
RANK -- a structural, not numeric, property -- is essentially never
changed by a perturbation this small, this module uses the standard
system Y-bus (one normal solve, loads at as-authored nominal values),
not a load-disabled variant. Documented here, not a silent shortcut.

Units: the state vector (observability.state_vector) is carried in
actual volts/radians. Each row converts to whatever unit that
registry entry's own h() function reports: P/Q rows divide by 1000
(W,VAR -> kW,kvar, matching sensors.measurement_functions), V_magnitude
rows divide by that node's own V_base (volts -> per-unit), I_magnitude
rows are left in amps (matching CktElement.CurrentsMagAng() directly).
"""

import math

import numpy as np
import opendssdirect as dss

from observability.state_vector import flat_start_complex_v

PHASE_LETTER_TO_NUM = {"A": 1, "B": 2, "C": 3}
PHASE_NUM_TO_ANGLE_DEG = {1: 0.0, 2: -120.0, 3: 120.0}


def _scatter_injection(row_p, row_q, target_name, y_row_complex, state, flat_v):
    """Adds one bus-injection measurement's contribution to dense
    row vectors row_p, row_q (length dim(x) each), returning (P, Q)
    forward values (actual watts/vars). y_row_complex: {node_name (or
    None for a fixed/non-state far end): complex Y}; a None key's
    "voltage" must be supplied via the special '__fixed__' convention
    handled by the caller instead (bus-injection rows never use this;
    only the head-branch builder does, see _scatter_branch_like)."""

    Vt_c = flat_v[target_name]
    Vt = abs(Vt_c)
    dt = math.atan2(Vt_c.imag, Vt_c.real)
    t_v_idx = state.v_index[target_name]
    t_d_idx = state.delta_index.get(target_name)

    P = 0.0
    Q = 0.0

    for k_name, Yk in y_row_complex.items():
        if Yk == 0:
            continue
        Gk, Bk = Yk.real, Yk.imag
        Vk_c = flat_v[k_name]
        Vk = abs(Vk_c)
        dk = math.atan2(Vk_c.imag, Vk_c.real)
        k_v_idx = state.v_index[k_name]
        k_d_idx = state.delta_index.get(k_name)

        if k_name == target_name:
            # self term: theta == 0 identically, product rule on Vi*Vi
            P += Vt * Vt * Gk
            Q += -Vt * Vt * Bk
            row_p[t_v_idx] += 2.0 * Vt * Gk
            row_q[t_v_idx] += -2.0 * Vt * Bk
            continue

        theta = dt - dk
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        P += Vt * Vk * (Gk * cos_t + Bk * sin_t)
        Q += Vt * Vk * (Gk * sin_t - Bk * cos_t)

        dP_dVt = Vk * (Gk * cos_t + Bk * sin_t)
        dP_dVk = Vt * (Gk * cos_t + Bk * sin_t)
        dP_ddt = Vt * Vk * (-Gk * sin_t + Bk * cos_t)
        dP_ddk = -dP_ddt

        dQ_dVt = Vk * (Gk * sin_t - Bk * cos_t)
        dQ_dVk = Vt * (Gk * sin_t - Bk * cos_t)
        # BUG FIXED HERE (found via finite-difference verification,
        # RES.md D10 Phase 7/8 Q-angle Jacobian bug): Q = Vt*Vk*(Gk*sin
        # theta - Bk*cos theta), so dQ/dtheta = Vt*Vk*(Gk*cos theta +
        # Bk*sin theta) -- this line carried an erroneous extra unary
        # negative on the whole expression, sign-flipping dQ/d(delta)
        # for every Q-type measurement in the system. Confirmed by FD
        # agreement at every eps from 1e-3 to 1e-8.
        dQ_ddt = Vt * Vk * (Gk * cos_t + Bk * sin_t)
        dQ_ddk = -dQ_ddt

        row_p[t_v_idx] += dP_dVt
        row_p[k_v_idx] += dP_dVk
        if t_d_idx is not None:
            row_p[t_d_idx] += dP_ddt
        if k_d_idx is not None:
            row_p[k_d_idx] += dP_ddk

        row_q[t_v_idx] += dQ_dVt
        row_q[k_v_idx] += dQ_dVk
        if t_d_idx is not None:
            row_q[t_d_idx] += dQ_ddt
        if k_d_idx is not None:
            row_q[k_d_idx] += dQ_ddk

    return P, Q


def _system_y_row(target_name, Y, node_index):
    """{node_name: complex Y} for every nonzero entry in Y's row for
    target_name.

    BUG FIXED HERE (found in Phase 7, tracked down from a Gauss-Newton
    divergence): dss.Circuit.AllNodeNames() (state.all_node_names, the
    STATE VECTOR's canonical order) and dss.Circuit.YNodeOrder() (what
    `Y`'s OWN rows/columns are actually indexed by, via node_index) are
    NOT the same permutation -- confirmed directly, not assumed: they
    disagree at exactly 6 of 278 positions, all clustered around buses
    35/36/67/68 (AllNodeNames has ...35.3, 36.1, 36.2... where YNodeOrder
    has ...36.1, 36.2, 35.3... at the same three index slots, and
    similarly for 67/68). Reverse-indexing Y's columns via
    state.all_node_names (the old code) silently attributed a nonzero
    Y entry to the WRONG neighbour node at exactly those 6 positions --
    invisible to Phase 4's rank-only checks (a mislabelled but still
    nonzero, well-formed entry doesn't change RANK), but fatal to Phase
    7's Gauss-Newton solve, which evaluates h_zi(x) against a target of
    exactly 0: confirmed by an independent OpenDSS Powers()-sum power-
    balance check at bus 67 (exactly 0 to numerical precision, proving
    the NETWORK MODEL was never the problem) against this function's
    own h_zi(x_true) there (off by ~700,000 kW before this fix).
    Building the reverse-index from node_index itself (inverted, so it
    is guaranteed to use THE SAME permutation Y's columns actually use,
    whatever that permutation is) rather than from a second, separately-
    ordered list is the general fix -- not a hardcoded correction for
    these 6 positions specifically."""

    idx = node_index[target_name.lower()]
    row = Y.getrow(idx).tocoo()
    reverse_index = {i: name for name, i in node_index.items()}

    out = {}
    for col, val in zip(row.col, row.data):
        name = reverse_index[col]
        if val != 0:
            out[name] = out.get(name, 0.0) + val
    return out


def build_meter_pq_row(entry, Y, node_index, state, flat_v):
    node = f"{entry['node']}.{PHASE_LETTER_TO_NUM[entry['phase']]}"
    y_row = _system_y_row(node, Y, node_index)

    row_p = np.zeros(state.dim)
    row_q = np.zeros(state.dim)
    P, Q = _scatter_injection(row_p, row_q, node, y_row, state, flat_v)

    row_p /= 1000.0  # W -> kW
    row_q /= 1000.0
    P /= 1000.0
    Q /= 1000.0

    if entry["quantity"] == "P_injection":
        return row_p, P
    return row_q, Q


def build_voltage_row(entry, state, flat_v):
    """BUG FIXED HERE (RES.md D10 Phase 7/8 Q-angle Jacobian bug
    investigation, found alongside it): this function previously took
    no at_v/flat_v parameter at all and always read state.x0_v --
    meaning every V_magnitude measurement's forward value (h0) stayed
    frozen at the SOLVE'S STARTING GUESS for the entire Gauss-Newton
    iteration, never updated to the current iterate, at any point
    where the solver moves away from its start. The Jacobian row
    itself (1.0/v_base, a constant) was always correct; only h0 was
    stale. Now computed from flat_v, the SAME at_v every other row
    builder in this module receives."""

    node = f"{entry['node']}.{PHASE_LETTER_TO_NUM[entry['phase']]}"
    row = np.zeros(state.dim)
    row[state.v_index[node]] = 1.0 / state.v_base[node]
    h_value = abs(flat_v[node]) / state.v_base[node]
    return row, h_value


# --------------------------------------------------
# Branch / head flow -- element-local YPrim
# --------------------------------------------------

def _element_yprim_map(element_ref):
    """{(terminal_bus_phase_name or None): {(terminal_bus_phase_name or
    None): complex}} keyed by GLOBAL 'bus.phase' node names (lowercase),
    built from CktElement.YPrim() + NodeOrder() + BusNames(). A ground
    reference (phase 0, e.g. Vsource's terminal 2) maps to the key
    None -- callers combine it with the FIXED source EMF instead of a
    state lookup."""

    dss.Circuit.SetActiveElement(element_ref)
    node_order = dss.CktElement.NodeOrder()
    bus_names = dss.CktElement.BusNames()
    num_phases = dss.CktElement.NumPhases()
    n_conductors = len(node_order)
    n_terminals = n_conductors // num_phases if num_phases else 0

    local_names = []
    for t in range(n_terminals):
        bus_root = bus_names[t].split(".")[0].lower() if t < len(bus_names) else bus_names[0].split(".")[0].lower()
        for c in range(num_phases):
            phase_num = node_order[t * num_phases + c]
            local_names.append(None if phase_num == 0 else f"{bus_root}.{phase_num}")

    yprim_flat = dss.CktElement.YPrim()
    n = len(local_names)
    arr = np.array(yprim_flat, dtype=float).reshape(n, n, 2)
    Ycplx = arr[:, :, 0] + 1j * arr[:, :, 1]

    return local_names, Ycplx


def _fixed_source_ev(entry_phase_letter):
    """(magnitude volts, angle radians) of the Vsource's ideal internal
    EMF for one phase -- fixed, not a state variable."""

    dss.Vsources.Name("source")
    pu = dss.Vsources.PU()
    base_kv_ll = dss.Vsources.BasekV()
    v_ln = pu * base_kv_ll * 1000.0 / math.sqrt(3.0)
    ang = math.radians(dss.Vsources.AngleDeg() + PHASE_NUM_TO_ANGLE_DEG[PHASE_LETTER_TO_NUM[entry_phase_letter]])
    return v_ln, ang


def build_branch_or_head_row(entry, state, flat_v, want="P"):
    """want in {'P','Q','I'}. Handles P_flow/Q_flow/I_magnitude_branch
    (element_kind='Line', two real buses) and P_head/Q_head/
    I_magnitude_head (Vsource.source, terminal 2 is the fixed EMF)."""

    is_head = entry["quantity"] in ("P_head", "Q_head", "I_magnitude_head")
    element_ref = f"Vsource.{entry.get('element_name', 'source')}" if is_head else f"{entry.get('element_kind', 'Line')}.{entry['element_name']}"

    local_names, Ycplx = _element_yprim_map(element_ref)

    target_phase_num = PHASE_LETTER_TO_NUM[entry["phase"]]
    target_bus = entry["node"].lower()
    target_local_idx = None
    for i, name in enumerate(local_names):
        if name == f"{target_bus}.{target_phase_num}":
            target_local_idx = i
            break
    if target_local_idx is None:
        raise ValueError(f"could not locate {entry['id']} ({target_bus}.{target_phase_num}) among element terminals {local_names}")

    row_p = np.zeros(state.dim)
    row_q = np.zeros(state.dim)

    target_name = local_names[target_local_idx]
    Vt_c = flat_v[target_name]
    Vt = abs(Vt_c)
    dt = math.atan2(Vt_c.imag, Vt_c.real)
    t_v_idx = state.v_index[target_name]
    t_d_idx = state.delta_index.get(target_name)

    P = 0.0
    Q = 0.0

    for j, other_name in enumerate(local_names):
        Yk = Ycplx[target_local_idx, j]
        if Yk == 0:
            continue
        Gk, Bk = Yk.real, Yk.imag

        if other_name is None:
            # fixed source EMF -- one-sided (no state column for "other")
            Vk, dk = _fixed_source_ev(entry["phase"])
            k_v_idx = None
            k_d_idx = None
        elif other_name == target_name:
            P += Vt * Vt * Gk
            Q += -Vt * Vt * Bk
            row_p[t_v_idx] += 2.0 * Vt * Gk
            row_q[t_v_idx] += -2.0 * Vt * Bk
            continue
        else:
            Vk_c = flat_v[other_name]
            Vk = abs(Vk_c)
            dk = math.atan2(Vk_c.imag, Vk_c.real)
            k_v_idx = state.v_index[other_name]
            k_d_idx = state.delta_index.get(other_name)

        theta = dt - dk
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        P += Vt * Vk * (Gk * cos_t + Bk * sin_t)
        Q += Vt * Vk * (Gk * sin_t - Bk * cos_t)

        dP_dVt = Vk * (Gk * cos_t + Bk * sin_t)
        dP_ddt = Vt * Vk * (-Gk * sin_t + Bk * cos_t)
        dQ_dVt = Vk * (Gk * sin_t - Bk * cos_t)
        # BUG FIXED HERE -- same sign error as _scatter_injection's
        # dQ_ddt, see that function's comment for the derivation.
        dQ_ddt = Vt * Vk * (Gk * cos_t + Bk * sin_t)

        row_p[t_v_idx] += dP_dVt
        row_q[t_v_idx] += dQ_dVt
        if t_d_idx is not None:
            row_p[t_d_idx] += dP_ddt
            row_q[t_d_idx] += dQ_ddt

        if k_v_idx is not None:
            dP_dVk = Vt * (Gk * cos_t + Bk * sin_t)
            dQ_dVk = Vt * (Gk * sin_t - Bk * cos_t)
            row_p[k_v_idx] += dP_dVk
            row_q[k_v_idx] += dQ_dVk
        if k_d_idx is not None:
            row_p[k_d_idx] += -dP_ddt
            row_q[k_d_idx] += -dQ_ddt

    if want in ("P", "Q"):
        row_p /= 1000.0
        row_q /= 1000.0
        P /= 1000.0
        Q /= 1000.0
        return (row_p, P) if want == "P" else (row_q, Q)

    # I_magnitude: |I| where I = sum_j Y[target,j] * V[j] (complex).
    # d|I|/dx via d|I|^2 = 2*Re(conj(I) dI); dI/dVk = Y*e^{j dk} (if
    # varying Vk), dI/ddk = Y*Vk*j*e^{j dk} (if varying dk); target's
    # OWN V,delta are just one term (k==target) in this same sum, no
    # special-casing needed unlike the power case (no Vt^2 product).
    I_c = 0j
    for j, other_name in enumerate(local_names):
        Yk = Ycplx[target_local_idx, j]
        if Yk == 0:
            continue
        if other_name is None:
            Vk, dk = _fixed_source_ev(entry["phase"])
            Vk_c = Vk * complex(math.cos(dk), math.sin(dk))
        else:
            Vk_c = flat_v[other_name]
        I_c += Yk * Vk_c

    I_mag = abs(I_c)
    if I_mag == 0:
        return np.zeros(state.dim), 0.0

    row_i = np.zeros(state.dim)
    for j, other_name in enumerate(local_names):
        Yk = Ycplx[target_local_idx, j]
        if Yk == 0 or other_name is None:
            continue
        Vk_c = flat_v[other_name]
        dk = math.atan2(Vk_c.imag, Vk_c.real)
        Vk = abs(Vk_c)

        dI_dVk = Yk * complex(math.cos(dk), math.sin(dk))
        dI_ddk = Yk * Vk * complex(-math.sin(dk), math.cos(dk))

        d_absI_dVk = (I_c.conjugate() * dI_dVk).real / I_mag
        d_absI_ddk = (I_c.conjugate() * dI_ddk).real / I_mag

        row_i[state.v_index[other_name]] += d_absI_dVk
        d_idx = state.delta_index.get(other_name)
        if d_idx is not None:
            row_i[d_idx] += d_absI_ddk

    return row_i, I_mag


def build_head_p_row(entry, state, flat_v):
    """P_head only (RES.md D10 Phase 8 fix; Q_head is untouched -- see
    below). The OLD formula (build_branch_or_head_row, is_head=True,
    element Vsource.source) evaluates P as a function of theta = dt -
    dk between bus 150 and the Vsource's own internal EMF -- but bus
    150's angle is PINNED at its fixed per-phase reference value
    (observability/state_vector.py's complex_v_from_state: bus 150 is
    in state.reference_nodes, so dt is never read from x_delta), and
    the EMF's angle is ALSO fixed at that same reference (_fixed_
    source_ev). theta is therefore identically 0 for every state x,
    and since the Vsource branch is purely reactive (Gk=0 -- confirmed
    via CktElement.YPrim()), P = Vt*Vk*Bk*sin(0) = 0 for every x.
    Confirmed empirically: reports/model_consistency_at_truth.md,
    h_true=0.0 at all 120 checked timesteps against z0 ~930-1400 kW.

    Fix: bus 150 has exactly one non-source neighbour, Transformer.
    reg1a (150 -> 150r); 150r in turn connects only to reg1a and
    Line.sw1 (150r -> 149) -- no load, no other branch (confirmed via
    opendssdirect: dss.Lines/Transformers/Loads.AllNames() filtered to
    150r). Bus 150r is NOT a hard zero-injection node in this model
    (it is a REGZI_150r_* regulator-pseudo WEIGHTED measurement,
    observability/zero_injection.py) but its net injection is small
    enough that the power flowing from 150r into sw1 tracks the true
    head power to within ~0.0001 kW at the checked points -- and,
    critically, 150r AND 149 both have FREE state angles (state.
    delta_index), so this computation never touches the pinned
    reference angle at bus 150 at all. Using reg1a (150->150r)
    directly instead of sw1 was tried and rejected: it still reads
    bus 150's pinned angle as one of its two terminals, and because
    reg1a's own branch is also very low-impedance (high G/B), the
    ~0.001 degree gap between bus 150's TRUE solved angle and its
    pinned reference value is amplified into a multi-thousand-kW
    error -- confirmed empirically, not assumed. Routing through
    sw1 (150r -> 149) avoids bus 150's angle entirely.

    Q_head is UNCHANGED (still build_branch_or_head_row, element
    Vsource.source): Q flows through a purely reactive branch even at
    theta=0 (Q = -Vt*Vk*Bk, Bk != 0), so Q_head's old formula was
    never structurally zero the way P_head's was -- RES.md D10 Phase 8
    task context confirms HEAD_Q is fine, and this function does not
    touch it."""

    synthetic = {
        "id": entry["id"], "quantity": "P_flow",
        "node": "150r", "phase": entry["phase"],
        "element_kind": "Line", "element_name": "sw1",
    }
    return build_branch_or_head_row(synthetic, state, flat_v, want="P")


def head_p_balance_residual(model, result, z):
    """Branch headp-excluded -- a standalone head-balance CHECK, not
    wired into the WLS fit, J, LNR or dof (P_head is excluded from
    model.weighted_meas_entries by default on this branch; this
    function is the only place its corrected value is still read).
    Uses build_head_p_row (branch fix-head-p-model's corrected
    formula) evaluated at the ESTIMATE x_hat (result.x_v/x_delta),
    not the true state -- so a nonzero residual here reflects the
    estimator's fit quality on this channel, not the measurement-
    model gap that build_head_p_row itself fixed.

    model: an observability.estimator_model.EstimatorModel.
    result: an estimation.state_estimator.EstimationResult.
    z: {registry_id: value} -- the real-channel measurement dict
    (same shape passed to StateEstimator.estimate()); must contain
    HEAD_P_A/B/C.

    Returns {phase_letter: {'z_kw', 'h_kw', 'r_kw', 'sigma', 'r_sigma'}}
    for every P_head entry in model.meas_entries (present regardless of
    exclude_dead_head_p, since meas_entries is the full registry list)."""

    from observability.state_vector import complex_v_from_state

    state = model.state
    at_v = complex_v_from_state(state, result.x_v, result.x_delta)

    out = {}
    for e in model.meas_entries:
        if e["quantity"] != "P_head":
            continue
        _, h_kw = build_head_p_row(e, state, at_v)
        z_kw = z[e["id"]]
        r_kw = z_kw - h_kw
        sigma = e["sigma"]
        out[e["phase"]] = {
            "z_kw": z_kw, "h_kw": h_kw, "r_kw": r_kw,
            "sigma": sigma, "r_sigma": r_kw / sigma,
        }
    return out


def build_jacobian(entries, Y, node_index, state, at_v=None):
    """H's row i corresponds to entries[i] -- built in ONE pass, same
    ordering invariant sensors/sensor_layer.py enforces for z. Returns
    (H: (len(entries), dim) ndarray, h0: forward values, same units as
    each entry's own quantity).

    at_v: {node_name: complex V in actual volts} to linearise/evaluate
    at -- defaults to flat start (Phase 4's original, and every existing
    caller's, use case) when omitted. RES.md D10 Phase 7's Gauss-Newton
    iteration passes state_vector.complex_v_from_state(state, x_v_k,
    x_delta_k) here to re-evaluate h(x_k)/H_k at each iterate x_k away
    from flat start -- the row-builder MATH (Sec module docstring) does
    not care where V,delta come from, only state_vector.py's flat-start
    convenience function hardcoded it before this parameter existed."""

    flat_v = at_v if at_v is not None else flat_start_complex_v(state)

    H = np.zeros((len(entries), state.dim))
    h0 = np.zeros(len(entries))
    row_ids = []

    for i, entry in enumerate(entries):
        q = entry["quantity"]

        if q in ("P_injection", "Q_injection"):
            row, val = build_meter_pq_row(entry, Y, node_index, state, flat_v)
        elif q == "V_magnitude":
            row, val = build_voltage_row(entry, state, flat_v)
        elif q in ("P_flow", "Q_flow"):
            want = "P" if q == "P_flow" else "Q"
            row, val = build_branch_or_head_row(entry, state, flat_v, want=want)
        elif q == "I_magnitude_branch":
            row, val = build_branch_or_head_row(entry, state, flat_v, want="I")
        elif q == "P_head":
            row, val = build_head_p_row(entry, state, flat_v)
        elif q == "Q_head":
            # UNCHANGED path -- Q_head was never structurally zero
            # (see build_head_p_row's docstring for why only P_head moved)
            row, val = build_branch_or_head_row(entry, state, flat_v, want="Q")
            row, val = -row, -val  # measure_head_q negates -- match sign convention
        elif q == "I_magnitude_head":
            row, val = build_branch_or_head_row(entry, state, flat_v, want="I")
        else:
            raise KeyError(f"observability.jacobian has no row-builder for quantity {q!r}")

        H[i, :] = row
        h0[i] = val
        row_ids.append(entry["id"])

    assert row_ids == [e["id"] for e in entries], "H row order diverged from entries order"

    return H, h0
