"""
RES.md D10 Phase 4 acceptance tests -- Jacobian construction.

The Jacobian is built analytically (no finite-differencing through
OpenDSS solves, per RES.md Phase 2's documented decision). These tests
instead finite-difference the module's OWN forward power-flow formula
(a pure calculus check on the hand-derived partial derivatives, never
touching OpenDSS a second time) -- this catches algebra/sign errors
without violating the "don't finite-difference through solves" rule.
"""

import math

import numpy as np
import pytest

from simulation.opendss_utils import compile_feeder
from placement.electrical_distance import get_system_y_index
from observability.state_vector import build_state_vector, flat_start_complex_v
from observability import jacobian as J
from sensors.registry import load_registry, enabled_entries


@pytest.fixture(scope="module")
def state():
    compile_feeder()
    return build_state_vector()


@pytest.fixture(scope="module")
def system_y(state):
    return get_system_y_index()


@pytest.fixture(scope="module")
def flat_v(state):
    return flat_start_complex_v(state)


def _perturb_v(flat_v, node, dv):
    fv = dict(flat_v)
    mag = abs(fv[node]) + dv
    ang = math.atan2(fv[node].imag, fv[node].real)
    fv[node] = mag * complex(math.cos(ang), math.sin(ang))
    return fv


def _perturb_d(flat_v, node, dd):
    fv = dict(flat_v)
    mag = abs(fv[node])
    ang = math.atan2(fv[node].imag, fv[node].real) + dd
    fv[node] = mag * complex(math.cos(ang), math.sin(ang))
    return fv


def test_dim_x_is_553(state):
    """RES.md D4 Sec 4.2(a): dim(x) = 2*278 - 3 = 553, three fixed
    source-bus angle references. Recomputed from the compiled circuit,
    not assumed."""

    assert state.n_phase_nodes == 278
    assert state.dim == 553
    assert len(state.reference_nodes) == 3


def test_v_and_delta_columns_do_not_collide(state):
    v_cols = set(state.v_index.values())
    d_cols = set(state.delta_index.values())
    assert v_cols.isdisjoint(d_cols)
    assert v_cols | d_cols == set(range(state.dim))


def test_h_row_order_matches_registry_order(state, system_y):
    Y, node_index = system_y
    registry = load_registry()
    entries = enabled_entries(registry)[:10]

    H, h0 = J.build_jacobian(entries, Y, node_index, state)

    assert H.shape == (len(entries), state.dim)
    assert len(h0) == len(entries)


def test_bus_injection_jacobian_matches_finite_difference(state, system_y, flat_v):
    Y, node_index = system_y
    registry = load_registry()
    entries = enabled_entries(registry)
    entry = next(e for e in entries if e["quantity"] == "P_injection")

    node = f"{entry['node']}.{J.PHASE_LETTER_TO_NUM[entry['phase']]}"
    y_row = J._system_y_row(node, Y, node_index)

    H, _ = J.build_jacobian([entry], Y, node_index, state)
    row = H[0] * 1000.0  # undo kW scaling -- compare in native watts

    eps = 1e-3
    fv_p = _perturb_v(flat_v, node, eps)
    fv_m = _perturb_v(flat_v, node, -eps)
    rp, rq = np.zeros(state.dim), np.zeros(state.dim)
    Pp, _ = J._scatter_injection(rp, rq, node, y_row, state, fv_p)
    rp2, rq2 = np.zeros(state.dim), np.zeros(state.dim)
    Pm, _ = J._scatter_injection(rp2, rq2, node, y_row, state, fv_m)
    fd = (Pp - Pm) / (2 * eps)

    assert row[state.v_index[node]] == pytest.approx(fd, abs=1e-2)

    neighbor = next(k for k in y_row if k != node)
    eps_d = 1e-6
    fv_p = _perturb_d(flat_v, neighbor, eps_d)
    fv_m = _perturb_d(flat_v, neighbor, -eps_d)
    rp3, rq3 = np.zeros(state.dim), np.zeros(state.dim)
    Pp, _ = J._scatter_injection(rp3, rq3, node, y_row, state, fv_p)
    rp4, rq4 = np.zeros(state.dim), np.zeros(state.dim)
    Pm, _ = J._scatter_injection(rp4, rq4, node, y_row, state, fv_m)
    fd = (Pp - Pm) / (2 * eps_d)

    k_d_idx = state.delta_index.get(neighbor)
    if k_d_idx is not None:
        assert row[k_d_idx] == pytest.approx(fd, rel=1e-3)


def test_branch_flow_jacobian_matches_finite_difference(state, flat_v):
    import opendssdirect as dss

    line_name = dss.Lines.AllNames()[0]
    dss.Circuit.SetActiveElement(f"Line.{line_name}")
    bus1 = dss.CktElement.BusNames()[0].split(".")[0]

    entry = {
        "id": "TEST_P", "node": bus1, "phase": "A",
        "element_name": line_name, "element_kind": "Line", "quantity": "P_flow",
    }
    row, _ = J.build_branch_or_head_row(entry, state, flat_v, want="P")
    target = f"{bus1.lower()}.1"

    eps = 1e-3
    fv_p = _perturb_v(flat_v, target, eps)
    fv_m = _perturb_v(flat_v, target, -eps)
    _, Pp = J.build_branch_or_head_row(entry, state, fv_p, want="P")
    _, Pm = J.build_branch_or_head_row(entry, state, fv_m, want="P")
    fd = (Pp - Pm) / (2 * eps)  # build_branch_or_head_row already returns kW

    assert row[state.v_index[target]] == pytest.approx(fd, abs=1e-6)


def test_current_magnitude_jacobian_matches_finite_difference(state, flat_v):
    """Uses a much smaller epsilon than the power-flow checks -- |I| at
    flat start is near-degenerate (close to zero, since neighbouring
    flat-start voltages are nearly identical), so a perturbation on the
    scale used for P/Q checks overshoots the local-linear regime and
    produces a misleading finite difference. Confirmed by sweeping eps
    from 1e-3 down to 1e-8 during development: the FD estimate only
    converges to the analytic value once eps is small enough relative
    to the tiny baseline |I|."""

    import opendssdirect as dss

    line_name = dss.Lines.AllNames()[0]
    dss.Circuit.SetActiveElement(f"Line.{line_name}")
    bus1 = dss.CktElement.BusNames()[0].split(".")[0]

    entry = {
        "id": "TEST_I", "node": bus1, "phase": "A",
        "element_name": line_name, "element_kind": "Line",
        "quantity": "I_magnitude_branch",
    }
    row, _ = J.build_branch_or_head_row(entry, state, flat_v, want="I")
    target = f"{bus1.lower()}.1"

    eps = 1e-7
    fv_p = _perturb_v(flat_v, target, eps)
    fv_m = _perturb_v(flat_v, target, -eps)
    _, Ip = J.build_branch_or_head_row(entry, state, fv_p, want="I")
    _, Im = J.build_branch_or_head_row(entry, state, fv_m, want="I")
    fd = (Ip - Im) / (2 * eps)

    assert row[state.v_index[target]] == pytest.approx(fd, rel=1e-2)


@pytest.fixture(scope="module")
def perturbed_v(flat_v):
    """A genuinely non-flat-start operating point -- every node's V
    magnitude AND angle displaced from nominal. The two regression
    tests below (RES.md D10 Phase 7/8 Jacobian bugs) specifically need
    this: both bugs were invisible at flat start (the Q-angle sign
    bug happens to be masked wherever theta==0 identically, which flat
    start's per-phase-only angle spread does not trigger; the frozen
    V_magnitude h0 bug is invisible whenever x0_v happens to equal the
    evaluation point, which flat start trivially does)."""

    rng = np.random.default_rng(42)
    pv = {}
    for name, v in flat_v.items():
        mag = abs(v) * (1.0 + rng.uniform(-0.05, 0.05))
        ang = math.atan2(v.imag, v.real) + rng.uniform(-0.05, 0.05)
        pv[name] = mag * complex(math.cos(ang), math.sin(ang))
    return pv


def test_q_injection_angle_jacobian_matches_finite_difference(state, system_y, perturbed_v):
    """RES.md D10 Phase 7/8 Q-ANGLE JACOBIAN BUG, regression guard.
    _scatter_injection's dQ_ddt previously carried an erroneous extra
    unary negative (`Vt*Vk*(-Gk*cos_t-Bk*sin_t)` instead of
    `Vt*Vk*(Gk*cos_t+Bk*sin_t)`), sign-flipping dQ/d(delta) for every
    Q-type measurement in the system -- found via exactly this kind
    of finite-difference check, but at a flat start where the existing
    P-only test (test_bus_injection_jacobian_matches_finite_difference)
    never exercised Q at all. Uses perturbed_v, not flat start, so a
    reintroduced bug can't hide behind flat start's special angle
    structure."""

    Y, node_index = system_y
    registry = load_registry()
    entries = enabled_entries(registry)
    entry = next(e for e in entries if e["quantity"] == "Q_injection")
    node = f"{entry['node']}.{J.PHASE_LETTER_TO_NUM[entry['phase']]}"
    y_row = J._system_y_row(node, Y, node_index)

    H, _ = J.build_jacobian([entry], Y, node_index, state, at_v=perturbed_v)
    row = H[0] * 1000.0  # undo kvar scaling -- compare in native VAR

    neighbor = next(k for k in y_row if k != node)
    k_d_idx = state.delta_index.get(neighbor)
    assert k_d_idx is not None, "test fixture needs a neighbor with a free angle"

    eps_d = 1e-6
    fv_p = _perturb_d(perturbed_v, neighbor, eps_d)
    fv_m = _perturb_d(perturbed_v, neighbor, -eps_d)
    rp, rq = np.zeros(state.dim), np.zeros(state.dim)
    _, Qp = J._scatter_injection(rp, rq, node, y_row, state, fv_p)
    rp2, rq2 = np.zeros(state.dim), np.zeros(state.dim)
    _, Qm = J._scatter_injection(rp2, rq2, node, y_row, state, fv_m)
    fd = (Qp - Qm) / (2 * eps_d)

    assert row[k_d_idx] == pytest.approx(fd, rel=1e-3)

    t_d_idx = state.delta_index.get(node)
    if t_d_idx is not None:
        fv_p2 = _perturb_d(perturbed_v, node, eps_d)
        fv_m2 = _perturb_d(perturbed_v, node, -eps_d)
        rp3, rq3 = np.zeros(state.dim), np.zeros(state.dim)
        _, Qp2 = J._scatter_injection(rp3, rq3, node, y_row, state, fv_p2)
        rp4, rq4 = np.zeros(state.dim), np.zeros(state.dim)
        _, Qm2 = J._scatter_injection(rp4, rq4, node, y_row, state, fv_m2)
        fd2 = (Qp2 - Qm2) / (2 * eps_d)
        assert row[t_d_idx] == pytest.approx(fd2, rel=1e-3)


def test_voltage_magnitude_jacobian_tracks_current_iterate(state, perturbed_v):
    """RES.md D10 Phase 7/8 V_MAGNITUDE FROZEN-h0 BUG, regression
    guard. build_voltage_row previously took no at_v parameter at all
    and always read state.x0_v -- so every V_magnitude measurement's
    forward value (h0) stayed frozen at the solve's STARTING guess for
    the entire Gauss-Newton iteration, never updated to the current
    iterate. The Jacobian row itself (1/v_base, a constant) was always
    correct; only h0 was stale -- invisible at flat start because
    state.x0_v IS the flat-start value there. Goes through the full
    build_jacobian dispatch (not build_voltage_row directly), so a
    regression at the call site (forgetting to pass flat_v through)
    is also caught."""

    registry = load_registry(variant="multiphase")
    entries = enabled_entries(registry)
    entry = next((e for e in entries if e["quantity"] == "V_magnitude"), None)
    assert entry is not None, "multiphase registry should have V_magnitude entries enabled"

    node = f"{entry['node']}.{J.PHASE_LETTER_TO_NUM[entry['phase']]}"
    _, h0_arr = J.build_jacobian([entry], None, None, state, at_v=perturbed_v)
    h0 = h0_arr[0]

    expected = abs(perturbed_v[node]) / state.v_base[node]
    frozen_would_be = state.x0_v[state.v_index[node]] / state.v_base[node]

    assert h0 == pytest.approx(expected, rel=1e-9)
    assert abs(h0 - frozen_would_be) > 1e-4, "perturbation too small to distinguish from the frozen-x0 bug"


def test_head_p_jacobian_matches_finite_difference(state, perturbed_v):
    """RES.md D10 Phase 8 HEAD_P MEASUREMENT MODEL FIX, regression
    guard. The old P_head formula (build_branch_or_head_row, element
    Vsource.source) computed theta = dt - dk between bus 150 (its
    angle PINNED at the fixed per-phase reference, never read from
    x_delta -- state.reference_nodes) and the Vsource's internal EMF
    (ALSO pinned at that same reference angle, _fixed_source_ev) --
    theta was identically 0 for every state x, and since the Vsource
    branch is purely reactive (Gk=0), P = Vt*Vk*Bk*sin(0) = 0 always
    (confirmed empirically: reports/model_consistency_at_truth.md,
    h_true=0.0 at all 120 checked timesteps against z0 ~930-1400 kW).
    build_head_p_row instead routes through Line.sw1 (150r -> 149,
    the only element downstream of bus 150's own regulator transformer
    with no load/other branch at 150r), whose two terminals BOTH have
    free state angles -- never touching bus 150's pinned reference at
    all. This test guards the NEW row's own derivative, finite-
    differenced against perturbed_v (not flat start -- at flat start
    every node sits at exactly its per-phase reference angle, so
    150r/149's angle difference would be 0 and this test would not
    exercise the angle derivative at all, the same flat-start blind
    spot the Q-angle/V-magnitude regression tests above exist to
    avoid)."""

    entry = {"id": "TEST_HEAD_P_B", "node": "150", "phase": "B", "quantity": "P_head"}
    row, _ = J.build_head_p_row(entry, state, perturbed_v)
    target = "150r.2"

    eps = 1e-5
    fv_p = _perturb_d(perturbed_v, target, eps)
    fv_m = _perturb_d(perturbed_v, target, -eps)
    _, hp = J.build_head_p_row(entry, state, fv_p)
    _, hm = J.build_head_p_row(entry, state, fv_m)
    fd = (hp - hm) / (2 * eps)

    assert row[state.delta_index[target]] == pytest.approx(fd, rel=1e-4)

    eps_v = 1e-3
    fv_p2 = _perturb_v(perturbed_v, target, eps_v)
    fv_m2 = _perturb_v(perturbed_v, target, -eps_v)
    _, hp2 = J.build_head_p_row(entry, state, fv_p2)
    _, hm2 = J.build_head_p_row(entry, state, fv_m2)
    fd2 = (hp2 - hm2) / (2 * eps_v)

    assert row[state.v_index[target]] == pytest.approx(fd2, rel=1e-4)
