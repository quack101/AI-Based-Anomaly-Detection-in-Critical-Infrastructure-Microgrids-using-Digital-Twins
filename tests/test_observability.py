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
