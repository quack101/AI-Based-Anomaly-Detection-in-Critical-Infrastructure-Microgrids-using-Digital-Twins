"""
RES.md D10 Phase 2 acceptance tests.
"""

import random

import opendssdirect as dss
import pytest

from core.ieee123_model import extract_all
from sensors.registry import load_registry, enabled_entries, entry_ids, validate_registry
from sensors.measurement_functions import measure, PHASE_LETTER_TO_NUM
from sensors.sensor_layer import build_measurement_vector, build_measurement_dict


@pytest.fixture(scope="module")
def solved_circuit():
    extract_all()  # compiles + solves; leaves the circuit active
    return True


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def entries(registry):
    return enabled_entries(registry)


# --------------------------------------------------
# Registry structure
# --------------------------------------------------

def test_registry_validates(registry):
    assert validate_registry(registry) is True


def test_current_configuration_is_98_enabled(entries):
    """RES.md D10 Phase 2: 'Start with the 49 metered nodes at (P, Q)
    -- the current configuration, m = 98.'"""

    assert len(entries) == 98
    assert all(e["quantity"] in ("P_injection", "Q_injection") for e in entries)


def test_every_id_unique_across_whole_registry(registry):
    ids = [e["id"] for e in registry]
    assert len(ids) == len(set(ids))


def test_disabled_categories_present_but_off(registry):
    """Head, branch and meter |V| entries exist (so Phase 4/5 can flip
    them on as a config edit) but are disabled by default."""

    head = [e for e in registry if e["id"].startswith("HEAD_")]
    branch = [e for e in registry if e["id"].startswith("BR_")]
    meter_v = [e for e in registry if e["quantity"] == "V_magnitude" and e["id"].startswith("M")]

    assert len(head) == 12
    assert all(not e["enabled"] for e in head)
    assert len(branch) > 0
    assert all(not e["enabled"] for e in branch)
    assert len(meter_v) == 49
    assert all(not e["enabled"] for e in meter_v)


def test_sigma_is_per_sensor_not_a_global_constant(registry):
    sigmas = {e["sigma"] for e in registry if e["quantity"] in ("P_injection", "Q_injection")}
    assert len(sigmas) > 10  # far more than 1 -- genuinely per-sensor


def test_every_registry_entry_has_a_known_quantity_function(registry):
    from sensors.measurement_functions import MEASUREMENT_FUNCTIONS

    unknown = [e["quantity"] for e in registry if e["quantity"] not in MEASUREMENT_FUNCTIONS]
    assert not unknown


# --------------------------------------------------
# THE key test: noise-free h(x_true) reproduces the OpenDSS solve
# --------------------------------------------------

def test_noise_free_h_matches_direct_opendss_query(solved_circuit, entries):
    """Computed INDEPENDENTLY of sensors/measurement_functions.py, by
    querying opendssdirect directly in this test, for a sample spanning
    every quantity type currently enabled/available -- not just calling
    measure() again and trivially agreeing with itself."""

    checked_quantities = set()

    for entry in entries:
        if entry["quantity"] == "P_injection":
            dss.Circuit.SetActiveElement(f"Load.{entry['load_name']}")
            node_order = dss.CktElement.NodeOrder()
            powers = dss.CktElement.Powers()
            idx = node_order.index(PHASE_LETTER_TO_NUM[entry["phase"]])
            expected = powers[2 * idx]
        elif entry["quantity"] == "Q_injection":
            dss.Circuit.SetActiveElement(f"Load.{entry['load_name']}")
            node_order = dss.CktElement.NodeOrder()
            powers = dss.CktElement.Powers()
            idx = node_order.index(PHASE_LETTER_TO_NUM[entry["phase"]])
            expected = powers[2 * idx + 1]
        else:
            continue

        got = measure(entry)
        assert got == pytest.approx(expected, abs=1e-9)
        checked_quantities.add(entry["quantity"])

    assert checked_quantities == {"P_injection", "Q_injection"}


def test_noise_free_h_for_voltage_head_and_branch(solved_circuit, registry):
    """Same independent-query check for the disabled categories (|V|,
    substation head, branch flow/current) -- enabled=False doesn't mean
    untested; it means not currently published."""

    # Voltage at a meter
    v_entry = next(e for e in registry if e["id"] == "M001_V")
    dss.Circuit.SetActiveBus(v_entry["node"])
    nodes = dss.Bus.Nodes()
    pu = dss.Bus.puVmagAngle()
    idx = nodes.index(PHASE_LETTER_TO_NUM[v_entry["phase"]])
    assert measure(v_entry) == pytest.approx(pu[2 * idx], abs=1e-9)

    # Substation head P
    head_entry = next(e for e in registry if e["id"] == "HEAD_P_A")
    dss.Circuit.SetActiveElement("Vsource.source")
    node_order = dss.CktElement.NodeOrder()[: dss.CktElement.NumPhases()]
    idx = node_order.index(PHASE_LETTER_TO_NUM["A"])
    expected = -dss.CktElement.Powers()[2 * idx]
    assert measure(head_entry) == pytest.approx(expected, abs=1e-9)

    # Branch flow
    branch_entry = next(e for e in registry if e["id"].startswith("BR_") and e["quantity"] == "P_flow")
    dss.Circuit.SetActiveElement(f"Line.{branch_entry['element_name']}")
    node_order = dss.CktElement.NodeOrder()[: dss.CktElement.NumPhases()]
    idx = node_order.index(PHASE_LETTER_TO_NUM[branch_entry["phase"]])
    expected = dss.CktElement.Powers()[2 * idx]
    assert measure(branch_entry) == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------
# CRITICAL: z / registry / (future H) ordering agreement
# --------------------------------------------------

def test_z_order_matches_registry_order(solved_circuit, entries):
    """z[i] must correspond to entries[i] -- built in ONE pass, not
    re-derived. Verified by recomputing each entry's h() value
    independently (via a SEPARATE call to measure(), not by reusing
    build_measurement_vector's internal loop) and checking positional
    agreement."""

    z, ids = build_measurement_vector(entries, add_noise=False)

    assert ids == entry_ids(entries)
    assert len(z) == len(entries)

    for i, entry in enumerate(entries):
        assert z[i] == pytest.approx(measure(entry), abs=1e-9), (
            f"z[{i}] does not match entries[{i}] ({entry['id']}) -- "
            f"ordering invariant violated"
        )


def test_registry_order_is_deterministic_across_independent_calls(registry):
    """Simulates 'two separate loops' building z and H from the raw
    registry independently -- they must still agree, because both are
    required to filter/order via enabled_entries(), which is a pure
    function of the (order-preserving) registry list."""

    entries_a = enabled_entries(registry)
    entries_b = enabled_entries(registry)

    assert entry_ids(entries_a) == entry_ids(entries_b)


def test_measurement_dict_keys_are_exactly_the_enabled_ids(solved_circuit, entries):
    """Acceptance: every published field traces to exactly one registry
    entry, and no more."""

    z_dict = build_measurement_dict(entries, add_noise=False)
    assert set(z_dict.keys()) == set(entry_ids(entries))
    assert len(z_dict) == len(entries)


# --------------------------------------------------
# Per-sensor noise model
# --------------------------------------------------

def test_noise_uses_each_entrys_own_sigma(solved_circuit, entries):
    rng = random.Random(0)

    low_sigma_entry = min(entries, key=lambda e: e["sigma"])
    high_sigma_entry = max(entries, key=lambda e: e["sigma"])

    n = 400

    low_draws = [
        build_measurement_vector([low_sigma_entry], add_noise=True, rng=rng)[0][0]
        for _ in range(n)
    ]
    high_draws = [
        build_measurement_vector([high_sigma_entry], add_noise=True, rng=rng)[0][0]
        for _ in range(n)
    ]

    import statistics
    low_std = statistics.pstdev(low_draws)
    high_std = statistics.pstdev(high_draws)

    assert low_std < high_std
    assert low_std == pytest.approx(low_sigma_entry["sigma"], rel=0.35)
    assert high_std == pytest.approx(high_sigma_entry["sigma"], rel=0.35)


def test_noise_free_disables_noise_exactly(solved_circuit, entries):
    a, _ = build_measurement_vector(entries[:5], add_noise=False)
    b, _ = build_measurement_vector(entries[:5], add_noise=False)
    assert a == b  # bit-identical, no RNG involved at all
