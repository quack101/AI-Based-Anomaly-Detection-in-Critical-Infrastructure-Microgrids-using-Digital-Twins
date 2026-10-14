"""
HybridV3Placement: valid selections at every budget, phase balance within
the same 5 pp tolerance as tests/test_placement_regression.py, and fewer
dead-end buses than v2 (the reason v3 exists).
"""

import pytest

from core.ieee123_model import extract_all
from placement.feeder_context import FeederContext
from placement.feeder_adapter import network_and_loads_from_core_extraction
from placement.hybrid_v2 import HybridV2Placement
from placement.hybrid_v3 import HybridV3Placement
from placement.sweep_metrics import compute_phase_dev_pp

TARGETS = [20, 30, 45, 49, 60, 70]
PHASE_TOLERANCE_PP = 5.0


@pytest.fixture(scope="module")
def context_and_loads():
    network, loads = network_and_loads_from_core_extraction(extract_all())
    context = FeederContext(network, loads)
    context.annotate_loads(loads)
    return context, loads


def _select(context_and_loads, strategy):
    context, loads = context_and_loads
    return strategy.select([dict(l) for l in loads], context)


def _dead_ends(context, selected):
    return sum(context.bus_degree(l["bus"]) < 2 for l in selected)


@pytest.mark.parametrize("target", TARGETS)
def test_valid_selection(context_and_loads, target):
    selected = _select(context_and_loads, HybridV3Placement(target))
    assert len(selected) == target
    assert len({l["bus"] for l in selected}) == target


@pytest.mark.parametrize("target", TARGETS)
def test_phase_balance_within_tolerance(context_and_loads, target):
    strategy = HybridV3Placement(target)
    phase_kw = strategy._selected_phase_kw(_select(context_and_loads, strategy))
    assert compute_phase_dev_pp(phase_kw["A"], phase_kw["B"], phase_kw["C"]) <= PHASE_TOLERANCE_PP


@pytest.mark.parametrize("target", TARGETS)
def test_fewer_dead_ends_than_v2(context_and_loads, target):
    context, _ = context_and_loads
    v2 = _dead_ends(context, _select(context_and_loads, HybridV2Placement(target)))
    v3 = _dead_ends(context, _select(context_and_loads, HybridV3Placement(target)))
    assert v3 < v2


def test_deterministic(context_and_loads):
    a = [l["name"] for l in _select(context_and_loads, HybridV3Placement(49))]
    b = [l["name"] for l in _select(context_and_loads, HybridV3Placement(49))]
    assert a == b


def test_unknown_connection_priority_rejected():
    with pytest.raises(ValueError):
        HybridV3Placement(49, connection_priority="degree")
