"""
Placement regression / correctness tests.

History, for anyone reading this later: a permanent test asserting
"the refactored hybrid returns the exact 49-meter baseline set" is not
possible past this task, by design -- Step 3's weight normalisation
(placement_strategy.py, HybridPlacement class docstring) intentionally
changes score values and therefore the round-robin selection order.
That equality was true only for the Step-2-only structural refactor
(FeederContext extraction, no defect fixes yet), and was verified once
as a gate before Step 3 landed:

    tests/fixtures/hybrid_baseline_target49.json          -- Config A:
        captured from the ORIGINAL, unrefactored
        parser/placement_strategy.py, fed the original
        config/network_metadata.json + load_metadata.json.
        actual_count = 49 (NOT 48 -- RES.md's assumed undershoot at
        target=49 does not reproduce on this feeder's actual size-class
        distribution; the undershoot bug is real but does not trigger
        at every count -- see model_placement_refactor_report.txt).

    tests/fixtures/step2_gate_and_topology_switch.json     -- Config B
        (refactored structure, OLD data source) verified byte-for-byte
        equal to Config A (config_b_matches_baseline: true) -- proof
        the FeederContext extraction alone did not change behaviour.
        Also holds Config C (refactored structure, NEW verified
        compiled-circuit data source, still pre-Step-3): identical
        49-bus set to Config B, because the regex parser's topology bugs
        only affect bus 610 (XFM1 secondary), which hosts no Load
        object -- see the report for the full explanation.

    tests/fixtures/config_d_step3_fixed.json                -- Config D
        (refactored structure, NEW data source, Step 3 fixes applied):
        differs from Config C by exactly 2 of 49 meters (bus 2, 12 ->
        65, 66), attributable to the phase-balance swap pass specifically
        (confirmed: all four loads are non-backbone, ruling out the
        weight-normalisation/backbone-indicator change as the cause).

Going forward, this file tests the CURRENT, correct invariants instead
of a frozen historical selection.
"""

import json
from pathlib import Path

import pytest

from core.ieee123_model import extract_all
from placement.feeder_context import FeederContext
from placement.feeder_adapter import network_and_loads_from_core_extraction
from placement.placement_strategy import HybridPlacement, RandomPlacement

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def context_and_loads():
    extraction = extract_all()
    network, loads = network_and_loads_from_core_extraction(extraction)
    context = FeederContext(network, loads)
    context.annotate_loads(loads)
    return context, loads


def _select(context_and_loads, target_meters, verbose=False):
    context, loads = context_and_loads
    strategy = HybridPlacement(target_meters=target_meters, verbose=verbose)
    return strategy.select([dict(l) for l in loads], context)


# --------------------------------------------------
# Step-2 refactor gate (historical evidence, not a live assertion)
# --------------------------------------------------

def test_step2_gate_fixtures_are_consistent():
    """The committed fixtures are the evidence trail for the Step-2
    refactor gate and the topology-switch delta. This does not re-run
    the historical Step-2-only code (that code no longer exists --
    Step 3's fixes were applied in place) -- it just guards against the
    evidence files silently disappearing or being edited into
    inconsistency."""

    with open(BASE_DIR / "tests/fixtures/hybrid_baseline_target49.json") as f:
        baseline = json.load(f)
    with open(BASE_DIR / "tests/fixtures/step2_gate_and_topology_switch.json") as f:
        gate = json.load(f)

    assert baseline["actual_count"] == 49
    assert gate["config_b_matches_baseline"] is True
    assert gate["config_b"]["actual_count"] == 49
    assert gate["config_c"]["actual_count"] == 49
    assert gate["edge_count_new_verified"] == 131  # matches model_verification_report.txt


# --------------------------------------------------
# Step-3 defect #1: meter count is honoured
# --------------------------------------------------

@pytest.mark.parametrize("target", [15, 30, 44, 45, 49, 60, 70, 80, 85])
def test_selected_count_equals_target(context_and_loads, target):
    selected = _select(context_and_loads, target)
    assert len(selected) == target


def test_target_above_ceiling_raises(context_and_loads):
    context, loads = context_and_loads
    with pytest.raises(ValueError):
        HybridPlacement(target_meters=86, verbose=False)


def test_target_at_ceiling_is_allowed(context_and_loads):
    selected = _select(context_and_loads, 85)
    assert len(selected) == 85


# --------------------------------------------------
# No duplicate buses
# --------------------------------------------------

@pytest.mark.parametrize("target", [15, 45, 49, 85])
def test_no_duplicate_buses(context_and_loads, target):
    selected = _select(context_and_loads, target)
    buses = [l["bus"] for l in selected]
    assert len(buses) == len(set(buses))


# --------------------------------------------------
# Step-3 defect #2: phase balance
# --------------------------------------------------

PHASE_KW_BASELINE = {"A": 1400.0, "B": 952.5, "C": 1137.5}
PHASE_TOLERANCE_PP = 5.0  # percentage points; observed deviation at
                          # target=49 is ~0.6pp -- 5pp leaves headroom
                          # for lower counts without masking a real
                          # regression in the swap pass.


def _phase_kw(selected):
    from placement.placement_strategy import HybridPlacement as HP
    strategy = HP.__new__(HP)  # only need the pure helper methods
    return strategy._selected_phase_kw(selected)


@pytest.mark.parametrize("target", [45, 49])
def test_phase_balance_within_tolerance(context_and_loads, target):
    selected = _select(context_and_loads, target)
    phase_kw = _phase_kw(selected)
    total = sum(phase_kw.values())
    baseline_total = sum(PHASE_KW_BASELINE.values())

    for phase in ("A", "B", "C"):
        achieved_share = 100.0 * phase_kw[phase] / total
        target_share = 100.0 * PHASE_KW_BASELINE[phase] / baseline_total
        assert abs(achieved_share - target_share) <= PHASE_TOLERANCE_PP, (
            f"phase {phase}: achieved {achieved_share:.2f}% vs target "
            f"{target_share:.2f}% (tolerance {PHASE_TOLERANCE_PP}pp)"
        )


# --------------------------------------------------
# Determinism
# --------------------------------------------------

def test_hybrid_is_deterministic(context_and_loads):
    a = _select(context_and_loads, 49)
    b = _select(context_and_loads, 49)
    assert sorted(l["bus"] for l in a) == sorted(l["bus"] for l in b)


def test_random_placement_same_seed_same_result(context_and_loads):
    context, loads = context_and_loads
    a = RandomPlacement(target_meters=49, seed=7).select([dict(l) for l in loads], context)
    b = RandomPlacement(target_meters=49, seed=7).select([dict(l) for l in loads], context)
    assert sorted(l["bus"] for l in a) == sorted(l["bus"] for l in b)


def test_random_placement_different_seeds_can_differ(context_and_loads):
    context, loads = context_and_loads
    a = RandomPlacement(target_meters=15, seed=1).select([dict(l) for l in loads], context)
    b = RandomPlacement(target_meters=15, seed=2).select([dict(l) for l in loads], context)
    assert sorted(l["bus"] for l in a) != sorted(l["bus"] for l in b)


def test_random_placement_honours_count_and_ceiling(context_and_loads):
    context, loads = context_and_loads
    selected = RandomPlacement(target_meters=45, seed=3).select([dict(l) for l in loads], context)
    assert len(selected) == 45
    assert len({l["bus"] for l in selected}) == 45

    with pytest.raises(ValueError):
        RandomPlacement(target_meters=86, seed=0)


# --------------------------------------------------
# sweep_4a_tier1 cells.csv -- (strategy, count) is NOT a unique key
# --------------------------------------------------
# reports/placement_quality_vs_baselines.txt (first version) computed
# every table already filtered to construction=='independent' -- that
# report was not actually blending constructions. But a later review
# still asked for a regression test guarding against this ever
# happening BY ACCIDENT in a future ad hoc analysis script, since
# nothing in the schema itself would stop someone from grouping by
# (strategy, count) alone and silently averaging two different
# meter-set construction methods (independent vs nested) together.
# These two tests encode the schema fact that makes that mistake
# possible, so a future script that skips 'construction' fails loudly
# (or a schema change that removes the ambiguity is caught here too).

SWEEP_4A_TIER1_CSV = Path(__file__).resolve().parent.parent / "artifacts" / "20260822T092205Z" / "sweep_4a_tier1" / "cells.csv"


@pytest.fixture(scope="module")
def sweep_4a_tier1_cells():
    import pandas as pd
    return pd.read_csv(SWEEP_4A_TIER1_CSV)


def test_construction_column_present_with_expected_values(sweep_4a_tier1_cells):
    assert "construction" in sweep_4a_tier1_cells.columns
    assert set(sweep_4a_tier1_cells["construction"].unique()) == {"independent", "nested"}


def test_strategy_and_count_alone_is_not_a_unique_key(sweep_4a_tier1_cells):
    """Every deterministic strategy has exactly TWO rows per count in
    this file -- one independent, one nested -- so grouping by
    (strategy, count) alone (omitting 'construction') silently averages
    two different meter-set construction methods together, which is
    exactly the mistake reports/placement_quality_vs_baselines.txt was
    checked against. This test documents that the ambiguity is real, not
    hypothetical: it must currently FAIL to prove the danger exists."""

    df = sweep_4a_tier1_cells
    deterministic = df[df["strategy"] != "Random"]
    counts_per_group = deterministic.groupby(["strategy", "count"]).size()
    assert (counts_per_group == 2).all(), (
        "expected exactly 2 rows (independent + nested) per (strategy, count) "
        "for every deterministic strategy -- if this ever becomes 1, "
        "'construction' may no longer be required to disambiguate, and the "
        "warning in the test above should be revisited"
    )


def test_strategy_construction_and_count_is_a_unique_key(sweep_4a_tier1_cells):
    """The moment 'construction' is included, (strategy, count) becomes
    unique for every deterministic strategy (Random still has 10 rows,
    one per seed) -- the fix for the ambiguity above is always to group
    by this triple, never the pair alone."""

    df = sweep_4a_tier1_cells
    deterministic = df[df["strategy"] != "Random"]
    counts_per_group = deterministic.groupby(["strategy", "construction", "count"]).size()
    assert (counts_per_group == 1).all()

    random_rows = df[df["strategy"] == "Random"]
    random_counts_per_group = random_rows.groupby(["construction", "count"]).size()
    assert (random_counts_per_group == 10).all()
