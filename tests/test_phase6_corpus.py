"""
RES.md D10 Phase 6 acceptance tests -- time-based split correctness and
x_true extraction, independent of the (expensive) full 60-day run.
"""

import pandas as pd
import pytest

from simulation.opendss_utils import compile_feeder, get_solution_metrics
from observability.state_vector import build_state_vector, extract_solved_state
from validation.generate_phase6_corpus import compute_splits


def _synthetic_timestamps(n_days, per_day=1440):
    return pd.date_range("2018-04-04", periods=n_days * per_day, freq="1min")


def test_split_is_contiguous_by_calendar_day_not_random():
    timestamps = _synthetic_timestamps(60)
    split_of_row, boundaries = compute_splits(timestamps)

    assert len(split_of_row) == len(timestamps)
    assert set(split_of_row) == {"train", "val", "test"}

    # contiguity: once a row belongs to "val", nothing before it (in
    # time) is "test", and once "test" starts nothing after it reverts
    # -- i.e. the split labels appear in train*, val*, test* order with
    # no interleaving, which a correct day-boundary split guarantees
    # and a random shuffle would violate.
    seen = []
    for s in split_of_row:
        if not seen or seen[-1] != s:
            seen.append(s)
    assert seen == ["train", "val", "test"]


def test_split_covers_every_row_exactly_once():
    timestamps = _synthetic_timestamps(60)
    split_of_row, boundaries = compute_splits(timestamps)

    counts = {"train": 0, "val": 0, "test": 0}
    for s in split_of_row:
        counts[s] += 1

    assert sum(counts.values()) == len(timestamps)
    assert all(c > 0 for c in counts.values())


def test_split_ratio_is_approximately_70_15_15():
    timestamps = _synthetic_timestamps(60)
    split_of_row, boundaries = compute_splits(timestamps)

    n = len(timestamps)
    train_frac = (split_of_row == "train").sum() / n
    val_frac = (split_of_row == "val").sum() / n
    test_frac = (split_of_row == "test").sum() / n

    assert train_frac == pytest.approx(0.70, abs=0.02)
    assert val_frac == pytest.approx(0.15, abs=0.02)
    assert test_frac == pytest.approx(0.15, abs=0.02)


def test_split_boundaries_are_reported_and_consistent_with_row_labels():
    timestamps = _synthetic_timestamps(60)
    split_of_row, boundaries = compute_splits(timestamps)

    for name in ("train", "val", "test"):
        lo, hi, n_days = boundaries[name]
        assert pd.Timestamp(lo) <= pd.Timestamp(hi)
        # every row whose date falls in [lo, hi] must be labelled `name`
        dates = pd.to_datetime(timestamps).date
        in_range = (dates >= pd.Timestamp(lo).date()) & (dates <= pd.Timestamp(hi).date())
        assert (split_of_row[in_range] == name).all()


def test_extract_solved_state_matches_flat_start_shape():
    compile_feeder()
    state = build_state_vector()
    x_v, x_delta = extract_solved_state(state)

    assert len(x_v) == len(state.x0_v) == state.n_phase_nodes
    assert len(x_delta) == len(state.delta_index)


def test_extract_solved_state_v_pu_matches_get_solution_metrics():
    """Cross-check against an INDEPENDENT source of the same solved
    state (get_solution_metrics's own v_pu_per_phase, already used and
    trusted throughout this project) -- not just internal consistency."""

    import opendssdirect as dss

    compile_feeder()
    dss.Solution.Solve()
    metrics = get_solution_metrics(0.0)
    per_phase = metrics["v_pu_per_phase"]
    expected_min = min(p["min"] for p in per_phase.values())
    expected_max = max(p["max"] for p in per_phase.values())

    state = build_state_vector()
    x_v, x_delta = extract_solved_state(state)

    v_pu = [x_v[state.v_index[name]] / state.v_base[name] for name in state.all_node_names]
    assert min(v_pu) == pytest.approx(expected_min, abs=1e-6)
    assert max(v_pu) == pytest.approx(expected_max, abs=1e-6)
