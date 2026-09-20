"""
RES.md D10 Phase 5 acceptance tests -- pseudo-measurement sigma
estimation and its distinctness from zero-injection.
"""

import pytest

from observability.pseudo_measurements import (
    estimate_sigma_pseudo, summarize_sigma_pseudo,
    build_pseudo_entries_with_estimated_sigma,
)


def test_sigma_pseudo_is_estimated_per_hour_not_flat():
    per_hour = estimate_sigma_pseudo("s2b")

    assert len(per_hour) == 24
    sigmas = [v["sigma_p"] for v in per_hour.values()]
    assert all(s is not None and s > 0 for s in sigmas)
    # genuinely per-bucket, not a single flat value copy-pasted 24 times
    assert len(set(round(s, 6) for s in sigmas)) > 5


def test_sigma_pseudo_uses_held_out_data_not_in_sample():
    """The train/held-out split must actually partition the series --
    train and held-out bucket counts should each be a meaningful
    fraction of the total, not one empty and one full (which would mean
    the split silently degenerated to in-sample residual estimation)."""

    per_hour = estimate_sigma_pseudo("s2b")
    hour0 = per_hour[0]
    assert hour0["n_train"] > 0
    assert hour0["n_heldout"] > 0
    assert hour0["n_train"] > hour0["n_heldout"]  # 75/25 split


def test_summarize_averages_across_hours():
    per_hour = estimate_sigma_pseudo("s107b")
    summary = summarize_sigma_pseudo(per_hour)

    assert summary["n_hours_estimated"] == 24
    assert summary["sigma_p"] > 0
    assert summary["sigma_q"] > 0
    # sanity: summary sigma should sit within the per-hour range
    hour_sigmas = [v["sigma_p"] for v in per_hour.values()]
    assert min(hour_sigmas) <= summary["sigma_p"] <= max(hour_sigmas)


def test_build_pseudo_entries_carries_estimated_sigma_not_a_flat_fraction():
    resolver = {"s2b": ("2", "B"), "s107b": ("107", "B")}
    entries, summaries = build_pseudo_entries_with_estimated_sigma(
        list(resolver.keys()), lambda name: resolver[name]
    )

    assert len(entries) == 4  # 2 loads x (P,Q)
    ids = {e["id"] for e in entries}
    assert ids == {"PSEUDO_s2b_P", "PSEUDO_s2b_Q", "PSEUDO_s107b_P", "PSEUDO_s107b_Q"}

    for e in entries:
        assert e["sigma"] > 0
        assert e["load_name"] in resolver

    # the two loads' sigmas must differ -- confirms genuine per-node
    # estimation, not one shared constant
    p_sigmas = {e["load_name"]: e["sigma"] for e in entries if e["quantity"] == "P_injection"}
    assert p_sigmas["s2b"] != p_sigmas["s107b"]


def test_pseudo_and_zero_injection_use_disjoint_id_namespaces():
    """RES.md: pseudo-measurements (real, unmeasured consumption) and
    zero-injection (load-free, exact) must never collapse into one
    mechanism. Cheap, durable guard: their id prefixes never overlap."""

    from observability.zero_injection import build_zero_injection_entries
    import opendssdirect as dss
    from simulation.opendss_utils import compile_feeder

    compile_feeder()
    zi_entries = build_zero_injection_entries(dss.Circuit.AllNodeNames())
    zi_ids = {e["id"] for e in zi_entries}

    resolver = {"s2b": ("2", "B")}
    pseudo_entries, _ = build_pseudo_entries_with_estimated_sigma(
        list(resolver.keys()), lambda name: resolver[name]
    )
    pseudo_ids = {e["id"] for e in pseudo_entries}

    assert zi_ids.isdisjoint(pseudo_ids)
    assert all(i.startswith("ZI_") for i in zi_ids)
    assert all(i.startswith("PSEUDO_") for i in pseudo_ids)

    # and structurally: no zero-injection node is ever an unmetered
    # LOAD node (they are load-free by definition) -- a load bus with
    # real consumption must never appear as a zero-injection constraint
    zi_buses = {e["node"] for e in zi_entries}
    pseudo_buses = {e["node"] for e in pseudo_entries}
    assert zi_buses.isdisjoint(pseudo_buses)
