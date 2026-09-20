"""
RES.md D10 Phase 7 acceptance tests -- the AC state estimator.

Structural checks (fast, no solve) that re-verify in code, not just
assert in a docstring, the three design properties this phase's task
description names as non-negotiable:
  1. zero-injection is HARD-constrained, never folded into the weighted
     (soft pseudo-measurement) block;
  2. Phase-5 load pseudo-measurements and Phase-7's regulator-adjacent
     pseudo-measurements both enter the weighted block with a real,
     positive, estimated sigma, and stay structurally distinct from
     zero-injection AND from each other (three id namespaces, three
     reasons for uncertainty);
  3. the plain-WLS alternative Sha25/this phase rejects really would be
     numerically unusable (a real cond() number, not an assertion).

Plus one real end-to-end convergence/accuracy check against a genuine
Phase 6 normal-operation timestamp -- the phase's own stated "basic
sanity bar" ("if it fails here, nothing downstream can be trusted").
The full-day, full-test-split run and its statistics live in
validation/phase7_fullday.py and
reports/phase7_state_estimator_report.txt, not here -- this file is the
fast, always-run acceptance gate, not the full validation campaign.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator, demonstrate_plain_wls_failure

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"


@pytest.fixture(scope="module")
def model():
    return build_estimator_model("multiphase")


@pytest.fixture(scope="module")
def estimator(model):
    return StateEstimator(model)


def test_zero_injection_entries_are_hard_not_weighted(model, estimator):
    """RES.md D10 Phase 7 acceptance: zero-injection uses HARD equality
    constraints, never soft pseudo-measurements -- explicitly verified,
    not assumed."""

    assert len(model.zi_entries) > 0
    for e in model.zi_entries:
        assert e["sigma"] == 0.0

    weighted_ids = {e["id"] for e in estimator._weighted_entries_for_hour(0)}
    zi_ids = {e["id"] for e in model.zi_entries}
    assert weighted_ids.isdisjoint(zi_ids), "zero-injection entries leaked into the weighted (soft) block"


def test_pseudo_and_regulator_entries_are_weighted_with_positive_sigma(model, estimator):
    """Phase 5 load pseudo-measurements and Phase 7's regulator-adjacent
    pseudo-measurements both enter the WEIGHTED block with a real,
    positive, estimated sigma -- distinct from zero-injection's exact 0."""

    weighted = estimator._weighted_entries_for_hour(12)
    weighted_by_id = {e["id"]: e for e in weighted}

    reg_ids = [e["id"] for e in model.reg_pseudo_entries]
    assert len(reg_ids) > 0
    for rid in reg_ids:
        assert rid.startswith("REGZI_")
        assert weighted_by_id[rid]["sigma"] > 0.0

    pseudo_ids = [eid for eid in weighted_by_id if eid.startswith("PSEUDO_")]
    assert len(pseudo_ids) == 2 * len(model.bus_phase_by_load)
    for pid in pseudo_ids:
        assert weighted_by_id[pid]["sigma"] > 0.0


def test_id_namespaces_are_disjoint(model, estimator):
    """Three structurally distinct categories of 'this should read
    zero/near-zero' claim -- ZI_ (exact), REGZI_ (tap-drift uncertain),
    PSEUDO_ (load uncertain) -- must never collide."""

    zi_ids = {e["id"] for e in model.zi_entries}
    reg_ids = {e["id"] for e in model.reg_pseudo_entries}
    pseudo_ids = {e["id"] for e in estimator._weighted_entries_for_hour(0) if e["id"].startswith("PSEUDO_")}

    assert zi_ids.isdisjoint(reg_ids)
    assert zi_ids.isdisjoint(pseudo_ids)
    assert reg_ids.isdisjoint(pseudo_ids)


def test_plain_wls_would_be_ill_conditioned(estimator):
    """RES.md D10 Phase 7 acceptance: show the REJECTED alternative
    (zero-injection folded in as a fake near-zero-sigma pseudo-
    measurement inside one plain unconstrained H^T W H) really would
    fail, as direct evidence for why the Lagrangian/null-space approach
    is necessary -- not merely asserted in a docstring."""

    cond_plain, cond_constrained = demonstrate_plain_wls_failure(estimator)
    assert cond_plain > 1e15  # beyond float64's usable ~1e16 precision -- numerically unusable
    assert cond_constrained < cond_plain


def test_recovers_x_true_on_normal_timestamp(model, estimator):
    """The phase's own stated basic sanity bar: 'confirm the estimator
    recovers x_true on a handful of Phase-6 normal timestamps to within
    a small, stated tolerance -- if it fails here, nothing downstream
    can be trusted.' Tolerance set from the full-test-split run
    (reports/phase7_state_estimator_report.txt): observed max per-node
    V error is consistently ~4-5% pu / ~0.05-0.09 rad, driven by
    ordinary WLS estimation error at sparsely-observed nodes (confirmed
    NOT concentrated at regulator-adjacent nodes, see the report) rather
    than a solver defect. Asserted here at a looser bound (10% pu / 0.3
    rad) than the typical observed max, so this fails only on a genuine
    regression, not ordinary run-to-run noise."""

    state = model.state
    v_base = np.array([state.v_base[n] for n in state.all_node_names])

    telem_cols = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1)
    npz = np.load(CORPUS_DIR / "test_x_true.npz", allow_pickle=True)

    row = df.iloc[0]
    z_real = {c: float(row[c]) for c in z_cols}
    result = estimator.estimate(z_real, row["real_timestamp"])

    assert result.converged
    assert result.dof >= 0

    err_v_pu = np.abs(result.x_v - npz["x_v"][0]) / v_base
    err_delta = np.abs(result.x_delta - npz["x_delta"][0])

    assert err_v_pu.max() < 0.10
    assert err_delta.max() < 0.30
