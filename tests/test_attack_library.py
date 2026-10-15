"""
attacks/library.py -- determinism, window/label consistency, and the
A3 load-redistribution invariants. Uses the first rows of the Phase 6
test telemetry (read-only) and one compiled feeder for the module.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attacks.library import attack, AttackSpec, AttackContext, profile_factors

CORPUS = Path(__file__).resolve().parent.parent / "simulation" / "phase6_corpus" / "test_telemetry.csv"
HEAD_P = ["HEAD_P_A", "HEAD_P_B", "HEAD_P_C"]


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(CORPUS, nrows=24)
    ts = df["real_timestamp"].tolist()
    z = df.drop(columns=["timestamp", "real_timestamp"])
    return z, ts, AttackContext()


def _changed_rows(z, za):
    return sorted(np.flatnonzero((za.values != z.values).any(axis=1)).tolist())


def test_deterministic_with_seed(data):
    z, ts, ctx = data
    spec = AttackSpec("A1", 3, 4, "step", magnitude=3.0, seed=7, k=20)
    za1, lb1 = attack(z, ts, spec, ctx)
    za2, lb2 = attack(z, ts, spec, ctx)
    pd.testing.assert_frame_equal(za1, za2)
    pd.testing.assert_frame_equal(lb1, lb2)
    za3, _ = attack(z, ts, AttackSpec("A1", 3, 4, "step", magnitude=3.0, seed=8, k=20), ctx)
    assert not za3.equals(za1)


@pytest.mark.parametrize("spec", [
    AttackSpec("A1", 2, 6, "step", magnitude=5.0, seed=1, k=5),
    AttackSpec("A1", 4, 12, "pulse", magnitude=5.0, seed=2, k=165),
    AttackSpec("A2", 5, 8, "ramp", magnitude=10.0, seed=3),
    AttackSpec("A3", 6, 3, "step", magnitude=0.2, seed=4, access="full"),
    AttackSpec("A3", 6, 2, "ramp", magnitude=0.5, seed=5, access="partial"),
])
def test_labels_match_window_and_outside_unchanged(data, spec):
    z, ts, ctx = data
    za, lb = attack(z, ts, spec, ctx)
    f = profile_factors(spec.profile, spec.duration)
    expected = [spec.start + w for w in range(spec.duration) if f[w] > 0]
    assert lb.index[lb["attacked"] == 1].tolist() == [z.index[p] for p in expected]
    assert _changed_rows(z, za) == expected
    outside = [p for p in range(len(z)) if p not in expected]
    assert (za.iloc[outside].values == z.iloc[outside].values).all()
    assert set(lb.loc[lb["attacked"] == 1, "attack_type"]) == {spec.type}
    for p in expected:
        targets = lb.iloc[p]["target_ids"].split(";")
        changed = z.columns[(za.iloc[p] != z.iloc[p]).values].tolist()
        assert sorted(targets) == sorted(changed)


def test_a2_targets_one_meter(data):
    z, ts, ctx = data
    _, lb = attack(z, ts, AttackSpec("A2", 0, 1, "step", magnitude=5.0, seed=9), ctx)
    ids = lb.iloc[0]["target_ids"].split(";")
    assert 1 <= len(ids) <= 3
    assert len({i.replace("_P", "").replace("_Q", "").replace("_V", "") for i in ids}) == 1


def test_a3_total_load_and_head_balance(data):
    z, ts, ctx = data
    for frac in (0.1, 0.2, 0.5):
        za, lb = attack(z, ts, AttackSpec("A3", 7, 1, "step", magnitude=frac, seed=11, access="full"), ctx)
        r = lb.iloc[7]
        assert r["a3_pf_converged"]
        assert abs(r["a3_dP_total_kw"]) < 1e-6          # set-point total unchanged
        a_head = sum(za.iloc[7][c] - z.iloc[7][c] for c in HEAD_P)
        # head P change = change in losses + change in power actually consumed
        # (voltage-dependent loads); checked to within 0.05 kW
        assert abs(a_head - r["a3_dLoss_kw"] - r["a3_dP_consumed_kw"]) < 0.05
        assert abs(a_head) <= abs(r["a3_dLoss_kw"]) + abs(r["a3_dP_consumed_kw"]) + 0.05


def test_a3_partial_only_touches_the_two_meters(data):
    z, ts, ctx = data
    za, lb = attack(z, ts, AttackSpec("A3", 7, 1, "step", magnitude=0.5, seed=11, access="partial"), ctx)
    r = lb.iloc[7]
    loads = {r["a3_load_up"], r["a3_load_down"]}
    allowed = {e["id"] for e in ctx.model.meas_entries if e.get("load_name") in loads}
    changed = set(z.columns[(za.iloc[7] != z.iloc[7]).values])
    assert changed and changed <= allowed
