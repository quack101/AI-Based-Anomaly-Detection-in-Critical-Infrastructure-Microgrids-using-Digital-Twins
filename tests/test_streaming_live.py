"""
Live-path unit tests (no Kafka broker needed): injector A1/A2 path, detector
consumer assembly + no-fabrication rule, conditional head publishing.
"""

import copy

import numpy as np
import pytest

import streaming.producer as producer_mod
from streaming.producer import Producer
from streaming.injector import Injector, apply_attack
from streaming.detector_consumer import DetectorConsumer
from attacks.library import AttackSpec, BiasContext, apply_bias_timestep


@pytest.fixture(scope="module")
def bias_ctx():
    return BiasContext("multiphase")


# ---------------- injector A1/A2 path ----------------

def _msg(t, meter, meas):
    return {"timestamp": t, "real_timestamp": "2018-05-25 00:00:00", "meter_id": meter, "measurements": dict(meas)}


def test_injector_applies_a1_and_labels_it(bias_ctx):
    spec = AttackSpec("A1", 5, 3, "step", seed=7, k=5, channel_class="PQ", pq_rel=0.10)
    # use the spec's own selection to know which channels will be hit
    ids = None
    probe = {c: 100.0 for c in bias_ctx.real_ids}
    _, info = apply_bias_timestep(probe, spec, 5, bias_ctx)
    ids = info["targets"]
    assert len(ids) == 5
    inj = Injector(expected_meters=2, attack_specs=[spec], bias_context=bias_ctx)
    half = len(bias_ctx.real_ids) // 2
    m1 = {c: 100.0 for c in bias_ctx.real_ids[:half]}
    m2 = {c: 100.0 for c in bias_ctx.real_ids[half:]}
    inj.ingest(_msg(5, "A", m1))
    delivered, label = inj.ingest(_msg(5, "B", m2))
    got = {}
    for d in delivered:
        got.update(d["measurements"])
    changed = sorted(c for c, v in got.items() if v != 100.0)
    assert changed == sorted(ids)
    assert label["attacked"] == 1 and label["attack_class"] == "A1"
    assert label["targets"] == sorted(ids)
    assert label["magnitude"] == 0.10
    # a timestep outside the window is untouched and labelled clean
    inj.ingest(_msg(9, "A", m1))
    delivered, label = inj.ingest(_msg(9, "B", m2))
    assert all(v == 100.0 for d in delivered for v in d["measurements"].values())
    assert label["attacked"] == 0 and label["attack_class"] is None and label["targets"] == []


def test_injector_does_not_mutate_the_raw_message(bias_ctx):
    spec = AttackSpec("A2", 0, 2, "step", seed=3, pq_rel=0.20, v_abs_pu=0.01)
    raw = _msg(0, "A", {c: 50.0 for c in bias_ctx.real_ids})
    before = copy.deepcopy(raw)
    delivered = apply_attack(raw, spec, bias_ctx)
    assert raw == before                      # raw untouched
    assert delivered != raw                   # delivered biased


def test_injector_still_rejects_a3_a4_a5_and_legacy_dicts(bias_ctx):
    raw = _msg(0, "A", {"M001_P": 1.0})
    for t in ("A3", "A4", "A5"):
        with pytest.raises(NotImplementedError):
            apply_attack(raw, AttackSpec(t, 0, 1, "step"), bias_ctx)
    with pytest.raises(NotImplementedError):
        apply_attack(raw, {"class": "A1"})


# ---------------- detector consumer: assembly + no fabrication ----------------

class _StubCore:
    """Stands in for DetectorCore: records what it was asked to score."""
    def __init__(self, cols):
        self.cols = cols
        self.calls = []

    def score_vector(self, window, real_ts):
        self.calls.append(np.array(window))
        return {"J": 1.0, "maxLNR": 1.0, "bdd_score": 0.5, "ae_mahalanobis": 2.0,
                "fusion_score": 0.1, "alarm": False, "top_channel": self.cols[0]}


def _feed(dc, t, cols, value, skip=()):
    out = None
    chunks = [cols[0::2], cols[1::2]]                     # two partial messages, like per-meter messages
    for ch in chunks:
        meas = {c: value for c in ch if c not in skip}
        r = dc.ingest({"timestamp": t, "real_timestamp": "2018-05-25 00:00:00", "meter_id": "x", "measurements": meas})
        out = r if r is not None else out
    return out


def test_detector_consumer_assembles_in_registry_order_and_waits_for_warmup():
    cols = [f"C{i}" for i in range(6)]
    core = _StubCore(cols)
    dc = DetectorConsumer(core, window_timeout_s=1e9)
    scored = []
    for t in range(12):
        r = _feed(dc, t, cols, float(t))
        scored.append(r)
    assert all(r["scored"] is False and r["reason"] == "warmup" for r in scored[:9])
    assert scored[9]["scored"] is True and scored[11]["scored"] is True
    w = core.calls[0]
    assert w.shape == (10, 6) and np.array_equal(w[:, 0], np.arange(10.0))   # steps 0..9, oldest first


def test_detector_consumer_never_fabricates_missing_channels():
    cols = [f"C{i}" for i in range(6)]
    core = _StubCore(cols)
    dc = DetectorConsumer(core, window_timeout_s=1e9)
    for t in range(10):
        _feed(dc, t, cols, 1.0)
    assert len(core.calls) == 1
    _feed(dc, 10, cols, 1.0, skip=("C3",))                 # one channel never arrives
    results = dc.flush_all()
    assert len(results) == 1
    assert results[0]["scored"] is False and results[0]["reason"] == "incomplete"
    assert results[0]["n_missing"] == 1
    assert len(core.calls) == 1                            # no score was produced from a filled-in vector
    # consecutiveness is broken: the next 9 complete steps are warmup again
    for t in range(11, 20):
        r = _feed(dc, t, cols, 1.0)
        assert r["scored"] is False and r["reason"] == "warmup"
    r = _feed(dc, 20, cols, 1.0)
    assert r["scored"] is True


# ---------------- conditional head publishing ----------------

def test_head_message_only_under_multiphase(tmp_path, monkeypatch):
    monkeypatch.setattr(producer_mod, "GROUND_TRUTH_LOG", tmp_path / "gt.csv")
    default = Producer(registry_variant="default", add_noise=False, seed=0)
    raw, _ = default.run_timestep(0)
    assert default.head_entries == []
    assert all(m["meter_id"] != "HEAD" for m in raw) and len(raw) == len(default.representative_meters)

    multi = Producer(registry_variant="multiphase", add_noise=False, seed=0)
    raw, _ = multi.run_timestep(0)
    head = [m for m in raw if m["meter_id"] == "HEAD"]
    assert len(head) == 1 and len(raw) == len(multi.representative_meters) + 1
    assert sorted(head[0]["measurements"]) == sorted(e["id"] for e in multi.head_entries)
    assert len(head[0]["measurements"]) == 9                      # HEAD_P/Q/V x A/B/C, HEAD_P included
    merged = {}
    for m in raw:
        merged.update(m["measurements"])
    assert len(merged) == 168
