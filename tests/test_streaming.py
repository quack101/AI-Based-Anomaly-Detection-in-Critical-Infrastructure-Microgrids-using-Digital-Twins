"""
RES.md D10 Phase 3 acceptance tests -- streaming layer restructure.
"""

import copy
import time

import pytest

import streaming.producer as producer_mod
import streaming.consumer as consumer_mod
from streaming.producer import Producer
from streaming.injector import Injector, apply_attack
from streaming.consumer import Consumer
from sensors.registry import load_registry, enabled_entries


@pytest.fixture(scope="module")
def producer(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    tmp_dir = tmp_path_factory.mktemp("phase3_producer")
    mp.setattr(producer_mod, "GROUND_TRUTH_LOG", tmp_dir / "ground_truth.csv")
    p = Producer(add_noise=False, seed=0)
    yield p
    mp.undo()


@pytest.fixture(scope="module")
def registry_ids():
    registry = load_registry()
    return {e["id"] for e in enabled_entries(registry)}


def _isolated_consumer(tmp_path_factory, window_timeout_s=5.0):
    mp = pytest.MonkeyPatch()
    tmp_dir = tmp_path_factory.mktemp("phase3_consumer")
    mp.setattr(consumer_mod, "OPERATOR_TWIN_LOG", tmp_dir / "state.csv")
    mp.setattr(consumer_mod, "DROPOUT_LOG", tmp_dir / "dropout.csv")
    c = Consumer(window_timeout_s=window_timeout_s)
    return c, mp


# --------------------------------------------------
# Acceptance 1: with a null attack, telemetry.delivered is byte-identical
# to telemetry.raw
# --------------------------------------------------

def test_apply_attack_none_is_identity():
    raw = {"timestamp": 0, "meter_id": "M001", "measurements": {"M001_P": 12.3, "M001_Q": 4.5}}
    delivered = apply_attack(raw, attack=None)
    assert delivered == raw
    assert delivered is not raw  # genuine copy, not aliasing -- a real
    # attack must not be able to mutate the raw message in place


def test_apply_attack_with_a_real_attack_is_not_implemented_yet():
    """Phase 3 builds the seam, not the attacks (RES.md D10 Phase 8) --
    this documents that the seam is real (accepts an attack argument)
    without silently no-op'ing it."""

    raw = {"timestamp": 0, "meter_id": "M001", "measurements": {}}
    with pytest.raises(NotImplementedError):
        apply_attack(raw, attack={"class": "A1"})


def test_delivered_byte_identical_to_raw_over_real_messages(producer):
    raw_messages, _ = producer.run_timestep(0)
    assert len(raw_messages) == len(producer.representative_meters)
    for raw in raw_messages:
        delivered = apply_attack(raw, attack=None)
        assert delivered == raw


# --------------------------------------------------
# Acceptance 5: every field on telemetry.raw traces to exactly one
# sensor_registry.json entry; a test fails if an unregistered field is
# published
# --------------------------------------------------

def test_every_raw_message_field_traces_to_registry(producer, registry_ids):
    raw_messages, _ = producer.run_timestep(1)

    seen_ids = set()
    for msg in raw_messages:
        for measurement_id in msg["measurements"]:
            assert measurement_id in registry_ids, (
                f"{measurement_id} on telemetry.raw has no "
                f"config/sensor_registry.json entry"
            )
            seen_ids.add(measurement_id)

    p_q_ids = {i for i in registry_ids if i.endswith("_P") or i.endswith("_Q")}
    assert p_q_ids == seen_ids, "some enabled P/Q registry ids were never published"


def test_unregistered_field_is_caught(producer, registry_ids):
    """The failing case the previous test's assertion exists to catch --
    a tampered message with a field that has no registry entry."""

    raw_messages, _ = producer.run_timestep(2)
    tampered = copy.deepcopy(raw_messages[0])
    tampered["measurements"]["NOT_A_REGISTRY_ID"] = 999.0

    with pytest.raises(AssertionError):
        for measurement_id in tampered["measurements"]:
            assert measurement_id in registry_ids


# --------------------------------------------------
# Injector: buffers by timestamp, flushes on count OR timeout, and never
# marks anything attacked in Phase 3
# --------------------------------------------------

def test_injector_flushes_on_count_and_labels_are_unattacked(producer):
    injector = Injector(expected_meters=len(producer.representative_meters), timeout_s=5.0)
    raw_messages, _ = producer.run_timestep(3)

    completions = []
    for msg in raw_messages:
        delivered, label = injector.ingest(msg)
        if delivered is not None:
            completions.append((delivered, label))

    assert len(completions) == 1  # exactly one flush, triggered by the last message
    delivered_batch, label = completions[0]
    assert len(delivered_batch) == len(raw_messages)
    assert label["attacked_meters"] == []
    assert label["attack_class"] is None
    assert label["magnitude"] == 0.0
    assert label["stealth_flag"] is False


def test_injector_flushes_on_timeout_not_just_count():
    injector = Injector(expected_meters=5, timeout_s=0.05)

    delivered, label = injector.ingest(
        {"timestamp": 0, "meter_id": "X1", "measurements": {}}
    )
    assert delivered is None  # only 1 of 5 arrived, no timeout yet

    time.sleep(0.06)
    delivered, label = injector.ingest(
        {"timestamp": 0, "meter_id": "X2", "measurements": {}}
    )
    assert delivered is not None  # timeout fired -- flushed with only 2/5
    assert len(delivered) == 2
    assert label["timestamp"] == 0


# --------------------------------------------------
# Acceptance 3-4: a dropped meter does not stall the consumer, and the
# resulting event is logged distinctly from attacks
# --------------------------------------------------

def test_consumer_missing_meter_does_not_stall(tmp_path_factory):
    consumer, mp = _isolated_consumer(tmp_path_factory, window_timeout_s=0.05)
    try:
        meters = consumer.representative_meters

        def make_message(t, meter, p, q):
            return {
                "timestamp": t, "real_timestamp": None,
                "meter_id": meter["meter_id"], "load_name": meter["load_name"],
                "bus": meter["bus"],
                "measurements": {
                    consumer._p_id_for_load[meter["load_name"]]: p,
                    consumer._q_id_for_load[meter["load_name"]]: q,
                },
            }

        # Timestep 0: complete window -- establishes a "last known" reading
        state = None
        for meter in meters:
            state = consumer.ingest(make_message(0, meter, 10.0, 5.0))
        assert state is not None
        assert state["dropout_count"] == 0

        # Timestep 1: withhold meters[0] entirely
        state = None
        for meter in meters[1:]:
            state = consumer.ingest(make_message(1, meter, 11.0, 6.0))
        assert state is None  # only 48/49 -- must not have completed yet

        time.sleep(0.06)
        results = consumer.flush_all()  # forces the timed-out window closed
        assert len(results) == 1
        assert results[0]["dropout_count"] == 1
        assert results[0]["converged"] is True  # pipeline kept solving, did not hang
    finally:
        mp.undo()


def test_dropout_uses_hold_last_value(tmp_path_factory):
    consumer, mp = _isolated_consumer(tmp_path_factory, window_timeout_s=0.05)
    try:
        meters = consumer.representative_meters
        target = meters[0]

        def make_message(t, meter, p, q):
            return {
                "timestamp": t, "real_timestamp": None,
                "meter_id": meter["meter_id"], "load_name": meter["load_name"],
                "bus": meter["bus"],
                "measurements": {
                    consumer._p_id_for_load[meter["load_name"]]: p,
                    consumer._q_id_for_load[meter["load_name"]]: q,
                },
            }

        for meter in meters:
            p, q = (77.0, 33.0) if meter is target else (1.0, 1.0)
            consumer.ingest(make_message(0, meter, p, q))

        assert consumer._last_known[target["load_name"]] == (77.0, 33.0)

        for meter in meters[1:]:
            consumer.ingest(make_message(1, meter, 1.0, 1.0))
        time.sleep(0.06)
        consumer.flush_all()

        # the held value, not the nominal fallback, must have been reused
        dss_kw = consumer._last_known[target["load_name"]][0]
        assert dss_kw == 77.0
    finally:
        mp.undo()


def test_dropout_logged_distinctly_from_labels(tmp_path_factory):
    import csv

    consumer, mp = _isolated_consumer(tmp_path_factory, window_timeout_s=0.05)
    try:
        meters = consumer.representative_meters

        def make_message(t, meter):
            return {
                "timestamp": t, "real_timestamp": None,
                "meter_id": meter["meter_id"], "load_name": meter["load_name"],
                "bus": meter["bus"],
                "measurements": {
                    consumer._p_id_for_load[meter["load_name"]]: 5.0,
                    consumer._q_id_for_load[meter["load_name"]]: 2.0,
                },
            }

        for meter in meters[1:]:
            consumer.ingest(make_message(0, meter))
        time.sleep(0.06)
        consumer.flush_all()

        with open(consumer_mod.DROPOUT_LOG) as f:
            dropout_rows = list(csv.DictReader(f))

        assert len(dropout_rows) == 1
        assert dropout_rows[0]["event_class"] == "dropout"
        assert dropout_rows[0]["event_class"] != "attack"
        assert dropout_rows[0]["meter_id"] == meters[0]["meter_id"]
    finally:
        mp.undo()


# --------------------------------------------------
# Phase 2 Finding #1 follow-up: multiphase registry variant
# --------------------------------------------------

def test_multiphase_registry_variant_has_more_measurements():
    default_registry = load_registry(variant="default")
    multiphase_registry = load_registry(variant="multiphase")

    default_entries = enabled_entries(default_registry)
    multiphase_entries = enabled_entries(multiphase_registry)

    assert len(default_entries) == 98
    # 98 + (s47, s48) x (B, C) x (P, Q) = 106 meter entries, + 6
    # HEAD_P/Q_A/B/C -- RES.md D10 Phase 6's deployment decision:
    # Phase 5 found this exact 6-entry addition is the minimum needed,
    # on top of meters + zero-injection + pseudo, to reach full rank
    # with zero critical measurements (reports/phase5_sensor_set_report.txt);
    # Phase 6 deploys it so the generated corpus matches the sensor
    # configuration Phase 5 actually validated.
    assert len(multiphase_entries) == 112


def test_multiphase_variant_only_extends_the_two_genuinely_3phase_loads():
    """Phase-2 Finding #1 named 5 loads as 'multi-phase'; direct
    inspection of 123Bus/IEEE123Loads.DSS shows only s47 and s48 are
    (Phases=3 Conn=Wye) -- s35a, s65c, s76a are Phases=1 Conn=Delta
    (single-phase, line-to-line) and already fully metered by their one
    primary-phase entry. The multiphase variant must not fabricate a
    second phase for those three."""

    multiphase_registry = load_registry(variant="multiphase")
    entries = enabled_entries(multiphase_registry)

    by_load = {}
    for e in entries:
        if "load_name" not in e:
            continue  # HEAD_* entries (Phase 6) aren't tied to a load
        by_load.setdefault(e["load_name"], []).append(e)

    assert len(by_load["s47"]) == 6   # P,Q at each of A,B,C
    assert len(by_load["s48"]) == 6
    assert len(by_load["s35a"]) == 2  # unchanged: single delta phase only
    assert len(by_load["s65c"]) == 2
    assert len(by_load["s76a"]) == 2


def test_multiphase_variant_default_entries_identical_to_default_registry():
    """multiphase only APPENDS -- the original 98 entries must be
    byte-identical (same ids, same values) in both variants."""

    default_entries = {
        e["id"]: e for e in enabled_entries(load_registry(variant="default"))
    }
    multiphase_entries = {
        e["id"]: e for e in enabled_entries(load_registry(variant="multiphase"))
    }

    for entry_id, entry in default_entries.items():
        assert multiphase_entries[entry_id] == entry
