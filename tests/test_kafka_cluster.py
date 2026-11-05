"""
The fault-tolerant Kafka layer, without a broker: the window gate that lets a stage
read a multi-partition topic, the stages using it, and the deployed cluster settings.
The same behaviour on real brokers is checked by streaming/failover_check.py.
"""

import json
import random
import re
from pathlib import Path

import pytest
from kafka.partitioner.default import murmur2

from streaming.injector import Injector
from streaming.kafka_config import (BOOTSTRAP_SERVERS, MIN_INSYNC_REPLICAS, REPLICATION_FACTOR, TOPIC_ALARMS,
                                    TOPIC_DELIVERED, TOPIC_LABELS, TOPIC_PARTITIONS, TOPIC_RAW, TOPIC_STATE)
from streaming.failover_check import analyse
from streaming.windowing import WindowGate

BASE = Path(__file__).resolve().parent.parent
COMPOSE = (BASE / "streaming" / "docker-compose.yml").read_text(encoding="utf-8")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def partition_of(meter, n):
    return (murmur2(meter.encode()) & 0x7fffffff) % n


def gate_and_injector(meters, n_partitions=3, stall_timeout_s=60.0, clock=None):
    gate = WindowGate(range(n_partitions), stall_timeout_s=stall_timeout_s, clock=clock or Clock())
    return gate, Injector(expected_meters=len(meters), gate=gate)


def msg(t, meter):
    return {"timestamp": t, "real_timestamp": "2018-05-25 00:00:00", "meter_id": meter,
            "load_name": meter, "bus": "1", "measurements": {f"{meter}_P": 1.0}}


METERS = {"A": 0, "B": 0, "C": 1, "D": 2}


def feed(injector, t, meters=METERS):
    out = []
    for meter in meters:
        out += injector.ingest_stream(msg(t, meter), METERS[meter])
    return out


def test_complete_window_closes_at_once():
    _, inj = gate_and_injector(METERS)
    closed = feed(inj, 1)
    assert [label["timestamp"] for _, label in closed] == [1]
    assert len(closed[0][0]) == len(METERS)


def test_paused_partition_holds_the_window_however_long_it_takes():
    """A failover pause is not a dropped meter: nothing closes early and no reading is lost."""
    clock = Clock()
    _, inj = gate_and_injector(METERS, clock=clock)
    assert feed(inj, 1, ["A", "B", "C"]) == []           # partition 2 (D) has stopped
    clock.now += 30.0                                    # several times the old 5 s timeout
    assert feed(inj, 2, ["A", "B", "C"]) == []
    closed = inj.ingest_stream(msg(1, "D"), 2) + inj.ingest_stream(msg(2, "D"), 2)
    assert [label["timestamp"] for _, label in closed] == [1, 2]
    assert all(len(delivered) == len(METERS) for delivered, _ in closed)
    assert inj.gate.late == 0


def test_dropped_meter_closes_as_soon_as_all_partitions_move_on():
    _, inj = gate_and_injector(METERS)
    assert feed(inj, 1, ["A", "C", "D"]) == []           # B never publishes t=1; A shares its partition
    closed = feed(inj, 2, ["A", "C", "D"])
    assert [label["timestamp"] for _, label in closed] == [1]
    assert {m["meter_id"] for m in closed[0][0]} == {"A", "C", "D"}


def test_complete_window_also_closes_every_older_window_in_order():
    _, inj = gate_and_injector(METERS)
    feed(inj, 1, ["A", "C"])                             # B and D lost for t=1
    closed = feed(inj, 2)
    assert [label["timestamp"] for _, label in closed] == [1, 2]


@pytest.mark.parametrize("seed", range(5))
def test_any_interleaving_of_partitions_gives_each_step_once_in_order(seed):
    """Partitions are read in bursts of different lengths when a stage is behind."""
    steps, pending = 40, {}
    for t in range(steps):
        for meter, p in METERS.items():
            pending.setdefault(p, []).append(msg(t, meter))
    rng = random.Random(seed)
    _, inj = gate_and_injector(METERS)
    closed = []
    while any(pending.values()):
        p = rng.choice([q for q, v in pending.items() if v])
        burst = rng.randint(1, 25)
        for m in pending[p][:burst]:
            closed += inj.ingest_stream(m, p)
        pending[p] = pending[p][burst:]
    assert [label["timestamp"] for _, label in closed] == list(range(steps))
    assert all(len(delivered) == len(METERS) for delivered, _ in closed)


def test_partition_with_no_leader_cannot_stall_the_stream_forever():
    clock = Clock()
    _, inj = gate_and_injector(METERS, stall_timeout_s=60.0, clock=clock)
    assert feed(inj, 1, ["A", "B", "C"]) == []
    clock.now += 61.0
    # partition 2 is set aside, so windows the others have read past can close
    closed = feed(inj, 2, ["A", "B", "C"])
    assert [label["timestamp"] for _, label in closed] == [1]
    assert {m["meter_id"] for m in closed[0][0]} == {"A", "B", "C"}


def test_message_after_its_window_closed_is_dropped_not_reopened():
    clock = Clock()
    gate, inj = gate_and_injector(METERS, stall_timeout_s=60.0, clock=clock)
    feed(inj, 1, ["A", "B", "C"])
    clock.now += 61.0
    feed(inj, 2, ["A", "B", "C"])                        # t=1 closes without partition 2
    assert inj.ingest_stream(msg(1, "D"), 2) == []
    assert gate.late == 1
    assert 1 not in inj._buffer


def test_quiet_time_with_no_window_open_is_not_a_stall():
    """Waiting for the first message, or a pause between steps, sets no partition aside."""
    clock = Clock()
    gate, inj = gate_and_injector(METERS, stall_timeout_s=60.0, clock=clock)
    clock.now += 600.0
    assert feed(inj, 1, ["A"]) == []
    assert gate.set_aside == set()
    feed(inj, 1, ["B", "C", "D"])
    clock.now += 600.0
    assert feed(inj, 2, ["A"]) == []
    assert gate.set_aside == set()


def test_retry_outlasts_a_client_that_cannot_connect_yet(monkeypatch):
    from kafka.errors import NoBrokersAvailable
    from streaming import demo_tools
    monkeypatch.setattr(demo_tools.time, "sleep", lambda s: None)
    attempts = []

    def connect():
        attempts.append(1)
        if len(attempts) < 3:
            raise NoBrokersAvailable()
        return 42
    assert demo_tools._retrying(connect) == 42
    assert len(attempts) == 3


def test_operator_twin_does_not_count_head_as_late():
    from streaming.consumer import Consumer
    consumer = Consumer.__new__(Consumer)
    consumer.gate = WindowGate(range(2), clock=Clock())
    consumer._meter_ids = {"A"}
    consumer.expected_meters = 1
    consumer._buffer, consumer._first_seen, consumer._clock = {}, {}, Clock()
    consumer._close_window = lambda t: consumer._buffer.pop(t) and t
    assert consumer.ingest_stream({"timestamp": 1, "meter_id": "A"}, 0) == [1]
    assert consumer.ingest_stream({"timestamp": 1, "meter_id": "HEAD"}, 1) == []
    assert consumer.gate.late == 0
    assert consumer._buffer == {}


def test_gate_refuses_a_topic_with_no_partitions():
    with pytest.raises(ValueError, match="create the topics"):
        WindowGate(None)


def test_detector_consumer_scores_consecutively_across_partitions():
    from streaming.detector_consumer import DetectorConsumer

    class Core:
        cols = ["C1", "C2", "C3"]
        thr_f = 0.5

        def score_vector(self, window, real_ts):
            return {"J": 1.0, "maxLNR": 1.0, "bdd_score": 1.0, "ae_mahalanobis": 1.0, "fusion_score": 0.1,
                    "alarm": False, "top_channel": "C1"}

    dc = DetectorConsumer(Core(), gate=WindowGate(range(3), clock=Clock()))
    results = []
    # partition 0 races ahead by a whole window before the others are read
    for t in range(12):
        results += dc.ingest_stream({"timestamp": t, "real_timestamp": "r", "measurements": {"C1": 1.0}}, 0)
    for t in range(12):
        results += dc.ingest_stream({"timestamp": t, "real_timestamp": "r", "measurements": {"C2": 1.0}}, 1)
    for t in range(12):
        results += dc.ingest_stream({"timestamp": t, "real_timestamp": "r", "measurements": {"C3": 1.0}}, 2)
    assert [r["timestamp"] for r in results] == list(range(12))
    assert [r["timestamp"] for r in results if r["scored"]] == [9, 10, 11]


def test_failover_check_analysis_catches_each_failure():
    keys = ["A", "B"]
    good = [("A", 0), ("B", 1), ("A", 2), ("B", 3)]
    assert analyse(good, 4, keys) == (0, 0, [])
    assert analyse(good[:-1], 4, keys) == (1, 0, [])
    assert analyse(good + [("A", 2)], 4, keys)[1] == 1
    assert analyse([("A", 2), ("A", 0), ("B", 1), ("B", 3)], 4, keys)[2] == ["A"]


# ---------- what is deployed


def test_all_meters_land_on_every_telemetry_partition():
    """The read-past-t rule needs every partition to carry messages."""
    meters = [m["meter_id"] for m in json.load(open(BASE / "simulation" / "representative_meters.json"))] + ["HEAD"]
    for topic in (TOPIC_RAW, TOPIC_DELIVERED):
        n = TOPIC_PARTITIONS[topic]
        assert {partition_of(m, n) for m in meters} == set(range(n))


def test_every_live_topic_is_multi_partition_and_replicated():
    assert set(TOPIC_PARTITIONS) == {TOPIC_RAW, TOPIC_DELIVERED, TOPIC_LABELS, TOPIC_ALARMS, TOPIC_STATE}
    assert all(n > 1 for n in TOPIC_PARTITIONS.values())
    assert REPLICATION_FACTOR == 3 and MIN_INSYNC_REPLICAS == 2
    assert len(BOOTSTRAP_SERVERS) == 3


def test_cluster_is_three_voters_with_replicated_internal_topics():
    assert "1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093" in COMPOSE
    for setting in ("KAFKA_DEFAULT_REPLICATION_FACTOR", "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR",
                    "KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR"):
        assert f"{setting}: {REPLICATION_FACTOR}" in COMPOSE
    assert f"KAFKA_MIN_INSYNC_REPLICAS: {MIN_INSYNC_REPLICAS}" in COMPOSE
    assert 'KAFKA_AUTO_CREATE_TOPICS_ENABLE: "false"' in COMPOSE


def test_kafka_image_is_pinned_by_digest():
    images = re.findall(r"image:\s*(apache/kafka\S*)", COMPOSE)
    assert images and all(re.fullmatch(r"apache/kafka:\d+\.\d+\.\d+@sha256:[0-9a-f]{64}", i) for i in images)


def test_every_stage_has_its_own_dockerfile_that_exists():
    used = set(re.findall(r"dockerfile:\s*(docker/\S+?\.Dockerfile)", COMPOSE))
    assert {f"docker/{s}.Dockerfile" for s in ("producer", "injector", "detector", "consumer", "viewer", "report")} == used
    for path in used:
        text = (BASE / path).read_text(encoding="utf-8")
        assert re.search(r"^FROM python:\d+\.\d+\.\d+-slim$", text, re.M), path
        for source in re.findall(r"^COPY (?!--)(.+) \S+$", text, re.M):
            for item in source.split():
                assert (BASE / item).exists(), f"{path}: {item}"
