"""
RES.md D10 Phase 3 / D6.0 -- the attack injector. Consumes telemetry.raw,
emits telemetry.delivered + labels. Sits ONLY on the wire between
producer and consumer, per D6.0's explicit injection point ("between
producer and consumer, on the wire... implement it as a Kafka stream
processor: consume telemetry.raw, emit telemetry.delivered + labels").

Phase 3 scope is the SEAM ONLY: apply_attack() with attack=None is the
identity transform, so telemetry.delivered is byte-identical to
telemetry.raw (Phase 3 acceptance criterion 1). No attack class (A1-A9,
RES.md D10 Phase 8) is implemented here -- attack is a parameter this
module is built to accept, not a feature it exercises yet.

Layer boundary: this module NEVER touches OpenDSS. It only reads and
re-emits the (node, phase, quantity, id, value) measurements a raw
message already carries -- exactly what D6.0 permits an attacker to
touch (Layer 2 only, in transit).
"""

import copy
import json


def apply_attack(raw_message, attack=None):
    """z_delivered = z_true + a. attack=None => a=0 => delivered is a
    deep copy of raw, byte-identical once serialized. The seam for
    Phase 8: a real attack function would take (raw_message, attack)
    and perturb raw_message['measurements'][id] for the targeted ids,
    returning a new dict -- this function's signature already supports
    that; only the attack logic itself is deferred."""

    if attack is None:
        return copy.deepcopy(raw_message)

    raise NotImplementedError(
        "Attack injection is RES.md D10 Phase 8 scope -- Phase 3 builds "
        "the pass-through seam only."
    )


class Injector:
    """Per-timestamp buffering of telemetry.raw, mirroring (but
    independent of) the consumer's own buffering -- the injector must
    not stall on a missing meter either, since it sits upstream of the
    consumer's timeout logic. On flush (count reached or its own
    timeout), forwards every buffered raw message unchanged to
    telemetry.delivered and emits exactly one labels record for that
    timestamp. This phase never marks anything attacked; attacked_meters
    is always []."""

    def __init__(self, expected_meters, timeout_s=5.0, clock=None):
        self.expected_meters = expected_meters
        self.timeout_s = timeout_s
        self._clock = clock or _default_clock
        self._buffer = {}   # timestamp -> {meter_id: raw_message}
        self._first_seen = {}  # timestamp -> clock time of first message

    def ingest(self, raw_message):
        """Buffer one raw message. Returns (delivered_messages, label)
        if this completed (or timed out on) its timestamp, else
        (None, None)."""

        t = raw_message["timestamp"]
        meter_id = raw_message["meter_id"]

        if t not in self._buffer:
            self._buffer[t] = {}
            self._first_seen[t] = self._clock()

        self._buffer[t][meter_id] = raw_message

        complete = len(self._buffer[t]) >= self.expected_meters
        timed_out = (self._clock() - self._first_seen[t]) >= self.timeout_s

        if not (complete or timed_out):
            return None, None

        return self._flush(t)

    def flush_all(self):
        """Force-flush every still-open timestamp -- used at end of
        stream (direct-replay / shutdown) so no buffered timestamp is
        silently dropped."""

        results = []
        for t in sorted(self._buffer.keys()):
            results.append(self._flush(t))
        return results

    def _flush(self, t):
        raw_messages = self._buffer.pop(t)
        self._first_seen.pop(t, None)

        delivered_messages = [
            apply_attack(msg, attack=None) for msg in raw_messages.values()
        ]

        label = {
            "timestamp": t,
            "attacked_meters": [],
            "attack_class": None,
            "magnitude": 0.0,
            "stealth_flag": False,
        }

        return delivered_messages, label


def _default_clock():
    import time
    return time.monotonic()


def main():
    """Live mode -- requires a running Kafka broker."""

    from kafka import KafkaConsumer, KafkaProducer
    from pathlib import Path
    from .kafka_config import (
        BOOTSTRAP_SERVERS, TOPIC_RAW, TOPIC_DELIVERED, TOPIC_LABELS,
    )

    BASE_DIR = Path(__file__).resolve().parent.parent
    with open(BASE_DIR / "simulation" / "representative_meters.json") as f:
        expected_meters = len(json.load(f))

    consumer = KafkaConsumer(
        TOPIC_RAW,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="injector",
    )
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    injector = Injector(expected_meters=expected_meters)

    print(f"Injector waiting on {TOPIC_RAW} (pass-through only, no attack class active)...")

    for record in consumer:
        delivered_messages, label = injector.ingest(record.value)
        if delivered_messages is None:
            continue

        for msg in delivered_messages:
            producer.send(TOPIC_DELIVERED, key=msg["meter_id"].encode(), value=msg)
        producer.send(TOPIC_LABELS, key=str(label["timestamp"]).encode(), value=label)
        producer.flush()


if __name__ == "__main__":
    main()
