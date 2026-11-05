"""
streaming/failover_check.py -- proves on the real cluster that the stream survives a
broker failure. Needs the brokers up (docker compose up -d kafka-1 kafka-2 kafka-3).

  python -m streaming.failover_check [--broker kafka-1] [--messages 3000] [--out FILE]

Publishes numbered messages on a scratch topic laid out like telemetry (same keys,
partitions, replication), through the same reliable_producer the stages use. A third
of the way in it kills a broker outright (docker compose kill, no clean shutdown),
and two thirds of the way in it starts that broker again. Exit status 0 only if:

  * every message arrived exactly once and each key's messages stayed in order,
  * partition leaders moved off the dead broker while it was down,
  * after the restart every partition is back to three in-sync replicas.
"""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

from kafka import KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic

from streaming.kafka_config import (BOOTSTRAP_SERVERS, MIN_INSYNC_REPLICAS, REPLICATION_FACTOR, TOPIC_PARTITIONS,
                                    TOPIC_RAW, reliable_producer)

BASE = Path(__file__).resolve().parent.parent
COMPOSE = BASE / "streaming" / "docker-compose.yml"
TOPIC = "failover.check"
PARTITIONS = TOPIC_PARTITIONS[TOPIC_RAW]
KEYS = [f"M{i:03d}" for i in range(1, 50)] + ["HEAD"]


def compose(*args):
    subprocess.run(["docker", "compose", "-f", str(COMPOSE), *args], check=True, capture_output=True)


def layout(admin):
    """(leader per partition, in-sync replica count per partition)."""
    parts = admin.describe_topics([TOPIC])[0]["partitions"]
    return ({p["partition"]: p["leader"] for p in parts}, {p["partition"]: len(p["isr"]) for p in parts})


def wait_for(condition, timeout_s, what):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if condition():
                return True
        except Exception:
            pass                     # the metadata request itself can hit the dead broker
        time.sleep(0.5)
    print(f"  timed out waiting for {what}", flush=True)
    return False


def analyse(received, total, keys):
    """Compare what was read with what was sent: (missing, duplicates, out-of-order keys)."""
    per_key = {}
    for key, seq in received:
        per_key.setdefault(key, []).append(seq)
    sent = {k: list(range(i, total, len(keys))) for i, k in enumerate(keys)}
    missing = sum(len(set(sent[k]) - set(per_key.get(k, []))) for k in keys)
    duplicates = sum(len(v) - len(set(v)) for v in per_key.values())
    out_of_order = sorted(k for k, v in per_key.items() if v != sorted(v))
    return missing, duplicates, out_of_order


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--broker", default="kafka-1", choices=["kafka-1", "kafka-2", "kafka-3"])
    ap.add_argument("--messages", type=int, default=3000)
    ap.add_argument("--rate", type=float, default=50.0, help="messages per second")
    ap.add_argument("--out", default=None, help="also write the result to this JSON file")
    a = ap.parse_args()
    node_id = int(a.broker[-1])

    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP_SERVERS)
    if TOPIC in admin.list_topics():
        admin.delete_topics([TOPIC])
        wait_for(lambda: TOPIC not in admin.list_topics(), 30, "old scratch topic to go")
        time.sleep(2)
    admin.create_topics([NewTopic(TOPIC, PARTITIONS, REPLICATION_FACTOR,
                                  topic_configs={"min.insync.replicas": str(MIN_INSYNC_REPLICAS)})])
    wait_for(lambda: len(layout(admin)[0]) == PARTITIONS, 30, "scratch topic")
    leaders_before, _ = layout(admin)
    print(f"leaders before: {leaders_before}", flush=True)

    received, done = [], threading.Event()

    def consume():
        c = KafkaConsumer(TOPIC, bootstrap_servers=BOOTSTRAP_SERVERS, group_id=f"failover-{time.time():.0f}",
                          auto_offset_reset="earliest", consumer_timeout_ms=1000)
        idle_since = None
        while True:
            got = False
            for rec in c:
                received.append((rec.key.decode(), int(rec.value)))
                got = True
            if not done.is_set():
                continue
            idle_since = None if got else (idle_since or time.time())
            if len(received) >= a.messages or (idle_since and time.time() - idle_since > 60):
                break
        c.close()

    reader = threading.Thread(target=consume, daemon=True)
    reader.start()

    producer = reliable_producer()
    futures, slowest = [], 0.0
    leaders_during, isr_after = None, None
    killed_at = restarted_at = None
    for i in range(a.messages):
        t0 = time.time()
        futures.append(producer.send(TOPIC, key=KEYS[i % len(KEYS)].encode(), value=str(i).encode()))
        if i % len(KEYS) == len(KEYS) - 1:
            producer.flush()             # the producer stage flushes once per timestep too
        slowest = max(slowest, time.time() - t0)
        if i == a.messages // 3:
            print(f"message {i}: killing {a.broker}", flush=True)
            compose("kill", a.broker)
            killed_at = time.time()
        elif i == 2 * a.messages // 3:
            ok = wait_for(lambda: node_id not in layout(admin)[0].values(), 60, "leaders to move")
            leaders_during = layout(admin)[0] if ok else None
            print(f"message {i}: leaders while down: {leaders_during}; starting {a.broker}", flush=True)
            compose("start", a.broker)
            restarted_at = time.time()
        time.sleep(1.0 / a.rate)
    producer.flush()
    undelivered = sum(1 for f in futures if f.failed())
    producer.close()
    done.set()
    reader.join(timeout=180)

    recovered = wait_for(lambda: set(layout(admin)[1].values()) == {REPLICATION_FACTOR}, 120, "replicas to catch up")
    isr_after = layout(admin)[1]
    missing, duplicates, out_of_order = analyse(received, a.messages, KEYS)
    admin.close()

    moved = leaders_during is not None and node_id not in leaders_during.values()
    passed = not (missing or duplicates or out_of_order or undelivered) and moved and recovered
    result = {"killed": a.broker, "messages_sent": a.messages, "messages_received": len(received),
              "send_failures": undelivered, "missing": missing, "duplicates": duplicates,
              "keys_out_of_order": out_of_order, "leaders_before": leaders_before, "leaders_while_down": leaders_during,
              "leaders_moved_off_dead_broker": moved, "in_sync_replicas_after_restart": isr_after,
              "replicas_recovered": recovered, "broker_down_seconds": round(restarted_at - killed_at, 1),
              "longest_single_send_blocked_seconds": round(slowest, 2), "passed": passed}
    print(json.dumps(result, indent=2), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"FAILOVER CHECK: {'PASS' if passed else 'FAIL'}", flush=True)
    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP_SERVERS)   # the old one still names the dead controller
    admin.delete_topics([TOPIC])
    admin.close()
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
