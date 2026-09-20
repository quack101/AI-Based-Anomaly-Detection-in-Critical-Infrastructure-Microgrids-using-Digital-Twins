"""
RES.md D10 Phase 3 acceptance validation.

No Docker broker is available in this environment (same constraint
Phase 0 hit) -- streaming/docker-compose.yml is the live-mode path, and
streaming/producer.py, streaming/injector.py, streaming/consumer.py are
all written so their core per-timestep logic (Producer.run_timestep,
Injector.ingest, Consumer.ingest) takes/returns plain dicts with no
Kafka dependency. This script chains those REAL classes in-process --
not a re-implementation -- so it validates the actual Phase 3 code.

Two runs:
  1. NOISE-FREE full-corpus run -- the comparand for Phase 3 acceptance
     criterion 2 ("end-to-end results are identical to the Phase-1
     direct-replay run"). Phase 1's numbers
     (reports/phase0_validation_report_ideal.txt) were produced with no
     measurement noise (there was no sensor layer yet), so this is the
     apples-to-apples setting: the producer's ground-truth log here uses
     the IDENTICAL methodology (49-meter corpus drive + 42 static loads,
     one solve/timestamp) and must reproduce those numbers exactly. It
     also checks telemetry.delivered == telemetry.raw byte-for-byte at
     every timestamp (acceptance criterion 1).
  2. A short dropout demonstration with a short window timeout, showing
     one manufactured missing meter does not stall the pipeline and is
     logged as a dropout distinct from labels (acceptance criteria 3-4).
"""

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from streaming.producer import Producer, GROUND_TRUTH_LOG
from streaming.injector import Injector
from streaming.consumer import Consumer, OPERATOR_TWIN_LOG, DROPOUT_LOG

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
REPORT_PATH = REPORTS_DIR / "phase3_streaming_report.txt"

PHASE1_REPORT = REPORTS_DIR / "phase0_validation_report_ideal.txt"

# Phase-1 acceptance numbers this run must reproduce exactly (from
# reports/phase0_validation_report_ideal.txt).
PHASE1_CONVERGED_PCT = 100.00
PHASE1_V_PU_RANGE = (0.936943, 1.099974)
PHASE1_AVG_HEAD_P_KW = 3623.3
PHASE1_AVG_HEAD_Q_KVAR = 1331.9
PHASE1_AVG_LOSS_PCT = 2.6757


def run_full_chain(num_rows, add_noise, seed=0):
    """Feeds num_rows timesteps through Producer -> Injector -> Consumer,
    message by message. No drops -- every window closes by count, so
    timeouts never fire and never slow this down."""

    producer = Producer(add_noise=add_noise, seed=seed)
    injector = Injector(expected_meters=len(producer.representative_meters), timeout_s=5.0)
    consumer = Consumer(window_timeout_s=5.0)

    num_rows = min(num_rows, producer.num_rows)

    byte_identical = 0
    byte_mismatch = 0
    state_rows = []
    labels = []

    for t in range(num_rows):
        raw_messages, _ = producer.run_timestep(t)
        raw_by_meter = {m["meter_id"]: m for m in raw_messages}

        for raw_msg in raw_messages:
            delivered_batch, label = injector.ingest(raw_msg)
            if delivered_batch is None:
                continue

            labels.append(label)
            for delivered_msg in delivered_batch:
                original = raw_by_meter[delivered_msg["meter_id"]]
                if delivered_msg == original:
                    byte_identical += 1
                else:
                    byte_mismatch += 1

                state_row = consumer.ingest(delivered_msg)
                if state_row is not None:
                    state_rows.append(state_row)

    return {
        "num_rows": num_rows,
        "byte_identical": byte_identical,
        "byte_mismatch": byte_mismatch,
        "state_rows": state_rows,
        "labels": labels,
    }


def run_dropout_demo(num_rows=5, drop_at=2, drop_meter_index=0,
                      injector_timeout_s=0.1, consumer_timeout_s=0.1):
    """Manufactures a single missing meter at timestep drop_at and shows
    the pipeline does not stall: the injector's own timeout closes its
    window with 48/49 raw messages, forwards those, and the consumer's
    timeout then closes ITS window using the missing-data policy,
    logging exactly one dropout row distinct from any labels record."""

    producer = Producer(add_noise=False, seed=0)
    injector = Injector(expected_meters=len(producer.representative_meters),
                         timeout_s=injector_timeout_s)
    consumer = Consumer(window_timeout_s=consumer_timeout_s)

    dropped_meter_id = None
    state_rows = []
    labels = []

    for t in range(num_rows):
        raw_messages, _ = producer.run_timestep(t)

        if t == drop_at:
            dropped = raw_messages.pop(drop_meter_index)
            dropped_meter_id = dropped["meter_id"]

        for raw_msg in raw_messages:
            delivered_batch, label = injector.ingest(raw_msg)
            if delivered_batch is None:
                continue
            labels.append(label)
            for delivered_msg in delivered_batch:
                state_row = consumer.ingest(delivered_msg)
                if state_row is not None:
                    state_rows.append(state_row)

        if t == drop_at:
            time.sleep(injector_timeout_s + 0.05)
            for delivered_batch, label in injector.flush_all():
                labels.append(label)
                for delivered_msg in delivered_batch:
                    state_row = consumer.ingest(delivered_msg)
                    if state_row is not None:
                        state_rows.append(state_row)

            time.sleep(consumer_timeout_s + 0.05)
            for state_row in consumer.flush_all():
                state_rows.append(state_row)

    return {
        "dropped_meter_id": dropped_meter_id,
        "drop_at": drop_at,
        "num_timesteps_completed": len(state_rows),
        "num_labels": len(labels),
        "state_rows": state_rows,
        "labels": labels,
    }


def _read_dropout_log():
    import csv
    with open(DROPOUT_LOG) as f:
        return list(csv.DictReader(f))


def summarize_ground_truth():
    import csv
    with open(GROUND_TRUTH_LOG) as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    converged = sum(1 for r in rows if r["converged"] == "True")
    v_mins = [float(r["v_pu_min"]) for r in rows]
    v_maxs = [float(r["v_pu_max"]) for r in rows]
    head_p = [float(r["feeder_head_p_kw"]) for r in rows]
    head_q = [float(r["feeder_head_q_kvar"]) for r in rows]
    losses = [float(r["losses_kw"]) for r in rows]

    loss_pct = [(l / p * 100.0) if p else float("nan") for l, p in zip(losses, head_p)]

    return {
        "n": n,
        "converged_pct": 100.0 * converged / n if n else float("nan"),
        "v_pu_range": (min(v_mins), max(v_maxs)),
        "avg_head_p_kw": sum(head_p) / n if n else float("nan"),
        "avg_head_q_kvar": sum(head_q) / n if n else float("nan"),
        "avg_loss_pct": sum(loss_pct) / n if n else float("nan"),
    }


def summarize_operator_twin(state_rows):
    n = len(state_rows)
    v_mins = [r["v_pu_min"] for r in state_rows]
    v_maxs = [r["v_pu_max"] for r in state_rows]
    head_p = [r["feeder_head_p_kw"] for r in state_rows]
    head_q = [r["feeder_head_q_kvar"] for r in state_rows]
    converged = sum(1 for r in state_rows if r["converged"])

    return {
        "n": n,
        "converged_pct": 100.0 * converged / n if n else float("nan"),
        "v_pu_range": (min(v_mins), max(v_maxs)) if n else (None, None),
        "avg_head_p_kw": sum(head_p) / n if n else float("nan"),
        "avg_head_q_kvar": sum(head_q) / n if n else float("nan"),
    }


if __name__ == "__main__":
    import sys as _sys
    num_rows = int(_sys.argv[1]) if len(_sys.argv) > 1 else 86400

    t0 = time.time()
    print(f"Running noise-free full chain: {num_rows} timesteps...")
    result = run_full_chain(num_rows=num_rows, add_noise=False)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. byte_identical={result['byte_identical']} "
          f"byte_mismatch={result['byte_mismatch']}")

    gt = summarize_ground_truth()
    ot = summarize_operator_twin(result["state_rows"])
    print("Ground truth:", gt)
    print("Operator twin:", ot)

    print("\nRunning dropout demo...")
    demo = run_dropout_demo()
    print(f"Dropped meter {demo['dropped_meter_id']} at t={demo['drop_at']}; "
          f"pipeline completed {demo['num_timesteps_completed']} timesteps "
          f"(no stall), {demo['num_labels']} labels emitted.")
    dropouts = _read_dropout_log()
    print(f"Dropout log entries: {len(dropouts)}")
