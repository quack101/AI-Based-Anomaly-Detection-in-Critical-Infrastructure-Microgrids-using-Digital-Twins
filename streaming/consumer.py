"""
RES.md D10 Phase 3 -- the consumer is the L3c "operator twin" (RES.md
D2.4): it reconstructs a state estimate from telemetry.delivered alone,
the way a real control-centre would, and never touches ground truth.

Per timestamp:
  1. Buffer delivered messages by timestamp, keyed by meter_id, with a
     WINDOW TIMEOUT (Phase 3 acceptance: a missing/dropped meter must
     not stall the twin -- the pre-Phase-3 consumer blocked forever on
     `if len(buffer[timestamp]) < EXPECTED_METERS: continue`, which is
     exactly the single-meter DoS RES.md flags).
  2. MISSING-DATA POLICY: hold-last-value. For any meter absent when the
     window closes, reuse that load's most recently received
     (P_injection, Q_injection) reading; before any reading has ever
     arrived for a load, fall back to its as-authored nominal DSS value
     (same fallback validate_phase0.py's nominal_total_kw capture uses).
     Chosen over pseudo-measurement substitution because Phase 5 (the
     sigma_pseudo framework for pseudo-measurements) does not exist yet
     -- hold-last-value needs no new machinery and is a physically
     reasonable approximation for a single missed 1-minute reading.
  3. Every dropout is logged to DROPOUT_LOG, a stream DISTINCT from
     telemetry.labels -- Irfan Sec 8.7 names cyber-vs-physical
     discrimination as an open problem; conflating "meter didn't arrive
     in time" with "meter was attacked" in the same log would silently
     prejudge that open question. A dropout is never written to labels
     and an attack (Phase 8+) will never be written to DROPOUT_LOG.
  4. Apply each (possibly substituted) meter's P,Q as that load's
     setpoint (same convention as validation/validate_phase0.py) and
     solve. This is DELIBERATELY NOT a re-derivation of x_true: for the
     16 of 49 representative loads whose OpenDSS Model != 1 (constant
     power), re-applying a SOLVED terminal reading as a fresh setpoint
     is a voltage-dependent approximation, not an identity -- confirmed
     empirically (feeder-head P differs by ~54 kW / 0.7% from the
     producer's own clean solve at the corpus peak timestamp, even
     noise-free and under a null attack). This is the correct, expected
     behaviour of an AMI-reconstructed "operator twin" (RES.md D2 item 4
     draws exactly this clean-vs-operator distinction) -- an exact
     inverse would require a real WLS state estimator, which is Phase 7.
     reports/phase3_streaming_report.txt quantifies this gap explicitly
     rather than treating it as a bug.

Layer boundary: this module never touches OpenDSS via anything other
than simulation.opendss_utils (compile/solve/metrics) -- it never calls
sensors.measurement_functions or reads a "clean" circuit; the only
circuit it ever solves is the one IT drives from delivered telemetry.
It reads sensors.registry for METADATA ONLY (id -> load_name/quantity
mapping), which tests/test_import_boundaries.py's own docstring
confirms is not a boundary violation (no circuit access).
"""

import csv
import json
import time
from pathlib import Path

import opendssdirect as dss

from simulation.opendss_utils import compile_feeder, get_solution_metrics
from sensors.registry import load_registry, enabled_entries

BASE_DIR = Path(__file__).resolve().parent.parent

REPRESENTATIVE_METERS = BASE_DIR / "simulation" / "representative_meters.json"
OPERATOR_TWIN_LOG = BASE_DIR / "simulation" / "phase3_operator_twin_log.csv"
DROPOUT_LOG = BASE_DIR / "simulation" / "phase3_dropout_log.csv"

STATE_FIELDS = [
    "timestamp", "real_timestamp", "converged", "iterations",
    "v_pu_min", "v_pu_max", "losses_kw", "losses_kvar",
    "feeder_head_p_kw", "feeder_head_q_kvar", "solve_time_s",
    "dropout_count",
]

DROPOUT_FIELDS = [
    "timestamp", "meter_id", "load_name", "policy",
    "p_kw_used", "q_kvar_used", "event_class",
]

DEFAULT_WINDOW_TIMEOUT_S = 5.0


class Consumer:
    def __init__(self, registry_variant="default", window_timeout_s=DEFAULT_WINDOW_TIMEOUT_S, clock=None):
        with open(REPRESENTATIVE_METERS) as f:
            self.representative_meters = json.load(f)
        self.expected_meters = len(self.representative_meters)

        registry = load_registry(variant=registry_variant)
        entries = enabled_entries(registry)

        # id -> (load_name, quantity) metadata only -- no circuit access
        # (tests/test_import_boundaries.py's FORBIDDEN_MODULES list
        # explicitly excludes sensors.registry for exactly this reason).
        self._p_id_for_load = {}
        self._q_id_for_load = {}
        for entry in entries:
            load_name = entry.get("load_name")
            if load_name is None:
                continue
            if entry["quantity"] == "P_injection" and load_name not in self._p_id_for_load:
                self._p_id_for_load[load_name] = entry["id"]
            elif entry["quantity"] == "Q_injection" and load_name not in self._q_id_for_load:
                self._q_id_for_load[load_name] = entry["id"]

        self.window_timeout_s = window_timeout_s
        self._clock = clock or _default_clock

        self._buffer = {}       # timestamp -> {meter_id: delivered_message}
        self._first_seen = {}   # timestamp -> clock time of first message
        self._last_known = {}   # load_name -> (p_kw, q_kvar)

        compile_feeder()

        self._nominal = {}
        for meter in self.representative_meters:
            dss.Loads.Name(meter["load_name"])
            self._nominal[meter["load_name"]] = (dss.Loads.kW(), dss.Loads.kvar())

        OPERATOR_TWIN_LOG.parent.mkdir(exist_ok=True)
        with open(OPERATOR_TWIN_LOG, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=STATE_FIELDS).writeheader()
        with open(DROPOUT_LOG, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=DROPOUT_FIELDS).writeheader()

    def ingest(self, delivered_message):
        """Buffer one delivered message. Returns a state row dict once
        its timestamp's window closes (complete or timed out), else
        None."""

        t = delivered_message["timestamp"]
        meter_id = delivered_message["meter_id"]

        if t not in self._buffer:
            self._buffer[t] = {}
            self._first_seen[t] = self._clock()

        self._buffer[t][meter_id] = delivered_message

        complete = len(self._buffer[t]) >= self.expected_meters
        timed_out = (self._clock() - self._first_seen[t]) >= self.window_timeout_s

        if not (complete or timed_out):
            return None

        return self._close_window(t)

    def flush_all(self):
        results = []
        for t in sorted(self._buffer.keys()):
            results.append(self._close_window(t))
        return results

    def _close_window(self, t):
        messages = self._buffer.pop(t)
        self._first_seen.pop(t, None)

        present_meter_ids = set(messages.keys())
        all_meter_ids = {m["meter_id"] for m in self.representative_meters}
        missing_meter_ids = all_meter_ids - present_meter_ids

        real_ts = None
        for meter in self.representative_meters:
            load_name = meter["load_name"]
            meter_id = meter["meter_id"]

            if meter_id in messages:
                msg = messages[meter_id]
                real_ts = real_ts or msg.get("real_timestamp")
                p_id = self._p_id_for_load[load_name]
                q_id = self._q_id_for_load[load_name]
                p_kw = msg["measurements"][p_id]
                q_kvar = msg["measurements"][q_id]
                self._last_known[load_name] = (p_kw, q_kvar)
                policy = None
            else:
                p_kw, q_kvar = self._last_known.get(
                    load_name, self._nominal[load_name]
                )
                policy = "hold_last_value" if load_name in self._last_known else "nominal_fallback"
                self._log_dropout(t, meter_id, load_name, policy, p_kw, q_kvar)

            dss.Loads.Name(load_name)
            dss.Loads.kW(float(p_kw))
            dss.Loads.kvar(float(q_kvar))

        solve_start = time.perf_counter()
        dss.Solution.Solve()
        solve_time_s = time.perf_counter() - solve_start

        metrics = get_solution_metrics(solve_time_s)
        state_row = self._log_state(t, real_ts, metrics, len(missing_meter_ids))

        return state_row

    def _log_dropout(self, t, meter_id, load_name, policy, p_kw_used, q_kvar_used):
        row = {
            "timestamp": t,
            "meter_id": meter_id,
            "load_name": load_name,
            "policy": policy,
            "p_kw_used": p_kw_used,
            "q_kvar_used": q_kvar_used,
            "event_class": "dropout",  # NEVER "attack" -- distinct event class, see module docstring
        }
        with open(DROPOUT_LOG, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=DROPOUT_FIELDS).writerow(row)

    def _log_state(self, t, real_ts, metrics, dropout_count):
        per_phase = metrics["v_pu_per_phase"]
        row = {
            "timestamp": t,
            "real_timestamp": real_ts,
            "converged": metrics["converged"],
            "iterations": metrics["iterations"],
            "v_pu_min": min(p["min"] for p in per_phase.values()),
            "v_pu_max": max(p["max"] for p in per_phase.values()),
            "losses_kw": metrics["losses_kw"],
            "losses_kvar": metrics["losses_kvar"],
            "feeder_head_p_kw": metrics["feeder_head_p_kw"],
            "feeder_head_q_kvar": metrics["feeder_head_q_kvar"],
            "solve_time_s": metrics["solve_time_s"],
            "dropout_count": dropout_count,
        }
        with open(OPERATOR_TWIN_LOG, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=STATE_FIELDS).writerow(row)
        return row


def _default_clock():
    return time.monotonic()


def main():
    """Live mode -- requires a running Kafka broker."""

    from kafka import KafkaConsumer, KafkaProducer
    from .kafka_config import BOOTSTRAP_SERVERS, TOPIC_DELIVERED, TOPIC_STATE

    kafka_consumer = KafkaConsumer(
        TOPIC_DELIVERED,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="smartgrid",
    )
    kafka_producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    consumer = Consumer()
    print(f"Loaded {consumer.expected_meters} representative meters. "
          f"Window timeout {consumer.window_timeout_s}s, policy=hold_last_value.")
    print("Waiting for messages...")

    for record in kafka_consumer:
        state_row = consumer.ingest(record.value)
        if state_row is None:
            continue

        kafka_producer.send(TOPIC_STATE, key=str(state_row["timestamp"]).encode(), value=state_row)
        kafka_producer.flush()

        if state_row["timestamp"] % 100 == 0:
            print(f"  t={state_row['timestamp']}  converged={state_row['converged']}  "
                  f"dropouts={state_row['dropout_count']}")


if __name__ == "__main__":
    main()
