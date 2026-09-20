"""
RES.md D10 Phase 3 -- the producer is now the L1 ground-truth generator
+ L2 sensor layer (RES.md D2 item 4 / item 5), not a raw-CSV publisher.

Per timestamp:
  1. Drive the 49 representative loads from dataset/node_loads_91/ (the
     IDEAL corpus) exactly as validation/validate_phase0.py does -- same
     name-keyed lookup, same 42-unmetered-loads-stay-at-nominal
     architecture. Solve once. This is x_true, the CLEAN/ground-truth
     twin (RES.md D2 item 4's "run the twin once on clean inputs").
  2. Log x_true's solution metrics with the SAME fields
     validate_phase0.py logs, to GROUND_TRUTH_LOG -- this is what
     reports/phase3_streaming_report.txt compares against
     reports/phase0_validation_report_ideal.txt for Phase 3 acceptance
     criterion 2. It is computed by literally the same methodology, so
     it reproduces those numbers exactly; the DELIVERED/consumer path
     (a separate, later solve driven by AMI-style measurements) is a
     distinct, approximate reconstruction by design -- see
     streaming/consumer.py's docstring.
  3. Call sensors/sensor_layer.py -- ONE pass over
     sensors.registry.enabled_entries(registry) -- to build z_true.
     Every field this producer ever publishes traces to exactly one
     registry entry, by construction (RES.md D10 Phase 2 acceptance,
     now actually wired into the live pipeline, closing the one Phase-2
     criterion that could not pass until Phase 3).
  4. Publish one telemetry.raw message per meter, keyed by meter_id,
     containing that meter's registry-id-keyed measurements.

Layer boundary: this module reads the solved circuit only via
simulation.opendss_utils (already an approved ground-truth-generator
module) and sensors.sensor_layer/measurement_functions -- the only
modules RES.md D2 / tests/test_import_boundaries.py permit to do so.
The injector and consumer never do this (see their own docstrings).
"""

import csv
import json
import random
import time
from pathlib import Path

import pandas as pd
import opendssdirect as dss

from simulation.opendss_utils import compile_feeder, get_solution_metrics
from sensors.registry import load_registry, enabled_entries
from sensors.sensor_layer import build_measurement_dict

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "node_loads_91"
REPRESENTATIVE_METERS = BASE_DIR / "simulation" / "representative_meters.json"

GROUND_TRUTH_LOG = BASE_DIR / "simulation" / "phase3_ground_truth_log.csv"

GROUND_TRUTH_FIELDS = [
    "timestamp", "real_timestamp", "converged", "iterations",
    "v_pu_min", "v_pu_max", "losses_kw", "losses_kvar",
    "feeder_head_p_kw", "feeder_head_q_kvar", "solve_time_s",
]

ADD_NOISE_DEFAULT = True  # live/production default -- see run_timestep(add_noise=)


def load_inputs():
    """Same name-keyed corpus lookup as validation/validate_phase0.py
    (duplicated intentionally, not imported from it -- that script is
    the frozen, already-validated Phase 0/1 acceptance artifact and
    must not gain a new caller that could motivate future edits to it)."""

    with open(REPRESENTATIVE_METERS, "r") as f:
        representative_meters = json.load(f)

    available = {
        f.stem[len("node_"):].lower(): f
        for f in DATASET_DIR.glob("node_*.csv")
    }

    missing = [
        m["load_name"] for m in representative_meters
        if m["load_name"].lower() not in available
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} representative meter(s) have no corpus file "
            f"in {DATASET_DIR}: {missing}"
        )

    datasets = [
        pd.read_csv(available[m["load_name"].lower()])
        for m in representative_meters
    ]

    return representative_meters, datasets


def _entries_by_load_name(entries):
    by_load = {}
    for entry in entries:
        load_name = entry.get("load_name")
        if load_name is None:
            continue  # head/branch entries aren't per-meter
        by_load.setdefault(load_name, []).append(entry)
    return by_load


class Producer:
    """Framework-agnostic per-timestep driver. run_timestep() does the
    OpenDSS solve + sensor-layer measurement and returns plain dicts --
    no Kafka dependency -- so it can be exercised identically in live
    mode (main(), which wraps it with a real KafkaProducer) and in
    direct-replay validation (validation/validate_phase3.py, used
    because no Docker broker is available in this environment)."""

    def __init__(self, registry_variant="default", add_noise=ADD_NOISE_DEFAULT, seed=None):
        self.representative_meters, self.datasets = load_inputs()
        self.registry = load_registry(variant=registry_variant)
        self.entries = enabled_entries(self.registry)
        self.entries_by_load = _entries_by_load_name(self.entries)
        self.add_noise = add_noise
        self.rng = random.Random(seed) if add_noise else None

        compile_feeder()

        self.num_rows = min(len(df) for df in self.datasets)

        GROUND_TRUTH_LOG.parent.mkdir(exist_ok=True)
        with open(GROUND_TRUTH_LOG, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=GROUND_TRUTH_FIELDS).writeheader()

    def run_timestep(self, t):
        """Returns (raw_messages, ground_truth_row). raw_messages is a
        list of one dict per representative meter, ready to publish to
        TOPIC_RAW keyed by meter_id."""

        real_ts = None
        for i, (meter, df) in enumerate(zip(self.representative_meters, self.datasets)):
            row = df.iloc[t]
            if i == 0:
                real_ts = str(row["timestamp"])

            dss.Loads.Name(meter["load_name"])
            dss.Loads.kW(float(row["P_kW"]))
            dss.Loads.kvar(float(row["Q_kvar"]))

        solve_start = time.perf_counter()
        dss.Solution.Solve()
        solve_time_s = time.perf_counter() - solve_start

        metrics = get_solution_metrics(solve_time_s)
        self._log_ground_truth(t, real_ts, metrics)

        z_dict = build_measurement_dict(self.entries, add_noise=self.add_noise, rng=self.rng)

        raw_messages = []
        for meter in self.representative_meters:
            load_name = meter["load_name"]
            measurements = {
                entry["id"]: z_dict[entry["id"]]
                for entry in self.entries_by_load.get(load_name, [])
            }
            raw_messages.append({
                "timestamp": t,
                "real_timestamp": real_ts,
                "meter_id": meter["meter_id"],
                "load_name": load_name,
                "bus": meter["bus"],
                "measurements": measurements,
            })

        return raw_messages, metrics

    def _log_ground_truth(self, t, real_ts, metrics):
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
        }
        with open(GROUND_TRUTH_LOG, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=GROUND_TRUTH_FIELDS).writerow(row)


def main():
    """Live mode -- requires a running Kafka broker (streaming/docker-compose.yml)."""

    from kafka import KafkaProducer
    from .kafka_config import BOOTSTRAP_SERVERS, TOPIC_RAW

    kafka_producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    producer = Producer()
    print(f"Loaded {len(producer.representative_meters)} representative meters, "
          f"{len(producer.entries)} enabled registry entries")
    print(f"Streaming {producer.num_rows} timestamps to {TOPIC_RAW}...")

    for t in range(producer.num_rows):
        raw_messages, metrics = producer.run_timestep(t)

        for msg in raw_messages:
            kafka_producer.send(TOPIC_RAW, key=msg["meter_id"].encode(), value=msg)
        kafka_producer.flush()

        if t % 100 == 0:
            print(f"  t={t}  converged={metrics['converged']}  "
                  f"published {len(raw_messages)} raw messages")

        time.sleep(1)  # 1 second = 1 simulated minute, matching the prior cadence


if __name__ == "__main__":
    main()
