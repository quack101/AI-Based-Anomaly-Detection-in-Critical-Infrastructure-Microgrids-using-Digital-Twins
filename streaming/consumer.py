import csv
from pathlib import Path
import json
from collections import defaultdict

import opendssdirect as dss
from kafka import KafkaConsumer

from .kafka_config import BOOTSTRAP_SERVERS, TOPIC
from simulation.opendss_utils import (
    compile_feeder,
    get_load_telemetry,
)

BASE_DIR = Path(__file__).resolve().parent.parent

DIGITAL_TWIN_OUTPUT = (
    BASE_DIR /
    "simulation" /
    "digital_twin_telemetry.csv"
)

# --------------------------------------------------
# Load representative meters
# --------------------------------------------------

REPRESENTATIVE_METERS = (
    BASE_DIR /
    "simulation" /
    "representative_meters.json"
)

with open(REPRESENTATIVE_METERS, "r") as f:
    representative_meters = json.load(f)

EXPECTED_METERS = len(representative_meters)

print(f"Loaded {EXPECTED_METERS} representative meters")

# --------------------------------------------------
# Compile IEEE-123 feeder
# --------------------------------------------------
#Without this, your consumer never loads the OpenDSS model.
compile_feeder()

# --------------------------------------------------
# Create output CSV
# --------------------------------------------------

if not DIGITAL_TWIN_OUTPUT.exists():

    with open(DIGITAL_TWIN_OUTPUT, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "datetime",
            "meter_id",
            "load_name",
            "bus",

            "p_kw",
            "q_kvar",

            "input_voltage",
            "input_current",

            "bus_voltage_pu",
            "bus_voltage_v",
            "current_a",
        ])

# --------------------------------------------------
# Kafka Consumer
# --------------------------------------------------

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="earliest",
    group_id="smartgrid",
)

# --------------------------------------------------
# Buffer
# --------------------------------------------------

buffer = defaultdict(dict)

print("Waiting for messages...\n")

# ==================================================
# Main Loop
# ==================================================

for record in consumer:

    message = record.value

    print("Received:")
    print(message)
    print()

    timestamp = message["timestamp"]
    meter_id = message["meter_id"]

    # Store message
    buffer[timestamp][meter_id] = message

    # ---------------------------------------------
    # Wait until every representative meter arrives
    # ---------------------------------------------

    if len(buffer[timestamp]) < EXPECTED_METERS:
        continue

    print(f"\nTimestamp {timestamp} complete")

    messages = buffer[timestamp]

    # ---------------------------------------------
    # Update OpenDSS
    # ---------------------------------------------

    for msg in messages.values():

        dss.Loads.Name(msg["load_name"])

        dss.Loads.kW(msg["p_kw"])

        dss.Loads.kvar(msg["q_kvar"])

    # ---------------------------------------------
    # Solve ONLY ONCE
    # ---------------------------------------------

    dss.Solution.Solve()
    #debug
    if not dss.Solution.Converged():

        print(f"Power flow failed at timestamp {timestamp}")

        del buffer[timestamp]

        continue
    # ---------------------------------------------
    # Extract telemetry
    # ---------------------------------------------

    rows = []

    for msg in messages.values():

        telemetry = get_load_telemetry(msg["load_name"])

        row = {
            "timestamp": msg["timestamp"],
            "datetime": msg["datetime"],
            "meter_id": msg["meter_id"],
            "load_name": msg["load_name"],
            "bus": msg["bus"],

            # Input values
            "p_kw": msg["p_kw"],
            "q_kvar": msg["q_kvar"],
            "input_voltage": msg["voltage"],
            "input_current": msg["current"],

            # OpenDSS results
            "bus_voltage_pu": telemetry["voltage_pu"],
            "bus_voltage_v": telemetry["voltage_v"],
            "current_a": telemetry["current_a"],
        }

        rows.append(row)

    print(
        f"Timestamp {timestamp}: "
        f"generated {len(rows)} telemetry rows")

    with open(DIGITAL_TWIN_OUTPUT, "a", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "datetime",
                "meter_id",
                "load_name",
                "bus",

                "p_kw",
                "q_kvar",

                "input_voltage",
                "input_current",

                "bus_voltage_pu",
                "bus_voltage_v",
                "current_a",
            ]
        )

        writer.writerows(rows)
        f.flush()
print(f"Saved {len(rows)} rows")
   
    # ---------------------------------------------
    # Clear timestamp buffer
    # ---------------------------------------------
del buffer[timestamp]