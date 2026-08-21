from pathlib import Path
import json
import time

import pandas as pd
from kafka import KafkaProducer

from .kafka_config import BOOTSTRAP_SERVERS, TOPIC

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR /
    "data" /
    "scaled_datasets"
)

REPRESENTATIVE_METERS = (
    BASE_DIR /
    "simulation" /
    "representative_meters.json"
)

# --------------------------------------------------
# Kafka Producer
# --------------------------------------------------

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# --------------------------------------------------
# Load representative meters
# --------------------------------------------------

with open(REPRESENTATIVE_METERS, "r") as f:
    representative_meters = json.load(f)

print(f"Loaded {len(representative_meters)} representative meters")

# --------------------------------------------------
# Load datasets (RES.md D10 Phase 1: scaled per-load series, keyed by
# load_name rather than positional glob order -- load_name is the
# authoritative join key, not file-list position)
# --------------------------------------------------

datasets = []

for meter in representative_meters:

    csv_path = DATASET_DIR / f"{meter['load_name'].lower()}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No scaled dataset for {meter['load_name']} at {csv_path}. "
            f"Run simulation/generate_scaled_datasets.py first."
        )

    datasets.append(pd.read_csv(csv_path))

print(f"Loaded {len(datasets)} scaled datasets")

# --------------------------------------------------
# Stream data
# --------------------------------------------------

num_rows = min(len(df) for df in datasets)

print(f"Streaming {num_rows} timestamps...\n")

for t in range(num_rows):

    print(f"Timestamp {t}")

    for meter, df in zip(representative_meters, datasets):

        row = df.iloc[t]

        # RES.md D10 Phase 1 / CLAUDE.md "Data caution": Voltage and
        # current at a metered node come only from the OpenDSS solve.
        # They are never published here -- only the load input (P, Q).
        message = {

            "timestamp": t,

            "datetime": row["datetime"],

            "meter_id": meter["meter_id"],

            "load_name": meter["load_name"],

            "bus": meter["bus"],

            "p_kw": float(row["p_kw"]),

            "q_kvar": float(row["q_kvar"]),

        }

        producer.send(
            TOPIC,
            key=meter["meter_id"].encode(),
            value=message
        )

    producer.flush()

    print(f"Published {len(datasets)} messages\n")

    # 1 second = 1 simulated minute
    time.sleep(1)