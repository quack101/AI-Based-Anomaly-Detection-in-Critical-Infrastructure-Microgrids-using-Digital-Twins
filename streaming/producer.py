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
    "all_synthetic_datasets"
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
# Load datasets
# --------------------------------------------------

datasets = []

csv_files = sorted(DATASET_DIR.glob("synthetic_*.csv"))

if len(csv_files) != len(representative_meters):
    raise ValueError(
        f"Found {len(csv_files)} datasets but "
        f"{len(representative_meters)} representative meters."
    )

for csv in csv_files:

    df = pd.read_csv(csv)

    datasets.append(df)

print(f"Loaded {len(datasets)} synthetic datasets")

# --------------------------------------------------
# Stream data
# --------------------------------------------------

num_rows = min(len(df) for df in datasets)

print(f"Streaming {num_rows} timestamps...\n")

for t in range(num_rows):

    print(f"Timestamp {t}")

    for meter, df in zip(representative_meters, datasets):

        row = df.iloc[t]

        message = {

            "timestamp": t,

            "datetime": row["datetime"],

            "meter_id": meter["meter_id"],

            "load_name": meter["load_name"],

            "bus": meter["bus"],

            "p_kw": float(row["Global_active_power"]),

            "q_kvar": float(row["Global_reactive_power"]),

            "voltage": float(row["Voltage"]),

            "current": float(row["Global_intensity"])

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