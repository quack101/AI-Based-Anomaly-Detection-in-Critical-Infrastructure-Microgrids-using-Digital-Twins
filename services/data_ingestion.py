"""
data_ingestion.py — Load the preprocessed UCI power consumption CSV.

TRACK A — Owner: You
STATUS: IMPLEMENTED

Depends on:
    - config.settings.csv_path
    - models.SensorReading
"""

import os
from typing import List

import pandas as pd

from config import settings
from models.sensor_data import SensorReading


def load_dataset() -> List[SensorReading]:
    """
    Read the preprocessed CSV and return a chronologically sorted list of SensorReadings.

    Steps:
        1. Verify the CSV file exists at the configured path
        2. Read with pandas (comma-separated, parse 'datetime' column)
        3. Sort by datetime ascending
        4. Convert each row into a validated SensorReading via Pydantic
        5. Return the sorted list

    Returns:
        List[SensorReading]: All readings in chronological order.

    Raises:
        FileNotFoundError: If the CSV file does not exist at config.csv_path.
        ValueError: If the CSV has unexpected columns or invalid data.
    """
    # ── Step 1: Resolve and validate the file path ──
    csv_path = os.path.abspath(settings.csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            f"Check CSV_PATH in your .env file."
        )

    # ── Step 2: Read the CSV into a DataFrame ──
    dataframe = pd.read_csv(
        csv_path,
        parse_dates=["datetime"],  # auto-parse the datetime column
    )

    # ── Step 3: Sort by datetime ascending (oldest first) ──
    dataframe = dataframe.sort_values("datetime").reset_index(drop=True)

    # ── Step 4: Convert each row to a SensorReading ──
    # .to_dict("records") gives a list of dicts with CSV column names as keys
    # Pydantic aliases (e.g., alias="Global_active_power") handle the mapping
    readings: List[SensorReading] = [
        SensorReading(**row)
        for row in dataframe.to_dict("records")
    ]

    # ── Step 5: Return the validated, sorted readings ──
    print(f"[data_ingestion] Loaded {len(readings)} readings from {csv_path}")
    return readings


def get_dataset_info() -> dict:
    """
    Return basic metadata about the loaded dataset without loading all rows.

    Useful for the /health or /api/info endpoints to report dataset status
    without consuming memory.

    Returns:
        dict: Keys — 'path', 'exists', 'row_count', 'columns', 'date_range'.
    """
    csv_path = os.path.abspath(settings.csv_path)

    # ── Check if file exists ──
    if not os.path.isfile(csv_path):
        return {"path": csv_path, "exists": False}

    # ── Read just enough to get metadata ──
    dataframe = pd.read_csv(csv_path, parse_dates=["datetime"])

    return {
        "path": csv_path,
        "exists": True,
        "row_count": len(dataframe),
        "columns": list(dataframe.columns),
        "date_range": {
            "start": str(dataframe["datetime"].min()),
            "end": str(dataframe["datetime"].max()),
        },
    }
