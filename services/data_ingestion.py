"""
data_ingestion.py — Load the preprocessed UCI power consumption CSV.

TRACK A — Owner: You
STATUS: IMPLEMENTED

Depends on:
    - config.settings.csv_path
    - models.SensorReading
"""

import os
from typing import Any, Dict, List, cast

import pandas as pd

from config import settings
from models.sensor_data import SensorReading


REQUIRED_COLUMNS = [
    "datetime",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def _resolve_dataset_path(csv_path: str) -> str:
    """Resolve and validate the configured dataset path."""
    resolved_csv_path = os.path.abspath(csv_path)
    if not os.path.isfile(resolved_csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{resolved_csv_path}'. "
            f"Check CSV_PATH in your .env file."
        )
    return resolved_csv_path


def _validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Validate required columns and raise a clear error when any are missing."""
    missing_columns = [
        column_name
        for column_name in REQUIRED_COLUMNS
        if column_name not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{', '.join(missing_columns)}"
        )


def _clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize raw CSV data for stable Digital Twin processing."""
    cleaned_dataframe = dataframe.copy()

    # Normalize common missing markers and trim text whitespace.
    cleaned_dataframe = cleaned_dataframe.replace("?", pd.NA)
    cleaned_dataframe = cleaned_dataframe.apply(
        lambda column: column.str.strip() if column.dtype == "object" else column
    )

    # Parse datetime safely and coerce invalid values to NaT.
    cleaned_dataframe["datetime"] = pd.to_datetime(
        cleaned_dataframe["datetime"],
        errors="coerce",
    )

    # Convert numeric sensor columns and coerce invalid values to NaN.
    for numeric_column_name in NUMERIC_COLUMNS:
        cleaned_dataframe[numeric_column_name] = pd.to_numeric(
            cleaned_dataframe[numeric_column_name],
            errors="coerce",
        )

    # Sort chronologically before forward fill so imputation is time-consistent.
    cleaned_dataframe = cleaned_dataframe.sort_values("datetime").reset_index(drop=True)

    # Forward fill numeric gaps to stabilize short missing streaks.
    cleaned_dataframe[NUMERIC_COLUMNS] = cleaned_dataframe[NUMERIC_COLUMNS].ffill()

    # Remove rows that still cannot be used after cleaning.
    cleaned_dataframe = cleaned_dataframe.dropna(
        subset=["datetime", *NUMERIC_COLUMNS]
    ).reset_index(drop=True)

    return cleaned_dataframe


def load_dataset(csv_path: str = settings.csv_path) -> List[SensorReading]:
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
    # Resolve path and load raw CSV rows.
    csv_path_resolved = _resolve_dataset_path(csv_path)
    raw_dataframe = pd.read_csv(csv_path_resolved)

    # Validate schema before preprocessing.
    _validate_required_columns(raw_dataframe)

    # Clean, normalize, and sort rows for robust ingestion.
    cleaned_dataframe = _clean_dataset(raw_dataframe)

    # Convert cleaned rows into validated SensorReading objects.
    record_rows = cast(List[Dict[str, Any]], cleaned_dataframe.to_dict("records"))
    readings: List[SensorReading] = [
        SensorReading(**row)
        for row in record_rows
    ]

    dropped_row_count = len(raw_dataframe) - len(cleaned_dataframe)
    print(
        "[data_ingestion] "
        f"Loaded {len(readings)} cleaned readings from {csv_path_resolved} "
        f"(dropped {dropped_row_count} invalid rows)"
    )
    return readings


def get_dataset_info() -> dict:
    """
    Return basic metadata about the loaded dataset without loading all rows.

    Useful for the /health or /api/info endpoints to report dataset status
    without consuming memory.

    Returns:
        dict: Keys — 'path', 'exists', 'row_count', 'columns', 'date_range'.
    """
    try:
        csv_path = _resolve_dataset_path()
    except FileNotFoundError:
        csv_path = os.path.abspath(settings.csv_path)
        return {"path": csv_path, "exists": False}

    raw_dataframe = pd.read_csv(csv_path)
    _validate_required_columns(raw_dataframe)
    cleaned_dataframe = _clean_dataset(raw_dataframe)

    return {
        "path": csv_path,
        "exists": True,
        "raw_row_count": len(raw_dataframe),
        "cleaned_row_count": len(cleaned_dataframe),
        "dropped_row_count": len(raw_dataframe) - len(cleaned_dataframe),
        "columns": list(raw_dataframe.columns),
        "date_range": {
            "start": str(cleaned_dataframe["datetime"].min()),
            "end": str(cleaned_dataframe["datetime"].max()),
        },
    }
