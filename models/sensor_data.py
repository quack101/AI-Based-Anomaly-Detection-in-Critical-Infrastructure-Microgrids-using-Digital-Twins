"""
sensor_data.py — Pydantic model for a single sensor reading.

Maps to one row of the preprocessed UCI power consumption CSV.
CSV columns:
    datetime, Global_active_power, Global_reactive_power,
    Voltage, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3
"""

from datetime import datetime as dt

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """A single timestamped power consumption reading from the preprocessed dataset."""

    # ── Timestamp (single 'datetime' column from the CSV) ──
    timestamp: dt = Field(
        ...,
        alias="datetime",
        description="Timestamp of the reading in ISO format"
    )

    # ── Power measurements ──
    global_active_power: float = Field(
        ...,
        alias="Global_active_power",
        description="Household global minute-averaged active power (kilowatt)"
    )
    global_reactive_power: float = Field(
        ...,
        alias="Global_reactive_power",
        description="Household global minute-averaged reactive power (kilowatt)"
    )
    voltage: float = Field(
        ...,
        alias="Voltage",
        description="Minute-averaged voltage (volt)"
    )
    global_intensity: float = Field(
        ...,
        alias="Global_intensity",
        description="Household global minute-averaged current intensity (ampere)"
    )

    # ── Sub-metering readings ──
    sub_metering_1: float = Field(
        ...,
        alias="Sub_metering_1",
        description="Energy sub-metering No.1 — kitchen (watt-hour)"
    )
    sub_metering_2: float = Field(
        ...,
        alias="Sub_metering_2",
        description="Energy sub-metering No.2 — laundry room (watt-hour)"
    )
    sub_metering_3: float = Field(
        ...,
        alias="Sub_metering_3",
        description="Energy sub-metering No.3 — water heater + AC (watt-hour)"
    )

    class Config:
        # Allow using both alias (CSV column name) and field name
        populate_by_name = True
        from_attributes = True
