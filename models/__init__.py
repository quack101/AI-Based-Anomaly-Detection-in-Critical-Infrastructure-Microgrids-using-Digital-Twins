"""
models package — Pydantic data models shared across the application.

Exports:
    SensorReading  — a single row from the UCI power consumption dataset
    TwinState      — the WebSocket message schema (actual vs expected vs residual)
"""

from models.sensor_data import SensorReading
from models.twin_state import TwinState

__all__ = ["SensorReading", "TwinState"]
