"""
twin_state.py — Pydantic model for the WebSocket message schema.

Every frame sent to connected clients follows this structure.
It wraps actual readings, the twin's expected baseline, the residual
(difference), and an anomaly classification.

Required JSON schema:
{
    "timestamp": "ISO string",
    "actual":   { ...sensor fields },
    "expected": { ...twin baseline fields },
    "residual": { ...difference fields },
    "anomaly_flag": boolean,
    "anomaly_type": "none | spike | dropout | fdi"
}
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AnomalyType(str, Enum):
    """Classification of detected anomaly types."""
    NONE = "none"         # no anomaly detected
    SPIKE = "spike"       # sudden value surge
    DROPOUT = "dropout"   # value drops to zero / near-zero unexpectedly
    FDI = "fdi"           # false data injection (manipulated reading)


class SensorFields(BaseModel):
    """Numeric sensor fields shared by actual, expected, and residual blocks."""

    global_active_power: Optional[float] = None
    global_reactive_power: Optional[float] = None
    voltage: Optional[float] = None
    global_intensity: Optional[float] = None
    sub_metering_1: Optional[float] = None
    sub_metering_2: Optional[float] = None
    sub_metering_3: Optional[float] = None


class TwinState(BaseModel):
    """
    A single Digital Twin state frame sent over WebSocket.

    Contains the actual reading, the twin's expected baseline,
    the computed residual, and the anomaly classification.
    """

    # ── Node identifier ──
    node_id: int = Field(
        ...,
        description="Identifier for the node (1-49)"
    )

    # ── Timestamp of this frame ──
    timestamp: datetime = Field(
        ...,
        description="ISO-format timestamp of the reading"
    )

    # ── Sensor value blocks ──
    actual: SensorFields = Field(
        ...,
        description="Raw sensor values from the dataset"
    )
    expected: SensorFields = Field(
        ...,
        description="Digital Twin baseline (EMA) values"
    )
    residual: SensorFields = Field(
        ...,
        description="Difference: actual − expected"
    )

    # ── Anomaly classification ──
    anomaly_flag: bool = Field(
        False,
        description="True if any residual exceeds the anomaly threshold"
    )
    anomaly_type: AnomalyType = Field(
        AnomalyType.NONE,
        description="Type of anomaly detected"
    )

    model_config = ConfigDict(
        # Serialize enums as their string values in JSON output
        use_enum_values=True,
    )
