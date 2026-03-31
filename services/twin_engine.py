"""
twin_engine.py — Digital Twin baseline computation in teammate-ready mode.

TRACK A — Owner: Teammate (integration-ready)
STATUS: SIMPLIFIED MODE — expected baseline only, anomaly/residual deferred.

Depends on:
    - config.settings (ema_window)
    - models.SensorReading
    - models.TwinState, SensorFields, AnomalyType
"""

from typing import Optional

from config import settings
from models.sensor_data import SensorReading
from models.twin_state import AnomalyType, SensorFields, TwinState


class TwinEngine:
    """
    Stateful Digital Twin engine that maintains an EMA expected baseline.

    This simplified mode intentionally defers residual and anomaly logic
    to teammate-owned modules while preserving the stable WebSocket schema.
    """

    def __init__(self, ema_window: int = settings.ema_window):
        """Initialize EMA baseline state for per-field expected values."""
        self.ema_window = ema_window
        self._ema_alpha = 2 / (max(ema_window, 1) + 1)
        self._expected_state: Optional[SensorFields] = None

    def process_reading(self, reading: SensorReading) -> TwinState:
        """
        Process a single sensor reading and return the twin state.

        Simplified mode behavior:
        1. Compute expected values using EMA.
        2. Skip residual calculation (placeholder values only).
        3. Keep anomaly outputs fixed to false/none.
        """
        # Convert incoming model to a consistent numeric block.
        actual_fields = self._to_sensor_fields(reading)
        # Update EMA baseline state and expose it as the expected block.
        expected_fields = self._update_expected_state(actual_fields)
        # Keep residual as a schema placeholder until teammate logic is added.
        residual_fields = SensorFields()
        # Keep anomaly outputs deterministic and disabled in this phase.
        anomaly_flag, anomaly_type = self.predict_anomaly(actual_fields, expected_fields)

        return TwinState(
            timestamp=reading.timestamp,
            actual=actual_fields,
            expected=expected_fields,
            residual=residual_fields,
            anomaly_flag=anomaly_flag,
            anomaly_type=anomaly_type,
        )

    def predict_anomaly(
        self,
        actual_fields: SensorFields,
        expected_fields: SensorFields,
    ) -> tuple[bool, AnomalyType]:
        """
        Teammate integration hook for anomaly decision.

        Returns disabled anomaly outputs for now.
        """
        _ = actual_fields
        _ = expected_fields
        return False, AnomalyType.NONE

    def _to_sensor_fields(self, reading: SensorReading) -> SensorFields:
        """Helper to convert Model to SensorFields block."""
        return SensorFields(
            global_active_power=reading.global_active_power,
            global_reactive_power=reading.global_reactive_power,
            voltage=reading.voltage,
            global_intensity=reading.global_intensity,
            sub_metering_1=reading.sub_metering_1,
            sub_metering_2=reading.sub_metering_2,
            sub_metering_3=reading.sub_metering_3
        )

    def _update_expected_state(self, actual_fields: SensorFields) -> SensorFields:
        """Update and return per-field EMA expected values."""
        if self._expected_state is None:
            self._expected_state = actual_fields
            return self._expected_state

        self._expected_state = SensorFields(
            global_active_power=self._ema(
                actual_fields.global_active_power,
                self._expected_state.global_active_power,
            ),
            global_reactive_power=self._ema(
                actual_fields.global_reactive_power,
                self._expected_state.global_reactive_power,
            ),
            voltage=self._ema(actual_fields.voltage, self._expected_state.voltage),
            global_intensity=self._ema(
                actual_fields.global_intensity,
                self._expected_state.global_intensity,
            ),
            sub_metering_1=self._ema(
                actual_fields.sub_metering_1,
                self._expected_state.sub_metering_1,
            ),
            sub_metering_2=self._ema(
                actual_fields.sub_metering_2,
                self._expected_state.sub_metering_2,
            ),
            sub_metering_3=self._ema(
                actual_fields.sub_metering_3,
                self._expected_state.sub_metering_3,
            ),
        )
        return self._expected_state

    def _ema(self, actual_value: Optional[float], previous_expected: Optional[float]) -> Optional[float]:
        """Compute one EMA step for a single numeric field."""
        if actual_value is None:
            return previous_expected
        if previous_expected is None:
            return actual_value
        return (self._ema_alpha * actual_value) + ((1 - self._ema_alpha) * previous_expected)

