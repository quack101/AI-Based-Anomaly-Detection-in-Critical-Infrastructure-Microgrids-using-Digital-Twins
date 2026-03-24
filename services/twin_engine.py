"""
twin_engine.py — Digital Twin baseline computation and anomaly detection.

TRACK A — Owner: Teammate (Mocked for Track B Testing)
STATUS: FUNCTIONAL MOCK — implements simple baseline + random anomalies.

Depends on:
    - config.settings (ema_window, anomaly_threshold)
    - models.SensorReading
    - models.TwinState, SensorFields, AnomalyType
"""

import random
from typing import List, Optional

from models.sensor_data import SensorReading
from models.twin_state import TwinState, SensorFields, AnomalyType
from config import settings


class TwinEngine:
    """
    Stateful Digital Twin engine that maintains a rolling baseline
    and computes residuals for each incoming sensor reading.
    """

    def __init__(self, ema_window: int = settings.ema_window,
                 anomaly_threshold: float = settings.anomaly_threshold):
        """
        Initialize the twin engine.
        """
        self.ema_window = ema_window
        self.anomaly_threshold = anomaly_threshold
        self._history: List[SensorReading] = []

    def process_reading(self, reading: SensorReading) -> TwinState:
        """
        Process a single sensor reading and return the twin state.
        
        MOCK LOGIC: 
        1. 'Expected' is the average of the last N readings.
        2. 'Residual' is Actual - Average.
        3. 5% chance to force a SPIKE anomaly for testing the logger.
        """
        # ── 1. Update History ──
        self._history.append(reading)
        if len(self._history) > self.ema_window:
            self._history.pop(0)

        # ── 2. Calculate Expected Baseline (Simple Average) ──
        expected = self._calculate_average()

        # ── 3. Calculate Residuals ──
        actual_fields = self._to_sensor_fields(reading)
        residual_fields = self._calculate_residual(actual_fields, expected)

        # ── 4. Determine Anomaly (Mock Logic) ──
        # 5% chance to randomly flag an anomaly even if data is normal
        is_random_anomaly = random.random() < 0.05
        
        # ── 5. Build and Return State ──
        return TwinState(
            timestamp=reading.timestamp,
            actual=actual_fields,
            expected=expected,
            residual=residual_fields,
            anomaly_flag=is_random_anomaly,
            anomaly_type=AnomalyType.SPIKE if is_random_anomaly else AnomalyType.NONE
        )

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

    def _calculate_average(self) -> SensorFields:
        """Compute the average of the history window."""
        if not self._history:
            return SensorFields()

        count = len(self._history)
        return SensorFields(
            global_active_power=sum(r.global_active_power for r in self._history) / count,
            global_reactive_power=sum(r.global_reactive_power for r in self._history) / count,
            voltage=sum(r.voltage for r in self._history) / count,
            global_intensity=sum(r.global_intensity for r in self._history) / count,
            sub_metering_1=sum(r.sub_metering_1 for r in self._history) / count,
            sub_metering_2=sum(r.sub_metering_2 for r in self._history) / count,
            sub_metering_3=sum(r.sub_metering_3 for r in self._history) / count
        )

    def _calculate_residual(self, actual: SensorFields, expected: SensorFields) -> SensorFields:
        """Compute Actual - Expected."""
        return SensorFields(
            global_active_power=actual.global_active_power - (expected.global_active_power or 0),
            global_reactive_power=actual.global_reactive_power - (expected.global_reactive_power or 0),
            voltage=actual.voltage - (expected.voltage or 0),
            global_intensity=actual.global_intensity - (expected.global_intensity or 0),
            sub_metering_1=actual.sub_metering_1 - (expected.sub_metering_1 or 0),
            sub_metering_2=actual.sub_metering_2 - (expected.sub_metering_2 or 0),
            sub_metering_3=actual.sub_metering_3 - (expected.sub_metering_3 or 0)
        )

