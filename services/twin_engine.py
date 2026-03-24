"""
twin_engine.py — Digital Twin baseline computation and anomaly detection.

TRACK A — Owner: You
STATUS: SKELETON — implement the function bodies.

Depends on:
    - config.settings (ema_window, anomaly_threshold)
    - models.SensorReading
    - models.TwinState, SensorFields, AnomalyType
"""

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

        Args:
            ema_window: Number of past readings for the EMA baseline.
            anomaly_threshold: Std-dev multiplier to flag anomalies.
        """
        self.ema_window = ema_window
        self.anomaly_threshold = anomaly_threshold

        # TODO: Initialize internal state (running averages, std devs, etc.)
        self._history: List[SensorReading] = []
        self._baseline: Optional[SensorFields] = None

    def process_reading(self, reading: SensorReading) -> TwinState:
        """
        Process a single sensor reading and return the twin state.

        Steps (to implement):
            1. Append reading to history
            2. Compute EMA baseline from history window
            3. Calculate residuals (actual − expected)
            4. Check residuals against threshold to set anomaly flag/type
            5. Return a TwinState with all fields populated

        Args:
            reading: A single SensorReading from the dataset.

        Returns:
            TwinState: The complete twin state frame for WebSocket delivery.
        """
        # TODO: Implement — Track A, task A2
        raise NotImplementedError("TwinEngine.process_reading() not yet implemented")

    def predict_anomaly(self, residual: SensorFields) -> tuple:
        """
        Classify the anomaly type based on residual values.

        This is the HOOK POINT for ML teammates to replace with
        LSTM / RL-based anomaly detection.

        Args:
            residual: The computed residual (actual − expected).

        Returns:
            tuple: (anomaly_flag: bool, anomaly_type: AnomalyType)
        """
        # TODO: Implement basic threshold logic — Track A, task A2
        # ML teammates will replace this with their model later
        raise NotImplementedError("TwinEngine.predict_anomaly() not yet implemented")
