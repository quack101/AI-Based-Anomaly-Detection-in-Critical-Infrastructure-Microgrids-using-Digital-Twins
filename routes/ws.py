"""
ws.py — WebSocket endpoint for real-time Digital Twin streaming.

TRACK B — Owner: You
STATUS: IMPLEMENTED

Depends on:
    - services.twin_engine.TwinEngine
    - services.data_ingestion.load_dataset
    - services.logger.log_anomaly
    - config.settings.ws_frame_delay
"""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import settings
from services.data_ingestion import load_dataset
from services.twin_engine import TwinEngine
from services.logger import log_anomaly

# ── Create a router for WebSocket endpoints ──
router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/stream")
async def stream_twin_state(websocket: WebSocket):
    """
    Stream Digital Twin state frames to connected WebSocket clients.

    Simulates real-time data flow by iterating through the dataset
    and processing each reading through the Digital Twin engine.
    """
    # ── 1. Accept the incoming connection ──
    await websocket.accept()
    print("[websocket] Client connected to /ws/stream")

    try:
        # ── 2. Initialize dependencies ──
        # Load the entire processed dataset into memory
        readings = load_dataset()

        # Initialize the stateful twin engine (ema_window, threshold)
        engine = TwinEngine()

        # ── 3. Iterate through each sensor reading ──
        for reading in readings:
            # ── a. Process through the Digital Twin ──
            # This computes the expected baseline and residuals
            twin_state = engine.process_reading(reading)

            # ── b. Stream the frame to the client ──
            # We use model_dump(mode="json") to ensure datetime/enums are serialized
            await websocket.send_json(twin_state.model_dump(mode="json"))

            # ── c. Log to Supabase if an anomaly is detected ──
            if twin_state.anomaly_flag:
                # We fire-and-forget the log task so it doesn't block the stream
                asyncio.create_task(log_anomaly(twin_state))

            # ── d. Wait for the configured replay delay ──
            await asyncio.sleep(settings.ws_frame_delay)

    except WebSocketDisconnect:
        # ── 4. Handle client disconnection ──
        print("[websocket] Client disconnected from /ws/stream")

    except Exception as error:
        # ── 5. Clean up on server-side errors ──
        print(f"[websocket] [ERROR] Stream interrupted: {error}")
        try:
            await websocket.send_json({"error": str(error)})
            await websocket.close()
        except:
            pass

