"""
ws.py — WebSocket endpoint for real-time Digital Twin streaming.

TRACK B — Owner: Teammate
STATUS: SKELETON — implement the WebSocket handler body.

Depends on:
    - services.twin_engine.TwinEngine
    - services.data_ingestion.load_dataset
    - services.logger.log_anomaly
    - config.settings.ws_frame_delay
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# ── Create a router for WebSocket endpoints ──
router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/stream")
async def stream_twin_state(websocket: WebSocket):
    """
    Stream Digital Twin state frames to connected WebSocket clients.

    Steps (to implement):
        1. Accept the WebSocket connection
        2. Load the dataset via data_ingestion.load_dataset()
        3. Create a TwinEngine instance
        4. Iterate through each reading:
            a. Process reading through twin_engine.process_reading()
            b. Send the resulting TwinState as JSON to the client
            c. If anomaly_flag is True, call logger.log_anomaly()
            d. Sleep for ws_frame_delay seconds (replay speed)
        5. Handle WebSocketDisconnect gracefully

    Args:
        websocket: The incoming WebSocket connection.
    """
    # TODO: Implement — Track B, task B4
    await websocket.accept()
    await websocket.send_json({
        "message": "WebSocket connected — stream not yet implemented"
    })
    await websocket.close()
