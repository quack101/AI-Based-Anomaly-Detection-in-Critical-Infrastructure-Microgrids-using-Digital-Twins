import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from config import settings
from services.data_ingestion import load_dataset
from services.logger import log_anomaly, log_frame
from services.twin_engine import TwinEngine

router = APIRouter(tags=["WebSocket"])

_readings_list: list = []
_num_nodes = 49
_data_ready = asyncio.Event()

def _preload_sync():
    """Blocking CSV loads — runs in a thread pool, not the event loop."""
    result = []
    print("[websocket] Pre-loading all node datasets...")
    for node_id in range(1, _num_nodes + 1):
        csv_path = f"data/synthetic_{node_id:02d}.csv"
        result.append(load_dataset(csv_path))
    print("[websocket] All datasets ready.")
    return result

async def preload_on_startup():
    """Call this from your app lifespan/startup event."""
    global _readings_list
    loop = asyncio.get_event_loop()
    _readings_list = await loop.run_in_executor(None, _preload_sync)
    _data_ready.set()

@router.websocket("/ws/stream")
async def stream_twin_state(websocket: WebSocket):
    await websocket.accept()
    print("[websocket] Client connected to /ws/stream")

    # Wait for data to finish loading before streaming
    if not _data_ready.is_set():
        await websocket.send_json({"status": "loading", "message": "Server is loading data, please wait..."})
        await _data_ready.wait()

    try:
        engines = [TwinEngine() for _ in range(_num_nodes)]
        num_readings = len(_readings_list[0])

        for i in range(num_readings):
            node_states = []
            for node_id in range(1, _num_nodes + 1):
                reading = _readings_list[node_id - 1][i]
                twin_state = engines[node_id - 1].process_reading(reading, node_id)
                node_states.append(twin_state)

            await websocket.send_json(
                {"nodes": [state.model_dump(mode="json") for state in node_states]}
            )

            for state in node_states:
                asyncio.create_task(log_frame(state))
                if state.anomaly_flag:
                    asyncio.create_task(log_anomaly(state))

            await asyncio.sleep(settings.ws_frame_delay)

    except WebSocketDisconnect:
        print("[websocket] Client disconnected from /ws/stream")
    except Exception as error:
        print(f"[websocket] [ERROR] Stream interrupted: {error}")
        try:
            await websocket.send_json({"error": str(error)})
            await websocket.close()
        except:
            pass