"""
api.py — REST API endpoints.

TRACK B — Owner: Teammate
STATUS: SKELETON — implement the endpoint bodies.

Endpoints:
    GET /health       — Simple health check
    GET /api/history  — Retrieve recent anomaly logs from Supabase
"""

from fastapi import APIRouter

# ── Create a router with the /api prefix for non-WebSocket endpoints ──
router = APIRouter(tags=["API"])


@router.get("/health")
async def health_check():
    """
    Return a simple health status to confirm the server is running.

    Returns:
        dict: {"status": "ok"}
    """
    # Health check always works — no TODO needed
    return {"status": "ok"}


@router.get("/api/history")
async def get_anomaly_history(limit: int = 50):
    """
    Retrieve the most recent anomaly log entries from Supabase.

    Steps (to implement):
        1. Get the Supabase client
        2. Query the 'anomaly_logs' table, ordered by timestamp DESC
        3. Limit results to the 'limit' parameter
        4. Return the rows as a JSON list

    Args:
        limit: Maximum number of records to return (default 50).

    Returns:
        dict: {"logs": [...], "count": int}
    """
    # TODO: Implement Supabase query — Track B, task B3
    return {"logs": [], "count": 0, "message": "Not yet implemented"}
