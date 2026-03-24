"""
api.py — REST API endpoints.

TRACK B — Owner: You
STATUS: IMPLEMENTED

Endpoints:
    GET /health       — Simple health check
    GET /api/history  — Retrieve recent anomaly logs from Supabase
"""

from fastapi import APIRouter, HTTPException

from services.supabase_client import get_supabase_client

# ── Create a router with the /api prefix for non-WebSocket endpoints ──
router = APIRouter(tags=["API"])


@router.get("/health")
async def health_check():
    """
    Return a simple health status to confirm the server is running.

    Returns:
        dict: {"status": "ok"}
    """
    return {"status": "ok"}


@router.get("/api/history")
async def get_anomaly_history(limit: int = 50):
    """
    Retrieve the most recent anomaly log entries from Supabase.

    Queries the 'anomaly_logs' table, ordered by timestamp DESC.

    Args:
        limit: Maximum number of records to return (default 50).

    Returns:
        dict: {"logs": list, "count": int}

    Raises:
        HTTPException: if Supabase query fails.
    """
    try:
        # ── 1. Get the authenticated Supabase client ──
        supabase = get_supabase_client()

        # ── 2. Query the 'anomaly_logs' table ──
        result = (
            supabase.table("anomaly_logs")
            .select("*")
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )

        return {
            "logs": result.data,
            "count": len(result.data)
        }

    except Exception as error:
        # ── 3. Handle query errors ──
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch anomaly history: {error}"
        )

