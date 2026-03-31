"""
api.py — REST API endpoints.

TRACK B — Owner: You
STATUS: IMPLEMENTED

Endpoints:
    GET /health       — Simple health check
    GET /api/history  — Retrieve recent anomaly logs from Supabase
"""

from fastapi import APIRouter, Query

from config import settings
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
async def get_anomaly_history(limit: int = Query(default=settings.history_default_limit, ge=1)):
    """
    Retrieve the most recent anomaly log entries from Supabase.

    Queries the 'anomaly_logs' table, ordered by timestamp DESC.

    Args:
        limit: Maximum number of records to return (default 50).

    Returns:
        dict: {"logs": list, "count": int}

    """
    requested_limit = limit
    effective_limit = min(requested_limit, settings.history_max_limit)

    try:
        # ── 1. Get the authenticated Supabase client ──
        supabase = get_supabase_client()

        # ── 2. Query the 'anomaly_logs' table ──
        result = (
            supabase.table(settings.anomaly_logs_table_name)
            .select("*")
            .order("timestamp", desc=True)
            .limit(effective_limit)
            .execute()
        )

        return {
            "logs": result.data,
            "count": len(result.data),
            "requested_limit": requested_limit,
            "effective_limit": effective_limit,
            "fallback": False,
        }

    except Exception as error:
        # ── 3. Return a stable fallback payload when Supabase is unavailable ──
        print(f"[api] [WARN] Failed to fetch anomaly history: {error}")
        return {
            "logs": [],
            "count": 0,
            "requested_limit": requested_limit,
            "effective_limit": effective_limit,
            "fallback": True,
            "message": "History unavailable. Returning empty logs.",
        }

