"""
logger.py — Write anomaly events to the Supabase anomaly_logs table.

TRACK B — Owner: You
STATUS: IMPLEMENTED

Depends on:
    - services.supabase_client.get_supabase_client
    - models.TwinState
"""

import json
from datetime import datetime

from models.twin_state import TwinState
from services.supabase_client import get_supabase_client


async def log_anomaly(twin_state: TwinState) -> None:
    """
    Insert an anomaly event into the Supabase 'anomaly_logs' table.

    This function should only be called when twin_state.anomaly_flag is True.
    It serializes the TwinState into a flattened structure for the database.

    Table Schema (Postgres / Supabase):
        timestamp: timestamptz
        anomaly_type: text
        actual_data: jsonb
        expected_data: jsonb
        residual_data: jsonb

    Args:
        twin_state: The TwinState frame that triggered the anomaly.
    """
    try:
        # ── 1. Get the authenticated Supabase client ──
        supabase = get_supabase_client()

        # ── 2. Prepare the database row ──
        # We store the nested sensor blocks as JSONB for flexibility
        row = {
            "timestamp": twin_state.timestamp.isoformat(),
            "anomaly_type": twin_state.anomaly_type,
            "actual_data": twin_state.actual.model_dump(),
            "expected_data": twin_state.expected.model_dump(),
            "residual_data": twin_state.residual.model_dump(),
        }

        # ── 3. Perform the async insert into the 'anomaly_logs' table ──
        # Note: supabase-py handles the serialization to JSON for us
        response = supabase.table("anomaly_logs").insert(row).execute()

        print(f"[logger] Successfully logged {twin_state.anomaly_type} anomaly "
              f"at {twin_state.timestamp}")

    except Exception as error:
        # ── 4. Graceful error handling ──
        # We print the error but don't crash the main stream if logging fails
        print(f"[logger] [ERROR] Failed to log anomaly to Supabase: {error}")

