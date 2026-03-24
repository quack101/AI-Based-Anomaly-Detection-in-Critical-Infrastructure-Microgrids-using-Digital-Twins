"""
logger.py — Write anomaly events to the Supabase anomaly_logs table.

TRACK B — Owner: Teammate
STATUS: SKELETON — implement the function body.

Depends on:
    - services.supabase_client.get_supabase_client
    - models.TwinState
"""

from models.twin_state import TwinState


async def log_anomaly(twin_state: TwinState) -> None:
    """
    Insert an anomaly event into the Supabase 'anomaly_logs' table.

    Should only be called when twin_state.anomaly_flag is True.

    Steps (to implement):
        1. Get the Supabase client
        2. Build the row dict from twin_state fields
        3. Insert into the 'anomaly_logs' table
        4. Handle any Supabase errors gracefully (log, don't crash)

    Args:
        twin_state: The TwinState frame that triggered the anomaly.
    """
    # TODO: Implement — Track B, task B2
    raise NotImplementedError("log_anomaly() not yet implemented")
