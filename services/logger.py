"""
logger.py — Persist Digital Twin frames and anomaly events to Supabase.

TRACK B — Owner: You
STATUS: IMPLEMENTED

Depends on:
    - config.settings
    - services.supabase_client.get_supabase_client
    - models.TwinState
"""

import asyncio
from typing import Any, Dict, List, Optional

from config import settings
from models.twin_state import TwinState
from services.supabase_client import get_supabase_client


_frame_buffer: List[Dict[str, Any]] = []
_buffer_lock = asyncio.Lock()
_flush_task: Optional[asyncio.Task] = None


def _build_frame_row(twin_state: TwinState) -> Dict[str, Any]:
    """Build a database row payload from a TwinState frame."""
    return {
        "node_id": twin_state.node_id,
        "timestamp": twin_state.timestamp.isoformat(),
        "anomaly_flag": twin_state.anomaly_flag,
        "anomaly_type": twin_state.anomaly_type,
        "actual_data": twin_state.actual.model_dump(),
        "expected_data": twin_state.expected.model_dump(),
        "residual_data": twin_state.residual.model_dump(),
    }


async def _insert_rows_with_retry(table_name: str, rows: List[Dict[str, Any]]) -> bool:
    """Insert rows with configured retry policy and backoff."""
    if not rows:
        return True

    retry_count = max(settings.frame_log_retry_count, 0)
    backoff_seconds = max(settings.frame_log_retry_backoff_seconds, 0.0)

    for attempt_index in range(retry_count + 1):
        try:
            await asyncio.to_thread(_insert_rows_sync, table_name, rows)
            return True
        except Exception as error:
            is_last_attempt = attempt_index >= retry_count
            if is_last_attempt:
                print(
                    f"[logger] [ERROR] Insert failed for table '{table_name}' "
                    f"after {attempt_index + 1} attempts: {error}"
                )
                return False

            sleep_seconds = backoff_seconds * (2 ** attempt_index)
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)

    return False


def _insert_rows_sync(table_name: str, rows: List[Dict[str, Any]]) -> None:
    """Execute Supabase insert in a thread to avoid blocking the event loop."""
    supabase = get_supabase_client()
    supabase.table(table_name).insert(rows).execute()


async def _flush_frame_batch(force: bool = False) -> None:
    """Flush up to configured batch size from the in-memory frame buffer."""
    if not settings.enable_frame_logging_to_supabase:
        return

    batch_size = max(settings.frame_log_batch_size, 1)

    async with _buffer_lock:
        if not _frame_buffer:
            return
        if not force and len(_frame_buffer) < batch_size:
            return

        rows_to_insert = _frame_buffer[:batch_size]
        del _frame_buffer[:batch_size]

    insertion_succeeded = await _insert_rows_with_retry(
        settings.twin_frames_table_name,
        rows_to_insert,
    )

    if not insertion_succeeded:
        print(
            "[logger] [WARN] Dropped "
            f"{len(rows_to_insert)} frame rows after retry exhaustion"
        )


async def _frame_flush_worker() -> None:
    """Background worker that periodically flushes frame batches."""
    flush_interval_seconds = max(settings.frame_log_flush_interval_seconds, 0.1)

    while True:
        await asyncio.sleep(flush_interval_seconds)
        await _flush_frame_batch(force=False)


def _ensure_frame_flush_worker_started() -> None:
    """Start the frame flush worker once for the running process."""
    global _flush_task

    if _flush_task is None or _flush_task.done():
        _flush_task = asyncio.create_task(_frame_flush_worker())


async def log_frame(twin_state: TwinState) -> None:
    """Queue a TwinState frame for batched persistence in Supabase."""
    if not settings.enable_frame_logging_to_supabase:
        return

    _ensure_frame_flush_worker_started()

    frame_row = _build_frame_row(twin_state)
    async with _buffer_lock:
        _frame_buffer.append(frame_row)
        current_buffer_size = len(_frame_buffer)

    if current_buffer_size >= max(settings.frame_log_batch_size, 1):
        await _flush_frame_batch(force=True)


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
    row = {
        "timestamp": twin_state.timestamp.isoformat(),
        "anomaly_type": twin_state.anomaly_type,
        "actual_data": twin_state.actual.model_dump(),
        "expected_data": twin_state.expected.model_dump(),
        "residual_data": twin_state.residual.model_dump(),
    }

    insertion_succeeded = await _insert_rows_with_retry(
        settings.anomaly_logs_table_name,
        [row],
    )

    if insertion_succeeded:
        print(
            f"[logger] Logged {twin_state.anomaly_type} anomaly "
            f"at {twin_state.timestamp}"
        )

