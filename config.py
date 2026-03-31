"""
config.py — Centralized application settings.

Reads environment variables from a .env file and exposes them
as typed attributes via Pydantic BaseSettings.
All modules import settings from here instead of reading env vars directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables."""

    # ── Supabase credentials ──
    supabase_url: str = "https://your-project-id.supabase.co"
    supabase_key: str = "your-anon-or-service-role-key"

    # ── Dataset location ──
    csv_path: str = "data/processed_data.csv"

    # ── Twin engine parameters ──
    ema_window: int = 30          # rolling window size for baseline EMA
    anomaly_threshold: float = 3.0  # std-dev multiplier for anomaly flagging

    # ── WebSocket replay speed ──
    ws_frame_delay: float = 1.0   # seconds between frames sent to clients

    # ── Supabase logging behavior ──
    enable_frame_logging_to_supabase: bool = True
    twin_frames_table_name: str = "twin_frames"
    anomaly_logs_table_name: str = "anomaly_logs"

    # ── Frame log batching controls ──
    frame_log_batch_size: int = 100
    frame_log_flush_interval_seconds: float = 1.0
    frame_log_retry_count: int = 3
    frame_log_retry_backoff_seconds: float = 0.5

    # ── History endpoint controls ──
    history_default_limit: int = 50
    history_max_limit: int = 500

    model_config = SettingsConfigDict(
        env_file=".env",            # auto-load from .env if present
        env_file_encoding="utf-8",
    )


# ── Singleton instance used across all modules ──
settings = Settings()
