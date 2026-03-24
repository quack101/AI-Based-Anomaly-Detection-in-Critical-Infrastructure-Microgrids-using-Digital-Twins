"""
config.py — Centralized application settings.

Reads environment variables from a .env file and exposes them
as typed attributes via Pydantic BaseSettings.
All modules import settings from here instead of reading env vars directly.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables."""

    # ── Supabase credentials ──
    supabase_url: str = "https://your-project-id.supabase.co"
    supabase_key: str = "your-anon-or-service-role-key"

    # ── Dataset location ──
    csv_path: str = "data/household_power_consumption.csv"

    # ── Twin engine parameters ──
    ema_window: int = 30          # rolling window size for baseline EMA
    anomaly_threshold: float = 3.0  # std-dev multiplier for anomaly flagging

    # ── WebSocket replay speed ──
    ws_frame_delay: float = 1.0   # seconds between frames sent to clients

    class Config:
        env_file = ".env"          # auto-load from .env if present
        env_file_encoding = "utf-8"


# ── Singleton instance used across all modules ──
settings = Settings()
