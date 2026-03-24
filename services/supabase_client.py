"""
supabase_client.py — Singleton Supabase client.

TRACK B — Owner: You
STATUS: IMPLEMENTED

Depends on:
    - config.settings (supabase_url, supabase_key)
"""

from functools import lru_cache

from supabase import create_client, Client

from config import settings


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Create and return a cached Supabase client instance.

    Uses @lru_cache so the client is created exactly once on first call,
    and all subsequent calls return the same instance (singleton pattern).

    Returns:
        supabase.Client: An authenticated Supabase client.

    Raises:
        Exception: If Supabase URL or key are invalid / unreachable.
    """
    # ── Create the Supabase client using project credentials from .env ──
    client: Client = create_client(
        settings.supabase_url,
        settings.supabase_key,
    )
    print(f"[supabase_client] Connected to {settings.supabase_url}")
    return client
