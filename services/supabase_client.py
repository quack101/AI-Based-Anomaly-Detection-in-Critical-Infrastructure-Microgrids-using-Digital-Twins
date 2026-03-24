"""
supabase_client.py — Singleton Supabase client.

TRACK B — Owner: Teammate
STATUS: SKELETON — implement the function body.

Depends on:
    - config.settings (supabase_url, supabase_key)
"""

from config import settings


def get_supabase_client():
    """
    Create and return a cached Supabase client instance.

    Steps (to implement):
        1. Import supabase.create_client
        2. Create client using settings.supabase_url and settings.supabase_key
        3. Cache the client so subsequent calls reuse the same instance

    Returns:
        supabase.Client: An authenticated Supabase client.
    """
    # TODO: Implement — Track B, task B1
    raise NotImplementedError("get_supabase_client() not yet implemented")
