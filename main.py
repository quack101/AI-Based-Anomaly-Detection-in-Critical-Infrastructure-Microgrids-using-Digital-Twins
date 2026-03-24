"""
main.py — FastAPI application entry point.

This file only creates the app and includes routers.
NO business logic belongs here.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Import route modules ──
from routes.api import router as api_router
from routes.ws import router as ws_router

# ── Create the FastAPI app ──
app = FastAPI(
    title="Digital Twin — Microgrid Anomaly Detection",
    description=(
        "Real-time Digital Twin engine for AI-based anomaly detection "
        "in critical infrastructure microgrids."
    ),
    version="0.1.0",
)

# ── CORS middleware (allow Three.js frontend to connect) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──
app.include_router(api_router)   # REST endpoints (Track B)
app.include_router(ws_router)    # WebSocket endpoint (Track B)
