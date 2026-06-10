from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routes.api import router as api_router
from routes.ws import router as ws_router, preload_on_startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    await preload_on_startup()  # loads all 49 CSVs on startup
    yield

app = FastAPI(
    title="Digital Twin — Microgrid Anomaly Detection",
    description=(
        "Real-time Digital Twin engine for AI-based anomaly detection "
        "in critical infrastructure microgrids."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)